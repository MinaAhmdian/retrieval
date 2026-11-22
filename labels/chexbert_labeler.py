import csv
import shutil
import sys

import pandas as pd

from config import Config
from labels.report_selection import ReportSelection
from provenance import ProvenanceRecords, Unit, chunked, sha256_file

CHEXBERT_TO_LABEL = {
    "Enlarged Cardiomediastinum": "enlarged_cardiomediastinum", "Cardiomegaly": "cardiomegaly",
    "Lung Opacity": "lung_opacity", "Lung Lesion": "lung_lesion", "Edema": "edema",
    "Consolidation": "consolidation", "Pneumonia": "pneumonia", "Atelectasis": "atelectasis",
    "Pneumothorax": "pneumothorax", "Pleural Effusion": "pleural_effusion", "Pleural Other": "pleural_other",
    "Fracture": "fracture", "Support Devices": "support_devices", "No Finding": "no_finding",
}
RAW_VALUES = {"1.0", "0.0", "-1.0", ""}


class CheXbertLabeler:

    step = "chexbert"

    def __init__(self, config: Config, corpus_name: str):
        self.config = config
        self.corpus_name = corpus_name
        self.selection = ReportSelection(config, corpus_name)
        self.out_dir = config.path("paths.cohort_dir") / corpus_name / "chexbert"
        self.chunk_size = config.get("cohort.chexbert.chunk_size")
        self.checkpoint = config.path("tools.chexbert.checkpoint")
        self.records = ProvenanceRecords(self.out_dir, self.step, self.step_settings(config, corpus_name))

    @staticmethod
    def step_settings(config: Config, corpus_name: str) -> dict:
        return {"checkpoint_sha256": config.get("tools.chexbert.checkpoint_sha256"),
                "repo_src": config.get("tools.chexbert.repo_src"), "chunk_size": config.get("cohort.chexbert.chunk_size")}

    def units(self, reports: pd.DataFrame | None = None) -> list[Unit]:
        reports = self.selection.all_reports() if reports is None else reports
        return [Unit(f"labels_{chunk:05d}", part["exam_id"], part["label_text"],
                     [self.out_dir / f"labels_{chunk:05d}.csv"]) for chunk, part in chunked(reports, self.chunk_size)]

    @staticmethod
    def log_key(unit: Unit) -> str:
        return str(int(unit.name.split("_")[-1]))

    def run(self) -> None:
        reports = self.selection.all_reports()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        units = self.units(reports)
        todo = self.records.pending(units, self.corpus_name)
        if todo:
            label_module = self.chexbert_module()
            parts = dict(chunked(reports, self.chunk_size))
            for unit in todo:
                part = parts[int(self.log_key(unit))]
                self.label_chunk(label_module, unit, part)
                self.records.write(unit)
        merged = pd.concat(pd.read_csv(unit.outputs[0], dtype={"exam_id": str}) for unit in units)
        if merged["exam_id"].tolist() != reports["exam_id"].tolist():
            raise RuntimeError(f"merged CheXbert output has {len(merged)} rows for {len(reports)} reports")
        merged.to_csv(self.out_dir.parent / "chexbert_raw.csv", index=False)

    def chexbert_module(self):
        if sha256_file(self.checkpoint) != self.config.get("tools.chexbert.checkpoint_sha256"):
            raise RuntimeError(f"{self.checkpoint} is not the configured CheXbert checkpoint")
        sys.path.insert(0, str(self.config.path("tools.chexbert.repo_src")))
        import label as label_module
        return label_module

    def label_chunk(self, label_module, unit: Unit, part: pd.DataFrame) -> None:
        output = self.chexbert_outputs(label_module, unit, part)
        output.insert(0, "exam_id", part["exam_id"].to_numpy())
        output.to_csv(unit.outputs[0], index=False)

    def chexbert_outputs(self, label_module, unit: Unit, part: pd.DataFrame) -> pd.DataFrame:
        work_dir = self.out_dir / f"work_{unit.name}"
        work_dir.mkdir(exist_ok=True)
        input_csv = work_dir / "reports.csv"
        texts = part["label_text"].where(part["label_text"].str.strip() != "", " ")
        texts.to_frame("Report Impression").to_csv(input_csv, index=False, quoting=csv.QUOTE_ALL)
        predictions = label_module.label(str(self.checkpoint), str(input_csv))
        label_module.save_preds(predictions, str(input_csv), str(work_dir))
        labelled = pd.read_csv(work_dir / "labeled_reports.csv")
        if len(labelled) != len(part):
            raise RuntimeError(f"CheXbert returned {len(labelled)} rows for {len(part)} reports")
        shutil.rmtree(work_dir)
        return labelled[list(CHEXBERT_TO_LABEL)].rename(columns=CHEXBERT_TO_LABEL)
