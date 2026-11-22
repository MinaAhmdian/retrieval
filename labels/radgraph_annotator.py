import json

import pandas as pd

from config import Config
from labels.report_selection import ReportSelection
from provenance import ProvenanceRecords, Unit, chunked


class RadGraphAnnotator:

    step = "radgraph"

    def __init__(self, config: Config, corpus_name: str):
        self.corpus_name = corpus_name
        self.selection = ReportSelection(config, corpus_name)
        self.out_dir = config.path("paths.cohort_dir") / corpus_name / "radgraph"
        self.settings = config.get("cohort.radgraph")
        self.records = ProvenanceRecords(self.out_dir, self.step, self.step_settings(config, corpus_name))

    @staticmethod
    def step_settings(config: Config, corpus_name: str) -> dict:
        return {"model_type": config.get("cohort.radgraph.model_type"),
                "chunk_size": config.get("cohort.radgraph.chunk_size")}

    def reports(self) -> pd.DataFrame:
        return self.selection.test_reports()

    def units(self, reports: pd.DataFrame | None = None) -> list[Unit]:
        reports = self.reports() if reports is None else reports
        return [Unit(f"counts_{chunk:05d}", part["exam_id"], part["label_text"],
                     [self.out_dir / f"entities_{chunk:05d}.jsonl", self.out_dir / f"counts_{chunk:05d}.csv"])
                for chunk, part in chunked(reports, self.settings["chunk_size"])]

    @staticmethod
    def log_key(unit: Unit) -> str:
        return str(int(unit.name.split("_")[-1]))

    def run(self) -> None:
        reports = self.reports()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        units = self.units(reports)
        todo = self.records.pending(units, self.corpus_name)
        if todo:
            from radgraph import RadGraph
            model = RadGraph(model_type=self.settings["model_type"])
            parts = dict(chunked(reports, self.settings["chunk_size"]))
            for unit in todo:
                self.annotate_chunk(model, unit, parts[int(self.log_key(unit))])
                self.records.write(unit)
        counts = pd.concat(pd.read_csv(unit.outputs[1], dtype={"exam_id": str}) for unit in units)
        if counts["exam_id"].tolist() != reports["exam_id"].tolist():
            raise RuntimeError(f"merged RadGraph output has {len(counts)} rows for {len(reports)} reports")
        counts.to_csv(self.out_dir.parent / "radgraph_counts.csv", index=False)

    def annotate_chunk(self, model, unit: Unit, part: pd.DataFrame) -> None:
        texts = part["label_text"].tolist()
        non_empty = [i for i, text in enumerate(texts) if text.strip()]
        output = model([texts[i] for i in non_empty]) if non_empty else {}
        entity_sets = [set() for _ in texts]
        for position, i in enumerate(non_empty):
            for entity in output[str(position)]["entities"].values():
                entity_sets[i].add((entity["tokens"].lower(), entity["label"]))
        entities_path, counts_path = unit.outputs
        with open(entities_path, "w") as handle:
            for exam_id, entities in zip(part["exam_id"], entity_sets):
                handle.write(json.dumps({"exam_id": exam_id, "entities": sorted(entities)}) + "\n")
        pd.DataFrame({"exam_id": part["exam_id"].to_numpy(), "n_entities": [len(s) for s in entity_sets]}) \
            .to_csv(counts_path, index=False)
