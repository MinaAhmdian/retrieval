import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lib.exam_encoder import LABELS
from config import Config
from controls.permutations import LateralPermutations
from encoders.models import compute_device
from provenance import FileDigests, ProvenanceRecords, Unit, sha256_file
from trained_encoder.backbones import backbone_settings
from trained_encoder.context_permutations import ContextPermutations
from trained_encoder.exam_batches import ExamItems, collate
from trained_encoder.exam_inputs import SLOT_COLUMNS, VIEW_COLUMNS, ExamInputs
from trained_encoder.exam_model import ExamModel, build_model, configuration_settings
from trained_encoder.fit_data import ContextEmbeddings
from trained_encoder.slot_images import SlotImages
from trained_encoder.runs import all_run_names, run_root
from trained_encoder.trainer import LIB_DIR

SIDES = ("query", "database")
DATABASE_INPUTS = "all"


def slot_paths(table: pd.DataFrame, png: pd.Series) -> np.ndarray:
    paths = np.full((len(table), len(SLOT_COLUMNS)), "", dtype=object)
    for slot, column in enumerate(SLOT_COLUMNS):
        filled = (table[column] != "").to_numpy()
        if not table.loc[filled, column].isin(png.index).all():
            raise ValueError("a slot image has no resized PNG")
        paths[filled, slot] = png.loc[table.loc[filled, column]].to_numpy()
    return paths


class TestExams:

    split = "test"

    def __init__(self, config: Config, corpus: str, configuration: str, query_inputs: str, side: str):
        self.config, self.corpus = config, corpus
        slot_images = SlotImages(config, corpus)
        exams = self.split_exams(slot_images)
        slots = slot_images.slots()
        inputs = ExamInputs(config)
        table = inputs.base(exams, slots[slots["exam_id"].isin(exams["exam_id"])])
        operations = config.get(f"trained_encoder.input_configurations.{configuration}")
        donors, lateral_donors = None, None
        if operations.get("context_from_donor"):
            donors = ContextPermutations(config).donors(corpus, self.split)
        if operations.get("lateral_from_donor"):
            lateral_donors = LateralPermutations(config).donors_of(corpus, self.split, table)
        table = inputs.configured(table, configuration, query_inputs, donors=donors, lateral_donors=lateral_donors)
        self.table = table[self.side_mask(exams, side)].reset_index(drop=True)
        self.context, self.context_mask = ContextEmbeddings(config, corpus).rows(self.table["context_exam"])
        png = slot_images.resized().set_index("image_id")["png_path"]
        self.slot_paths = slot_paths(self.table, png)
        self.labels = np.zeros((len(self.table), len(LABELS)), dtype=np.int64)
        self.reports, self.report_keys = None, None

    def split_exams(self, slot_images: SlotImages) -> pd.DataFrame:
        exams = slot_images.exams()
        return exams[exams["split"] == self.split].reset_index(drop=True)

    def side_mask(self, exams: pd.DataFrame, side: str) -> np.ndarray:
        return (exams["query_or_database"] == side).to_numpy()

    def __len__(self) -> int:
        return len(self.table)

    def contents(self, digests: FileDigests) -> list[str]:
        paths = [path for path in self.slot_paths.ravel() if path]
        checksums = dict(zip(paths, digests.digests(paths)))
        rows = []
        for i, row in enumerate(self.table.itertuples(index=False)):
            context = (hashlib.sha256(self.context[i].tobytes()).hexdigest()[:16] if self.context_mask[i] else "")
            rows.append(f"{','.join(checksums.get(p, '') for p in self.slot_paths[i])}|"
                        f"{','.join(str(getattr(row, c)) for c in VIEW_COLUMNS)}|{context}|{row.age_id},{row.sex_id}")
        return rows


def input_keys(exams) -> list[str]:
    keys = []
    for i, row in enumerate(exams.table.itertuples(index=False)):
        context = hashlib.sha256(exams.context[i].tobytes()).hexdigest() if exams.context_mask[i] else ""
        keys.append(f"{','.join(exams.slot_paths[i])}|{','.join(str(getattr(row, c)) for c in VIEW_COLUMNS)}|"
                    f"{context}|{row.age_id},{row.sex_id}")
    return keys


def identical_inputs(target, source) -> np.ndarray:
    if target.table["exam_id"].tolist() != source.table["exam_id"].tolist():
        raise ValueError("the two query inputs do not hold the same exams in the same order")
    return np.array([a == b for a, b in zip(input_keys(target), input_keys(source))], dtype=bool)


def reuse_rows(outputs: tuple[np.ndarray, ...], source: tuple[np.ndarray, ...], same: np.ndarray) -> tuple:
    return tuple(np.where(same[:, None], s, o) for o, s in zip(outputs, source))


class TestEmbedder:

    step = "embed-test-exams"

    def __init__(self, config: Config, run: str):
        if run not in all_run_names(config):
            raise ValueError(f"{run} is not a run")
        self.config = config
        self.run_name = run
        self.backbone_name, self.configuration, _ = run.split("__")
        self.settings = config.get("trained_encoder.test_outputs")
        root = run_root(config, run)
        self.chosen = root / config.get("trained_encoder.training.folder") / run / "chosen.pt"
        self.out_dir = root / self.settings["folder"] / run
        self.query_inputs = config.get(f"trained_encoder.test_query_inputs.{self.configuration}")
        self.reuse = {target: source for target, source in self.settings.get("reuse_outputs", {}).items()
                      if target in self.query_inputs}
        for target, source in self.reuse.items():
            if source not in self.query_inputs or self.query_inputs.index(source) > self.query_inputs.index(target):
                raise ValueError(f"reuse_outputs: {source} must be a query input of {self.configuration} listed "
                                 f"before {target}")
        head = config.get("controls.replacement_heads.configurations").get(self.configuration, {})
        self.capture_norm = head.get("features") == "norm"
        if self.capture_norm and self.reuse:
            raise ValueError(f"{self.configuration}: reuse_outputs cannot be combined with stored norm features")

    def corpora(self) -> list[str]:
        return self.config.get("cohort.corpora")

    def units_of(self, corpus: str) -> list[tuple[str, str]]:
        return [("query", name) for name in self.query_inputs] + [("database", DATABASE_INPUTS)]

    @staticmethod
    def unit_name(corpus: str, side: str, query_inputs: str) -> str:
        return f"{corpus}__{side}__{query_inputs}"

    def path(self, corpus: str, side: str, query_inputs: str) -> Path:
        return self.out_dir / f"{self.unit_name(corpus, side, query_inputs)}.npz"

    def reuse_source(self, side: str, query_inputs: str) -> str | None:
        return self.reuse.get(query_inputs) if side == "query" else None

    def step_settings(self, query_inputs: str, corpus: str | None = None, side: str = "query") -> dict:
        settings = self.base_settings(query_inputs)
        source = self.reuse_source(side, query_inputs)
        if source is not None:
            settings["reuse_outputs_of"] = source
            settings["reused_unit_sha256"] = sha256_file(self.path(corpus, "query", source))
        return settings

    def base_settings(self, query_inputs: str) -> dict:
        settings = {**self.common_settings(query_inputs), **configuration_settings(self.config, self.configuration)}
        if self.capture_norm:
            settings["stored_features"] = "norm"
        return settings

    def common_settings(self, query_inputs: str) -> dict:
        return {"run": self.run_name, "chosen_sha256": sha256_file(self.chosen),
                "configuration": self.config.get(f"trained_encoder.input_configurations.{self.configuration}"),
                "query_inputs": self.config.get(f"trained_encoder.query_inputs.{query_inputs}"),
                "backbone": backbone_settings(self.config, self.backbone_name, self.configuration),
                "encoder": self.config.get("trained_encoder.encoder"),
                "mixed_precision": self.config.get("trained_encoder.training.mixed_precision"),
                "context_permutation_seed": self.config.get("trained_encoder.context_permutation.seed"),
                "lib_sha256": sha256_file(LIB_DIR / "exam_encoder.py")}

    def exams(self, corpus: str, side: str, query_inputs: str) -> TestExams:
        return TestExams(self.config, corpus, self.configuration, query_inputs, side)

    def load_model(self, device) -> ExamModel:
        model = build_model(self.config, self.backbone_name, self.configuration)
        model.load_weights(torch.load(self.chosen, map_location="cpu", weights_only=False)["weights"])
        return model.to(device).eval()

    def embed(self, model: ExamModel, exams: TestExams, device) -> tuple[np.ndarray, np.ndarray]:
        outputs = self.embed_outputs(model, exams, device)
        return outputs["alignment"], outputs["logits"]

    @torch.no_grad()
    def embed_outputs(self, model: ExamModel, exams, device) -> dict[str, np.ndarray]:
        loader = DataLoader(ExamItems(exams, model.transform), batch_size=self.settings["batch_size"],
                            num_workers=self.settings["num_workers"], collate_fn=collate,
                            pin_memory=device.type == "cuda")
        dtype = self.config.get("trained_encoder.training.mixed_precision")
        parts = {"alignment": [], "logits": [], "features": []}
        hook = None
        if self.capture_norm:
            hook = model.encoder.norm.register_forward_hook(
                lambda module, inputs, output: parts["features"].append(output.float().cpu().numpy()))
        try:
            for batch in loader:
                batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                         for key, value in batch.items()}
                with torch.autocast(device.type, dtype=getattr(torch, dtype) if dtype else None,
                                    enabled=bool(dtype)):
                    alignment, logit = model(batch)
                parts["alignment"].append(alignment.float().cpu().numpy())
                parts["logits"].append(logit.float().cpu().numpy())
        finally:
            if hook is not None:
                hook.remove()
        outputs = {"alignment": np.concatenate(parts["alignment"]), "logits": np.concatenate(parts["logits"])}
        if self.capture_norm:
            outputs["features"] = np.concatenate(parts["features"])
            if len(outputs["features"]) != len(outputs["alignment"]):
                raise RuntimeError("the norm hook did not run once per batch")
        return outputs

    def run(self) -> None:
        if not self.chosen.exists():
            raise FileNotFoundError(f"[{self.run_name}] no chosen weights; train the run first")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        digests = FileDigests(self.config.get("provenance.digest_workers"))
        model, device = None, None
        for corpus in self.corpora():
            for side, query_inputs in self.units_of(corpus):
                exams = self.exams(corpus, side, query_inputs)
                ids = exams.table["exam_id"].tolist()
                if not ids:
                    raise ValueError(f"[{self.run_name} {self.unit_name(corpus, side, query_inputs)}] no exams")
                unit = Unit(self.unit_name(corpus, side, query_inputs), ids, exams.contents(digests),
                            [self.path(corpus, side, query_inputs)])
                source = self.reuse_source(side, query_inputs)
                records = ProvenanceRecords(self.out_dir, self.step, self.step_settings(query_inputs, corpus, side))
                if records.pending([unit], f"{self.run_name} {unit.name}"):
                    if model is None:
                        device = compute_device(self.settings["device"])
                        model = self.load_model(device)
                    outputs = self.embed_outputs(model, exams, device)
                    alignment, logits = outputs["alignment"], outputs["logits"]
                    extra = {"features": outputs["features"]} if "features" in outputs else {}
                    reused = np.zeros(len(ids), dtype=bool)
                    if source is not None:
                        reused = identical_inputs(exams, self.exams(corpus, side, source))
                        source_outputs = load_outputs(self.path(corpus, side, source), exams.table["exam_id"])
                        alignment, logits = reuse_rows((alignment, logits), source_outputs, reused)
                    partial = unit.outputs[0].with_name(unit.outputs[0].stem + ".partial.npz")
                    np.savez(partial, exam_ids=np.array(ids, dtype=str), alignment=alignment, logits=logits,
                             reused=reused, **extra)
                    os.replace(partial, unit.outputs[0])
                    records.write(unit)


def load_outputs(path: Path, exam_ids: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as saved:
        if saved["exam_ids"].tolist() != exam_ids.tolist():
            raise ValueError(f"{path.name} does not hold these exams in this order")
        return saved["alignment"], saved["logits"]


def load_features(path: Path, exam_ids: pd.Series) -> np.ndarray:
    with np.load(path) as saved:
        if saved["exam_ids"].tolist() != exam_ids.tolist():
            raise ValueError(f"{path.name} does not hold these exams in this order")
        if "features" not in saved:
            raise ValueError(f"{path.name} holds no features")
        return saved["features"]
