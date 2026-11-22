import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from lib import metrics, scoring
from lib.exam_encoder import LABELS
from config import Config
from encoders.embedding_store import EmbeddingStore
from encoders.exam_images import ExamImages
from encoders.query_metrics import metric_frame
from provenance import ProvenanceRecords, Unit, sha256_file

EVIDENCE_COLUMNS = ["retriever", "test_corpus", "archive", "rule", "level", "finding", "metric", "exam_id", "value"]
DOCUMENTED, UNDOCUMENTED = "documented", "undocumented"
IMAGE_RULES = ["image_image", "two_view_mean"]
IMAGE_REPORT, PROFILE = "image_report", "profile"
LIB_DIR = Path(__file__).resolve().parents[1] / "lib"


def l2_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def row_digest(row: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(row, dtype=np.float64).tobytes()).hexdigest()


class RetrievalInputs:

    def __init__(self, config: Config, corpus_name: str):
        self.config = config
        self.corpus_name = corpus_name
        self.exam_images = ExamImages(config, corpus_name)
        self.store = EmbeddingStore(config, corpus_name)
        exams = self.exam_images.exams()
        test = exams[exams["split"] == "test"]
        self.queries = test[test["query_or_database"] == "query"].reset_index(drop=True)
        self.database = test[test["query_or_database"] == "database"].reset_index(drop=True)
        self.entities_dir = config.path("paths.cohort_dir") / corpus_name / "radgraph"
        self.probabilities_file = config.path("paths.retrieval_dir") / corpus_name / "classifier_probabilities.npz"
        self.empty_code = config.get("retrieval.evaluation.empty_label_code")
        self.cache = {}

    def side(self, side: str) -> pd.DataFrame:
        return self.queries if side == "query" else self.database

    def label_codes(self, side: str) -> np.ndarray:
        cells = self.side(side)[LABELS]
        return cells.replace("", str(self.empty_code)).astype(int).to_numpy()

    def label_cells(self, side: str) -> list[str]:
        return self.side(side)[LABELS].agg(",".join, axis=1).tolist()

    def profile_labels(self) -> np.ndarray:
        return (self.database[LABELS] == "1").astype(float).to_numpy()

    def label_mask(self) -> np.ndarray:
        test = pd.concat([self.queries, self.database])
        return (test[LABELS] != "").any(axis=0).to_numpy()

    def images(self, encoder: str, side: str, two_view: bool) -> np.ndarray:
        key = ("images", encoder)
        if key not in self.cache:
            selection = self.exam_images.selection()
            selection = selection[selection["split"] == "test"]
            stored = self.store.images(encoder)
            self.cache[key] = {view: pd.DataFrame(stored.loc[rows["image_id"]].to_numpy(), index=rows["exam_id"])
                               for view, rows in selection.groupby("view")}
        views = self.cache[key]
        exam_ids = self.side(side)["exam_id"]
        frontal = views["frontal"].loc[exam_ids].to_numpy()
        if not two_view or "lateral" not in views:
            return frontal
        lateral = views["lateral"].reindex(exam_ids)
        has_lateral = lateral.notna().all(axis=1).to_numpy()
        combined = frontal.copy()
        combined[has_lateral] = l2_rows((frontal[has_lateral] + lateral.to_numpy()[has_lateral]) / 2)
        return combined

    def reports(self, encoder: str) -> np.ndarray:
        ids, embeddings = self.store.load(self.store.text_file(encoder, "reports"))
        frame = pd.DataFrame(embeddings, index=ids)
        missing = ~self.database["exam_id"].isin(frame.index)
        if missing.any():
            raise ValueError(f"{self.corpus_name}: {int(missing.sum())} database exams have no {encoder} report")
        return frame.loc[self.database["exam_id"]].to_numpy()

    def probabilities(self, side: str) -> np.ndarray:
        if "probabilities" not in self.cache:
            with np.load(self.probabilities_file) as data:
                if data["labels"].tolist() != LABELS:
                    raise ValueError(f"{self.probabilities_file} does not hold the labels in LABELS order")
                self.cache["probabilities"] = pd.DataFrame(data["probabilities"], index=data["exam_ids"])
        return self.cache["probabilities"].loc[self.side(side)["exam_id"]].to_numpy()

    def entities(self, side: str) -> list[frozenset]:
        if "entities" not in self.cache:
            sets = {}
            for path in sorted(self.entities_dir.glob("entities_[0-9][0-9][0-9][0-9][0-9].jsonl")):
                with open(path) as handle:
                    for line in handle:
                        row = json.loads(line)
                        sets[row["exam_id"]] = frozenset(tuple(entity) for entity in row["entities"])
            self.cache["entities"] = sets
        exam_ids = self.side(side)["exam_id"]
        missing = int((~exam_ids.isin(self.cache["entities"].keys())).sum())
        if missing:
            raise ValueError(f"{self.corpus_name}: {missing} {side} exams have no entity set")
        return [self.cache["entities"][exam_id] for exam_id in exam_ids]


def evaluate_unit(spec: dict, scores_input: dict, labels: dict, report: dict | None, evaluation: dict,
                  out_path: Path) -> int:
    if spec["rule"] == PROFILE:
        scores = scoring.profile_scores(scores_input["query"], scores_input["archive"],
                                        finding_mask=scores_input["mask"])
    else:
        scores = scoring.alignment_scores(scores_input["query"], scores_input["archive"])
    evidence = metric_frame(scores, labels, report, evaluation)
    for column in ["retriever", "test_corpus", "archive", "rule"]:
        evidence[column] = spec[column]
    partial = out_path.with_name(out_path.name + ".partial")
    with open(partial, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
        evidence[EVIDENCE_COLUMNS].to_csv(handle, index=False)
    os.replace(partial, out_path)
    return len(evidence)


def write_unit(spec, scores_input, labels, report, evaluation, unit, records) -> tuple[str, int]:
    n_rows = evaluate_unit(spec, scores_input, labels, report, evaluation, unit.outputs[0])
    records.write(unit)
    return unit.name, n_rows


class Retrieval:

    step = "retrieval"

    def __init__(self, config: Config, corpus_name: str, inputs: RetrievalInputs | None = None):
        self.config = config
        self.corpus_name = corpus_name
        self.inputs = RetrievalInputs(config, corpus_name) if inputs is None else inputs
        self.evaluation = config.get("retrieval.evaluation")
        self.out_dir = config.path("paths.retrieval_dir") / corpus_name / "evidence"
        self.documented = corpus_name in config.get("retrieval.documented_corpora")
        self.classifier = config.get("retrieval.classifier_retriever")

    @staticmethod
    def step_settings(config: Config, archive: str, rule: str, mask: list[bool] | None) -> dict:
        evaluation = {key: value for key, value in config.get("retrieval.evaluation").items() if key != "n_jobs"}
        return {"evaluation": evaluation, "archive": archive, "rule": rule, "label_mask": mask,
                "lib_sha256": {name: sha256_file(LIB_DIR / name) for name in ("metrics.py", "scoring.py")}}

    def specs(self) -> list[dict]:
        archives = [DOCUMENTED, UNDOCUMENTED] if self.documented else [UNDOCUMENTED]
        specs = []
        for encoder, settings in self.config.get("retrieval.encoders").items():
            for archive in archives:
                rules = IMAGE_RULES + ([IMAGE_REPORT] if archive == DOCUMENTED and "text_max_tokens" in settings else [])
                specs += [{"retriever": encoder, "archive": archive, "rule": rule} for rule in rules]
        specs += [{"retriever": self.classifier, "archive": archive, "rule": PROFILE} for archive in archives]
        return [dict(spec, test_corpus=self.corpus_name) for spec in specs]

    @staticmethod
    def unit_name(spec: dict) -> str:
        return f"{spec['retriever']}__{spec['archive']}__{spec['rule']}"

    def scores_input(self, spec: dict) -> dict:
        inputs, rule, retriever = self.inputs, spec["rule"], spec["retriever"]
        if rule == PROFILE:
            archive = inputs.profile_labels() if spec["archive"] == DOCUMENTED else inputs.probabilities("database")
            return {"query": inputs.probabilities("query"), "archive": archive, "mask": inputs.label_mask()}
        query = inputs.images(retriever, "query", two_view=rule == "two_view_mean")
        if rule == IMAGE_REPORT:
            return {"query": query, "archive": inputs.reports(retriever)}
        return {"query": query, "archive": inputs.images(retriever, "database", two_view=rule == "two_view_mean")}

    def labels(self) -> dict:
        return {"query": self.inputs.label_codes("query"), "archive": self.inputs.label_codes("database"),
                "query_ids": self.inputs.queries["exam_id"].tolist()}

    def report(self) -> dict | None:
        if not self.documented:
            return None
        if "report" not in self.inputs.cache:
            query_sets, archive_sets = self.inputs.entities("query"), self.inputs.entities("database")
            keep = np.array([len(entities) > 0 for entities in query_sets])
            gains = metrics.entity_f1_matrix([s for s, k in zip(query_sets, keep) if k], archive_sets)
            self.inputs.cache["report"] = {"keep": keep, "gains": gains}
        return self.inputs.cache["report"]

    def unit(self, spec: dict, scores_input: dict) -> tuple[Unit, ProvenanceRecords]:
        mask = scores_input.get("mask")
        records = ProvenanceRecords(self.out_dir, self.step, self.step_settings(
            self.config, spec["archive"], spec["rule"], None if mask is None else mask.tolist()))
        ids, contents = [], []
        for side, matrix in (("query", scores_input["query"]), ("database", scores_input["archive"])):
            entities = self.inputs.entities(side) if self.documented else None
            for position, (exam_id, cells) in enumerate(zip(self.inputs.side(side)["exam_id"],
                                                             self.inputs.label_cells(side))):
                entity_part = "" if entities is None else ";".join(sorted("|".join(e) for e in entities[position]))
                ids.append(f"{side}:{exam_id}")
                contents.append(f"{row_digest(matrix[position])}|{cells}|{entity_part}")
        name = self.unit_name(spec)
        return Unit(name, ids, contents, [self.out_dir / f"{name}.csv.gz"]), records

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        labels, report = self.labels(), self.report()
        jobs = []
        for spec in self.specs():
            scores_input = self.scores_input(spec)
            unit, records = self.unit(spec, scores_input)
            if records.pending([unit], f"{self.corpus_name} {unit.name}"):
                jobs.append(delayed(write_unit)(spec, scores_input, labels, report, self.evaluation, unit, records))
        if jobs:
            Parallel(n_jobs=self.evaluation["n_jobs"])(jobs)
