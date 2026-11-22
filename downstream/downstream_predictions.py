import gzip
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from lib.exam_encoder import LABELS
from config import Config
from controls.controls_retrieval import EvidenceUnits
from encoders.retrieval import DOCUMENTED, UNDOCUMENTED, LIB_DIR, RetrievalInputs, row_digest
from provenance import ProvenanceRecords, Unit, sha256_file
from trained_encoder.test_outputs import TestEmbedder
from trained_encoder.trained_retrieval import PROFILE, RULE_PARTS, TWO_STAGE, rule_scores

EVIDENCE_COLUMNS = ["retriever", "seed", "test_corpus", "archive", "rule", "k", "task", "finding", "exam_id", "target",
                    "score"]
NO_SEED = "none"
NO_DEPTH = 0


def downstream_dir(config: Config) -> Path:
    return config.path("paths.downstream_dir")


def unit_name(spec: dict) -> str:
    return f"{spec['retriever']}__{spec['seed']}__{spec['archive']}__{spec['task']}"


def neighbor_positions(scores: np.ndarray, n: int) -> np.ndarray:
    return np.argsort(-np.asarray(scores, dtype=np.float64), axis=1, kind="stable")[:, :n]


def predictions(vectors: np.ndarray, neighbors: np.ndarray, mean: bool) -> np.ndarray:
    return vectors[neighbors].mean(axis=1) if mean else vectors[neighbors[:, 0]]


def evidence_rows(spec: dict, query_ids: pd.Series, codes: np.ndarray, mask: np.ndarray,
                  predicted: np.ndarray) -> pd.DataFrame:
    valid = (codes == 0) | (codes == 1)
    valid &= mask[None, :]
    rows, columns = np.nonzero(valid)
    frame = pd.DataFrame({"finding": np.asarray(LABELS)[columns], "exam_id": query_ids.to_numpy()[rows],
                          "target": codes[rows, columns].astype(int), "score": predicted[rows, columns]})
    for column in ("retriever", "seed", "test_corpus", "archive", "rule", "k", "task"):
        frame[column] = spec[column]
    return frame[EVIDENCE_COLUMNS]


def write_unit(spec: dict, query: dict, archive: dict, vectors: np.ndarray, mask: np.ndarray, codes: np.ndarray,
               query_ids: pd.Series, database_ids: pd.Series, unit: Unit, records: ProvenanceRecords) -> tuple[str, int]:
    scores = rule_scores(spec["rule"], query, archive, mask, spec["k"])
    neighbors = neighbor_positions(scores, spec["neighbors"])
    predicted = predictions(vectors, neighbors, spec["mean"])
    evidence = evidence_rows(spec, query_ids, codes, mask, predicted)
    evidence_path, neighbors_path = unit.outputs
    partial = evidence_path.with_name(evidence_path.name + ".partial")
    with open(partial, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
        evidence.to_csv(handle, index=False)
    os.replace(partial, evidence_path)
    partial = neighbors_path.with_name(neighbors_path.stem + ".partial.npz")
    np.savez(partial, query_ids=query_ids.to_numpy(dtype=str), database_ids=database_ids.to_numpy(dtype=str),
             neighbors=neighbors, scores=np.take_along_axis(scores, neighbors, axis=1))
    os.replace(partial, neighbors_path)
    records.write(unit)
    return unit.name, len(evidence)


class DownstreamPredictions:

    step = "downstream-predictions"

    def __init__(self, config: Config, corpus_name: str):
        from controls.depth_selection import DepthSelection
        self.config = config
        self.settings = config.get("downstream")
        if corpus_name not in self.settings["test_corpora"]:
            raise ValueError(f"{corpus_name} is not one of downstream.test_corpora")
        self.corpus_name = corpus_name
        self.inputs = RetrievalInputs(config, corpus_name)
        self.documented = corpus_name in config.get("retrieval.documented_corpora")
        self.depth = DepthSelection(config).depth()
        self.out_dir = downstream_dir(config) / self.settings["evidence_folder"] / corpus_name
        self.report_encoder = config.get("trained_encoder.retrieval.documented_report_encoder")
        self.classifier = config.get("retrieval.classifier_retriever")
        self.cache = {}


    def specs(self) -> list[dict]:
        archives = [DOCUMENTED, UNDOCUMENTED] if self.documented else [UNDOCUMENTED]
        methods = [{"retriever": self.settings["encoder_retriever"], "seed": run.split("__")[2], "run": run,
                    "rule": TWO_STAGE, "k": self.depth} for run in self.settings["encoder_runs"]]
        methods.append({"retriever": self.classifier, "seed": NO_SEED, "run": None, "rule": PROFILE, "k": NO_DEPTH})
        specs = []
        for method in methods:
            for task, task_settings in self.settings["tasks"].items():
                for archive in task_settings["archives"]:
                    if archive in archives:
                        specs.append({**method, "test_corpus": self.corpus_name, "archive": archive, "task": task,
                                      "neighbors": task_settings["neighbors"], "mean": task_settings["mean"],
                                      "query_inputs": self.settings["query_inputs"]})
        return specs

    def paths(self, spec: dict) -> list[Path]:
        name = unit_name(spec)
        return [self.out_dir / f"{name}.csv.gz", self.out_dir / f"{name}__neighbors.npz"]

    def needed(self, spec: dict) -> list[Path]:
        if spec["run"] is None:
            return [self.inputs.probabilities_file]
        embedder = TestEmbedder(self.config, spec["run"])
        paths = [embedder.path(self.corpus_name, "query", spec["query_inputs"])]
        if spec["archive"] == UNDOCUMENTED:
            paths.append(embedder.path(self.corpus_name, "database", "all"))
        return paths

    def sides(self, spec: dict) -> tuple[dict, dict, np.ndarray]:
        queries, database = self.inputs.queries["exam_id"], self.inputs.database["exam_id"]
        if spec["run"] is None:
            query = {"profile": self.inputs.probabilities("query")}
            archive = {"profile": self.labels() if spec["archive"] == DOCUMENTED
                       else self.inputs.probabilities("database")}
            return query, archive, archive["profile"]
        query = EvidenceUnits.model_side(self.needed(spec)[0], queries)
        if spec["archive"] == DOCUMENTED:
            archive = {"profile": self.labels(), "alignment": self.reports()}
        else:
            archive = EvidenceUnits.model_side(self.needed(spec)[1], database)
        return query, archive, archive["profile"]

    def labels(self) -> np.ndarray:
        if "labels" not in self.cache:
            self.cache["labels"] = self.inputs.profile_labels()
        return self.cache["labels"]

    def reports(self) -> np.ndarray:
        if "reports" not in self.cache:
            self.cache["reports"] = self.inputs.reports(self.report_encoder)
        return self.cache["reports"]

    def step_settings(self, spec: dict, mask: np.ndarray) -> dict:
        return {**{key: spec[key] for key in ("retriever", "seed", "run", "test_corpus", "archive", "rule", "k", "task",
                                              "neighbors", "mean", "query_inputs")},
                "label_mask": mask.tolist(), "documented_report_encoder": self.report_encoder,
                "lib_sha256": {name: sha256_file(LIB_DIR / name) for name in ("metrics.py", "scoring.py")}}

    def unit(self, spec: dict, query: dict, archive: dict, mask: np.ndarray) -> tuple[Unit, ProvenanceRecords]:
        records = ProvenanceRecords(self.out_dir, self.step, self.step_settings(spec, mask))
        parts = RULE_PARTS[spec["rule"]]
        ids, contents = [], []
        for side, matrices in (("query", query), ("database", archive)):
            for position, (exam_id, cells) in enumerate(zip(self.inputs.side(side)["exam_id"],
                                                             self.inputs.label_cells(side))):
                ids.append(f"{side}:{exam_id}")
                contents.append(",".join(row_digest(matrices[part][position]) for part in parts) + f"|{cells}")
        return Unit(unit_name(spec), ids, contents, self.paths(spec)), records

    def status(self, spec: dict, mask: np.ndarray) -> tuple[str, str]:
        query, archive, _ = self.sides(spec)
        unit, records = self.unit(spec, query, archive, mask)
        record = records.read(unit)
        if record is None:
            return "no record", ""
        reason = records.stale_reason(unit)
        if reason is None and record["code"]["uncommitted_changes"]:
            reason = "written with uncommitted changes"
        return reason or "valid", record["code"]["commit"]

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        mask = self.inputs.label_mask()
        codes = self.inputs.label_codes("query")
        queries, database = self.inputs.queries["exam_id"], self.inputs.database["exam_id"]
        jobs, missing, specs = [], [], self.specs()
        for spec in specs:
            if not all(path.exists() for path in self.needed(spec)):
                missing.append(unit_name(spec))
                continue
            query, archive, vectors = self.sides(spec)
            unit, records = self.unit(spec, query, archive, mask)
            if records.pending([unit], f"{self.corpus_name} {unit.name}"):
                jobs.append(delayed(write_unit)(spec, query, archive, vectors, mask, codes, queries, database, unit,
                                                records))
        if jobs:
            Parallel(n_jobs=self.settings["n_jobs"])(jobs)
        if missing:
            raise RuntimeError(f"[{self.corpus_name}] units with missing inputs: {missing}")
