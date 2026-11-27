import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from lib import scoring
from lib.exam_encoder import LABELS
from config import Config
from controls.controls_retrieval import EvidenceUnits
from encoders.query_metrics import metric_frame
from encoders.retrieval import DOCUMENTED, LIB_DIR, PROFILE, Retrieval, row_digest
from provenance import RECORD_SUFFIX, ProvenanceRecords, Unit, sha256_file, sha256_json, sha256_rows
from trained_encoder.test_outputs import TestEmbedder

EVIDENCE_COLUMNS = ["method", "first_stage", "reranker", "seed", "test_corpus", "archive", "k", "query_inputs",
                    "level", "finding", "metric", "exam_id", "value"]
SPEC_COLUMNS = EVIDENCE_COLUMNS[:8]
CLASSIFIER_RERANK, PROFILE_RERANK = "classifier_rerank", "profile_rerank"
CLASSIFIER_RERANK_OURS, TWO_STAGE_OURS = "classifier_rerank_ours", "two_stage_ours"
METHODS = [CLASSIFIER_RERANK, PROFILE_RERANK, CLASSIFIER_RERANK_OURS, TWO_STAGE_OURS]
CLASSIFIER, PROFILE_STAGE = "classifier", "profile"
FIRST_STAGES = [CLASSIFIER, PROFILE_STAGE]
OURS = "ours"
NO_SEED = "none"
METHOD_PARTS = {CLASSIFIER_RERANK: (CLASSIFIER, False), PROFILE_RERANK: (PROFILE_STAGE, False),
                CLASSIFIER_RERANK_OURS: (CLASSIFIER, True), TWO_STAGE_OURS: (PROFILE_STAGE, True)}


def reranker_name(retriever: str, rule: str) -> str:
    return f"{retriever}:{rule}"


def share_frame(scores: np.ndarray, labels: dict, report: dict, share: dict) -> pd.DataFrame:
    query, archive = np.asarray(labels["query"]), np.asarray(labels["archive"])
    counted = np.array([label not in share["excluded"] for label in LABELS])
    positive = (query == 1) & counted[None, :]
    rows = np.flatnonzero(np.asarray(report["keep"]) & positive.any(axis=1))
    top = np.argsort(-np.asarray(scores[rows], dtype=np.float64), axis=1, kind="stable")[:, :share["k"]]
    shares = (archive[top] == 1).mean(axis=1)
    query_index, label_index = np.nonzero(positive[rows])
    return pd.DataFrame({"level": "finding", "finding": np.array(LABELS)[label_index], "metric": share["name"],
                         "exam_id": np.asarray(labels["query_ids"])[rows][query_index],
                         "value": shares[query_index, label_index]})


def ranking_scores(first: dict, reranker: dict, mask: np.ndarray, k: int) -> np.ndarray:
    profile = scoring.profile_scores(first["query"], first["archive"], finding_mask=mask)
    alignment = scoring.alignment_scores(reranker["query"], reranker["archive"])
    return scoring.two_stage_scores(profile, alignment, k)


def evidence_frame(spec: dict, first: dict, reranker: dict, mask: np.ndarray, labels: dict, report: dict,
                   evaluation: dict, share: dict, metrics: list[str]) -> pd.DataFrame:
    scores = ranking_scores(first, reranker, mask, spec["k"])
    evidence = pd.concat([metric_frame(scores, labels, report, evaluation), share_frame(scores, labels, report, share)],
                         ignore_index=True)
    evidence = evidence[evidence["metric"].isin(metrics)].copy()
    for column in SPEC_COLUMNS:
        evidence[column] = spec[column]
    return evidence[EVIDENCE_COLUMNS]


def write_unit(spec, first, reranker, mask, labels, report, evaluation, share, metrics, unit: Unit,
               records: ProvenanceRecords) -> None:
    evidence = evidence_frame(spec, first, reranker, mask, labels, report, evaluation, share, metrics)
    partial = unit.outputs[0].with_name(unit.outputs[0].name + ".partial")
    with open(partial, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
        evidence.to_csv(handle, index=False)
    os.replace(partial, unit.outputs[0])
    records.write(unit)


def unit_name(spec: dict) -> str:
    reranker = spec["reranker"].replace(":", "-")
    return (f"{spec['method']}__{spec['first_stage']}__{reranker}__{spec['seed']}__{spec['archive']}__k{spec['k']}__"
            f"{spec['query_inputs']}")


def read_rerank_evidence(path) -> pd.DataFrame:
    dtypes = {"exam_id": str, "seed": str, "k": int,
              **{c: "category" for c in EVIDENCE_COLUMNS if c not in ("exam_id", "seed", "k", "value")}}
    return pd.read_csv(path, dtype=dtypes, keep_default_na=False, na_values={"value": [""]},
                       float_precision="round_trip")


class TwoStageUnits:

    group = ""
    step = ""

    def __init__(self, config: Config, corpus_name: str):
        from controls.depth_selection import DepthSelection
        self.config = config
        self.corpus_name = corpus_name
        self.settings = config.get(self.group)
        if corpus_name not in self.settings["test_corpora"]:
            raise ValueError(f"{corpus_name} is not a test corpus of {self.group}: {self.settings['test_corpora']}")
        self.retrieval = Retrieval(config, corpus_name)
        if not self.retrieval.documented:
            raise ValueError(f"{corpus_name} is not a documented corpus")
        self.inputs = self.retrieval.inputs
        self.evaluation = config.get("retrieval.evaluation")
        self.report_encoder = config.get("trained_encoder.retrieval.documented_report_encoder")
        self.out_dir = config.path(f"paths.{self.group}_dir") / self.settings["evidence_folder"] / corpus_name
        self.depth = DepthSelection(config).depth()
        self.runs = {run.split("__")[2]: run for run in self.settings["encoder_runs"]}
        self.cache = {}

    def specs(self) -> list[dict]:
        raise NotImplementedError

    def spec(self, method: str, reranker: str, seed: str, archive: str, k: int) -> dict:
        return {"method": method, "first_stage": METHOD_PARTS[method][0], "reranker": reranker, "seed": seed,
                "test_corpus": self.corpus_name, "archive": archive, "k": int(k),
                "query_inputs": self.settings["query_inputs"], "run": self.runs.get(seed)}

    def frozen_rerankers(self, archive: str) -> list[str]:
        return [reranker_name(s["retriever"], s["rule"]) for s in self.retrieval.specs()
                if s["archive"] == archive and s["rule"] != PROFILE]

    def output_path(self, run: str, side: str) -> Path:
        query_inputs = self.settings["query_inputs"] if side == "query" else "all"
        return TestEmbedder(self.config, run).path(self.corpus_name, side, query_inputs)

    def needed(self, spec: dict) -> list[Path]:
        if spec["run"] is None:
            return []
        sides = ["query", "database"] if spec["archive"] != DOCUMENTED else ["query"]
        return [self.output_path(spec["run"], side) for side in sides]

    def model_sides(self, run: str, archive: str) -> tuple[dict, dict]:
        key = ("model", run, archive)
        if key not in self.cache:
            query = EvidenceUnits.model_side(self.output_path(run, "query"), self.inputs.queries["exam_id"])
            if archive == DOCUMENTED:
                database = {"profile": self.inputs.profile_labels(),
                            "alignment": self.documented_reports()}
            else:
                database = EvidenceUnits.model_side(self.output_path(run, "database"), self.inputs.database["exam_id"])
            self.cache[key] = (query, database)
        return self.cache[key]

    def documented_reports(self) -> np.ndarray:
        if "reports" not in self.cache:
            self.cache["reports"] = self.inputs.reports(self.report_encoder)
        return self.cache["reports"]

    def sides(self, spec: dict) -> tuple[dict, dict]:
        if spec["first_stage"] == CLASSIFIER:
            key = ("classifier", spec["archive"])
            if key not in self.cache:
                classifier = self.retrieval.scores_input({"retriever": self.retrieval.classifier,
                                                          "archive": spec["archive"], "rule": PROFILE})
                self.cache[key] = {"query": classifier["query"], "archive": classifier["archive"]}
            first = self.cache[key]
        else:
            query, database = self.model_sides(spec["run"], spec["archive"])
            first = {"query": query["profile"], "archive": database["profile"]}
        if spec["reranker"] == OURS:
            query, database = self.model_sides(spec["run"], spec["archive"])
            reranker = {"query": query["alignment"], "archive": database["alignment"]}
        else:
            key = ("reranker", spec["reranker"], spec["archive"])
            if key not in self.cache:
                retriever, rule = spec["reranker"].split(":")
                self.cache[key] = self.retrieval.scores_input({"retriever": retriever, "archive": spec["archive"],
                                                               "rule": rule})
            reranker = self.cache[key]
        return first, reranker

    def path(self, spec: dict) -> Path:
        return self.out_dir / f"{unit_name(spec)}.csv.gz"

    def step_settings(self, spec: dict, mask: np.ndarray) -> dict:
        evaluation = {key: value for key, value in self.evaluation.items() if key != "n_jobs"}
        return {"evaluation": evaluation, "metrics": self.settings["metrics"], "share": self.settings["share"],
                **{c: spec[c] for c in SPEC_COLUMNS}, "run": spec["run"], "label_mask": mask.tolist(),
                "documented_report_encoder": self.report_encoder, **spec.get("extra_settings", {}),
                "lib_sha256": {name: sha256_file(LIB_DIR / name) for name in ("metrics.py", "scoring.py")}}

    def unit(self, spec: dict, first: dict, reranker: dict, mask: np.ndarray) -> tuple[Unit, ProvenanceRecords]:
        records = ProvenanceRecords(self.out_dir, self.step, self.step_settings(spec, mask))
        ids, contents = [], []
        for side, key in (("query", "query"), ("database", "archive")):
            entities = self.inputs.entities(side)
            for position, (exam_id, cells) in enumerate(zip(self.inputs.side(side)["exam_id"],
                                                             self.inputs.label_cells(side))):
                rows = f"{row_digest(first[key][position])},{row_digest(reranker[key][position])}"
                entity_part = ";".join(sorted("|".join(e) for e in entities[position]))
                ids.append(f"{side}:{exam_id}")
                contents.append(f"{rows}|{cells}|{entity_part}")
        return Unit(unit_name(spec), ids, contents, [self.path(spec)]), records

    def status(self, spec: dict, mask: np.ndarray) -> tuple[str, str]:
        output = self.path(spec)
        record_path = output.with_name(f"{unit_name(spec)}{RECORD_SUFFIX}")
        if not record_path.exists():
            return "no record", ""
        record = json.loads(record_path.read_text())
        ids = [f"{side}:{exam_id}" for side in ("query", "database") for exam_id in self.inputs.side(side)["exam_id"]]
        checks = [
            ("record of another step", record["step"] == self.step),
            ("queries or database differ", record["inputs"]["n"] == len(ids)
             and record["inputs"]["ids_sha256"] == sha256_rows(ids)),
            ("settings differ", record["settings_sha256"] == sha256_json(self.step_settings(spec, mask))),
            ("output file missing", output.exists()),
            ("output file changed", output.exists() and record["outputs"].get(output.name) == sha256_file(output)),
            ("written with uncommitted changes", not record["code"]["uncommitted_changes"]),
        ]
        failed = [reason for reason, ok in checks if not ok]
        return (failed[0] if failed else "valid"), record["code"]["commit"]

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        mask = self.inputs.label_mask()
        labels, report = self.retrieval.labels(), self.retrieval.report()
        jobs, missing = [], []
        for spec in self.specs():
            if not all(path.exists() for path in self.needed(spec)):
                missing.append(unit_name(spec))
                continue
            first, reranker = self.sides(spec)
            unit, records = self.unit(spec, first, reranker, mask)
            if records.pending([unit], f"{self.corpus_name} {unit.name}"):
                jobs.append(delayed(write_unit)(spec, first, reranker, mask, labels, report, self.evaluation,
                                                self.settings["share"], self.settings["metrics"], unit, records))
        if jobs:
            Parallel(n_jobs=self.settings["n_jobs"])(jobs)
        if missing:
            raise RuntimeError(f"[{self.corpus_name}] {len(missing)} units have missing model outputs: {missing}")


class RerankRetrieval(TwoStageUnits):

    group = "rerank"
    step = "rerank-retrieval"

    def specs(self) -> list[dict]:
        specs = []
        for archive in self.settings["archives"]:
            frozen = self.frozen_rerankers(archive)
            specs += [self.spec(CLASSIFIER_RERANK, reranker, NO_SEED, archive, self.depth) for reranker in frozen]
            for seed in self.runs:
                specs += [self.spec(PROFILE_RERANK, reranker, seed, archive, self.depth) for reranker in frozen]
                specs.append(self.spec(CLASSIFIER_RERANK_OURS, OURS, seed, archive, self.depth))
                specs.append(self.spec(TWO_STAGE_OURS, OURS, seed, archive, self.depth))
        return specs


class DepthRetrieval(TwoStageUnits):

    group = "depth"
    step = "depth-retrieval"

    def specs(self) -> list[dict]:
        from rerank.rerank_files import RerankFiles
        files = RerankFiles(self.config)
        selections_sha256 = sha256_file(files.selections_path)
        specs = []
        for archive in self.settings["archives"]:
            selected = files.selected(CLASSIFIER, self.corpus_name, archive, self.settings["reranker_metric"])
            for k in self.settings["ks"]:
                spec = self.spec(CLASSIFIER_RERANK, selected, NO_SEED, archive, k)
                spec["extra_settings"] = {"selections_sha256": selections_sha256,
                                          "selected_for": self.settings["reranker_metric"], "selected_at": self.depth}
                specs.append(spec)
                specs += [self.spec(TWO_STAGE_OURS, OURS, seed, archive, k) for seed in self.runs]
        return specs
