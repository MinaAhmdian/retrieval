import gzip
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from lib import scoring
from config import Config
from encoders.query_metrics import metric_frame
from encoders.retrieval import DOCUMENTED, UNDOCUMENTED, LIB_DIR, Retrieval, row_digest
from provenance import ProvenanceRecords, Unit, sha256_file
from trained_encoder.test_outputs import TestEmbedder, load_outputs
from trained_encoder.runs import run_names

EVIDENCE_COLUMNS = ["configuration", "backbone", "seed", "test_corpus", "archive", "rule", "query_inputs", "level",
                    "finding", "metric", "exam_id", "value"]
PROFILE, ALIGNMENT, TWO_STAGE = "profile", "alignment", "two_stage"
RULES = [PROFILE, ALIGNMENT, TWO_STAGE]
RULE_PARTS = {PROFILE: ["profile"], ALIGNMENT: ["alignment"], TWO_STAGE: ["profile", "alignment"]}


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))


def rule_scores(rule: str, query: dict, archive: dict, mask: np.ndarray, k: int) -> np.ndarray:
    if rule == PROFILE:
        return scoring.profile_scores(query["profile"], archive["profile"], finding_mask=mask)
    alignment = scoring.alignment_scores(query["alignment"], archive["alignment"])
    if rule == ALIGNMENT:
        return alignment
    profile = scoring.profile_scores(query["profile"], archive["profile"], finding_mask=mask)
    return scoring.two_stage_scores(profile, alignment, k)


def write_unit(spec: dict, query: dict, archive: dict, mask, k, labels, report, evaluation, unit: Unit,
               records: ProvenanceRecords) -> tuple[str, int]:
    evidence = metric_frame(rule_scores(spec["rule"], query, archive, mask, k), labels, report, evaluation)
    for column in EVIDENCE_COLUMNS[:7]:
        evidence[column] = spec[column]
    partial = unit.outputs[0].with_name(unit.outputs[0].name + ".partial")
    with open(partial, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
        evidence[EVIDENCE_COLUMNS].to_csv(handle, index=False)
    os.replace(partial, unit.outputs[0])
    records.write(unit)
    return unit.name, len(evidence)


class TrainedRetrieval:

    step = "trained-retrieval"

    def __init__(self, config: Config, corpus_name: str, retrieval: Retrieval | None = None):
        self.config = config
        self.corpus_name = corpus_name
        self.retrieval = Retrieval(config, corpus_name) if retrieval is None else retrieval
        self.inputs = self.retrieval.inputs
        self.settings = config.get("trained_encoder.retrieval")
        self.evaluation = config.get("retrieval.evaluation")
        self.out_dir = config.path("paths.trained_dir") / self.settings["evidence_folder"] / corpus_name
        self.archives = [DOCUMENTED, UNDOCUMENTED] if self.retrieval.documented else [UNDOCUMENTED]

    def specs(self) -> list[dict]:
        specs = []
        for run in run_names(self.config):
            backbone, configuration, seed = run.split("__")
            for query_inputs in self.config.get(f"trained_encoder.test_query_inputs.{configuration}"):
                for archive in self.archives:
                    specs += [{"run": run, "configuration": configuration, "backbone": backbone, "seed": int(seed),
                               "test_corpus": self.corpus_name, "archive": archive, "rule": rule,
                               "query_inputs": query_inputs} for rule in RULES]
        return specs

    @staticmethod
    def unit_name(spec: dict) -> str:
        return f"{spec['run']}__{spec['archive']}__{spec['rule']}__{spec['query_inputs']}"

    def step_settings(self, spec: dict, mask: np.ndarray) -> dict:
        evaluation = {key: value for key, value in self.evaluation.items() if key != "n_jobs"}
        return {"evaluation": evaluation, "run": spec["run"], "archive": spec["archive"], "rule": spec["rule"],
                "query_inputs": spec["query_inputs"], "label_mask": mask.tolist(),
                "two_stage_k": self.settings["two_stage_k"],
                "documented_report_encoder": self.settings["documented_report_encoder"],
                "lib_sha256": {name: sha256_file(LIB_DIR / name) for name in ("metrics.py", "scoring.py")}}

    def sides(self, spec: dict, embedder: TestEmbedder) -> tuple[dict, dict]:
        queries, database = self.inputs.queries["exam_id"], self.inputs.database["exam_id"]
        alignment, logits = load_outputs(embedder.path(self.corpus_name, "query", spec["query_inputs"]), queries)
        query = {"profile": sigmoid(logits), "alignment": alignment}
        if spec["archive"] == DOCUMENTED:
            archive = {"profile": self.inputs.profile_labels(),
                       "alignment": self.inputs.reports(self.settings["documented_report_encoder"])}
        else:
            alignment, logits = load_outputs(embedder.path(self.corpus_name, "database", "all"), database)
            archive = {"profile": sigmoid(logits), "alignment": alignment}
        return query, archive

    def unit(self, spec: dict, query: dict, archive: dict, mask: np.ndarray) -> tuple[Unit, ProvenanceRecords]:
        records = ProvenanceRecords(self.out_dir, self.step, self.step_settings(spec, mask))
        parts = RULE_PARTS[spec["rule"]]
        ids, contents = [], []
        for side, matrices in (("query", query), ("database", archive)):
            entities = self.inputs.entities(side) if self.retrieval.documented else None
            for position, (exam_id, cells) in enumerate(zip(self.inputs.side(side)["exam_id"],
                                                             self.inputs.label_cells(side))):
                rows = ",".join(row_digest(matrices[part][position]) for part in parts)
                entity_part = "" if entities is None else ";".join(sorted("|".join(e) for e in entities[position]))
                ids.append(f"{side}:{exam_id}")
                contents.append(f"{rows}|{cells}|{entity_part}")
        name = self.unit_name(spec)
        return Unit(name, ids, contents, [self.out_dir / f"{name}.csv.gz"]), records

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        mask = self.inputs.label_mask()
        labels, report = self.retrieval.labels(), self.retrieval.report()
        embedders = {run: TestEmbedder(self.config, run) for run in run_names(self.config)}
        jobs, missing, specs = [], set(), self.specs()
        for spec in specs:
            embedder = embedders[spec["run"]]
            needed = [embedder.path(self.corpus_name, "query", spec["query_inputs"])]
            if spec["archive"] == UNDOCUMENTED:
                needed.append(embedder.path(self.corpus_name, "database", "all"))
            if not all(path.exists() for path in needed):
                missing.add(spec["run"])
                continue
            query, archive = self.sides(spec, embedder)
            unit, records = self.unit(spec, query, archive, mask)
            if records.pending([unit], f"{self.corpus_name} {unit.name}"):
                jobs.append(delayed(write_unit)(spec, query, archive, mask, self.settings["two_stage_k"], labels,
                                                report, self.evaluation, unit, records))
        if jobs:
            Parallel(n_jobs=self.settings["n_jobs"])(jobs)
        if missing:
            raise RuntimeError(f"[{self.corpus_name}] runs without test outputs: {sorted(missing)}; run "
                               "embed-test-exams for them")
