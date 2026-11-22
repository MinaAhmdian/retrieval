import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config import Config
from controls.permutations import ReportPermutations
from controls.replacement_heads import replaced_representation
from controls.validation import ValidEmbedder, ValidRetrievalInputs, validation_dir
from encoders.query_metrics import metric_frame
from encoders.retrieval import DOCUMENTED, UNDOCUMENTED, LIB_DIR, Retrieval, row_digest
from provenance import RECORD_SUFFIX, ProvenanceRecords, Unit, sha256_file, sha256_json, sha256_rows
from trained_encoder.model_files import COMBINATION as REUSED_COMBINATION, ROW as REUSED_ROW, ModelFiles
from trained_encoder.runs import run_names
from trained_encoder.test_outputs import TestEmbedder, load_outputs
from trained_encoder.trained_retrieval import (ALIGNMENT, PROFILE, RULE_PARTS, TWO_STAGE, TrainedRetrieval,
                                                       rule_scores, sigmoid)

EVIDENCE_COLUMNS = ["configuration", "backbone", "seed", "test_corpus", "split", "archive", "rule", "k", "query_inputs",
                    "level", "finding", "metric", "exam_id", "value"]
SPEC_COLUMNS = EVIDENCE_COLUMNS[:9]
REUSED_KEY = [*REUSED_COMBINATION, *REUSED_ROW]
VALID, TEST = "valid", "test"
NO_DEPTH = 0


def write_unit(spec: dict, query: dict, archive: dict, mask, labels, report, evaluation, unit: Unit,
               records: ProvenanceRecords) -> tuple[str, int]:
    if spec.get("reuse_path"):
        evidence = read_reused(spec["reuse_path"])
        for column in ("configuration", "backbone", "seed", "test_corpus", "archive", "rule", "query_inputs"):
            if set(evidence[column].astype(str)) != {str(spec[column])}:
                raise ValueError(f"{unit.name}: the reused unit's {column} is not {spec[column]}")
        evidence = evidence.assign(split=spec["split"], k=spec["k"])
    else:
        evidence = metric_frame(rule_scores(spec["rule"], query, archive, mask, spec["k"]), labels, report, evaluation)
        for column in SPEC_COLUMNS:
            evidence[column] = spec[column]
    partial = unit.outputs[0].with_name(unit.outputs[0].name + ".partial")
    with open(partial, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
        evidence[EVIDENCE_COLUMNS].to_csv(handle, index=False)
    os.replace(partial, unit.outputs[0])
    records.write(unit)
    return unit.name, len(evidence)


def unit_name(spec: dict) -> str:
    return (f"{spec['backbone']}__{spec['configuration']}__{spec['seed']}__{spec['split']}__{spec['archive']}__"
            f"{spec['rule']}__k{spec['k']}__{spec['query_inputs']}")


def read_controls_evidence(path) -> pd.DataFrame:
    dtypes = {"exam_id": str, "seed": int, "k": int,
              **{c: "category" for c in EVIDENCE_COLUMNS if c not in ("exam_id", "seed", "k", "value")}}
    return pd.read_csv(path, dtype=dtypes, keep_default_na=False, na_values={"value": [""]},
                       float_precision="round_trip")


def read_reused(path) -> pd.DataFrame:
    dtypes = {"exam_id": str, "seed": int, **{column: "category" for column in [*REUSED_KEY]}}
    return pd.read_csv(path, dtype=dtypes, keep_default_na=False, na_values={"value": [""]},
                       float_precision="round_trip")


class EvidenceUnits:

    step = ""

    def __init__(self, config: Config, corpus_name: str, retrieval: Retrieval, out_dir: Path):
        self.config = config
        self.corpus_name = corpus_name
        self.retrieval = retrieval
        self.inputs = retrieval.inputs
        self.evaluation = config.get("retrieval.evaluation")
        self.out_dir = out_dir
        self.report_encoder = config.get("trained_encoder.retrieval.documented_report_encoder")
        self.n_jobs = config.get("controls.retrieval.n_jobs")

    def specs(self) -> list[dict]:
        raise NotImplementedError

    def sides(self, spec: dict) -> tuple[dict, dict]:
        raise NotImplementedError

    def needed(self, spec: dict) -> list[Path]:
        raise NotImplementedError

    def path(self, spec: dict) -> Path:
        return self.out_dir / f"{unit_name(spec)}.csv.gz"

    def step_settings(self, spec: dict, mask: np.ndarray) -> dict:
        evaluation = {key: value for key, value in self.evaluation.items() if key != "n_jobs"}
        settings = {"evaluation": evaluation, **{c: spec[c] for c in SPEC_COLUMNS}, "run": spec["run"],
                    "label_mask": mask.tolist(), "documented_report_encoder": self.report_encoder,
                    "lib_sha256": {name: sha256_file(LIB_DIR / name) for name in ("metrics.py", "scoring.py")}}
        if "reuse_sha256" in spec:
            settings["reuse_sha256"] = spec["reuse_sha256"]
        return settings

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

    def pending_units(self) -> tuple[list, list[dict], set]:
        mask = self.inputs.label_mask()
        labels, report = self.retrieval.labels(), self.retrieval.report()
        jobs, missing, specs = [], set(), self.specs()
        for spec in specs:
            if not all(path.exists() for path in self.needed(spec)):
                missing.add(spec["run"])
                continue
            query, archive = self.sides(spec)
            unit, records = self.unit(spec, query, archive, mask)
            if records.pending([unit], f"{self.corpus_name} {unit.name}"):
                jobs.append(delayed(write_unit)(spec, query, archive, mask, labels, report, self.evaluation, unit,
                                                records))
        return jobs, specs, missing

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        jobs, specs, missing = self.pending_units()
        if jobs:
            Parallel(n_jobs=self.n_jobs)(jobs)
        if missing:
            raise RuntimeError(f"[{self.corpus_name}] runs with missing inputs: {sorted(missing)}")

    @staticmethod
    def model_side(path: Path, exam_ids: pd.Series) -> dict:
        alignment, logits = load_outputs(path, exam_ids)
        return {"profile": sigmoid(logits), "alignment": alignment}


class ValidRetrieval(EvidenceUnits):

    step = "valid-retrieval"

    def __init__(self, config: Config, corpus_name: str):
        self.settings = config.get("controls.validation")
        inputs = ValidRetrievalInputs(config, corpus_name)
        super().__init__(config, corpus_name, Retrieval(config, corpus_name, inputs=inputs),
                         validation_dir(config) / self.settings["evidence_folder"] / corpus_name)
        if self.settings["archive"] != DOCUMENTED:
            raise ValueError("the validation retrieval reads the documented archive only")
        self.embedders = {run: ValidEmbedder(config, run) for run in self.settings["weight_runs"]}

    def specs(self) -> list[dict]:
        specs = []
        for run in self.settings["weight_runs"]:
            backbone, configuration, seed = run.split("__")
            base = {"run": run, "configuration": configuration, "backbone": backbone, "seed": int(seed),
                    "test_corpus": self.corpus_name, "split": VALID, "archive": self.settings["archive"],
                    "query_inputs": self.settings["query_inputs"]}
            specs += [dict(base, rule=PROFILE, k=NO_DEPTH), dict(base, rule=ALIGNMENT, k=NO_DEPTH)]
            specs += [dict(base, rule=TWO_STAGE, k=int(k)) for k in self.settings["depths"]]
        return specs

    def needed(self, spec: dict) -> list[Path]:
        return [self.embedders[spec["run"]].path(self.corpus_name, "query", spec["query_inputs"])]

    def sides(self, spec: dict) -> tuple[dict, dict]:
        query = self.model_side(self.needed(spec)[0], self.inputs.queries["exam_id"])
        archive = {"profile": self.inputs.profile_labels(), "alignment": self.inputs.reports(self.report_encoder)}
        return query, archive


class ControlsRetrieval(EvidenceUnits):

    step = "controls-retrieval"

    def __init__(self, config: Config, corpus_name: str):
        from controls.depth_selection import DepthSelection
        settings = config.get("controls.retrieval")
        super().__init__(config, corpus_name, Retrieval(config, corpus_name),
                         config.path("paths.controls_dir") / settings["evidence_folder"] / corpus_name)
        self.depth = DepthSelection(config).depth()
        self.reevaluated_rules = settings["reevaluated_rules"]
        self.shuffled = config.get("controls.shuffled_report")
        self.archives = [DOCUMENTED, UNDOCUMENTED] if self.retrieval.documented else [UNDOCUMENTED]
        self.reused_k = config.get("trained_encoder.retrieval.two_stage_k")
        self.heads = config.get("controls.replacement_heads.configurations")
        self.embedders = {}
        self.cache = {}

    def embedder(self, run: str) -> TestEmbedder:
        if run not in self.embedders:
            self.embedders[run] = TestEmbedder(self.config, run)
        return self.embedders[run]

    @staticmethod
    def base(run: str, corpus: str, archive: str, query_inputs: str, configuration: str | None = None) -> dict:
        backbone, run_configuration, seed = run.split("__")
        return {"run": run, "configuration": configuration or run_configuration, "backbone": backbone,
                "seed": int(seed), "test_corpus": corpus, "split": TEST, "archive": archive,
                "query_inputs": query_inputs}

    def rules(self, rules: list[str]) -> list[tuple[str, int]]:
        return [(rule, self.depth if rule == TWO_STAGE else NO_DEPTH) for rule in rules]

    def specs(self) -> list[dict]:
        specs = []
        for run in run_names(self.config, "controls.runs"):
            configuration = run.split("__")[1]
            for query_inputs in self.config.get(f"trained_encoder.test_query_inputs.{configuration}"):
                for archive in self.archives:
                    specs += [dict(self.base(run, self.corpus_name, archive, query_inputs), rule=rule, k=k,
                                   source="controls") for rule, k in self.rules([PROFILE, ALIGNMENT, TWO_STAGE])]
        if self.retrieval.documented:
            for run in self.shuffled["weight_runs"]:
                for archive in self.shuffled["archives"]:
                    specs += [dict(self.base(run, self.corpus_name, archive, self.shuffled["query_inputs"],
                                             self.shuffled["configuration"]), rule=rule, k=k, source="shuffled_report")
                              for rule, k in self.rules([PROFILE, ALIGNMENT, TWO_STAGE])]
        for run in run_names(self.config):
            configuration = run.split("__")[1]
            for query_inputs in self.config.get(f"trained_encoder.test_query_inputs.{configuration}"):
                for archive in self.archives:
                    for rule, k in self.rules(self.reevaluated_rules):
                        spec = dict(self.base(run, self.corpus_name, archive, query_inputs), rule=rule, k=k,
                                    source="reevaluated")
                        if k == self.reused_k:
                            spec["reuse_path"] = self.reused_unit(spec)
                            spec["reuse_sha256"] = sha256_file(spec["reuse_path"])
                        specs.append(spec)
        return specs

    def reused_unit(self, spec: dict) -> Path:
        if "reused" not in self.cache:
            self.cache["reused"] = (TrainedRetrieval(self.config, self.corpus_name, retrieval=self.retrieval),
                                      ModelFiles(self.config))
        job, files = self.cache["reused"]
        reused = {key: spec[key] for key in ("run", "configuration", "backbone", "seed", "test_corpus", "archive",
                                               "rule", "query_inputs")}
        status, _, output = files.unit_status(job, reused, self.inputs.label_mask())
        if status != "valid":
            raise RuntimeError(f"[{self.corpus_name}] the reused unit {output.name} is not valid: {status}")
        return output

    def needed(self, spec: dict) -> list[Path]:
        embedder = self.embedder(spec["run"])
        paths = [embedder.path(self.corpus_name, "query", spec["query_inputs"])]
        if spec["archive"] == UNDOCUMENTED:
            paths.append(embedder.path(self.corpus_name, "database", "all"))
        if spec["configuration"] in self.heads:
            from controls.replacement_heads import FitReplacementHeads
            heads = FitReplacementHeads(self.config, spec["run"])
            paths.append(heads.applied_path(self.corpus_name, "query", spec["query_inputs"]))
            if spec["archive"] == UNDOCUMENTED:
                paths.append(heads.applied_path(self.corpus_name, "database", "all"))
        if spec["source"] == "shuffled_report":
            paths.append(ReportPermutations(self.config).path(self.corpus_name, TEST))
        return paths

    def documented_archive(self) -> dict:
        if "documented" not in self.cache:
            self.cache["documented"] = {"profile": self.inputs.profile_labels(),
                                        "alignment": self.inputs.reports(self.report_encoder)}
        return self.cache["documented"]

    def donor_positions(self) -> np.ndarray:
        if "donors" not in self.cache:
            database = self.inputs.database["exam_id"]
            donors = ReportPermutations(self.config).donors(self.corpus_name, TEST)
            if sorted(donors.index) != sorted(database) or sorted(donors) != sorted(database):
                raise ValueError(f"[{self.corpus_name}] the report permutation is not a bijection of the database")
            position = pd.Series(np.arange(len(database)), index=database.to_numpy())
            self.cache["donors"] = position.loc[donors.loc[database].to_numpy()].to_numpy()
        return self.cache["donors"]

    def sides(self, spec: dict) -> tuple[dict, dict]:
        embedder = self.embedder(spec["run"])
        queries, database = self.inputs.queries["exam_id"], self.inputs.database["exam_id"]
        query = self.model_side(embedder.path(self.corpus_name, "query", spec["query_inputs"]), queries)
        if spec["archive"] == DOCUMENTED:
            archive = dict(self.documented_archive())
            if spec["source"] == "shuffled_report":
                donors = self.donor_positions()
                archive = {part: matrix[donors] for part, matrix in archive.items()}
        else:
            archive = self.model_side(embedder.path(self.corpus_name, "database", "all"), database)
        if spec["configuration"] in self.heads:
            part, rows = replaced_representation(self.config, spec["run"], self.corpus_name, "query",
                                                 spec["query_inputs"], queries)
            query[part] = rows
            if spec["archive"] == UNDOCUMENTED:
                archive[part] = replaced_representation(self.config, spec["run"], self.corpus_name, "database", "all",
                                                        database)[1]
        return query, archive
