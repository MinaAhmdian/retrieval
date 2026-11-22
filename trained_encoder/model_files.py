import gzip
import json
import os

import numpy as np
import pandas as pd

from lib.exam_encoder import LABELS
from config import Config
from encoders.encoder_files import RESULT_COLUMNS
from provenance import RECORD_SUFFIX, sha256_file, sha256_json, sha256_rows
from trained_encoder.runs import run_names
from trained_encoder.trained_retrieval import EVIDENCE_COLUMNS, RULES, TrainedRetrieval

COMBINATION = ["configuration", "backbone", "test_corpus", "archive", "rule", "query_inputs"]
ROW = ["level", "finding", "metric"]
KEY = [*COMBINATION, *ROW, "exam_id"]
UNIT_COLUMNS = ["test_corpus", "run", "unit", "status", "commit", "rows"]


def read_evidence(path) -> pd.DataFrame:
    dtypes = {"exam_id": str, "seed": int, **{column: "category" for column in [*COMBINATION, *ROW]}}
    return pd.read_csv(path, dtype=dtypes, keep_default_na=False, na_values={"value": [""]})


def aggregate(evidence: pd.DataFrame, group: list[str], settings: dict) -> pd.DataFrame:
    rows = evidence.groupby([*group, *ROW], sort=False, observed=True)["value"].agg(mean_raw="mean", n="size")
    rows = rows.reset_index()
    ap = evidence[evidence["metric"] == "ap"]
    aggregates = []
    for key, part in ap.groupby(group, sort=False, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        per_finding = part.groupby("finding", observed=True)["value"].mean()
        sets = {settings["macro_name"]: list(per_finding.index)}
        if set(settings["core_findings"]) <= set(per_finding.index):
            sets[settings["core_name"]] = settings["core_findings"]
        for name, findings in sets.items():
            aggregates.append({**dict(zip(group, key)), "level": "finding", "finding": name,
                               "metric": settings["macro_metric"], "mean_raw": per_finding.loc[findings].mean(),
                               "n": part.loc[part["finding"].isin(findings), "exam_id"].nunique()})
    return pd.concat([rows, pd.DataFrame(aggregates, columns=rows.columns)], ignore_index=True)


def result_rows(evidence: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = config.get("retrieval.files")
    files = config.get("trained_encoder.files")
    seeds = config.get("trained_encoder.runs.seeds")
    per_key = evidence.groupby(KEY, sort=False, observed=True).agg(value=("value", "mean"), seeds=("seed", "nunique"))
    per_key = per_key.reset_index()
    short = per_key[per_key["seeds"] < len(seeds)]
    left_out = short.groupby(COMBINATION, observed=True).size().rename("keys_without_every_seed").reset_index()
    complete = per_key.merge(left_out[COMBINATION], on=COMBINATION, how="left", indicator=True)
    complete = complete[complete["_merge"] == "left_only"].drop(columns=["_merge", "seeds"])
    combined = aggregate(complete, COMBINATION, settings).assign(seeds=files["seeds_separator"].join(map(str, seeds)))
    parts = [combined]
    wanted = pd.DataFrame(files["per_seed_rows"])
    for seed in seeds:
        part = evidence[evidence["seed"] == seed]
        if len(part):
            rows = aggregate(part, COMBINATION, settings).merge(wanted, on=ROW)
            parts.append(rows.assign(seeds=str(seed)))
    result = pd.concat(parts, ignore_index=True)
    result["mean"] = result["mean_raw"] * settings["percent_scale"]
    for column, value in files["fixed_columns"].items():
        result[column] = value
    return ordered(result[RESULT_COLUMNS], config), left_out


def ordered(result: pd.DataFrame, config: Config) -> pd.DataFrame:
    settings = config.get("retrieval.files")
    evaluation = config.get("retrieval.evaluation")
    runs = config.get("trained_encoder.runs")
    configurations = list(dict.fromkeys(c for cs in runs["configurations"].values() for c in cs))
    seeds = [config.get("trained_encoder.files.seeds_separator").join(map(str, runs["seeds"])), *map(str, runs["seeds"])]
    precision = [m for m in result["metric"].unique() if m.startswith("p") and "_f1_" in m]
    order = {"configuration": configurations, "backbone": list(runs["configurations"]), "seeds": seeds,
             "test_corpus": config.get("cohort.corpora"), "archive": ["documented", "undocumented"], "rule": RULES,
             "query_inputs": list(config.get("trained_encoder.query_inputs")), "level": ["finding", "report"],
             "finding": [*LABELS, settings["macro_name"], settings["core_name"], "pooled"],
             "metric": ["ap", settings["macro_metric"], *[f"hit{k}" for k in evaluation["hit_ks"]],
                        *[f"ndcg{k}" for k in evaluation["ndcg_ks"]], *sorted(precision)]}
    keys = pd.DataFrame({column: pd.Categorical(result[column], categories=list(dict.fromkeys(values)), ordered=True)
                         for column, values in order.items()})
    if keys.isna().any().any():
        raise ValueError("a result row has a value outside the expected categories")
    return result.loc[keys.sort_values(list(order)).index].reset_index(drop=True)


class ModelFiles:

    def __init__(self, config: Config):
        self.config = config
        self.files_dir = config.path("paths.trained_files_dir")
        trained_dir = config.path("paths.trained_dir")
        self.units_csv = trained_dir / "evidence_units.csv"
        self.left_out_csv = trained_dir / "combinations_left_out.csv"
        self.skipped = set(config.get("trained_encoder.files.skipped_runs") or [])
        unknown = self.skipped - set(run_names(config))
        if unknown:
            raise ValueError(f"skipped_runs names runs that do not exist: {sorted(unknown)}")

    @property
    def evidence_path(self):
        return self.files_dir / "model__evidence.csv.gz"

    @property
    def result_path(self):
        return self.files_dir / "model__result.csv"

    def unit_status(self, job: TrainedRetrieval, spec: dict, mask: np.ndarray) -> tuple[str, str, object]:
        name = job.unit_name(spec)
        output = job.out_dir / f"{name}.csv.gz"
        record_path = job.out_dir / f"{name}{RECORD_SUFFIX}"
        if not record_path.exists():
            return "no record", "", output
        record = json.loads(record_path.read_text())
        ids = [f"{side}:{exam_id}" for side in ("query", "database") for exam_id in job.inputs.side(side)["exam_id"]]
        checks = [
            ("record of another step", record["step"] == job.step),
            ("queries or database differ", record["inputs"]["n"] == len(ids)
             and record["inputs"]["ids_sha256"] == sha256_rows(ids)),
            ("settings differ", record["settings_sha256"] == sha256_json(job.step_settings(spec, mask))),
            ("output file missing", output.exists()),
            ("output file changed", output.exists() and record["outputs"].get(output.name) == sha256_file(output)),
            ("written with uncommitted changes", not record["code"]["uncommitted_changes"]),
        ]
        failed = [reason for reason, ok in checks if not ok]
        return (failed[0] if failed else "valid"), record["code"]["commit"], output

    def run(self) -> None:
        units, parts = [], []
        for corpus in self.config.get("cohort.corpora"):
            job = TrainedRetrieval(self.config, corpus)
            mask = job.inputs.label_mask()
            for spec in job.specs():
                status, commit, output = self.unit_status(job, spec, mask)
                rows = 0
                if status == "valid":
                    part = read_evidence(output)
                    rows = len(part)
                    parts.append(part)
                units.append({"test_corpus": corpus, "run": spec["run"], "unit": job.unit_name(spec),
                              "status": status, "commit": commit, "rows": rows})
        units = pd.DataFrame(units, columns=UNIT_COLUMNS)
        units.to_csv(self.units_csv, index=False)
        invalid = units[(units["status"] != "valid") & ~units["run"].isin(self.skipped)]
        if len(invalid):
            raise RuntimeError(f"{len(invalid)} evidence units of runs not listed as skipped are not valid; see "
                               f"{self.units_csv.name}")
        evidence = pd.concat(parts, ignore_index=True)[EVIDENCE_COLUMNS]
        self.files_dir.mkdir(parents=True, exist_ok=True)
        partial = self.evidence_path.with_name(self.evidence_path.name + ".partial")
        with open(partial, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            evidence.to_csv(handle, index=False)
        os.replace(partial, self.evidence_path)
        result, left_out = result_rows(read_evidence(self.evidence_path), self.config)
        result.to_csv(self.result_path, index=False)
        left_out.to_csv(self.left_out_csv, index=False)
