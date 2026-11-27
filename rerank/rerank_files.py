import gzip
import os

import numpy as np
import pandas as pd

from lib.exam_encoder import LABELS
from config import Config
from controls.controls_files import Bootstrap
from rerank.rerank_retrieval import (CLASSIFIER, CLASSIFIER_RERANK, EVIDENCE_COLUMNS, FIRST_STAGES, METHODS, NO_SEED,
                                     OURS, PROFILE_RERANK, TWO_STAGE_OURS, DepthRetrieval, RerankRetrieval,
                                     read_rerank_evidence, unit_name)

COMBINATION = ["method", "first_stage", "reranker", "test_corpus", "archive", "k", "query_inputs"]
ROW = ["level", "finding", "metric"]
RESULT_COLUMNS = ["method", "first_stage", "reranker", "seeds", "test_corpus", "archive", "k", "query_inputs", "level",
                  "finding", "metric", "mean", "std", "ci_low", "ci_high", "mean_raw", "n", "unit"]
UNIT_COLUMNS = ["test_corpus", "step", "unit", "status", "commit", "rows"]
FROZEN_METHOD = {CLASSIFIER: CLASSIFIER_RERANK, "profile": PROFILE_RERANK}


class RerankResultRows:

    def __init__(self, config: Config, group: str):
        self.config = config
        self.settings = config.get(group)
        self.files = config.get("retrieval.files")
        self.share = self.settings["share"]
        self.separator = config.get("trained_encoder.files.seeds_separator")
        self.seeds = [run.split("__")[2] for run in self.settings["encoder_runs"]]
        self.bootstrap = Bootstrap(config.get("controls.files.bootstrap"), self.files["percent_scale"])
        self.unit = config.get("controls.files.unit")

    def macro_task(self, values: pd.DataFrame, metric: str, name: str, findings: list[str], macro_metric: str):
        table = values[values["metric"] == metric].pivot(index="exam_id", columns="finding", values="value")
        part = table.sort_index()[findings].dropna(how="all")
        return {"level": "finding", "finding": name, "metric": macro_metric, "queries": part.index.to_numpy(),
                "values": part.to_numpy(dtype=np.float64)}

    def row_tasks(self, values: pd.DataFrame) -> list[dict]:
        tasks = []
        for (level, finding, metric), part in values.groupby(ROW, sort=False, observed=True):
            part = part.sort_values("exam_id")
            tasks.append({"level": level, "finding": finding, "metric": metric, "queries": part["exam_id"].to_numpy(),
                          "values": part["value"].to_numpy(dtype=np.float64)})
        ap = values[values["metric"] == "ap"]
        if len(ap):
            labels = [label for label in LABELS if label in set(ap["finding"])]
            tasks.append(self.macro_task(values, "ap", self.files["macro_name"], labels, self.files["macro_metric"]))
            if set(self.files["core_findings"]) <= set(labels):
                tasks.append(self.macro_task(values, "ap", self.files["core_name"], list(self.files["core_findings"]),
                                             self.files["macro_metric"]))
        share = values[values["metric"] == self.share["name"]]
        if len(share):
            counts = share.groupby("finding", observed=True)["exam_id"].nunique()
            findings = [label for label in LABELS if counts.get(label, 0) >= self.share["min_queries"]]
            if findings:
                tasks.append(self.macro_task(values, self.share["name"], self.files["macro_name"], findings,
                                             self.share["macro_metric"]))
        return tasks

    def summaries(self, tasks: list[dict], keys: dict) -> list[dict]:
        rows = []
        for task in tasks:
            values = task["values"]
            estimates = (self.bootstrap.macro_estimates(values) if values.ndim == 2
                         else self.bootstrap.mean_estimates(values))
            summary = self.bootstrap.summary(estimates)
            if summary is None:
                raise ValueError(f"{keys} {task['finding']} {task['metric']}: no finite replicate estimate")
            rows.append({**keys, "level": task["level"], "finding": task["finding"], "metric": task["metric"],
                         "n": len(task["queries"]), **summary})
        return rows

    def compute(self, evidence: pd.DataFrame) -> pd.DataFrame:
        wanted = {tuple(row[c] for c in ROW) for row in self.settings["files"]["per_seed_rows"]}
        rows = []
        for key, part in evidence.groupby(COMBINATION, sort=False, observed=True):
            keys = dict(zip(COMBINATION, key))
            seeds = sorted(part["seed"].unique(), key=lambda s: self.seeds.index(s) if s in self.seeds else -1)
            if seeds == [NO_SEED]:
                rows += self.summaries(self.row_tasks(part), {**keys, "seeds": NO_SEED})
                continue
            if seeds != self.seeds:
                raise ValueError(f"{keys}: seeds {seeds}, expected {self.seeds}")
            per_key = part.groupby([*ROW, "exam_id"], sort=False, observed=True)["value"].agg(["mean", "size"])
            if (per_key["size"] != len(seeds)).any():
                raise ValueError(f"{keys}: {int((per_key['size'] != len(seeds)).sum())} keys without every seed")
            combined = per_key["mean"].rename("value").reset_index()
            rows += self.summaries(self.row_tasks(combined), {**keys, "seeds": self.separator.join(seeds)})
            for seed in seeds:
                tasks = [t for t in self.row_tasks(part[part["seed"] == seed])
                         if (t["level"], t["finding"], t["metric"]) in wanted]
                rows += self.summaries(tasks, {**keys, "seeds": seed})
        result = pd.DataFrame(rows)
        result["unit"] = self.unit
        return self.ordered(result)[[*RESULT_COLUMNS, "finite"]]

    def ordered(self, result: pd.DataFrame) -> pd.DataFrame:
        frozen = [r for r in dict.fromkeys(result["reranker"]) if r != OURS]
        metrics = ["ap", self.files["macro_metric"], *self.settings["metrics"], self.share["macro_metric"]]
        order = {"method": METHODS, "first_stage": FIRST_STAGES, "reranker": [*frozen, OURS],
                 "seeds": [self.separator.join(self.seeds), *self.seeds, NO_SEED],
                 "test_corpus": self.config.get("cohort.corpora"), "archive": ["documented", "undocumented"],
                 "k": sorted(set(result["k"])), "query_inputs": list(self.config.get("trained_encoder.query_inputs")),
                 "level": ["finding", "report"],
                 "finding": [*LABELS, self.files["macro_name"], self.files["core_name"], "pooled"],
                 "metric": list(dict.fromkeys(metrics))}
        keys = pd.DataFrame({column: pd.Categorical(result[column], categories=list(dict.fromkeys(values)),
                                                    ordered=True) for column, values in order.items()})
        if keys.isna().any().any():
            raise ValueError("a result row has a value outside the expected categories")
        return result.loc[keys.sort_values(list(order)).index].reset_index(drop=True)


def read_result(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"seeds": str, "k": int}, keep_default_na=False, float_precision="round_trip")


def best_rerankers(result: pd.DataFrame, config: Config) -> pd.DataFrame:
    settings = config.get("rerank")
    separator = config.get("trained_encoder.files.seeds_separator")
    combined = separator.join(run.split("__")[2] for run in settings["encoder_runs"])
    metric_rows = {row["metric"]: (row["level"], row["finding"]) for row in settings["files"]["per_seed_rows"]}
    rows = []
    for first_stage in FIRST_STAGES:
        method = FROZEN_METHOD[first_stage]
        seeds = NO_SEED if first_stage == CLASSIFIER else combined
        part = result[(result["method"] == method) & (result["seeds"] == seeds)]
        for metric in settings["selection_metrics"]:
            level, finding = metric_rows[metric]
            rows_m = part[(part["metric"] == metric) & (part["level"] == level) & (part["finding"] == finding)]
            for (corpus, archive), group in rows_m.groupby(["test_corpus", "archive"], sort=False):
                ranked = group.sort_values("mean_raw", ascending=False, kind="stable")
                top = ranked.iloc[0]
                tied = int((ranked["mean_raw"] == top["mean_raw"]).sum())
                rows.append({"first_stage": first_stage, "method": method, "seeds": seeds, "test_corpus": corpus,
                             "archive": archive, "metric": metric, "reranker": top["reranker"],
                             "mean_raw": top["mean_raw"], "runner_up": ranked.iloc[1]["reranker"],
                             "runner_up_mean_raw": ranked.iloc[1]["mean_raw"], "candidates": len(ranked),
                             "tied_at_top": tied})
    selections = pd.DataFrame(rows)
    if (selections["tied_at_top"] > 1).any():
        raise ValueError("a best-reranker selection is tied at the top")
    return selections


def two_stage_equality(config: Config) -> pd.DataFrame:
    from controls.controls_retrieval import ControlsRetrieval, read_controls_evidence
    rows = []
    for corpus in config.get("rerank.test_corpora"):
        job = RerankRetrieval(config, corpus)
        controls = ControlsRetrieval(config, corpus)
        mask = controls.inputs.label_mask()
        reevaluated = {(s["run"], s["archive"], s["query_inputs"]): s for s in controls.specs()
                       if s["source"] == "reevaluated" and s["rule"] == "two_stage" and s["k"] == job.depth}
        for spec in job.specs():
            if spec["method"] != TWO_STAGE_OURS:
                continue
            control = reevaluated[(spec["run"], spec["archive"], spec["query_inputs"])]
            status, _ = controls.status(control, mask)
            if status != "valid":
                raise RuntimeError(f"[{corpus}] the controls unit of {spec['run']} {spec['archive']} is {status}")
            ours = read_rerank_evidence(job.path(spec))
            theirs = read_controls_evidence(controls.path(control))
            key = ["level", "finding", "metric", "exam_id"]
            for metric in sorted(set(ours["metric"]) & set(theirs["metric"].astype(str))):
                a = ours[ours["metric"] == metric][[*key, "value"]].astype({"level": str, "finding": str, "metric": str})
                b = theirs[theirs["metric"] == metric][[*key, "value"]].astype({"level": str, "finding": str,
                                                                                 "metric": str})
                merged = a.merge(b, on=key, how="outer", suffixes=("_rerank", "_controls"), indicator=True)
                both = merged[merged["_merge"] == "both"]
                rows.append({"test_corpus": corpus, "archive": spec["archive"], "seed": spec["seed"], "metric": metric,
                             "rows_rerank": len(a), "rows_controls": len(b), "rows_matched": len(both),
                             "rows_equal": int((both["value_rerank"] == both["value_controls"]).sum()),
                             "max_abs_difference": float((both["value_rerank"] - both["value_controls"]).abs().max())})
    table = pd.DataFrame(rows)
    table["equal"] = ((table["rows_rerank"] == table["rows_controls"]) & (table["rows_matched"] == table["rows_rerank"])
                      & (table["rows_equal"] == table["rows_rerank"]))
    return table


class GroupFiles:

    group = ""
    retrieval_job = None

    def __init__(self, config: Config):
        self.config = config
        self.settings = config.get(self.group)
        self.root = config.path(f"paths.{self.group}_dir")
        self.files_dir = self.root / self.settings["files"]["folder"]
        self.units_csv = self.root / "evidence_units.csv"
        self.replicates_csv = self.root / "bootstrap_replicates.csv"

    @property
    def evidence_path(self):
        return self.files_dir / f"{self.group}__evidence.csv.gz"

    @property
    def result_path(self):
        return self.files_dir / f"{self.group}__result.csv"

    def jobs(self) -> list:
        return [self.retrieval_job(self.config, corpus) for corpus in self.settings["test_corpora"]]

    def unit_parts(self) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
        units, parts = [], []
        for job in self.jobs():
            mask = job.inputs.label_mask()
            for spec in job.specs():
                status, commit = job.status(spec, mask)
                rows = 0
                if status == "valid":
                    part = read_rerank_evidence(job.path(spec))
                    rows = len(part)
                    parts.append(part)
                units.append({"test_corpus": job.corpus_name, "step": job.step, "unit": unit_name(spec),
                              "status": status, "commit": commit, "rows": rows})
        return pd.DataFrame(units, columns=UNIT_COLUMNS), parts

    def before_result(self) -> None:
        pass

    def after_result(self, result: pd.DataFrame) -> None:
        pass

    def run(self) -> None:
        units, parts = self.unit_parts()
        units.to_csv(self.units_csv, index=False)
        invalid = units[units["status"] != "valid"]
        if len(invalid):
            raise RuntimeError(f"{len(invalid)} evidence units are not valid; see {self.units_csv.name}")
        self.before_result()
        evidence = pd.concat(parts, ignore_index=True)[EVIDENCE_COLUMNS]
        self.files_dir.mkdir(parents=True, exist_ok=True)
        partial = self.evidence_path.with_name(self.evidence_path.name + ".partial")
        with open(partial, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            evidence.to_csv(handle, index=False)
        os.replace(partial, self.evidence_path)
        result = RerankResultRows(self.config, self.group).compute(read_rerank_evidence(self.evidence_path))
        result[RESULT_COLUMNS].to_csv(self.result_path, index=False)
        result[[c for c in RESULT_COLUMNS if c not in ("mean", "std", "ci_low", "ci_high", "mean_raw", "unit")]
               + ["finite"]].to_csv(self.replicates_csv, index=False)
        self.after_result(read_result(self.result_path))


class RerankFiles(GroupFiles):
    group = "rerank"
    retrieval_job = RerankRetrieval

    @property
    def selections_path(self):
        return self.root / "best_rerankers.csv"

    @property
    def equality_path(self):
        return self.root / "two_stage_equality.csv"

    def before_result(self) -> None:
        table = two_stage_equality(self.config)
        table.to_csv(self.equality_path, index=False)
        if not table["equal"].all():
            raise RuntimeError(f"two_stage_ours differs from the controls units; see {self.equality_path.name}")

    def after_result(self, result: pd.DataFrame) -> None:
        best_rerankers(result, self.config).to_csv(self.selections_path, index=False)

    def selected(self, first_stage: str, corpus: str, archive: str, metric: str) -> str:
        selections = pd.read_csv(self.selections_path, dtype=str, keep_default_na=False)
        row = selections[(selections["first_stage"] == first_stage) & (selections["test_corpus"] == corpus)
                         & (selections["archive"] == archive) & (selections["metric"] == metric)]
        if len(row) != 1:
            raise LookupError(f"{len(row)} selections for {first_stage} {corpus} {archive} {metric}")
        return row["reranker"].iloc[0]


class DepthFiles(GroupFiles):
    group = "depth"
    retrieval_job = DepthRetrieval
