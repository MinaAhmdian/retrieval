import gzip
import os

import numpy as np
import pandas as pd

from lib.exam_encoder import LABELS
from config import Config
from controls.controls_files import Bootstrap
from downstream.downstream_metrics import f1, f1_counts, weighted_auroc
from downstream.downstream_predictions import (EVIDENCE_COLUMNS, NO_SEED, DownstreamPredictions,
                                                       downstream_dir, unit_name)

COMBINATION = ["retriever", "test_corpus", "archive", "rule", "k", "task"]
RESULT_COLUMNS = ["retriever", "seeds", "test_corpus", "archive", "rule", "k", "task", "finding", "metric", "mean", "std",
                  "ci_low", "ci_high", "mean_raw", "n", "unit"]
RESULT_KEY = ["retriever", "seeds", "test_corpus", "archive", "rule", "k", "task", "finding", "metric"]
UNIT_COLUMNS = ["test_corpus", "step", "unit", "status", "commit", "rows"]
AUROC, F1 = "auroc", "f1"


def read_downstream_evidence(path) -> pd.DataFrame:
    dtypes = {"exam_id": str, "seed": str, "k": int, "target": int,
              **{c: "category" for c in ("retriever", "test_corpus", "archive", "rule", "task", "finding")}}
    return pd.read_csv(path, dtype=dtypes, keep_default_na=False, float_precision="round_trip")


class FindingData:

    def __init__(self, part: pd.DataFrame, seeds: list[str]):
        table = part.pivot(index="exam_id", columns="seed", values="score").sort_index()
        if table[seeds].isna().any().any():
            raise ValueError("a query of a finding lacks a seed's score")
        self.queries = table.index.to_numpy()
        targets = part.drop_duplicates("exam_id").set_index("exam_id")["target"].loc[self.queries]
        if part.groupby("exam_id")["target"].nunique().max() > 1:
            raise ValueError("a query's target differs between seeds")
        self.targets = targets.to_numpy()
        self.scores = {seed: table[seed].to_numpy(dtype=np.float64) for seed in seeds}


class DownstreamResult:

    def __init__(self, config: Config):
        self.config = config
        self.settings = config.get("downstream")
        self.files = self.settings["files"]
        self.metrics = {task: settings["metric"] for task, settings in self.settings["tasks"].items()}
        self.separator = config.get("trained_encoder.files.seeds_separator")
        self.bootstrap = Bootstrap(config.get("controls.files.bootstrap"),
                                   config.get("retrieval.files.percent_scale"))

    def eligible(self, data: FindingData, metric: str) -> str | None:
        positive = int((data.targets == 1).sum())
        negative = int((data.targets == 0).sum())
        if positive < self.files["min_positive"]:
            return f"fewer than {self.files['min_positive']} positive queries ({positive})"
        if metric == AUROC and negative < self.files["min_negative"]:
            return f"fewer than {self.files['min_negative']} negative queries ({negative})"
        return None

    @staticmethod
    def seed_metric(metric: str, data: FindingData, seed: str, weights: np.ndarray) -> np.ndarray:
        if metric == AUROC:
            return weighted_auroc(data.scores[seed], data.targets, weights)
        return f1_counts(data.scores[seed], data.targets, weights)

    @staticmethod
    def combined(per_seed: list[np.ndarray]) -> np.ndarray:
        return np.mean(np.stack(per_seed), axis=0)

    def finding_estimates(self, metric: str, data: FindingData, seeds: list[str]) -> np.ndarray:
        weights = self.bootstrap.counts(len(data.queries))
        values = [self.seed_metric(metric, data, seed, weights) for seed in seeds]
        if metric == F1:
            values = [f1(v) for v in values]
        return self.combined(values)

    def union_estimates(self, metric: str, findings: dict, seeds: list[str], micro: bool) -> tuple[np.ndarray, int]:
        union = np.unique(np.concatenate([data.queries for data in findings.values()]))
        counts = self.bootstrap.counts(len(union))
        per_seed = []
        for seed in seeds:
            values = []
            for data in findings.values():
                weights = counts[:, np.searchsorted(union, data.queries)]
                values.append(self.seed_metric(metric, data, seed, weights))
            if micro:
                per_seed.append(f1(np.sum(values, axis=0)))
            elif metric == F1:
                per_seed.append(np.mean([f1(v) for v in values], axis=0))
            else:
                per_seed.append(np.mean(values, axis=0))
        return self.combined(per_seed), len(union)

    def summarize(self, keys: dict, finding: str, metric: str, estimates: np.ndarray, n: int, rows: list,
                  omitted: list) -> None:
        summary = self.bootstrap.summary(estimates)
        row = {**keys, "finding": finding, "metric": metric, "n": n}
        if summary is None:
            omitted.append({**row, "cause": "no finite resample estimate"})
        else:
            rows.append({**row, **summary})

    def seed_sets(self, seeds: list[str]) -> list[list[str]]:
        if seeds == [NO_SEED]:
            return [seeds]
        if len(seeds) < 2:
            raise ValueError(f"the encoder has the seeds {seeds}")
        return [seeds] + [[seed] for seed in seeds]

    def compute(self, evidence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows, omitted = [], []
        for key, part in evidence.groupby(COMBINATION, sort=False, observed=True):
            keys = dict(zip(COMBINATION, key))
            metric = self.metrics[keys["task"]]
            all_seeds = sorted(part["seed"].unique(), key=lambda s: (s == NO_SEED, int(s) if s.isdigit() else 0))
            by_finding = {finding: group for finding, group in part.groupby("finding", sort=False, observed=True)}
            for seeds in self.seed_sets(all_seeds):
                row_keys = {**keys, "seeds": self.separator.join(seeds)}
                findings = {}
                for finding in [label for label in LABELS if label in by_finding]:
                    group = by_finding[finding]
                    data = FindingData(group[group["seed"].isin(seeds)], seeds)
                    cause = self.eligible(data, metric)
                    if cause is not None:
                        omitted.append({**row_keys, "finding": finding, "metric": metric, "n": len(data.queries),
                                        "cause": cause})
                        continue
                    findings[finding] = data
                    self.summarize(row_keys, finding, metric, self.finding_estimates(metric, data, seeds),
                                   len(data.queries), rows, omitted)
                if not findings:
                    omitted.append({**row_keys, "finding": self.files["macro_name"], "metric": metric, "n": 0,
                                    "cause": "no eligible finding"})
                    continue
                estimates, n = self.union_estimates(metric, findings, seeds, micro=False)
                self.summarize(row_keys, self.files["macro_name"], metric, estimates, n, rows, omitted)
                if metric == F1:
                    estimates, n = self.union_estimates(metric, findings, seeds, micro=True)
                    self.summarize(row_keys, self.files["micro_name"], metric, estimates, n, rows, omitted)
        result = pd.DataFrame(rows)
        result["unit"] = self.config.get("controls.files.unit")
        return self.ordered(result)[[*RESULT_COLUMNS, "finite"]], pd.DataFrame(omitted)

    def ordered(self, result: pd.DataFrame) -> pd.DataFrame:
        runs = [run.split("__")[2] for run in self.settings["encoder_runs"]]
        order = {"retriever": [self.settings["encoder_retriever"], self.config.get("retrieval.classifier_retriever")],
                 "seeds": [self.separator.join(sorted(runs, key=int)), *sorted(runs, key=int), NO_SEED],
                 "test_corpus": self.settings["test_corpora"], "archive": ["documented", "undocumented"],
                 "task": list(self.settings["tasks"]),
                 "finding": [*LABELS, self.files["macro_name"], self.files["micro_name"]]}
        keys = pd.DataFrame({column: pd.Categorical(result[column].astype(str), categories=values, ordered=True)
                             for column, values in order.items()})
        if keys.isna().any().any():
            raise ValueError("a result row has a value outside the expected categories")
        return result.loc[keys.sort_values(list(order)).index].reset_index(drop=True)


class DownstreamFiles:

    def __init__(self, config: Config):
        self.config = config
        root = downstream_dir(config)
        self.files_dir = root / config.get("downstream.files.folder")
        self.units_csv = root / "evidence_units.csv"
        self.omitted_csv = root / "omitted_summaries.csv"
        self.replicates_csv = root / "bootstrap_replicates.csv"

    @property
    def evidence_path(self):
        return self.files_dir / "downstream__evidence.csv.gz"

    @property
    def result_path(self):
        return self.files_dir / "downstream__result.csv"

    def jobs(self) -> list[DownstreamPredictions]:
        return [DownstreamPredictions(self.config, corpus) for corpus in self.config.get("downstream.test_corpora")]

    def run(self) -> None:
        units, parts = [], []
        for job in self.jobs():
            mask = job.inputs.label_mask()
            for spec in job.specs():
                status, commit = job.status(spec, mask)
                rows = 0
                if status == "valid":
                    part = read_downstream_evidence(job.paths(spec)[0])
                    rows = len(part)
                    parts.append(part)
                units.append({"test_corpus": job.corpus_name, "step": job.step, "unit": unit_name(spec),
                              "status": status, "commit": commit, "rows": rows})
        units = pd.DataFrame(units, columns=UNIT_COLUMNS)
        units.to_csv(self.units_csv, index=False)
        invalid = units[units["status"] != "valid"]
        if len(invalid):
            raise RuntimeError(f"{len(invalid)} evidence units are not valid; see {self.units_csv.name}")
        evidence = pd.concat(parts, ignore_index=True)[EVIDENCE_COLUMNS]
        self.files_dir.mkdir(parents=True, exist_ok=True)
        partial = self.evidence_path.with_name(self.evidence_path.name + ".partial")
        with open(partial, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            evidence.to_csv(handle, index=False)
        os.replace(partial, self.evidence_path)
        result, omitted = DownstreamResult(self.config).compute(read_downstream_evidence(self.evidence_path))
        result[RESULT_COLUMNS].to_csv(self.result_path, index=False)
        result[[*RESULT_KEY, "n", "finite"]].to_csv(self.replicates_csv, index=False)
        omitted.to_csv(self.omitted_csv, index=False)
