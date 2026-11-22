import gzip
import os

import numpy as np
import pandas as pd

from lib.exam_encoder import LABELS
from config import Config
from controls.controls_retrieval import (EVIDENCE_COLUMNS, ControlsRetrieval, ValidRetrieval,
                                                 read_controls_evidence, unit_name)
from trained_encoder.trained_retrieval import RULES

COMBINATION = ["configuration", "backbone", "test_corpus", "split", "archive", "rule", "k", "query_inputs"]
ROW = ["level", "finding", "metric"]
RESULT_COLUMNS = ["retriever", "configuration", "backbone", "seeds", "test_corpus", "split", "archive", "rule", "k",
                  "query_inputs", "level", "finding", "metric", "mean", "std", "ci_low", "ci_high", "mean_raw", "n",
                  "unit"]
RESULT_KEY = [c for c in RESULT_COLUMNS if c not in ("mean", "std", "ci_low", "ci_high", "mean_raw", "n", "unit")]
UNIT_COLUMNS = ["split", "test_corpus", "step", "unit", "status", "commit", "rows"]


class Bootstrap:

    def __init__(self, settings: dict, scale: float):
        self.resamples = settings["resamples"]
        self.seed = settings["seed"]
        self.ddof = settings["ddof"]
        self.percentiles = settings["percentiles"]
        self.decimals = settings["decimals"]
        self.scale = scale
        self.cached = (None, None)

    def draws(self, n: int) -> np.ndarray:
        return np.random.RandomState(self.seed).randint(0, n, size=(self.resamples, n))

    def counts(self, n: int) -> np.ndarray:
        if self.cached[0] != n:
            draws = self.draws(n)
            flat = (draws + n * np.arange(self.resamples)[:, None]).ravel()
            self.cached = (n, np.bincount(flat, minlength=self.resamples * n).reshape(self.resamples, n)
                           .astype(np.float64))
        return self.cached[1]

    def mean_estimates(self, values: np.ndarray) -> np.ndarray:
        return self.counts(len(values)) @ values / len(values)

    def macro_estimates(self, values: np.ndarray) -> np.ndarray:
        counts = self.counts(len(values))
        present = ~np.isnan(values)
        sums = counts @ np.where(present, values, 0.0)
        drawn = counts @ present.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_label = np.where(drawn > 0, sums / np.where(drawn > 0, drawn, 1), np.nan)
        return per_label.mean(axis=1)

    def summary(self, estimates: np.ndarray) -> dict | None:
        finite = estimates[np.isfinite(estimates)]
        if not len(finite):
            return None
        low, high = np.percentile(finite, self.percentiles)
        mean_raw = float(finite.mean())
        std = float(np.std(finite, ddof=self.ddof)) if len(finite) > 1 else np.nan

        def scaled(value):
            return round(value * self.scale, self.decimals) if np.isfinite(value) else np.nan

        return {"mean": scaled(mean_raw), "std": scaled(std), "ci_low": scaled(float(low)),
                "ci_high": scaled(float(high)), "mean_raw": mean_raw, "finite": int(len(finite))}


def per_query(evidence: pd.DataFrame) -> pd.DataFrame:
    return evidence[[*ROW, "exam_id", "value"]]


class ResultRows:

    def __init__(self, config: Config):
        self.config = config
        self.files = config.get("controls.files")
        self.retrieval_files = config.get("retrieval.files")
        self.separator = config.get("trained_encoder.files.seeds_separator")
        self.bootstrap = Bootstrap(self.files["bootstrap"], self.retrieval_files["percent_scale"])
        self.single_seed = {c for cs in config.get("controls.runs.configurations").values() for c in cs}

    def row_tasks(self, values: pd.DataFrame) -> list[dict]:
        tasks = []
        for (level, finding, metric), part in values.groupby(ROW, sort=False, observed=True):
            part = part.sort_values("exam_id")
            tasks.append({"level": level, "finding": finding, "metric": metric, "queries": part["exam_id"].to_numpy(),
                          "values": part["value"].to_numpy(dtype=np.float64)})
        ap = values[values["metric"] == "ap"]
        if len(ap):
            table = ap.pivot(index="exam_id", columns="finding", values="value").sort_index()
            labels = [label for label in LABELS if label in table.columns]
            sets = {self.retrieval_files["macro_name"]: labels}
            if set(self.retrieval_files["core_findings"]) <= set(labels):
                sets[self.retrieval_files["core_name"]] = list(self.retrieval_files["core_findings"])
            for name, findings in sets.items():
                part = table[findings].dropna(how="all")
                tasks.append({"level": "finding", "finding": name, "metric": self.retrieval_files["macro_metric"],
                              "queries": part.index.to_numpy(), "values": part.to_numpy(dtype=np.float64)})
        return tasks

    def summaries(self, tasks: list[dict], keys: dict) -> tuple[list[dict], list[dict]]:
        rows, omitted = [], []
        for task in sorted(tasks, key=lambda t: len(t["queries"])):
            values = task["values"]
            estimates = (self.bootstrap.macro_estimates(values) if values.ndim == 2
                         else self.bootstrap.mean_estimates(values))
            summary = self.bootstrap.summary(estimates)
            row = {**keys, "level": task["level"], "finding": task["finding"], "metric": task["metric"],
                   "n": len(task["queries"])}
            if summary is None:
                omitted.append({**row, "cause": "no finite replicate estimate"})
                continue
            rows.append({**row, **summary})
        return rows, omitted

    def compute(self, evidence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        wanted = {tuple(row[c] for c in ROW) for row in self.files["per_seed_rows"]}
        rows, omitted, left_out = [], [], []
        for key, part in evidence.groupby(COMBINATION, sort=False, observed=True):
            keys = dict(zip(COMBINATION, key))
            seeds = sorted(part["seed"].unique())
            if keys["configuration"] in self.single_seed:
                if len(seeds) != 1:
                    raise ValueError(f"{keys}: a run of controls.runs with seeds {seeds}")
                done, skipped = self.summaries(self.row_tasks(per_query(part)), {**keys, "seeds": str(seeds[0])})
                rows += done
                omitted += skipped
                continue
            if len(seeds) < 2:
                raise ValueError(f"{keys}: one seed {seeds} for a configuration outside controls.runs")
            per_key = part.groupby([*ROW, "exam_id"], sort=False, observed=True)["value"].agg(["mean", "size"])
            short = int((per_key["size"] < len(seeds)).sum())
            if short:
                left_out.append({**keys, "seeds": self.separator.join(map(str, seeds)),
                                 "keys_without_every_seed": short})
            else:
                combined = per_key["mean"].rename("value").reset_index()
                done, skipped = self.summaries(self.row_tasks(combined),
                                               {**keys, "seeds": self.separator.join(map(str, seeds))})
                rows += done
                omitted += skipped
            for seed in seeds:
                tasks = [t for t in self.row_tasks(per_query(part[part["seed"] == seed]))
                         if (t["level"], t["finding"], t["metric"]) in wanted]
                done, skipped = self.summaries(tasks, {**keys, "seeds": str(seed)})
                rows += done
                omitted += skipped
        result = pd.DataFrame(rows)
        result["retriever"] = self.files["retriever"]
        result["unit"] = self.files["unit"]
        return (self.ordered(result)[[*RESULT_COLUMNS, "finite"]], pd.DataFrame(left_out),
                pd.DataFrame(omitted))

    def ordered(self, result: pd.DataFrame) -> pd.DataFrame:
        evaluation = self.config.get("retrieval.evaluation")
        runs = self.config.get("trained_encoder.runs")
        controls = self.config.get("controls.runs")
        configurations = [c for cs in controls["configurations"].values() for c in cs]
        configurations += [self.config.get("controls.shuffled_report.configuration")]
        configurations += [c for cs in runs["configurations"].values() for c in cs]
        backbones = [*controls["configurations"], *runs["configurations"]]
        seeds = [self.separator.join(map(str, runs["seeds"])), *map(str, runs["seeds"]), *map(str, controls["seeds"])]
        precision = sorted(m for m in result["metric"].unique() if m.startswith("p") and "_f1_" in m)
        order = {"split": ["valid", "test"], "configuration": configurations, "backbone": backbones, "seeds": seeds,
                 "test_corpus": self.config.get("cohort.corpora"), "archive": ["documented", "undocumented"],
                 "rule": RULES, "k": sorted(set(result["k"])), "query_inputs": list(self.config.get(
                     "trained_encoder.query_inputs")), "level": ["finding", "report"],
                 "finding": [*LABELS, self.retrieval_files["macro_name"], self.retrieval_files["core_name"], "pooled"],
                 "metric": ["ap", self.retrieval_files["macro_metric"], *[f"hit{k}" for k in evaluation["hit_ks"]],
                            *[f"ndcg{k}" for k in evaluation["ndcg_ks"]], *precision]}
        keys = pd.DataFrame({column: pd.Categorical(result[column], categories=list(dict.fromkeys(values)),
                                                    ordered=True) for column, values in order.items()})
        if keys.isna().any().any():
            raise ValueError("a result row has a value outside the expected categories")
        return result.loc[keys.sort_values(list(order)).index].reset_index(drop=True)


class ControlsFiles:

    def __init__(self, config: Config):
        self.config = config
        controls_dir = config.path("paths.controls_dir")
        self.files_dir = controls_dir / config.get("controls.files.folder")
        self.units_csv = controls_dir / "evidence_units.csv"
        self.left_out_csv = controls_dir / "combinations_left_out.csv"
        self.omitted_csv = controls_dir / "omitted_summaries.csv"
        self.replicates_csv = controls_dir / "bootstrap_replicates.csv"

    @property
    def evidence_path(self):
        return self.files_dir / "controls__evidence.csv.gz"

    @property
    def result_path(self):
        return self.files_dir / "controls__result.csv"

    def jobs(self) -> list:
        return ([ValidRetrieval(self.config, c) for c in self.config.get("trained_encoder.train_corpora")]
                + [ControlsRetrieval(self.config, c) for c in self.config.get("cohort.corpora")])

    def result_rows(self) -> ResultRows:
        return ResultRows(self.config)

    def run(self) -> None:
        units, parts = [], []
        for job in self.jobs():
            mask = job.inputs.label_mask()
            for spec in job.specs():
                status, commit = job.status(spec, mask)
                rows = 0
                if status == "valid":
                    part = read_controls_evidence(job.path(spec))
                    rows = len(part)
                    parts.append(part)
                units.append({"split": spec["split"], "test_corpus": job.corpus_name, "step": job.step,
                              "unit": unit_name(spec), "status": status, "commit": commit, "rows": rows})
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
        result, left_out, omitted = self.result_rows().compute(read_controls_evidence(self.evidence_path))
        result[RESULT_COLUMNS].to_csv(self.result_path, index=False)
        result[[*RESULT_KEY, "n", "finite"]].to_csv(self.replicates_csv, index=False)
        left_out.to_csv(self.left_out_csv, index=False)
        omitted.to_csv(self.omitted_csv, index=False)
