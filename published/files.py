import gzip
import os

import numpy as np
import pandas as pd

from lib.exam_encoder import LABELS
from config import Config
from controls.controls_files import Bootstrap, ResultRows
from published.population import published_dir
from published.retrieval import EVIDENCE_COLUMNS, PublishedRetrieval
from published.study_common import StudyCommon

RESULT_COLUMNS = ["protocol", "method", "seeds", "test_corpus", "archive", "rule", "query_inputs", "level", "finding",
                  "metric", "mean", "std", "ci_low", "ci_high", "mean_raw", "n", "unit"]
KEY = ["protocol", "method", "seeds", "test_corpus", "archive", "rule", "query_inputs", "level", "finding", "metric"]
COMBINATION = ["method", "test_corpus", "archive", "rule", "query_inputs"]
ALL, COMMON = "published_all", "study_common"


def read_evidence(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={c: str for c in EVIDENCE_COLUMNS if c != "value"}, keep_default_na=False,
                       na_values={"value": [""]}, float_precision="round_trip")


class ImageSummaries:

    def __init__(self, bootstrap: Bootstrap, findings: list[str], names: dict, zero_metric: str):
        self.bootstrap, self.findings, self.names, self.zero_metric = bootstrap, findings, names, zero_metric

    def matrices(self, part: pd.DataFrame, seeds: list[str], metric: str, queries: np.ndarray) -> np.ndarray:
        out = np.full((len(seeds), len(queries), len(self.findings)), np.nan)
        position = {q: i for i, q in enumerate(queries)}
        column = {f: i for i, f in enumerate(self.findings)}
        for s, seed in enumerate(seeds):
            rows = part[(part["seed"] == seed) & (part["metric"] == metric)]
            out[s, rows["image_id"].map(position).to_numpy(), rows["finding"].map(column).to_numpy()] = rows["value"]
        return out

    @staticmethod
    def finding_means(counts: np.ndarray, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        sums = counts @ np.where(mask, values, 0.0)
        drawn = counts @ mask.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(drawn > 0, sums / np.where(drawn > 0, drawn, 1.0), np.nan)

    def rows(self, part: pd.DataFrame, method: str, seeds: list[str], keys: dict) -> tuple[list[dict], list[dict]]:
        queries = np.array(sorted(part["image_id"].unique()))
        zero = self.matrices(part, seeds, self.zero_metric, queries)
        positive = ~np.isnan(zero[0])
        if any((~np.isnan(zero[s]) != positive).any() for s in range(len(seeds))):
            raise ValueError(f"{method}: the seeds do not have the same positive pairs")
        rows, omitted = [], []
        protocol = ALL
        masks = [positive for s in range(len(seeds))]
        for metric in self.names["metrics"]:
            values = self.matrices(part, seeds, metric, queries)
            base = {**keys, "protocol": protocol, "metric": metric}
            undefined = [f for i, f in enumerate(self.findings) if not all(m[:, i].any() for m in masks)]
            for i, finding in enumerate(self.findings):
                row = {**base, "finding": finding}
                members = np.flatnonzero(positive[:, i])
                if finding in undefined:
                    omitted.append({**row, "n": 0, "cause": "no query left in a seed"})
                    continue
                counts = self.bootstrap.counts(len(members))
                per_seed = [self.finding_means(counts, values[s][members][:, [i]], masks[s][members][:, [i]])[:, 0]
                            for s in range(len(seeds))]
                n = int(np.any([m[members, i] for m in masks], axis=0).sum())
                self.add(rows, omitted, row, np.mean(per_seed, axis=0), n)
            counts = self.bootstrap.counts(len(queries))
            weights = counts @ positive.astype(np.float64)
            macro, weighted = [], []
            for s in range(len(seeds)):
                means = self.finding_means(counts, values[s], masks[s])
                complete = ~np.isnan(means).any(axis=1)
                macro.append(np.where(complete, np.nanmean(np.where(complete[:, None], means, 0.0), axis=1),
                                      np.nan))
                with np.errstate(invalid="ignore", divide="ignore"):
                    weighted.append(np.where(complete, (np.where(complete[:, None], means, 0.0) * weights).sum(axis=1)
                                             / weights.sum(axis=1), np.nan))
            n = int(np.any([m.any(axis=1) for m in masks], axis=0).sum())
            for name, estimates in ((self.names["macro"], macro), (self.names["weighted"], weighted)):
                row = {**base, "finding": name}
                if undefined:
                    omitted.append({**row, "n": n, "cause": f"requires {', '.join(undefined)}"})
                    continue
                self.add(rows, omitted, row, np.mean(estimates, axis=0), n)
        return rows, omitted

    def add(self, rows: list, omitted: list, row: dict, estimates: np.ndarray, n: int) -> None:
        summary = self.bootstrap.summary(estimates)
        if summary is None:
            omitted.append({**row, "n": n, "cause": "no finite replicate estimate"})
        else:
            rows.append({**row, **summary, "n": n})


class PublishedFiles:

    def __init__(self, config: Config):
        self.config = config
        self.settings = config.get("published.files")
        root = published_dir(config)
        self.files_dir = root / self.settings["folder"]
        self.units_csv = root / "evidence_units.csv"
        self.omitted_csv = root / "omitted_summaries.csv"
        self.replicates_csv = root / "bootstrap_replicates.csv"
        self.separator = config.get("trained_encoder.files.seeds_separator")
        self.bootstrap = Bootstrap(config.get("controls.files.bootstrap"), config.get("retrieval.files.percent_scale"))

    @property
    def evidence_path(self):
        return self.files_dir / "published__evidence.csv.gz"

    @property
    def result_path(self):
        return self.files_dir / "published__result.csv"

    def units(self) -> pd.DataFrame:
        rows = PublishedRetrieval(self.config).status() + StudyCommon(self.config).status()
        return pd.DataFrame(rows)

    def seed_sets(self, seeds: list[str]) -> list[list[str]]:
        seeds = sorted(seeds, key=lambda seed: (not seed.isdigit(), int(seed) if seed.isdigit() else 0, seed))
        return ([seeds] if len(seeds) > 1 else []) + [[seed] for seed in seeds]

    def image_rows(self, evidence: pd.DataFrame) -> tuple[list[dict], list[dict]]:
        settings = self.config.get("published.retrieval")
        names = {"metrics": list(settings["metrics"]), "macro": self.settings["macro_name"],
                 "weighted": self.settings["weighted_name"]}
        summaries = ImageSummaries(self.bootstrap, self.config.get("published.findings"), names,
                                   self.settings["zero_hit_metric"])
        rows, omitted = [], []
        for key, part in evidence[evidence["protocol"] == ALL].groupby(COMBINATION, sort=False):
            keys = {**dict(zip(COMBINATION, key)), "level": settings["level"]}
            for seeds in self.seed_sets(sorted(part["seed"].unique())):
                done, skipped = summaries.rows(part[part["seed"].isin(seeds)], key[0], seeds,
                                               {**keys, "seeds": self.separator.join(seeds)})
                rows += done
                omitted += skipped
        return rows, omitted

    def common_rows(self, evidence: pd.DataFrame) -> tuple[list[dict], list[dict]]:
        result_rows = ResultRows(self.config)
        rows, omitted = [], []
        for key, part in evidence[evidence["protocol"] == COMMON].groupby(COMBINATION, sort=False):
            keys = {**dict(zip(COMBINATION, key)), "protocol": COMMON}
            for seeds in self.seed_sets(sorted(part["seed"].unique())):
                values = part[part["seed"].isin(seeds)]
                per_key = values.groupby(["level", "finding", "metric", "exam_id"], sort=False)["value"].agg(
                    ["mean", "size"])
                if (per_key["size"] < len(seeds)).any():
                    omitted.append({**keys, "seeds": self.separator.join(seeds), "level": "", "finding": "",
                                    "metric": "", "n": 0, "cause": "a query-metric key without every seed"})
                    continue
                combined = per_key["mean"].rename("value").reset_index()
                done, skipped = result_rows.summaries(result_rows.row_tasks(combined),
                                                      {**keys, "seeds": self.separator.join(seeds)})
                rows += done
                omitted += [{**row, "cause": row.get("cause", "")} for row in skipped]
        return rows, omitted

    def ordered(self, result: pd.DataFrame) -> pd.DataFrame:
        evaluation = self.config.get("retrieval.evaluation")
        files = self.config.get("retrieval.files")
        precision = sorted(m for m in result["metric"].unique() if m.startswith("p") and "_f1_" in m)
        seeds = list(dict.fromkeys(sorted(result["seeds"].unique(), key=lambda s: (self.separator not in s, s))))
        order = {"protocol": [ALL, COMMON], "method": self.settings["method_order"], "seeds": seeds,
                 "archive": ["documented", "undocumented"], "rule": self.settings["rule_order"],
                 "query_inputs": ["all", "frontal_only"], "level": ["finding", "report"],
                 "finding": [*self.config.get("published.findings"), *LABELS, self.settings["macro_name"],
                             self.settings["weighted_name"], files["core_name"], "pooled"],
                 "metric": [*self.config.get("published.retrieval.metrics"), "ap", files["macro_metric"],
                            *[f"hit{k}" for k in evaluation["hit_ks"]], *[f"ndcg{k}" for k in evaluation["ndcg_ks"]],
                            *precision]}
        keys = pd.DataFrame({c: pd.Categorical(result[c], categories=list(dict.fromkeys(v)), ordered=True)
                             for c, v in order.items()})
        if keys.isna().any().any():
            raise ValueError("a result row has a value outside the expected categories")
        return result.loc[keys.sort_values(list(order)).index].reset_index(drop=True)

    def run(self) -> None:
        units = self.units()
        units.drop(columns="path").to_csv(self.units_csv, index=False)
        invalid = units[units["status"] != "valid"]
        if len(invalid):
            raise RuntimeError(f"{len(invalid)} evidence units are not valid; see {self.units_csv.name}")
        evidence = pd.concat([read_evidence(path) for path in units["path"]], ignore_index=True)[EVIDENCE_COLUMNS]
        self.files_dir.mkdir(parents=True, exist_ok=True)
        partial = self.evidence_path.with_name(self.evidence_path.name + ".partial")
        with open(partial, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            evidence.to_csv(handle, index=False)
        os.replace(partial, self.evidence_path)
        evidence = read_evidence(self.evidence_path)
        image_rows, image_omitted = self.image_rows(evidence)
        common_rows, common_omitted = self.common_rows(evidence)
        result = pd.DataFrame(image_rows + common_rows).assign(unit=self.settings["unit"])
        result = self.ordered(result)
        result[RESULT_COLUMNS].to_csv(self.result_path, index=False)
        result[[*KEY, "n", "finite"]].to_csv(self.replicates_csv, index=False)
        omitted = pd.DataFrame(image_omitted + common_omitted)
        omitted.to_csv(self.omitted_csv, index=False)
