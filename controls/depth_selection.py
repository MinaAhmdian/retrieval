import json

import numpy as np
import pandas as pd

from lib.fusion import select_depth
from config import Config
from controls.controls_retrieval import (EVIDENCE_COLUMNS, NO_DEPTH, ValidRetrieval, read_controls_evidence,
                                                 unit_name)
from controls.validation import validation_dir
from encoders.retrieval import LIB_DIR
from provenance import sha256_file
from trained_encoder.trained_retrieval import PROFILE, TWO_STAGE

QUERY_KEY = ["rule", "k", "level", "finding", "metric", "exam_id"]


def seed_means(evidence: pd.DataFrame, n_seeds: int) -> pd.DataFrame:
    grouped = evidence.groupby(QUERY_KEY, observed=True)["value"].agg(value="mean", seeds="size")
    grouped = grouped.reset_index()
    grouped.loc[grouped["seeds"] != n_seeds, "value"] = np.nan
    return grouped


def macro_ap(values: pd.DataFrame) -> float:
    ap = values[values["metric"] == "ap"]
    if not len(ap) or ap["value"].isna().any():
        return np.nan
    return float(ap.groupby("finding", observed=True)["value"].mean().mean())


def pooled(values: pd.DataFrame, metric: str) -> float:
    rows = values[(values["metric"] == metric) & (values["finding"] == "pooled")]
    return float(rows["value"].mean()) if len(rows) and not rows["value"].isna().any() else np.nan


class DepthSelection:

    step = "select-depth"

    def __init__(self, config: Config):
        self.config = config
        self.settings = config.get("controls.validation")
        self.corpora = config.get("trained_encoder.train_corpora")
        self.depths = [int(k) for k in self.settings["depths"]]
        self.path = validation_dir(config) / self.settings["depth_file"]

    def units(self, corpus: str) -> list[tuple[dict, object]]:
        job = ValidRetrieval(self.config, corpus)
        jobs, specs, missing = job.pending_units()
        if missing or jobs:
            raise RuntimeError(f"[{corpus}] validation units not current: {len(jobs)} to compute, runs with missing "
                               f"inputs {sorted(missing)}; run valid-retrieval --corpus {corpus}")
        return [(spec, job.path(spec)) for spec in specs]

    def means(self, corpus: str, units: list[tuple[dict, object]]) -> dict:
        evidence = pd.concat([read_controls_evidence(path) for _, path in units], ignore_index=True)[EVIDENCE_COLUMNS]
        values = seed_means(evidence, len(self.settings["weight_runs"]))
        ndcg, macro = [], []
        for k in self.depths:
            part = values[(values["rule"] == TWO_STAGE) & (values["k"] == k)]
            ndcg.append(pooled(part, self.settings["ndcg_metric"]))
            macro.append(macro_ap(part))
        profile = values[(values["rule"] == PROFILE) & (values["k"] == NO_DEPTH)]
        queries = evidence[evidence["metric"].astype(str).str.startswith("hit")]["exam_id"].nunique()
        report_queries = evidence[evidence["metric"] == self.settings["ndcg_metric"]]["exam_id"].nunique()
        return {"ndcg": ndcg, "macro_ap": macro, "profile_ap": macro_ap(profile),
                "labels_with_ap": int(profile[profile["metric"] == "ap"]["finding"].nunique()),
                "queries": int(queries), "report_queries": int(report_queries)}

    def run(self) -> None:
        inputs, sources = {}, {}
        for corpus in self.corpora:
            units = self.units(corpus)
            sources[corpus] = {unit_name(spec): sha256_file(path) for spec, path in units}
            inputs[corpus] = self.means(corpus, units)
        ndcg = np.array([inputs[c]["ndcg"] for c in self.corpora])
        macro = np.array([inputs[c]["macro_ap"] for c in self.corpora])
        profile = np.array([inputs[c]["profile_ap"] for c in self.corpora])
        finite = all(np.isfinite(x).all() for x in (ndcg, macro, profile))
        depth = select_depth(ndcg, macro, profile, depths=tuple(self.depths)) if finite else None
        saved = {"corpora": self.corpora, "depths": self.depths, "weight_runs": self.settings["weight_runs"],
                 "ndcg_metric": self.settings["ndcg_metric"], "ndcg": ndcg.tolist(), "macro_ap": macro.tolist(),
                 "profile_ap": profile.tolist(), "all_means_finite": finite, "depth": depth,
                 "depth_without": {c: select_depth(np.delete(ndcg, j, 0), np.delete(macro, j, 0), np.delete(profile, j), depths=tuple(self.depths)) for j, c in enumerate(self.corpora)} if finite else {},
                 "counts": {c: {k: inputs[c][k] for k in ("labels_with_ap", "queries", "report_queries")}
                            for c in self.corpora},
                 "fusion_lib_sha256": sha256_file(LIB_DIR / "fusion.py"), "units_sha256": sources}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_name(self.path.name + ".partial")
        partial.write_text(json.dumps(saved, indent=1))
        partial.replace(self.path)
        if depth is None:
            raise RuntimeError("no depth selected (a validation mean is undefined or select_depth returned None); "
                               "depth-dependent test evaluation stops")

    def saved(self) -> dict:
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path.name} does not exist; run select-depth first")
        return json.loads(self.path.read_text())

    def depth(self) -> int:
        depth = self.saved()["depth"]
        if depth is None:
            raise RuntimeError("select-depth selected no depth; depth-dependent test evaluation stops")
        return int(depth)
