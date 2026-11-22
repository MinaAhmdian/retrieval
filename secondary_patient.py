"""
secondary_patient.py
Created on November 22, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from downstream.downstream_differences import ARMS, SEEDS, boot_p, weighted_auroc
from final_differences import counts_matrix, summarize
from lib.stats_utils import bh_fdr
from patient_comparisons import cluster_difference


def common_chextriev(records, queries):
    data = records[(records.protocol == "study_common") & records.metric.isin(["ap", "ndcg10"])]
    rows = []
    for (archive, metric), group in data.groupby(["archive", "metric"]):
        group = group[group.method.isin(["encoder_frontal", "chextriev_rerun"])]
        if group.duplicated(["method", "seed", "exam_id", "finding"]).any():
            raise ValueError("Duplicate matched-subset scores.")
        cfg = group.groupby(["method", "exam_id", "finding"]).seed.agg(frozenset)
        for key, seeds in cfg.items():
            if seeds != (frozenset(SEEDS) if key[0] == "encoder_frontal" else {"42"}):
                raise ValueError("Incomplete matched-subset seeds.")
        avg = group.groupby(["method", "exam_id", "finding"], as_index=False).value.mean().rename(columns={"value": "score"})
        source_metric = "macro_ap" if metric == "ap" else metric
        cols = ["exam_id", "finding", "score"]
        vals = cluster_difference(avg.loc[avg.method == "encoder_frontal", cols], avg.loc[avg.method == "chextriev_rerun", cols],
                                  queries, source_metric)
        rows.append(dict(analysis="common_chextriev", corpus="mimic", archive=archive, metric=source_metric, **vals))
    return pd.DataFrame(rows)


def downstream_auroc(predictions, cohort):
    data = predictions[predictions.task == "knn10"]
    rows = []
    for (corpus, archive), group in data.groupby(["test_corpus", "archive"]):
        if set(group.seed) != set(ARMS):
            raise ValueError("Unexpected ranking seeds.")
        keys = ["exam_id", "finding"]
        scores = group.pivot(index=keys, columns="seed", values="score")
        targets = group.pivot(index=keys, columns="seed", values="target")
        if scores.isna().any().any() or targets.isna().any().any() or not targets.eq(targets["none"], axis=0).all().all():
            raise ValueError("Unmatched scores or targets across ranking methods.")
        y = targets["none"]
        keep = y.groupby("finding").transform("nunique") == 2
        scores, y = scores.loc[keep], y.loc[keep]
        subjects = cohort.loc[corpus].reindex(scores.index.get_level_values("exam_id"))
        if subjects.isna().any():
            raise ValueError("Missing query identifier.")
        patients, pos = np.unique(subjects, return_inverse=True)
        weights = np.vstack([np.ones(len(patients)), counts_matrix(len(patients))])
        findings = scores.groupby("finding").indices
        macro = np.zeros(len(weights))
        for idx in findings.values():
            truth = y.iloc[idx].to_numpy()
            def auc(arm):
                return weighted_auroc(truth, scores[arm].iloc[idx].to_numpy(), weights[:, pos[idx]])
            macro += (sum(auc(seed) for seed in SEEDS) / len(SEEDS) - auc("none")) / len(findings)
        observed, boot = macro[0], macro[1:]
        mean, sd, lo, hi = summarize(boot)
        rows.append(dict(analysis="downstream", corpus=corpus, archive=archive, metric="macro_auroc",
                         exams=scores.index.get_level_values("exam_id").nunique(), patients=len(patients),
                         findings=len(findings), finite=np.isfinite(boot).sum(), observed=observed,
                         mean=mean, std=sd, ci_low=lo, ci_high=hi, p_raw=boot_p(observed, boot)))
    return pd.DataFrame(rows)


def main_secondary_patient(config_path):
    cfg = Config(config_path).get("statistics")
    files = cfg["inputs"]
    cohort = pd.read_csv(files["cohort"], usecols=["corpus","exam_id","patient_id","query_or_database"], dtype=str)
    cohort = cohort[cohort.query_or_database == "query"].set_index(["corpus","exam_id"], verify_integrity=True).patient_id
    published = pd.read_csv(files["published_per_query"], dtype={"exam_id": str, "seed": str}, low_memory=False)
    chex = common_chextriev(published, cohort.loc["mimic"])
    pred = pd.read_csv(files["downstream_per_query"], usecols=["test_corpus","task","archive","exam_id","finding","target","seed","score"],
                       dtype={"exam_id": str, "seed": str}, low_memory=False)
    auc = downstream_auroc(pred, cohort)
    for frame in (chex, auc):
        frame["p_adjusted"] = bh_fdr(frame.p_raw.to_numpy())
    dest = Path(cfg["output_prefix"] + "_secondary_patient.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".csv.partial")
    pd.concat([chex,auc], ignore_index=True).to_csv(tmp, index=False)
    tmp.replace(dest)
