"""
paired_retrieval.py
Created on November 22, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from config import Config
from evidence_cache import ensure_evidence
from lib.stats_utils import bh_fdr
from patient_comparisons import cluster_difference


def exams_scores(db, corpus, metric, source, rule, inputs):
    k = 100 if rule == "two_stage" else 0
    result = db.execute("""SELECT exam_id, finding, avg(value) AS score, count(*) AS n, count(DISTINCT seed) AS s
        FROM ev WHERE archive='documented' AND split='test' AND src=? AND retriever='encoder' AND backbone='biovilt'
        AND configuration='full' AND test_corpus=? AND metric=? AND rule=? AND query_inputs=? AND k=?
        AND seed IN ('42','1337','2024') GROUP BY exam_id, finding""",
        [source, corpus, metric, rule, inputs, k]).df()
    if result.empty or (result.n != 3).any() or (result.s != 3).any():
        raise ValueError("Expected three distinct trained encoder runs per query and finding.")
    return result[["exam_id", "finding", "score"]]


def estimates(db, corpus, cases):
    patients = cases.set_index("exam_id", verify_integrity=True).patient_id
    rows = []
    for source_metric, metric in (("ap", "macro_ap"), ("ndcg10", "ndcg10")):
        full = exams_scores(db, corpus, source_metric, "controls", "two_stage", "all")
        profile = exams_scores(db, corpus, source_metric, "model", "profile", "all")
        for analysis, comparator, chosen in (
            ("two_stage_minus_profile", profile, cases),
            ("with_minus_without_available_lateral", exams_scores(db, corpus, source_metric, "controls", "two_stage", "no_lateral"), cases[cases.n_lateral > 0]),
            ("with_minus_without_available_context", exams_scores(db, corpus, source_metric, "controls", "two_stage", "no_context"), cases[cases.has_context_text]),
        ):
            a, b = (frame[frame.exam_id.isin(chosen.exam_id)] for frame in (full, comparator))
            if a.empty or (metric == "macro_ap" and a.groupby("finding").size().max() < 20):
                continue
            result = cluster_difference(a, b, patients, metric)
            rows.append(dict(corpus=corpus, analysis=analysis, metric=metric, subset_queries=len(chosen), **result))
    return rows


def main_paired_retrieval(config_path):
    cfg = Config(config_path)
    settings = cfg.get("statistics")
    raw = settings["inputs"]
    cases = pd.read_csv(raw["cohort"], usecols=["corpus", "exam_id", "patient_id", "query_or_database",
                         "n_lateral", "has_context_text"], dtype={"exam_id":str, "patient_id":str})
    if not cases.has_context_text.isin([True, False, 0, 1]).all():
        raise ValueError("Context availability must be binary.")
    cases["has_context_text"] = cases.has_context_text.astype(bool)
    database = Path(settings["database"])
    database.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with duckdb.connect(str(database)) as db:
        ensure_evidence(db, raw)
        for corpus in cfg.get("retrieval.documented_corpora"):
            subset = cases[(cases.corpus == corpus) & (cases.query_or_database == "query")]
            rows.extend(estimates(db, corpus, subset))
    result = pd.DataFrame(rows)
    result["p_adjusted"] = result.groupby("analysis")["p_raw"].transform(lambda values: bh_fdr(values.to_numpy()))
    dest = Path(settings["output_prefix"] + "_paired_retrieval.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".csv.partial")
    result.to_csv(tmp,index=False)
    tmp.replace(dest)


def score_archives(db, cohort, corpora, training, labels):
    fit = cohort[(cohort.split == "train") & cohort.corpus.isin(training)][labels]
    weights = (fit == 1).sum() / ((fit == 0) | (fit == 1)).sum()
    if not np.isfinite(weights).all():
        raise ValueError("Undefined training prevalence.")
    out = []
    for corpus in corpora:
        test = cohort[(cohort.corpus == corpus) & (cohort.split == "test")]
        pool = test[test.query_or_database == "database"].sort_values("exam_id", kind="stable")
        cases = test[test.query_or_database == "query"].set_index("exam_id",verify_integrity=True)
        model = exams_scores(db, corpus, "ap", "controls", "two_stage", "all")
        for rule, w in (("finding_count", np.ones(len(labels))), ("training_prevalence", weights.to_numpy())):
            order = np.argsort(-((pool[labels] == 1).to_numpy(dtype=float) @ w), kind="stable")
            values = {}
            for finding in model.finding.unique():
                judged = pool[finding].iloc[order].fillna(-2)
                relevant = (judged[judged != -1].to_numpy() == 1)
                ranks = np.flatnonzero(relevant) + 1
                values[finding] = (np.arange(1, len(ranks)+1)/ranks).mean()
            baseline = model[["exam_id","finding"]].copy()
            baseline["score"] = baseline.finding.map(values)
            result = cluster_difference(model, baseline, cases.patient_id, "macro_ap")
            out.append(dict(corpus=corpus,rule=rule,baseline_macro_ap=np.mean(list(values.values())),**result))
    res = pd.DataFrame(out)
    res["p_adjusted"] = bh_fdr(res.p_raw.to_numpy())
    return res


def main_burden(config_path):
    config = Config(config_path)
    settings = config.get("statistics")
    raw = settings["inputs"]
    labels = list(config.get("cohort.chexpert_label_names").values())
    cohort = pd.read_csv(raw["cohort"], usecols=["corpus","split","exam_id","patient_id","query_or_database",*labels],
                         dtype={"exam_id":str,"patient_id":str}, low_memory=False)
    database = Path(settings["database"])
    database.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database)) as db:
        ensure_evidence(db, raw)
        result = score_archives(db, cohort, config.get("retrieval.documented_corpora"), config.get("trained_encoder.train_corpora"), labels)
    dest = Path(settings["output_prefix"] + "_clinical_burden.csv")
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp = dest.with_suffix(".csv.partial")
    result.to_csv(tmp,index=False)
    tmp.replace(dest)
