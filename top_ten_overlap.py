"""
top_ten_overlap.py
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
from final_differences import MIN_POS, counts_matrix, summarize
from lib.stats_utils import bh_fdr
from patient_comparisons import cluster_difference


def top_ten_overlap(connection, predictions, cohort):
    labels = predictions[(predictions.archive == "documented") & (predictions.task == "knn10")
                         & (predictions.target == 1) & (predictions.finding != "no_finding")].copy()
    keys = ["retriever", "seed", "test_corpus", "exam_id"]
    reports = connection.execute("""SELECT retriever, seed, test_corpus, exam_id, value AS report_precision
        FROM ev WHERE split = 'test' AND archive = 'documented' AND metric = 'p10_f1_05'
        AND ((src = 'controls' AND retriever = 'encoder' AND configuration = 'full' AND backbone = 'biovilt'
              AND rule = 'two_stage' AND k = 100 AND query_inputs = 'all')
          OR (src = 'encoders' AND retriever = 'finding_classifier' AND rule = 'profile'))""").df()
    data = labels.merge(reports, on=keys, how="inner", validate="many_to_one")
    data = data.merge(cohort[["corpus", "exam_id", "patient_id"]].rename(columns={"corpus": "test_corpus"}),
                      on=["test_corpus", "exam_id"], validate="many_to_one")
    for name, target in (("score", "a"), ("report_precision", "b")):
        count = data[name].to_numpy() * 10
        if not np.allclose(count, np.rint(count), atol=1e-6) or not ((count >= 0) & (count <= 10)).all():
            raise ValueError("Expected frequencies of binary outcomes among ten retrieved exams.")
        data[target] = np.rint(count).astype(int)
    data["finding_precision"] = data.a / 10
    data["joint_lower"] = (data.a + data.b - 10).clip(lower=0) / 10
    data["joint_upper"] = np.minimum(data.a, data.b) / 10
    measures = ["finding_precision", "joint_lower", "joint_upper"]
    rows = []
    finding_scores = {}
    for (corpus, retriever), part in data.groupby(["test_corpus", "retriever"]):
        expected = {"42", "1337", "2024"} if retriever == "encoder" else {"none"}
        if set(part.seed) != expected or part.duplicated(["seed", "exam_id", "finding"]).any():
            raise ValueError("Unexpected evidence keys.")
        group = part.groupby(["exam_id", "patient_id", "finding"])
        if not (group.size() == len(expected)).all():
            raise ValueError("Some exam and finding pairs lack a training seed.")
        paired = group[measures].mean().reset_index()
        eligible = paired.groupby("finding").exam_id.nunique()
        paired = paired[paired.finding.isin(eligible[eligible >= MIN_POS].index)]
        if paired.empty:
            continue
        finding_scores[(corpus, retriever)] = paired[["exam_id", "patient_id", "finding", "finding_precision"]].rename(columns={"finding_precision": "score"})
        patient = paired.groupby(["patient_id", "finding"])
        den = patient.size().unstack(fill_value=0).to_numpy()
        weights = counts_matrix(len(den))
        for measure in measures:
            sums = patient[measure].sum().unstack(fill_value=0).to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                boot = ((weights @ sums) / (weights @ den)).mean(axis=1)
            observed = (sums.sum(axis=0)/den.sum(axis=0)).mean()
            mean, sd, low, high = summarize(boot)
            rows.append(dict(corpus=corpus, retriever=retriever, measure=measure, findings=den.shape[1],
                             queries=paired.exam_id.nunique(), patients=len(den), pairs=len(paired),
                             finite=int(np.isfinite(boot).sum()), observed=observed, mean=mean, std=sd, ci_low=low, ci_high=high))
    paired_rows = []
    for corpus in sorted({key[0] for key in finding_scores}):
        model = finding_scores[(corpus, "encoder")]
        baseline = finding_scores[(corpus, "finding_classifier")]
        cols = ["exam_id", "finding", "score"]
        patients = model[["exam_id", "patient_id"]].drop_duplicates().set_index("exam_id", verify_integrity=True).patient_id
        if not baseline.exam_id.map(patients).eq(baseline.patient_id).all():
            raise ValueError("The two ranking methods must use the same query patients.")
        result = cluster_difference(model[cols], baseline[cols], patients, "macro_ap")
        paired_rows.append(dict(corpus=corpus, retriever="encoder_minus_classifier", measure="finding_precision",
                                queries=result["exams"], pairs=len(model), **result))
    contrasts = pd.DataFrame(paired_rows)
    contrasts["p_adjusted"] = bh_fdr(contrasts.p_raw.to_numpy())
    ap = connection.execute("""SELECT test_corpus, retriever, exam_id, finding, avg(value) AS score,
               count(*) AS records, count(DISTINCT seed) AS seeds_present
        FROM ev WHERE archive='documented' AND split='test' AND metric='ap'
        AND ((src='controls' AND retriever='encoder' AND configuration='full' AND backbone='biovilt'
              AND rule='two_stage' AND k=100 AND query_inputs='all' AND seed IN ('42','1337','2024'))
          OR (src='encoders' AND retriever='finding_classifier' AND rule='profile'))
        GROUP BY test_corpus,retriever,exam_id,finding""").df()
    expected = np.where(ap.retriever == "encoder", 3, 1)
    if not ((ap.records == expected) & (ap.seeds_present == expected)).all():
        raise ValueError("Full-ranking AP has missing or duplicate seeds.")
    matched = []
    for corpus in contrasts.corpus:
        model = finding_scores[(corpus, "encoder")]
        comparator = finding_scores[(corpus, "finding_classifier")]
        patients = model[["exam_id", "patient_id"]].drop_duplicates().set_index("exam_id", verify_integrity=True).patient_id
        for measure in ("matched_full_ap", "pathology_full_ap", "pathology_first_ten"):
            arms = []
            for name, scores in (("encoder", model), ("finding_classifier", comparator)):
                eligible = scores if measure == "matched_full_ap" else scores[scores.finding != "support_devices"]
                if measure.endswith("full_ap"):
                    chosen = ap[(ap.test_corpus == corpus) & (ap.retriever == name)]
                    eligible = eligible[["exam_id", "finding"]].merge(chosen, how="left", validate="one_to_one",
                                                                     on=["exam_id", "finding"])
                    if eligible.score.isna().any():
                        raise ValueError("A top-ten finding and query pair lacks full-ranking AP.")
                arms.append(eligible[["exam_id", "finding", "score"]])
            result = cluster_difference(*arms, patients, "macro_ap")
            matched.append(dict(corpus=corpus, retriever="encoder_minus_classifier", measure=measure,
                                queries=result["exams"], pairs=len(arms[0]), **result))
    matched = pd.DataFrame(matched)
    matched["p_adjusted"] = matched.groupby("measure").p_raw.transform(lambda p: bh_fdr(p.to_numpy()))
    return pd.concat([pd.DataFrame(rows), contrasts, matched], ignore_index=True)


def main_top_ten_overlap(config_path):
    settings = Config(config_path).get("statistics")
    inputs = settings["inputs"]
    predictions = pd.read_csv(inputs["downstream_per_query"], usecols=["archive", "task", "target", "finding", "score",
                              "retriever", "seed", "test_corpus", "exam_id"], dtype={"seed": str, "exam_id": str},
                              low_memory=False)
    cohort = pd.read_csv(inputs["cohort"], usecols=["corpus", "exam_id", "patient_id"], dtype=str, low_memory=False)
    database = Path(settings["database"])
    database.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database)) as connection:
        ensure_evidence(connection, inputs)
        result = top_ten_overlap(connection, predictions, cohort)
    dest = Path(settings["output_prefix"] + "_top_ten_overlap.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".csv.partial")
    result.to_csv(tmp, index=False)
    tmp.replace(dest)
