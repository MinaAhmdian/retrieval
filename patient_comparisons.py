"""
patient_comparisons.py
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
from final_differences import MIN_POS, counts_matrix, perm_p, signs, summarize
from lib.stats_utils import bh_fdr


FROZEN = ("biovilt_frozen", "raddino_frozen", "biomedclip_frozen", "medsiglip_frozen",
          "cxrclip_frozen", "medimageinsight_frozen")
WITH_IMAGE_REPORT = ("biovilt_frozen", "biomedclip_frozen", "medsiglip_frozen", "cxrclip_frozen")


def cluster_difference(model, comparator, patients, metric):
    raw = model.merge(comparator, on=["exam_id", "finding"], validate="one_to_one", how="outer", indicator=True)
    if not (raw["_merge"] == "both").all() or raw.empty:
        raise ValueError("The two rankings must contain the same eligible query observations.")
    raw["difference"] = raw["score_x"] - raw["score_y"]
    if metric == "macro_ap":
        raw = raw[raw.groupby("finding")["exam_id"].transform("size") >= MIN_POS]
    raw["patient"] = raw.exam_id.map(patients)
    if raw.patient.isna().any() or raw.empty:
        raise ValueError("Missing patient identity or eligible finding.")
    grouped = raw.groupby(["patient", "finding"]).difference.agg(["sum", "size"])
    sums = grouped["sum"].unstack(fill_value=0).sort_index().to_numpy()
    sizes = grouped["size"].unstack(fill_value=0).sort_index().to_numpy()
    n = len(sums)
    counts = counts_matrix(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = ((counts @ sums) / (counts @ sizes)).mean(axis=1)
    observed = (sums.sum(axis=0) / sizes.sum(axis=0)).mean()
    permutations = ((signs(n) @ sums) / sizes.sum(axis=0)).mean(axis=1)
    mean, sd, low, high = summarize(boot)
    present = sizes > 0
    subject_means = np.divide(sums, sizes, out=np.zeros_like(sums), where=present)
    with np.errstate(invalid="ignore", divide="ignore"):
        equal_boot = ((counts @ subject_means) / (counts @ present.astype(float))).mean(axis=1)
    equal_mean, equal_sd, equal_low, equal_high = summarize(equal_boot)
    return dict(exams=raw.exam_id.nunique(), patients=n, findings=sizes.shape[1], finite=np.isfinite(boot).sum(),
                observed=observed, mean=mean, std=sd, ci_low=low, ci_high=high, p_raw=perm_p(observed, permutations),
                patient_weighted_mean=equal_mean, patient_weighted_sd=equal_sd,
                patient_weighted_low=equal_low, patient_weighted_high=equal_high)


def simultaneous_intervals(model, baselines, patients):
    columns = ["retriever", "rule"]
    wide = baselines.pivot(index="exam_id", columns=columns, values="score")
    full = model.set_index("exam_id", verify_integrity=True).score
    if wide.isna().any().any() or set(full.index) != set(wide.index):
        raise ValueError("Simultaneous intervals require matched queries for every candidate rule.")
    difference = wide.rsub(full, axis=0)
    subjects = patients.reindex(difference.index)
    if subjects.isna().any():
        raise ValueError("Missing sampling-unit identifier.")
    grouped = difference.groupby(subjects)
    sizes = grouped.size().to_numpy()
    weights = counts_matrix(len(sizes))
    observed = difference.mean().to_numpy()
    draws = (weights @ grouped.sum().to_numpy()) / (weights @ sizes)[:, None]
    sd = draws.std(axis=0, ddof=1)
    scaled = np.divide(draws - observed, sd, out=np.zeros_like(draws), where=sd > 0)
    critical = np.quantile(np.abs(scaled).max(axis=1), 0.95, method="higher")
    return pd.DataFrame({"simultaneous_low": observed - critical * sd,
                         "simultaneous_high": observed + critical * sd}, index=wide.columns)


def patient_comparisons(connection, cohort):
    query = cohort[cohort.query_or_database == "query"]
    rows = []
    for corpus, cases in query.groupby("corpus"):
        patients = cases.set_index("exam_id", verify_integrity=True).patient_id
        for archive in ("documented", "undocumented"):
            sql = """SELECT retriever, rule, exam_id, finding, metric, avg(value) AS score,
                            count(value) AS records, count(DISTINCT seed) AS seeds_present
                     FROM ev WHERE test_corpus = ? AND archive = ? AND split = 'test'
                       AND ((src = 'controls' AND configuration = 'full' AND backbone = 'biovilt' AND
                             rule = 'two_stage' AND k = 100 AND query_inputs = 'all' AND seed IN ('42','1337','2024'))
                         OR src = 'encoders')
                       AND (metric = 'ndcg10' OR (metric = 'ap' AND retriever IN ('encoder','finding_classifier')))
                     GROUP BY retriever, rule, exam_id, finding, metric"""
            values = connection.execute(sql, [corpus, archive]).df()
            expected = np.where(values.retriever == "encoder", 3, 1)
            if not ((values.records == expected) & (values.seeds_present == expected)).all():
                raise ValueError(f"Incomplete or repeated evidence for {corpus} {archive}.")
            for source_metric, metric in (("ap", "macro_ap"), ("ndcg10", "ndcg10")):
                evidence = values[values.metric == source_metric]
                full = evidence[evidence.retriever == "encoder"]
                if full.empty:
                    continue
                baselines = evidence[evidence.retriever == "finding_classifier"] if source_metric == "ap" else evidence[evidence.retriever.str.endswith("_frozen")]
                expected_rules = {("finding_classifier", "profile")}
                if source_metric == "ndcg10":
                    expected_rules = {(model, rule) for model in FROZEN for rule in ("image_image", "two_view_mean")}
                    if archive == "documented":
                        expected_rules.update((model, "image_report") for model in WITH_IMAGE_REPORT)
                if set(zip(baselines.retriever, baselines.rule)) != expected_rules:
                    raise ValueError(f"Incomplete comparator panel for {corpus} {archive} {metric}.")
                intervals = simultaneous_intervals(full, baselines, patients) if source_metric == "ndcg10" else None
                for (retriever, rule), baseline in baselines.groupby(["retriever", "rule"]):
                    cols = ["exam_id", "finding", "score"]
                    result = cluster_difference(full[cols], baseline[cols], patients, metric)
                    if intervals is not None:
                        result.update(intervals.loc[(retriever, rule)].to_dict())
                    rows.append(dict(corpus=corpus, archive=archive, metric=metric, retriever=retriever,
                                     rule=rule, **result))
    results = pd.DataFrame(rows)
    results["p_adjusted"] = bh_fdr(results.p_raw.to_numpy())
    return results


def main_patient_comparisons(config_path):
    settings = Config(config_path).get("statistics")
    cohort = pd.read_csv(settings["inputs"]["cohort"], dtype=str, low_memory=False,
                         usecols=["corpus", "exam_id", "patient_id", "query_or_database"])
    database = Path(settings["database"])
    database.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database)) as connection:
        ensure_evidence(connection, settings["inputs"])
        results = patient_comparisons(connection, cohort)
    destination = Path(settings["output_prefix"] + "_patient_comparisons.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.partial")
    results.to_csv(temporary, index=False)
    temporary.replace(destination)
