from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from lib.stats_utils import bh_fdr
from patient_comparisons import cluster_difference, simultaneous_intervals
from rerank.rerank_files import FROZEN_METHOD, RerankFiles, read_result
from rerank.rerank_retrieval import CLASSIFIER_RERANK_OURS, FIRST_STAGES, OURS, TWO_STAGE_OURS, read_rerank_evidence


TEST_COLUMNS = ["family", "comparison", "side_a", "side_b", "test_corpus", "archive", "metric", "units", "n_queries",
                "mean_diff", "sd", "ci_low", "ci_high", "p_raw", "p_fdr", "sim_ci_low", "sim_ci_high"]
SCORE_KEY = ["method", "reranker", "test_corpus", "archive", "metric"]


def method_scores(evidence):
    grouped = (evidence.groupby([*SCORE_KEY, "exam_id", "finding"], observed=True)["value"]
               .agg(["mean", "size"]).reset_index())
    seeds = evidence.groupby("method", observed=True)["seed"].nunique()
    if not (grouped["size"] == grouped["method"].map(seeds)).all():
        raise ValueError("A query value lacks one of its method's seeds.")
    scores = {}
    for key, part in grouped.groupby(SCORE_KEY, observed=True):
        scores[tuple(map(str, key))] = (part[["exam_id", "finding", "mean"]].rename(columns={"mean": "score"})
                                        .astype({"finding": str}).reset_index(drop=True))
    return scores


def source_metric(config, metric):
    share = config.get("rerank.share")
    if metric == config.get("retrieval.files.macro_metric"):
        return "ap", "macro_ap"
    if metric == share["macro_metric"]:
        return share["name"], "macro_ap"
    return metric, metric


def defined(config, result, corpus, archive, metric):
    row = {r["metric"]: r for r in config.get("rerank.files.per_seed_rows")}[metric]
    separator = config.get("trained_encoder.files.seeds_separator")
    return bool(((result["method"] == TWO_STAGE_OURS) & (result["test_corpus"] == corpus)
                 & (result["archive"] == archive) & (result["metric"] == metric) & (result["level"] == row["level"])
                 & (result["finding"] == row["finding"]) & result["seeds"].str.contains(separator, regex=False)).any())


def selected(selections, first_stage, corpus, archive, metric):
    row = selections[(selections["first_stage"] == first_stage) & (selections["test_corpus"] == corpus)
                     & (selections["archive"] == archive) & (selections["metric"] == metric)]
    if len(row) != 1:
        raise LookupError(f"{len(row)} selections for {first_stage} {corpus} {archive} {metric}.")
    return row["reranker"].iloc[0]


def rerank_tests(config, evidence, selections, result, cohort):
    settings = config.get("rerank")
    scores = method_scores(evidence)
    queries = cohort[cohort.query_or_database == "query"]
    rows, intervals = [], []
    for corpus in settings["test_corpora"]:
        patients = queries[queries.corpus == corpus].set_index("exam_id", verify_integrity=True).patient_id
        for archive in settings["archives"]:
            for metric in settings["selection_metrics"]:
                if not defined(config, result, corpus, archive, metric):
                    continue
                source, argument = source_metric(config, metric)
                model = scores[(TWO_STAGE_OURS, OURS, corpus, archive, source)]
                sides = [(f"vs_best_{FROZEN_METHOD[first_stage]}", FROZEN_METHOD[first_stage],
                          selected(selections, first_stage, corpus, archive, metric), first_stage)
                         for first_stage in FIRST_STAGES]
                sides.append((f"vs_{CLASSIFIER_RERANK_OURS}", CLASSIFIER_RERANK_OURS, OURS, ""))
                for comparison, method, reranker, first_stage in sides:
                    comparator = scores[(method, reranker, corpus, archive, source)]
                    rows.append(dict(comparison=comparison, side_b=f"{method}:{reranker}", test_corpus=corpus,
                                     archive=archive, metric=metric, first_stage=first_stage, reranker=reranker,
                                     **cluster_difference(model, comparator, patients, argument)))
                if metric not in settings["tests"]["simultaneous_metrics"]:
                    continue
                for first_stage in FIRST_STAGES:
                    method = FROZEN_METHOD[first_stage]
                    frozen = [(r, s) for (m, r, c, a, s_metric), s in scores.items()
                              if m == method and c == corpus and a == archive and s_metric == source]
                    baselines = pd.concat([s.assign(retriever=r, rule=method) for r, s in frozen])
                    table = simultaneous_intervals(model[["exam_id", "score"]],
                                                   baselines[["retriever", "rule", "exam_id", "score"]],
                                                   patients).reset_index()
                    intervals.append(table.assign(test_corpus=corpus, archive=archive, metric=metric,
                                                  first_stage=first_stage))
    comparisons = pd.DataFrame(rows)
    comparisons["p_fdr"] = bh_fdr(comparisons["p_raw"].to_numpy())
    intervals = (pd.concat(intervals, ignore_index=True) if intervals
                 else pd.DataFrame(columns=["retriever", "rule", "simultaneous_low", "simultaneous_high",
                                            "test_corpus", "archive", "metric", "first_stage"]))
    intervals = intervals.rename(columns={"retriever": "reranker"}).drop(columns="rule")
    table = comparisons.merge(intervals, on=["test_corpus", "archive", "metric", "first_stage", "reranker"],
                              how="left", validate="many_to_one")
    table = table.assign(family=settings["tests"]["family"], side_a=TWO_STAGE_OURS, units=table["patients"],
                         n_queries=table["exams"], mean_diff=table["mean"], sd=table["std"],
                         sim_ci_low=table["simultaneous_low"], sim_ci_high=table["simultaneous_high"])
    if table[["units", "n_queries"]].isna().any().any() or not np.isfinite(table["p_raw"]).all():
        raise ValueError("A comparison has no patients, queries or p value.")
    return table[TEST_COLUMNS]


def main_rerank_tests(config_path):
    config = Config(config_path)
    settings = config.get("statistics")
    files = RerankFiles(config)
    cohort = pd.read_csv(settings["inputs"]["cohort"], dtype=str, low_memory=False,
                         usecols=["corpus", "exam_id", "patient_id", "query_or_database"])
    selections = pd.read_csv(files.selections_path, dtype=str, keep_default_na=False)
    results = rerank_tests(config, read_rerank_evidence(files.evidence_path), selections, read_result(files.result_path),
                           cohort)
    destination = Path(settings["output_prefix"] + "_rerank_tests.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.partial")
    results.to_csv(temporary, index=False)
    temporary.replace(destination)
