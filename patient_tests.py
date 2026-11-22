"""
patient_tests.py
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


WITH_REPORTS = ("mimic", "chexpert", "rexgradient", "openi")
THREE = ("42", "1337", "2024")
ABLATIONS = {"frontal_copy": ("model", THREE), "shuffled_lateral": ("controls", ("42",)),
             "shuffled_context": ("model", THREE), "context_text_only": ("controls", ("42",)),
             "context_agesex_only": ("controls", ("42",)), "shuffled_report": ("controls", THREE),
             "profile_only": ("controls", ("42",)), "alignment_only": ("controls", ("42",)),
             "no_dropout": ("controls", ("42",)), "midlevel_fusion": ("controls", ("42",))}
METRICS = (("ap", "profile", "macro_ap"), ("ndcg10", "alignment", "ndcg10"))


def arm(src, configuration="full", rule="two_stage", k=100, seeds=THREE, retriever="encoder", query_inputs="all"):
    return dict(src=src, configuration=configuration, rule=rule, k=k, seeds=tuple(seeds), retriever=retriever,
                query_inputs=query_inputs)


TWO_STAGE = arm("controls")
CLASSIFIER = arm("encoders", "none", "profile", 0, ("none",), "finding_classifier")


def frozen(retriever, rule):
    return arm("encoders", "none", rule, 0, ("none",), retriever)


def scores(con, a, corpus, archive, metric):
    backbone = "none" if a["src"] == "encoders" else "biovilt"
    seeds = ",".join(f"'{s}'" for s in a["seeds"])
    sql = f"""SELECT exam_id, finding, avg(value) AS score FROM ev
              WHERE src = ? AND retriever = ? AND configuration = ? AND backbone = ? AND split = 'test' AND rule = ?
                AND k = ? AND query_inputs = ? AND seed IN ({seeds}) AND test_corpus = ? AND archive = ? AND metric = ?
              GROUP BY exam_id, finding HAVING count(*) = {len(a['seeds'])}"""
    return con.execute(sql, [a["src"], a["retriever"], a["configuration"], backbone, a["rule"], a["k"],
                             a["query_inputs"], corpus, archive, metric]).df()


def best_frozen(con, corpus, archive, metric):
    per = con.execute("""SELECT retriever, rule, finding, avg(value) AS mean_score, count(*) AS n FROM ev WHERE src = 'encoders'
                         AND split = 'test' AND retriever LIKE '%_frozen' AND test_corpus = ? AND archive = ? AND metric = ?
                         GROUP BY retriever, rule, finding""", [corpus, archive, metric]).df()
    if metric == "ap":
        per = per[per.n >= MIN_POS]
    levels = per.groupby(["retriever", "rule"])["mean_score"].mean()
    retriever, rule = levels.idxmax()
    return frozen(retriever, rule)


def compare(con, patients, a, b, corpus, archive, metric, subset=None, finding=None):
    sa, sb = scores(con, a, corpus, archive, metric), scores(con, b, corpus, archive, metric)
    for s in (sa, sb):
        if s.empty:
            raise ValueError(f"No scores for {corpus} {archive} {metric}.")
    if subset is not None:
        sa, sb = sa[sa.exam_id.isin(subset)], sb[sb.exam_id.isin(subset)]
    if finding is not None:
        sa, sb = sa[sa.finding == finding], sb[sb.finding == finding]
    kind = "macro_ap" if metric == "ap" and finding is None else "single"
    return cluster_difference(sa, sb, patients, kind)


def label(a):
    return "|".join(str(a[key]) for key in ("src", "retriever", "configuration", "rule", "k", "query_inputs")) + "|" + ";".join(a["seeds"])


def level(con, patients, a, corpus, archive, metric):
    s = scores(con, a, corpus, archive, metric)
    s["patient"] = s.exam_id.map(patients)
    if s.patient.isna().any():
        raise ValueError("Missing patient identity.")
    grouped = s.groupby("patient").score.agg(["sum", "size"])
    weights = counts_matrix(len(grouped))
    boot = (weights @ grouped["sum"].to_numpy()) / (weights @ grouped["size"].to_numpy())
    mean, sd, low, high = summarize(boot)
    return dict(exams=len(s), patients=len(grouped), observed=s.score.mean(), mean=mean, std=sd, ci_low=low, ci_high=high)


def protocol_tests(con, cohort, published):
    sql = f"""SELECT method, image_id AS exam_id, exam_id AS study, finding, metric, avg(CAST(value AS DOUBLE)) AS score,
                     count(*) AS n FROM read_csv('{published}', all_varchar=true)
              WHERE protocol = 'published_all' AND method IN ('encoder_frontal', 'chextriev_rerun') GROUP BY ALL"""
    d = con.execute(sql).df()
    expected = np.where(d.method == "encoder_frontal", 3, 1)
    if not (d.n == expected).all():
        raise ValueError("Missing or repeated training seeds in the protocol scores.")
    studies = cohort[cohort.corpus == "mimic"].set_index("exam_id", verify_integrity=True).patient_id
    images = d.drop_duplicates("exam_id").set_index("exam_id").study.map(studies)
    if images.isna().any():
        raise ValueError("A protocol image has no patient in the cohort.")
    rows = []
    for metric in ("ap5", "hr5", "rr5"):
        part = d[d.metric == metric]
        a = part[part.method == "encoder_frontal"][["exam_id", "finding", "score"]]
        b = part[part.method == "chextriev_rerun"][["exam_id", "finding", "score"]]
        result = cluster_difference(a, b, images, "macro_ap")
        rows.append(dict(analysis="protocol", item="encoder_frontal_vs_chextriev_rerun", corpus="mimic",
                         archive="undocumented", metric=metric, finding="macro", side_a="encoder_frontal|42;1337;2024",
                         side_b="chextriev_rerun|42", **result))
    return rows


def patient_tests(con, cohort, published):
    queries = cohort[cohort.query_or_database == "query"]
    rows, levels = [], []

    def add(analysis, item, a, b, corpus, archive, metric, finding="macro", **kw):
        result = compare(con, patients[corpus], a, b, corpus, archive, metric,
                         finding=None if finding == "macro" else finding, **kw)
        rows.append(dict(analysis=analysis, item=item, corpus=corpus, archive=archive,
                         metric="macro_ap" if metric == "ap" and finding == "macro" else ("ap" if metric == "ap" else metric),
                         finding=finding if metric == "ap" else "pooled", side_a=label(a), side_b=label(b), **result))

    patients = {c: q.set_index("exam_id", verify_integrity=True).patient_id for c, q in queries.groupby("corpus")}
    for corpus in WITH_REPORTS:
        for configuration, (src, seeds) in ABLATIONS.items():
            for metric, rule, _ in METRICS:
                if configuration == "shuffled_report" and metric == "ap":
                    continue
                full = arm("model", "full", rule, 0, seeds)
                control = arm(src, configuration, rule, 0, seeds)
                add("ablation", configuration, full, control, corpus, "documented", metric)
        best = {m: best_frozen(con, corpus, "documented", m) for m in ("ap", "ndcg10")}
        add("baselines", "classifier_vs_best_frozen", CLASSIFIER, best["ap"], corpus, "documented", "ap")
        add("baselines", "classifier_vs_best_frozen", CLASSIFIER, best["ndcg10"], corpus, "documented", "ndcg10")
        add("baselines", "profile_vs_classifier", arm("model", "full", "profile", 0), CLASSIFIER, corpus, "documented", "ap")
        add("baselines", "two_stage_vs_classifier", TWO_STAGE, CLASSIFIER, corpus, "documented", "ndcg10")
        add("baselines", "alignment_vs_best_frozen", arm("model", "full", "alignment", 0), best["ndcg10"], corpus,
            "documented", "ndcg10")
        if corpus != "openi":
            for name, b in (("vs_classifier", CLASSIFIER), ("vs_best_frozen", best["ap"])):
                found = scores(con, TWO_STAGE, corpus, "documented", "ap").groupby("finding").exam_id.nunique()
                for finding in sorted(found[found >= MIN_POS].index):
                    add("per_finding", name, TWO_STAGE, b, corpus, "documented", "ap", finding=finding)
            add("transfer", f"loso_without_{corpus}", arm("controls", seeds=("42",)),
                arm("transfer", f"loso_without_{corpus}", seeds=("42",)), corpus, "documented", "ap")
            add("transfer", f"loso_without_{corpus}", arm("controls", seeds=("42",)),
                arm("transfer", f"loso_without_{corpus}", seeds=("42",)), corpus, "documented", "ndcg10")
        for name, a in (("two_stage", TWO_STAGE), ("classifier", CLASSIFIER),
                        ("best_frozen", best_frozen(con, corpus, "undocumented", "ndcg10"))):
            levels.append(dict(analysis="level", item=name, corpus=corpus, archive="undocumented", metric="ndcg10",
                               finding="pooled", side_a=label(a), side_b="", **level(con, patients[corpus], a, corpus,
                                                                                       "undocumented", "ndcg10")))
    rows += protocol_tests(con, cohort, published)
    results = pd.DataFrame(rows)
    results["p_adjusted"] = results.groupby("analysis").p_raw.transform(lambda p: bh_fdr(p.to_numpy()))
    return pd.concat([results, pd.DataFrame(levels)], ignore_index=True)


def main_patient_tests(config_path):
    settings = Config(config_path).get("statistics")
    cohort = pd.read_csv(settings["inputs"]["cohort"], dtype=str, low_memory=False,
                         usecols=["corpus", "exam_id", "patient_id", "query_or_database"])
    database = Path(settings["database"])
    database.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database)) as connection:
        ensure_evidence(connection, settings["inputs"])
        results = patient_tests(connection, cohort, settings["inputs"]["published_per_query"])
    destination = Path(settings["output_prefix"] + "_patient_tests.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.partial")
    results.to_csv(temporary, index=False)
    temporary.replace(destination)
