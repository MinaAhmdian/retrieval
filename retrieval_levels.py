"""
retrieval_levels.py
Created on November 22, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from config import Config
from evidence_cache import ensure_evidence
from final_differences import CORE7, MIN_POS, counts_matrix, summarize


UNIT = ["retriever", "configuration", "backbone", "test_corpus", "split", "archive", "rule", "k", "query_inputs"]
METRIC = ["level", "finding", "metric"]


def retrieval_levels(connection, source, unit, seeds, metrics=None):
    if set(unit) != set(UNIT) or not seeds or len(set(map(str, seeds))) != len(seeds):
        raise ValueError("Supply one complete ranking configuration and a nonempty set of distinct seeds.")
    condition = " AND ".join(f'"{name}" = ?' for name in UNIT)
    slots = ", ".join("?" for _ in seeds)
    sql = f"""SELECT level, finding, metric, exam_id, avg(value) AS value,
                     count(value) AS records, count(DISTINCT seed) AS seeds_present
              FROM ev WHERE src = ? AND {condition} AND seed IN ({slots})
              GROUP BY level, finding, metric, exam_id"""
    values = connection.execute(sql, [source, *[unit[name] for name in UNIT], *map(str, seeds)]).df()
    if values.empty or not (values[["records", "seeds_present"]] == len(seeds)).all().all():
        raise ValueError("Missing, duplicated, or unmatched query evidence for the requested ranking.")
    tasks = []
    for key, part in values.groupby(METRIC):
        if metrics is None or key in metrics:
            tasks.append((key, part.sort_values("exam_id").value.to_numpy()))
    ap = values[values.metric == "ap"].pivot(index="exam_id", columns="finding", values="value").sort_index()
    eligible = ap.columns[ap.count() >= MIN_POS].tolist()
    for name, labels in (("macro", eligible), ("core7", CORE7)):
        key = ("finding", name, "macro_ap")
        if labels and set(labels) <= set(eligible) and (metrics is None or key in metrics):
            tasks.append((key, ap[labels].dropna(how="all").to_numpy()))
    rows = []
    counts = None
    for key, scores in sorted(tasks, key=lambda task: len(task[1])):
        n = len(scores)
        if counts is None or counts.shape[1] != n:
            counts = counts_matrix(n)
        if scores.ndim == 1:
            draws = counts @ scores / n
        else:
            present = np.isfinite(scores)
            with np.errstate(invalid="ignore", divide="ignore"):
                draws = ((counts @ np.where(present, scores, 0)) / (counts @ present.astype(float))).mean(axis=1)
        estimates = summarize(draws)
        if estimates is None:
            raise ValueError(f"No finite bootstrap summary for {key}.")
        mean, sd, low, high = estimates
        rows.append({**unit, "seeds": ";".join(map(str, seeds)), **dict(zip(METRIC, key)), "n": n,
                     "mean_raw": mean, "mean": round(100 * mean, 1), "std": round(100 * sd, 1),
                     "ci_low": round(100 * low, 1), "ci_high": round(100 * high, 1), "unit": "percent"})
    return pd.DataFrame(rows)


def paper_rankings(config):
    from patient_comparisons import FROZEN, WITH_IMAGE_REPORT
    labels = config.get("cohort.chexpert_label_names").values()

    with_reports = set(config.get("retrieval.documented_corpora"))
    training = set(config.get("trained_encoder.train_corpora"))
    for corpus in config.get("cohort.corpora"):
        archives = ["undocumented", "documented"] if corpus in with_reports else ["undocumented"]
        for archive in archives:
            methods = [("finding_classifier", "profile", "none")]
            for retriever in FROZEN:
                methods.extend((retriever, rule, "none") for rule in ("image_image", "two_view_mean"))
                if archive == "documented" and retriever in WITH_IMAGE_REPORT:
                    methods.append((retriever, "image_report", "none"))
            methods.append(("encoder", "two_stage", "biovilt"))
            if archive == "documented":
                methods.extend(("encoder", rule, bb) for bb in ("biovilt", "raddino")
                               for rule in ("profile", "alignment", "two_stage") if (bb, rule) != ("biovilt", "two_stage"))
            for retriever, rule, backbone in methods:
                model = retriever == "encoder"
                k = 100 if rule == "two_stage" else 0
                metrics = {("finding", "macro", "macro_ap")}
                if corpus in with_reports:
                    metrics.add(("report", "pooled", "ndcg10"))
                if corpus in training and archive == "documented":
                    metrics.update(("finding", f, "ap") for f in labels)
                yield dict(source="controls" if k else ("model" if model else "encoders"),
                           unit=dict(retriever=retriever, configuration="full" if model else "none", backbone=backbone,
                                     test_corpus=corpus, split="test", archive=archive, rule=rule, k=k, query_inputs="all"),
                           seeds=[42, 1337, 2024] if model else ["none"], metrics=metrics)


def main_retrieval_levels(config_path, rankings=None, output_path=None):
    config = Config(config_path)
    settings = config.get("statistics")
    if rankings is None:
        rankings = paper_rankings(config)
    if output_path is None:
        output_path = settings["output_prefix"] + "_retrieval_levels.csv"
    database = Path(settings["database"])
    database.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    with duckdb.connect(str(database)) as connection:
        ensure_evidence(connection, settings["inputs"])
        for ranking in rankings:
            frames.append(retrieval_levels(connection, **ranking))
    if not frames:
        raise ValueError("Specify at least one ranking.")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    pd.concat(frames, ignore_index=True).to_csv(temporary, index=False)
    os.replace(temporary, destination)
