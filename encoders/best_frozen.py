"""
encoders/best_frozen.py
Created on November 22, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

from functools import lru_cache
import os

import pandas as pd

from config import Config


TABLES = {}
CORPORA_DOC = ["mimic", "chexpert", "rexgradient", "openi"]
FROZEN = ["biovilt_frozen", "raddino_frozen", "biomedclip_frozen", "medsiglip_frozen", "cxrclip_frozen", "medimageinsight_frozen"]
LEVEL = {"macro_ap": ("finding", "macro"), "ndcg10": ("report", "pooled"), "ndcg50": ("report", "pooled"),
         "p10_f1_05": ("report", "pooled"), "hit1": ("finding", "pooled"), "hit5": ("finding", "pooled"),
         "hit10": ("finding", "pooled")}


@lru_cache(maxsize=None)
def table(name):
    return pd.read_csv(TABLES[name], low_memory=False, dtype=str)


def one(df, **keys):
    m = pd.Series(True, index=df.index)
    for k, v in keys.items():
        m &= (df[k].isna()) if v is None else (df[k] == str(v))
    rows = df[m]
    if len(rows) != 1:
        raise LookupError(f"{len(rows)} rows for {keys}")
    return rows.iloc[0]


def metric_row(retriever, rule, corpus, metric, archive="documented", backbone="none", configuration=None, seeds=None, k=None, query_inputs="all"):
    level, finding = LEVEL[metric]
    frozen = retriever.endswith("_frozen") or retriever == "finding_classifier"
    configuration = configuration or ("none" if frozen else "full")
    seeds = seeds or ("none" if frozen else "42;1337;2024")
    k = k if k is not None else (100 if rule == "two_stage" else 0)
    return one(table("retrieval_metrics"), retriever=retriever, rule=rule, test_corpus=corpus, metric=metric, archive=archive,
               backbone=backbone, configuration=configuration, seeds=seeds, k=k, query_inputs=query_inputs, split="test",
               level=level, finding=finding)


def value(row):
    return float(row["mean_raw"])


def best_frozen(corpus, metric, archive="documented"):
    if corpus not in CORPORA_DOC and (archive == "documented" or metric in ("ndcg10", "ndcg50", "p10_f1_05")):
        return None
    best = None
    for r in FROZEN:
        for rule in ("image_image", "image_report", "two_view_mean"):
            if rule == "image_report" and (archive == "undocumented" or r in ("raddino_frozen", "medimageinsight_frozen")):
                continue
            v = value(metric_row(r, rule, corpus, metric, archive))
            if best is None or v > best[2]:
                best = (r, rule, v)
    return best


def main_best_frozen(config_path):
    config = Config(config_path)
    settings = config.get("statistics")
    os.makedirs(os.path.dirname(settings["output_prefix"]), exist_ok=True)
    TABLES["retrieval_metrics"] = settings["inputs"]["retrieval_metrics"]
    table.cache_clear()
    rows = []
    for corpus in config.get("cohort.corpora"):
        for archive in ("documented", "undocumented"):
            for metric in LEVEL:
                best = best_frozen(corpus, metric, archive)
                if best is not None:
                    rows.append({"test_corpus": corpus, "archive": archive, "metric": metric, "retriever": best[0], "rule": best[1],
                                 "mean_raw": best[2]})
    pd.DataFrame(rows).to_csv(settings["output_prefix"] + "_best_frozen.csv", index=False)
