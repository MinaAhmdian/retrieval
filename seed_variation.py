"""
seed_variation.py
Created on November 22, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os

import numpy as np
import pandas as pd

from config import Config

SEEDS = ["42", "1337", "2024"]


def seed_variation(inputs, prefix):
    rows = []
    rm = pd.read_csv(inputs["retrieval_metrics"], dtype=str, keep_default_na=False)
    rm = rm[rm.seeds.isin(SEEDS)]
    key = ["retriever", "configuration", "backbone", "test_corpus", "split", "archive", "rule", "k", "query_inputs", "level", "finding", "metric"]
    sources = [("retrieval_metrics", rm, key)]
    pub = pd.read_csv(inputs["published"], dtype=str, keep_default_na=False)
    pub = pub[pub.seeds.isin(SEEDS)]
    sources.append(("published_protocol", pub, ["protocol", "method", "test_corpus", "archive", "rule", "query_inputs", "level", "finding", "metric"]))
    ds = pd.read_csv(inputs["downstream"], dtype=str, keep_default_na=False)
    ds = ds[ds.seeds.isin(SEEDS)]
    sources.append(("downstream", ds, ["retriever", "test_corpus", "archive", "rule", "k", "task", "finding", "metric"]))
    for table, df, k in sources:
        wide = df.pivot_table(index=k, columns="seeds", values="mean_raw", aggfunc="first")
        wide = wide.dropna(subset=SEEDS)
        for idx, r in wide.iterrows():
            v = np.array([float(r[s]) for s in SEEDS])
            sd = float(np.std(v, ddof=1))
            rows.append({"table": table, "key": "|".join(f"{a}={b}" for a, b in zip(k, idx)), "value_42_raw": v[0], "value_1337_raw": v[1],
                         "value_2024_raw": v[2], "sd_raw": sd, "sd": round(sd * 100, 1) + 0.0, "range": round((v.max() - v.min()) * 100, 1) + 0.0,
                         "unit": "percentage_points"})
    out = pd.DataFrame(rows)
    out.to_csv(prefix + "_seed_variation.csv", index=False)
    return out


def main_seed_variation(config_path):
    settings = Config(config_path).get("statistics")
    os.makedirs(os.path.dirname(settings["output_prefix"]), exist_ok=True)
    seed_variation(settings["inputs"], settings["output_prefix"])
