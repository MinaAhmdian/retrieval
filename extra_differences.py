"""
extra_differences.py
Created on November 22, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os

import duckdb
import numpy as np
import pandas as pd

from lib.stats_utils import bh_fdr
from final_differences import (DIFF_COLS, SEEDS3, STAT_COLS, arm, counts_matrix, diff_row, family_name,
                                            macro, perm_p, run_spec, signs, single, summarize)
from config import Config


def input_specs():
    out = []
    for bb, configs in (("biovilt", ["frontal_copy", "shuffled_context"]), ("raddino", ["frontal_copy"])):
        for rule, k in (("profile", 0), ("alignment", 0)):
            for c in configs:
                out.append({"name": f"full_minus_{c}", "a": arm(backbone=bb, rule=rule, k=k), "b": arm(configuration=c, backbone=bb, rule=rule, k=k)})
            for qi in ["no_lateral", "no_context"]:
                out.append({"name": f"all_minus_{qi}", "a": arm(backbone=bb, rule=rule, k=k), "b": arm(backbone=bb, rule=rule, k=k, query_inputs=qi)})
    return out


def published(con, diffs, stats, path):
    P = path
    con.execute(f"create or replace table pub as select * from read_csv('{P}', all_varchar=true)")
    con.execute("""create or replace view ev as select 'published' as src,
        case method when 'encoder_frontal' then 'encoder' when 'chextriev_rerun' then 'chextriev' else method end as retriever,
        case method when 'encoder_frontal' then 'full' else 'none' end as "configuration",
        case method when 'encoder_frontal' then 'biovilt' else 'none' end as backbone, seed, test_corpus, 'test' as split, archive, rule,
        0 as k, query_inputs, level, finding, metric, exam_id, cast(value as double) as value from pub where protocol = 'study_common'""")
    enc = arm(seeds=SEEDS3, rule="alignment", query_inputs="frontal_only")
    chx = arm(retriever="chextriev", configuration="none", backbone="none", seeds=("42",), rule="image_image", query_inputs="frontal_only")
    clf = arm(retriever="finding_classifier", configuration="none", backbone="none", seeds=("none",), rule="profile", query_inputs="all")
    for name, a, b in (("study_common_encoder_minus_chextriev", enc, chx), ("study_common_encoder_minus_finding_classifier", enc, clf),
                       ("study_common_finding_classifier_minus_chextriev", clf, chx)):
        run_spec(con, {"name": name, "a": a, "b": b}, diffs, stats)
    q = """with a as (select finding, metric, image_id, avg(cast(value as double)) v from pub where protocol = 'published_all' and method = 'encoder_frontal'
                      group by all having count(*) = 3),
                b as (select finding, metric, image_id, cast(value as double) v from pub where protocol = 'published_all' and method = 'chextriev_rerun' and seed = '42')
           select a.finding, a.metric, a.image_id, a.v - b.v d from a join b using (finding, metric, image_id)"""
    pr = con.execute(q).fetchdf()
    spec = {"name": "published_all_encoder_minus_chextriev", "a": dict(enc, query_inputs="frontal_only", rule="image_image"),
            "b": dict(chx)}
    for metric, g in pr.groupby("metric"):
        piv = g.pivot(index="image_id", columns="finding", values="d").sort_index()
        cols = list(piv.columns)
        D = piv.to_numpy(dtype=float)
        M = np.isfinite(D)
        fam = family_name(spec, "mimic", "undocumented", "finding", f"per_finding_{metric}")
        ps = []
        for j, f in enumerate(cols):
            d = D[M[:, j], j]
            o, boot, perm = single(d)
            diffs.append(diff_row(spec, "mimic", "undocumented", "finding", f, metric, summarize(boot), len(d)))
            ps.append((f, o, perm_p(o, perm), len(d)))
        padj = bh_fdr(np.array([p for _, _, p, _ in ps]))
        for (f, o, p, nq), qv in zip(ps, padj):
            stats.append({"family": fam, "item": f, "measure": "perm_p_mean_difference", "estimate": o, "n": nq, "p_raw": p, "p_fdr": qv,
                          "significant_fdr05": int(qv < 0.05)})
        o, boot, perm = macro(D, M)
        diffs.append(diff_row(spec, "mimic", "undocumented", "finding", "macro", metric, summarize(boot), len(D)))
        n = len(D)
        D0 = np.where(M, D, 0.0)
        C = counts_matrix(n)
        wboot = (C @ D0).sum(axis=1) / (C @ M.astype(float)).sum(axis=1)
        wperm = (signs(n) @ D0).sum(axis=1) / M.sum()
        wo = D0.sum() / M.sum()
        diffs.append(diff_row(spec, "mimic", "undocumented", "finding", "weighted", metric, summarize(wboot), n))
        fam2 = family_name(spec, "mimic", "undocumented", "finding", f"summary_{metric}")
        items = [("macro", o, perm_p(o, perm), n), ("weighted", wo, perm_p(wo, wperm), n)]
        padj = bh_fdr(np.array([p for _, _, p, _ in items]))
        for (it, ov, p, nq), qv in zip(items, padj):
            stats.append({"family": fam2, "item": f"{it}:{metric}", "measure": "perm_p_mean_difference", "estimate": ov, "n": nq, "p_raw": p,
                          "p_fdr": qv, "significant_fdr05": int(qv < 0.05)})


def main_extra_differences(config_path):
    settings = Config(config_path).get("statistics")
    os.makedirs(os.path.dirname(settings["database"]), exist_ok=True)
    os.makedirs(os.path.dirname(settings["output_prefix"]), exist_ok=True)
    from evidence_cache import ensure_evidence
    con = duckdb.connect(settings["database"])
    ensure_evidence(con, settings["inputs"])
    con.execute("PRAGMA threads=14")
    diffs, stats = [], []
    for spec in input_specs():
        run_spec(con, spec, diffs, stats)
    pd.DataFrame(diffs, columns=DIFF_COLS).to_csv(settings["output_prefix"] + "_input_diffs.csv", index=False)
    pd.DataFrame(stats, columns=STAT_COLS).to_csv(settings["output_prefix"] + "_input_stats.csv", index=False)
    con.close()
    d2, s2 = [], []
    with duckdb.connect(":memory:") as published_connection:
        published(published_connection, d2, s2, settings["inputs"]["published_per_query"])
    pd.DataFrame(d2, columns=DIFF_COLS).to_csv(settings["output_prefix"] + "_pub_diffs.csv", index=False)
    pd.DataFrame(s2, columns=STAT_COLS).to_csv(settings["output_prefix"] + "_pub_stats.csv", index=False)
