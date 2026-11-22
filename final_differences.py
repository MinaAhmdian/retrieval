"""
final_differences.py
Created on November 22, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os

import duckdb
import numpy as np
import pandas as pd

from lib.stats_utils import bh_fdr
from config import Config

N_BOOT = N_PERM = 1000
SEED = 0
MIN_POS = 20
CORE7 = ["atelectasis", "cardiomegaly", "consolidation", "edema", "pleural_effusion", "pneumonia", "pneumothorax"]
SEEDS3 = ("42", "1337", "2024")
DIFF_COLS = ["comparison", "retriever_a", "retriever_b", "backbone_a", "backbone_b", "seeds", "test_corpus", "split", "archive",
             "rule_a", "rule_b", "k_a", "k_b", "configuration_a", "query_inputs_a", "configuration_b", "query_inputs_b", "level",
             "finding", "metric", "mean", "std", "ci_low", "ci_high", "mean_raw", "n", "unit"]
STAT_COLS = ["family", "item", "measure", "estimate", "n", "p_raw", "p_fdr", "significant_fdr05"]


def build_db(con, inputs):
    con.execute("PRAGMA threads=14")
    con.execute(f"""create table ev as
      select 'encoders' as src, retriever, 'none' as "configuration", 'none' as backbone, 'none' as seed, test_corpus, 'test' as split, archive, rule,
             0 as k, 'all' as query_inputs, level, finding, metric, exam_id, cast(value as double) as value
        from read_csv('{inputs['encoders_per_query']}', all_varchar=true)
      union all
      select 'model', 'encoder', configuration, backbone, seed, test_corpus, 'test', archive, rule,
             case when rule = 'two_stage' then 50 else 0 end, query_inputs, level, finding, metric, exam_id, cast(value as double)
        from read_csv('{inputs['model_per_query']}', all_varchar=true)
      union all
      select g.src, 'encoder', configuration, backbone, seed, test_corpus, split, archive, rule, cast(k as integer), query_inputs, level,
             finding, metric, exam_id, cast(value as double)
        from (select 'controls' as src, * from read_csv('{inputs['controls_per_query']}', all_varchar=true)
              union all select 'transfer', * from read_csv('{inputs['transfer_per_query']}', all_varchar=true)) g""")


def arm_sql(arm):
    seeds = arm["seeds"]
    cond = [f"retriever = '{arm['retriever']}'", f"configuration = '{arm['configuration']}'", f"backbone = '{arm['backbone']}'",
            f"split = 'test'", f"rule = '{arm['rule']}'", f"k = {arm['k']}", f"query_inputs = '{arm['query_inputs']}'",
            "seed in (" + ",".join(f"'{s}'" for s in seeds) + ")"]
    return (f"select test_corpus, archive, level, finding, metric, exam_id, avg(value) v from ev where {' and '.join(cond)} "
            f"group by all having count(*) = {len(seeds)}")


def counts_matrix(n):
    idx = np.random.RandomState(SEED).randint(0, n, size=(N_BOOT, n))
    return np.bincount((np.arange(N_BOOT)[:, None] * n + idx).ravel(), minlength=N_BOOT * n).reshape(N_BOOT, n).astype(float)


def signs(n):
    return np.random.RandomState(SEED).choice([-1.0, 1.0], size=(N_PERM, n))


def single(d):
    n = len(d)
    boot = counts_matrix(n) @ d / n
    perm = signs(n) @ d / n
    return d.mean(), boot, perm


def macro(D, M):
    n = D.shape[0]
    D0 = np.where(M, D, 0.0)
    Mf = M.astype(float)
    C = counts_matrix(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = ((C @ D0) / (C @ Mf)).mean(axis=1)
    perm = ((signs(n) @ D0) / Mf.sum(axis=0)).mean(axis=1)
    obs = (D0.sum(axis=0) / Mf.sum(axis=0)).mean()
    return obs, boot, perm


def summarize(boot):
    b = boot[np.isfinite(boot)]
    if b.size < 2:
        return None
    return b.mean(), b.std(ddof=1), np.percentile(b, 2.5), np.percentile(b, 97.5)


def perm_p(obs, perm):
    if not np.isfinite(obs):
        return np.nan
    return (np.sum(np.abs(perm) >= abs(obs) - 1e-15) + 1) / (N_PERM + 1)


def diff_row(spec, corpus, archive, level, finding, metric, s, n):
    a, b = spec["a"], spec["b"]
    return {"comparison": spec["name"], "retriever_a": a["retriever"], "retriever_b": b["retriever"], "backbone_a": a["backbone"],
            "backbone_b": b["backbone"], "seeds": ";".join(a["seeds"]), "test_corpus": corpus, "split": "test", "archive": archive,
            "rule_a": a["rule"], "rule_b": b["rule"], "k_a": a["k"], "k_b": b["k"], "configuration_a": a["configuration"],
            "query_inputs_a": a["query_inputs"], "configuration_b": b["configuration"], "query_inputs_b": b["query_inputs"],
            "level": level, "finding": finding, "metric": metric, "mean": round(s[0] * 100, 1) + 0.0, "std": round(s[1] * 100, 1) + 0.0,
            "ci_low": round(s[2] * 100, 1) + 0.0, "ci_high": round(s[3] * 100, 1) + 0.0, "mean_raw": s[0], "n": int(n), "unit": "percentage_points"}


def family_name(spec, corpus, archive, level, kind):
    a, b = spec["a"], spec["b"]
    return (f"{corpus}|{archive}|{level}|{kind}|{spec['name']}|{a['backbone']}|{';'.join(a['seeds'])}|"
            f"{a['rule']}@{a['k']}|{b['rule']}@{b['k']}|{a['query_inputs']}|{b['query_inputs']}")


def run_spec(con, spec, diffs, stats):
    sql = (f"with a as ({arm_sql(spec['a'])}), b as ({arm_sql(spec['b'])}) "
           "select a.test_corpus, a.archive, a.level, a.finding, a.metric, a.exam_id, a.v - b.v d "
           "from a join b using (test_corpus, archive, level, finding, metric, exam_id)")
    paired = con.execute(sql).fetchdf()
    for (corpus, archive, level), g in paired.groupby(["test_corpus", "archive", "level"], sort=True):
        groups = []
        if level == "finding":
            ap = g[g.metric == "ap"]
            if len(ap):
                piv = ap.pivot(index="exam_id", columns="finding", values="d").sort_index()
                cols = list(piv.columns)
                M = piv.notna().to_numpy()
                counts = M.sum(axis=0)
                eligible = [c for c, k in zip(cols, counts) if k >= MIN_POS]
                if eligible:
                    piv = piv[eligible]
                    piv = piv[piv.notna().any(axis=1)]
                    groups.append(("per_finding_ap", piv, eligible))
            for metric in ["hit1", "hit5", "hit10"]:
                h = g[(g.metric == metric) & (g.finding == "pooled")]
                if len(h):
                    groups.append((metric, h.set_index("exam_id")[["d"]].sort_index(), None))
        else:
            for metric in ["ndcg10", "ndcg50", "p10_f1_05"]:
                h = g[(g.metric == metric) & (g.finding == "pooled")]
                if len(h):
                    groups.append((metric, h.set_index("exam_id")[["d"]].sort_index(), None))
        summary_items = []
        for kind, piv, eligible in groups:
            D = piv.to_numpy(dtype=float)
            M = np.isfinite(D)
            if kind == "per_finding_ap":
                fam = family_name(spec, corpus, archive, level, "per_finding_ap")
                ps = []
                for j, f in enumerate(eligible):
                    d = D[M[:, j], j]
                    o, boot, perm = single(d)
                    s = summarize(boot)
                    if s is not None:
                        diffs.append(diff_row(spec, corpus, archive, level, f, "ap", s, len(d)))
                    ps.append((f, o, perm_p(o, perm), len(d)))
                padj = bh_fdr(np.array([p for _, _, p, _ in ps]))
                for (f, o, p, nq), q in zip(ps, padj):
                    stats.append({"family": fam, "item": f, "measure": "perm_p_mean_difference", "estimate": o, "n": nq,
                                  "p_raw": p, "p_fdr": q, "significant_fdr05": int(q < 0.05)})
                for name, members in (("macro", eligible), ("core7", CORE7)):
                    if not set(members) <= set(eligible):
                        continue
                    sel = [eligible.index(c) for c in members]
                    keep = M[:, sel].any(axis=1)
                    o, boot, perm = macro(D[keep][:, sel], M[keep][:, sel])
                    s = summarize(boot)
                    if s is not None:
                        diffs.append(diff_row(spec, corpus, archive, level, name, "macro_ap", s, int(keep.sum())))
                    summary_items.append((f"{name}:macro_ap", o, perm_p(o, perm), int(keep.sum())))
            else:
                o, boot, perm = single(D[:, 0])
                s = summarize(boot)
                if s is not None:
                    diffs.append(diff_row(spec, corpus, archive, level, "pooled", kind, s, len(D)))
                summary_items.append((f"pooled:{kind}", o, perm_p(o, perm), len(D)))
        if summary_items:
            fam = family_name(spec, corpus, archive, level, "summary")
            padj = bh_fdr(np.array([p for _, _, p, _ in summary_items]))
            for (item, o, p, nq), q in zip(summary_items, padj):
                stats.append({"family": fam, "item": item, "measure": "perm_p_mean_difference", "estimate": o, "n": nq,
                              "p_raw": p, "p_fdr": q, "significant_fdr05": int(q < 0.05)})


def arm(retriever="encoder", configuration="full", backbone="biovilt", seeds=SEEDS3, rule="alignment", k=0, query_inputs="all"):
    return {"retriever": retriever, "configuration": configuration, "backbone": backbone, "seeds": tuple(seeds), "rule": rule,
            "k": k, "query_inputs": query_inputs}


def specs():
    out = []
    frozen = [("biovilt_frozen", ["image_image", "image_report", "two_view_mean"]), ("raddino_frozen", ["image_image", "two_view_mean"]),
              ("biomedclip_frozen", ["image_image", "image_report", "two_view_mean"]), ("medsiglip_frozen", ["image_image", "image_report", "two_view_mean"]),
              ("cxrclip_frozen", ["image_image", "image_report", "two_view_mean"]), ("medimageinsight_frozen", ["image_image", "two_view_mean"]),
              ("finding_classifier", ["profile"])]
    enc_rules = [("profile", 0), ("alignment", 0), ("two_stage", 100)]
    for bb in ["biovilt", "raddino"]:
        for rule, k in enc_rules:
            for fr, rules in frozen:
                for fr_rule in rules:
                    out.append({"name": f"encoder_minus_{fr}", "a": arm(backbone=bb, rule=rule, k=k),
                                "b": arm(retriever=fr, configuration="none", backbone="none", seeds=("none",), rule=fr_rule)})
    for c in ["shuffled_lateral", "no_dropout", "alignment_only", "profile_only", "midlevel_fusion", "context_text_only", "context_agesex_only",
              "loso_without_mimic", "loso_without_chexpert", "loso_without_rexgradient"]:
        for rule, k in enc_rules:
            out.append({"name": f"full_minus_{c}", "a": arm(seeds=("42",), rule=rule, k=k), "b": arm(configuration=c, seeds=("42",), rule=rule, k=k)})
    for rule, k in enc_rules:
        out.append({"name": "full_minus_shuffled_report", "a": arm(rule=rule, k=k), "b": arm(configuration="shuffled_report", rule=rule, k=k)})
    for bb, configs in (("biovilt", ["frontal_copy", "shuffled_context"]), ("raddino", ["frontal_copy"])):
        for c in configs:
            out.append({"name": f"full_minus_{c}", "a": arm(backbone=bb, rule="two_stage", k=100),
                        "b": arm(configuration=c, backbone=bb, rule="two_stage", k=100)})
        for qi in ["no_lateral", "no_context"]:
            out.append({"name": f"all_minus_{qi}", "a": arm(backbone=bb, rule="two_stage", k=100),
                        "b": arm(backbone=bb, rule="two_stage", k=100, query_inputs=qi)})
    return out


def main_final_differences(config_path):
    settings = Config(config_path).get("statistics")
    os.makedirs(os.path.dirname(settings["database"]), exist_ok=True)
    os.makedirs(os.path.dirname(settings["output_prefix"]), exist_ok=True)
    from evidence_cache import ensure_evidence
    con = duckdb.connect(settings["database"])
    ensure_evidence(con, settings["inputs"])
    con.execute("PRAGMA threads=14")
    diffs, stats = [], []
    for spec in specs():
        run_spec(con, spec, diffs, stats)
    pd.DataFrame(diffs, columns=DIFF_COLS).to_csv(settings["output_prefix"] + "_diffs.csv", index=False)
    pd.DataFrame(stats, columns=STAT_COLS).to_csv(settings["output_prefix"] + "_stats.csv", index=False)
