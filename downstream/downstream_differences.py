"""
downstream/downstream_differences.py
Created on November 22, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os

import numpy as np
import pandas as pd

from lib.stats_utils import bh_fdr
from config import Config

N_BOOT = 1000
SEEDS = ["42", "1337", "2024"]
ARMS = SEEDS + ["none"]
DIFF_COLS = ["comparison", "retriever_a", "retriever_b", "seeds", "test_corpus", "archive", "rule_a", "rule_b", "k_a", "k_b", "task",
             "finding", "metric", "mean", "std", "ci_low", "ci_high", "mean_raw", "n", "unit"]
STAT_COLS = ["family", "item", "measure", "estimate", "n", "p_raw", "p_fdr", "significant_fdr05"]


def multiplicities(n):
    idx = np.random.RandomState(0).randint(0, n, size=(N_BOOT, n))
    return np.bincount((np.arange(N_BOOT)[:, None] * n + idx).ravel(), minlength=N_BOOT * n).reshape(N_BOOT, n).astype(float)


def weighted_auroc(y, s, W):
    order = np.argsort(s, kind="stable")
    ys, ss, Ws = y[order], s[order], W[:, order]
    starts = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]
    P = np.add.reduceat(Ws * (ys == 1), starts, axis=1)
    N = np.add.reduceat(Ws * (ys == 0), starts, axis=1)
    below = np.cumsum(N, axis=1) - N
    num = (P * (below + 0.5 * N)).sum(axis=1)
    den = P.sum(axis=1) * N.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


def f1_counts(y, pred, W):
    tp = W @ ((y == 1) & (pred == 1)).astype(float)
    fp = W @ ((y == 0) & (pred == 1)).astype(float)
    fn = W @ ((y == 1) & (pred == 0)).astype(float)
    return tp, fp, fn


def f1(tp, fp, fn):
    den = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)


def summary(b):
    b = b[np.isfinite(b)]
    if b.size == 0:
        return None
    return b.mean(), (b.std(ddof=1) if b.size > 1 else np.nan), np.percentile(b, 2.5), np.percentile(b, 97.5), b.size


def boot_p(obs, b):
    b = b[np.isfinite(b)]
    if b.size == 0 or not np.isfinite(obs):
        return np.nan
    return max(float(np.mean(np.abs(b - obs) >= abs(obs))), 1.0 / b.size)


def disp(x):
    return round(x * 100, 1) + 0.0


def downstream_differences(path, prefix):
    e = pd.read_csv(path, dtype=str, keep_default_na=False)
    e["arm"] = np.where(e.retriever == "encoder", e.seed, "none")
    e["target"] = e.target.astype(int)
    e["score"] = e.score.astype(float)
    diffs, stats = [], []
    for (corpus, archive, task), g in e.groupby(["test_corpus", "archive", "task"], sort=True):
        tgt = g[g.arm == "none"].pivot(index="exam_id", columns="finding", values="target")
        sc = {a: g[g.arm == a].pivot(index="exam_id", columns="finding", values="score").reindex(index=tgt.index, columns=tgt.columns) for a in ARMS}
        for a in SEEDS:
            ta = g[g.arm == a].pivot(index="exam_id", columns="finding", values="target").reindex(index=tgt.index, columns=tgt.columns)
            assert ((ta == tgt) | (ta.isna() & tgt.isna())).all().all(), "targets differ between arms"
        tgt = tgt.sort_index()
        sc = {a: v.loc[tgt.index] for a, v in sc.items()}
        ids = tgt.index.to_numpy()
        T = tgt.to_numpy(dtype=float)
        S = {a: v.to_numpy(dtype=float) for a, v in sc.items()}
        M = np.isfinite(T)
        findings = list(tgt.columns)
        if task == "knn10":
            elig = [j for j, f in enumerate(findings) if np.nansum(T[:, j] == 1) > 0 and np.nansum(T[:, j] == 0) > 0]
            metric = "auroc"
        else:
            elig = [j for j, f in enumerate(findings) if np.nansum(T[:, j] == 1) > 0]
            metric = "f1"
        reps = {}

        def per_arm(Wf, j):
            y = np.nan_to_num(T[:, j]).astype(int)
            if task == "knn10":
                return {a: weighted_auroc(y, np.nan_to_num(S[a][:, j]), Wf) for a in ARMS}
            out = {}
            for a in ARMS:
                tp, fp, fn = f1_counts(y, np.nan_to_num(S[a][:, j]).astype(int), Wf)
                out[a] = (tp, fp, fn)
            return out

        for j in elig:
            keep = M[:, j]
            n = int(keep.sum())
            W = np.zeros((N_BOOT, len(ids)))
            W[:, keep] = multiplicities(n)
            W1 = keep[None, :].astype(float)
            r = per_arm(W, j)
            r1 = per_arm(W1, j)
            if task == "knn10":
                vals = {a: r[a] for a in ARMS}
                pts = {a: r1[a][0] for a in ARMS}
            else:
                vals = {a: f1(*r[a]) for a in ARMS}
                pts = {a: float(f1(*r1[a])[0]) for a in ARMS}
            reps[findings[j]] = (vals, pts, n)
        if elig:
            union = M[:, elig].any(axis=1)
            nU = int(union.sum())
            WU = np.zeros((N_BOOT, len(ids)))
            WU[:, union] = multiplicities(nU)
            W1 = union[None, :].astype(float)
            mac = {a: [] for a in ARMS}
            mac1 = {a: [] for a in ARMS}
            micro = {a: np.zeros((3, N_BOOT)) for a in ARMS}
            micro1 = {a: np.zeros((3, 1)) for a in ARMS}
            for j in elig:
                Wj = WU * M[:, j][None, :]
                W1j = W1 * M[:, j][None, :]
                r, r1 = per_arm(Wj, j), per_arm(W1j, j)
                for a in ARMS:
                    if task == "knn10":
                        mac[a].append(r[a]); mac1[a].append(r1[a])
                    else:
                        mac[a].append(f1(*r[a])); mac1[a].append(f1(*r1[a]))
                        micro[a] += np.vstack(r[a]); micro1[a] += np.vstack(r1[a])
            reps["macro"] = ({a: np.mean(np.vstack(mac[a]), axis=0) for a in ARMS}, {a: float(np.mean(np.vstack(mac1[a]), axis=0)[0]) for a in ARMS}, nU)
            if task == "top1_report":
                reps["micro"] = ({a: f1(*micro[a]) for a in ARMS}, {a: float(f1(*micro1[a])[0]) for a in ARMS}, nU)
        fam_items = {"per_finding": [], "summary": []}
        for f, (vals, pts, n) in reps.items():
            enc = np.mean(np.vstack([vals[s] for s in SEEDS]), axis=0)
            d = enc - vals["none"]
            sm = summary(d)
            if sm is None:
                continue
            obs = float(np.mean([pts[s] for s in SEEDS]) - pts["none"])
            diffs.append({"comparison": "encoder_minus_finding_classifier", "retriever_a": "encoder", "retriever_b": "finding_classifier",
                          "seeds": "42;1337;2024", "test_corpus": corpus, "archive": archive, "rule_a": "two_stage", "rule_b": "profile",
                          "k_a": 100, "k_b": 0, "task": task, "finding": f, "metric": metric, "mean": disp(sm[0]), "std": disp(sm[1]),
                          "ci_low": disp(sm[2]), "ci_high": disp(sm[3]), "mean_raw": sm[0], "n": n, "unit": "percentage_points"})
            kind = "summary" if f in ("macro", "micro") else "per_finding"
            fam_items[kind].append((f"{f}:{metric}" if kind == "summary" else f, obs, boot_p(obs, d), n))
        for kind, items in fam_items.items():
            if not items:
                continue
            fam = f"{corpus}|{archive}|{task}|{kind}|encoder_minus_finding_classifier|42;1337;2024|two_stage@100|profile@0"
            q = bh_fdr(np.array([p for _, _, p, _ in items]))
            for (it, o, p, n), qq in zip(items, q):
                stats.append({"family": fam, "item": it, "measure": "boot_p_difference", "estimate": o, "n": n, "p_raw": p, "p_fdr": qq,
                              "significant_fdr05": int(qq < 0.05)})
    pd.DataFrame(diffs, columns=DIFF_COLS).to_csv(prefix + "_downstream_diffs.csv", index=False)
    pd.DataFrame(stats, columns=STAT_COLS).to_csv(prefix + "_downstream_stats.csv", index=False)


def main_downstream_differences(config_path):
    settings = Config(config_path).get("statistics")
    os.makedirs(os.path.dirname(settings["output_prefix"]), exist_ok=True)
    downstream_differences(settings["inputs"]["downstream_per_query"], settings["output_prefix"])
