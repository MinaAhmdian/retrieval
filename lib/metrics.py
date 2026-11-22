"""
lib/metrics.py
Created on October 17, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import numpy as np


def split_patients(patient_ids, seed=42, fractions=(0.8, 0.1, 0.1)):
    ids = list(patient_ids)
    distinct = sorted(set(ids), key=str)
    rng = np.random.default_rng(seed)
    rng.shuffle(distinct)
    n = len(distinct)
    n_train = int(round(fractions[0] * n))
    n_valid = int(round(fractions[1] * n))
    part = {}
    for i, p in enumerate(distinct):
        part[p] = "train" if i < n_train else ("valid" if i < n_train + n_valid else "test")
    return [part[p] for p in ids]


def split_queries(patient_ids, labels, seed=42, fraction=0.2):
    ids = np.asarray(list(patient_ids))
    y = np.asarray(labels) == 1
    prev = y.mean(axis=0)
    groups = {}
    for p in sorted(set(ids.tolist()), key=str):
        rows = np.where(ids == p)[0]
        pos = np.where(y[rows].any(axis=0))[0]
        groups[p] = int(pos[np.argmin(prev[pos])]) if len(pos) else -1
    rng = np.random.default_rng(seed)
    chosen = set()
    for g in sorted(set(groups.values())):
        members = sorted([p for p, gg in groups.items() if gg == g], key=str)
        rng.shuffle(members)
        chosen.update(members[: int(round(fraction * len(members)))])
    return np.array([p in chosen for p in ids.tolist()])


def average_precision_rows(scores, relevant):
    order = np.argsort(-np.asarray(scores, dtype=np.float64), axis=1, kind="stable")
    rel = np.take_along_axis(np.asarray(relevant, dtype=np.float64), order, axis=1)
    cum = np.cumsum(rel, axis=1)
    ranks = np.arange(1, rel.shape[1] + 1)
    n_rel = rel.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rel * cum / ranks).sum(axis=1) / n_rel


def finding_level_ap(scores, query_labels, archive_labels, min_positive_queries=20):
    S = np.asarray(scores, dtype=np.float64)
    yq = np.asarray(query_labels)
    yd = np.asarray(archive_labels)
    out = np.full((S.shape[0], yq.shape[1]), np.nan)
    for c in range(yq.shape[1]):
        qpos = yq[:, c] == 1
        if qpos.sum() < min_positive_queries:
            continue
        keep = yd[:, c] != -1
        rel = np.repeat((yd[keep, c] == 1)[None, :], qpos.sum(), axis=0)
        out[qpos, c] = average_precision_rows(S[qpos][:, keep], rel)
    return out


def hit_at_k(scores, relevant, k):
    order = np.argsort(-np.asarray(scores, dtype=np.float64), axis=1, kind="stable")[:, :k]
    return np.take_along_axis(np.asarray(relevant, dtype=bool), order, axis=1).any(axis=1).astype(np.float64)


def ndcg_at_k(scores, gains, k):
    S = np.asarray(scores, dtype=np.float64)
    G = np.asarray(gains, dtype=np.float64)
    order = np.argsort(-S, axis=1, kind="stable")[:, :k]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (np.take_along_axis(G, order, axis=1) * disc).sum(axis=1)
    ideal = (np.sort(G, axis=1)[:, ::-1][:, :k] * disc).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return dcg / ideal


def precision_at_k(scores, gains, k, threshold=0.5):
    order = np.argsort(-np.asarray(scores, dtype=np.float64), axis=1, kind="stable")[:, :k]
    return (np.take_along_axis(np.asarray(gains, dtype=np.float64), order, axis=1) >= threshold).mean(axis=1)


def entity_f1_matrix(query_entities, archive_entities):
    vocab = {}
    for e in list(query_entities) + list(archive_entities):
        for t in e:
            vocab.setdefault(t, len(vocab))
    def mat(sets):
        M = np.zeros((len(sets), len(vocab)), dtype=np.float64)
        for i, e in enumerate(sets):
            for t in e:
                M[i, vocab[t]] = 1.0
        return M
    Mq, Md = mat(query_entities), mat(archive_entities)
    inter = Mq @ Md.T
    denom = Mq.sum(axis=1)[:, None] + Md.sum(axis=1)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(denom > 0, 2.0 * inter / denom, 0.0)
    return f1
