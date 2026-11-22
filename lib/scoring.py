"""
lib/scoring.py
Created on October 17, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import numpy as np


def profile_scores(query_probs, archive_labels, finding_mask=None):
    q = np.asarray(query_probs, dtype=np.float64).copy()
    d = np.asarray(archive_labels, dtype=np.float64).copy()
    d[d < 0] = 0.0
    if finding_mask is not None:
        q = q[:, finding_mask]
        d = d[:, finding_mask]
    return q @ d.T


def alignment_scores(query_emb, archive_emb):
    return np.asarray(query_emb, dtype=np.float64) @ np.asarray(archive_emb, dtype=np.float64).T


def two_stage_scores(profile, alignment, k):
    P = np.asarray(profile, dtype=np.float64)
    A = np.asarray(alignment, dtype=np.float64)
    out = P.copy()
    order = np.argsort(-P, axis=1, kind="stable")[:, :k]
    kth = np.take_along_axis(P, order[:, [-1]], axis=1)
    a = np.take_along_axis(A, order, axis=1)
    rank = np.argsort(np.argsort(-a, axis=1, kind="stable"), axis=1)
    lift = (P.max(axis=1, keepdims=True) - P.min(axis=1, keepdims=True)) + 1.0
    np.put_along_axis(out, order, kth + lift * (k - rank) / k, axis=1)
    return out
