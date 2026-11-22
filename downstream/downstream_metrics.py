import numpy as np


def weighted_auroc(scores: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="stable")
    ordered = scores[order]
    positive = targets[order].astype(np.float64)
    weights = weights[:, order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    pos = np.add.reduceat(weights * positive, starts, axis=1)
    neg = np.add.reduceat(weights * (1.0 - positive), starts, axis=1)
    below = np.cumsum(neg, axis=1) - neg
    numerator = (pos * (below + 0.5 * neg)).sum(axis=1)
    denominator = pos.sum(axis=1) * neg.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1.0), np.nan)


def f1_counts(predictions: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> np.ndarray:
    predicted = predictions.astype(bool)
    actual = targets.astype(bool)
    indicators = np.stack([predicted & actual, predicted & ~actual, ~predicted & actual], axis=1).astype(np.float64)
    return weights @ indicators


def f1(counts: np.ndarray) -> np.ndarray:
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    denominator = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, 2 * tp / np.where(denominator > 0, denominator, 1.0), 0.0)
