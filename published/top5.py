import numpy as np


def normalized(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError(f"{int((norms == 0).sum())} embeddings are zero")
    return values / norms


def top_neighbors(embeddings: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    if len(embeddings) <= k:
        raise ValueError(f"{len(embeddings)} images leave fewer than {k} neighbors")
    unit = normalized(embeddings)
    scores = unit[queries] @ unit.T
    scores[np.arange(len(queries)), queries] = -np.inf
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def top_k_metrics(relevance: np.ndarray) -> dict[str, np.ndarray]:
    relevance = np.asarray(relevance, dtype=np.float64)
    k = relevance.shape[1]
    ranks = np.arange(1, k + 1)
    hits = relevance.sum(axis=1)
    first = np.where(hits > 0, relevance.argmax(axis=1) + 1, 0)
    rr = np.where(first > 0, 1.0 / np.maximum(first, 1), 0.0)
    precision = np.cumsum(relevance, axis=1) / ranks
    ap = np.where(hits > 0, (precision * relevance).sum(axis=1) / np.maximum(hits, 1), 0.0)
    return {"hr": hits / k, "rr": rr, "ap": ap}


def pair_metrics(embeddings: np.ndarray, labels: np.ndarray, k: int) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    queries = np.flatnonzero(labels.sum(axis=1) > 0)
    neighbors = top_neighbors(embeddings, queries, k)
    query_rows, findings = np.nonzero(labels[queries] == 1)
    relevance = labels[neighbors[query_rows], findings[:, None]]
    return {"query": queries[query_rows], "finding": findings, **top_k_metrics(relevance)}
