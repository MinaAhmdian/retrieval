import numpy as np
import pandas as pd

from lib import metrics
from lib.exam_encoder import LABELS

METRIC_COLUMNS = ["level", "finding", "metric", "exam_id", "value"]


def precision_metric(k: int, threshold: float) -> str:
    return f"p{k}_f1_{f'{threshold:.1f}'.replace('.', '')}"


def metric_frame(scores: np.ndarray, labels: dict, report: dict | None, evaluation: dict) -> pd.DataFrame:
    query_ids = np.asarray(labels["query_ids"])
    parts = []

    def rows(level, finding, metric, exam_ids, values):
        parts.append(pd.DataFrame({"level": level, "finding": finding, "metric": metric, "exam_id": exam_ids,
                                   "value": values}))

    ap = metrics.finding_level_ap(scores, labels["query"], labels["archive"],
                                  min_positive_queries=evaluation["min_positive_queries"])
    query_index, label_index = np.nonzero(~np.isnan(ap))
    rows("finding", np.array(LABELS)[label_index], "ap", query_ids[query_index], ap[query_index, label_index])
    relevant = ((labels["query"] == 1).astype(np.float32) @ (labels["archive"] == 1).astype(np.float32).T) > 0
    for k in evaluation["hit_ks"]:
        rows("finding", "pooled", f"hit{k}", query_ids, metrics.hit_at_k(scores, relevant, k))
    if report is not None:
        kept_scores, report_ids = scores[report["keep"]], query_ids[report["keep"]]
        for k in evaluation["ndcg_ks"]:
            values = metrics.ndcg_at_k(kept_scores, report["gains"], k)
            defined = ~np.isnan(values)
            rows("report", "pooled", f"ndcg{k}", report_ids[defined], values[defined])
        k, threshold = evaluation["precision_k"], evaluation["precision_threshold"]
        rows("report", "pooled", precision_metric(k, threshold), report_ids,
             metrics.precision_at_k(kept_scores, report["gains"], k, threshold=threshold))
    return pd.concat(parts, ignore_index=True)[METRIC_COLUMNS]
