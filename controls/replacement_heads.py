import hashlib
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from lib.exam_encoder import LABELS
from config import Config
from encoders.models import compute_device
from provenance import FileDigests, ProvenanceRecords, Unit, sha256_file
from trained_encoder.fit_data import FitData
from trained_encoder.test_outputs import TestEmbedder, load_features, load_outputs

LOGISTIC, LEAST_SQUARES = "logistic_regression", "least_squares"
REPLACES = {LOGISTIC: "profile", LEAST_SQUARES: "alignment"}
FIT_SPLIT = "train"


def head_settings(config: Config, configuration: str) -> dict:
    heads = config.get("controls.replacement_heads.configurations")
    if configuration not in heads:
        raise ValueError(f"{configuration} has no replacement head (controls.replacement_heads.configurations)")
    return heads[configuration]


def row_digests(matrix: np.ndarray) -> list[str]:
    return [hashlib.sha256(row.tobytes()).hexdigest()[:16] for row in np.ascontiguousarray(matrix, dtype=np.float32)]


def fit_logistic(features: np.ndarray, labels: np.ndarray, settings: dict) -> tuple[dict, pd.DataFrame]:
    models, rows = [], []
    for j, label in enumerate(LABELS):
        used = np.isin(labels[:, j], (0, 1))
        y = labels[used, j].astype(int)
        row = {"label": label, "train_cells": int(used.sum()), "positives": int(y.sum()),
               "negatives": int(len(y) - y.sum()), "fitted": False, "n_iter": 0, "converged": False}
        model = None
        if len(np.unique(y)) >= 2:
            model = LogisticRegression(C=settings["C"], max_iter=settings["max_iter"])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(features[used], y)
            row.update(fitted=True, n_iter=int(model.n_iter_[0]),
                       converged=not any(issubclass(w.category, ConvergenceWarning) for w in caught))
        models.append(model)
        rows.append(row)
    return {"models": models}, pd.DataFrame(rows)


def apply_logistic(head: dict, features: np.ndarray) -> np.ndarray:
    return np.stack([model.predict_proba(features)[:, 1] for model in head["models"]], axis=1)


def fit_least_squares(features: np.ndarray, targets: np.ndarray) -> tuple[dict, pd.DataFrame]:
    design = np.hstack([np.asarray(features, dtype=np.float64), np.ones((len(features), 1))])
    weights, _, rank, _ = np.linalg.lstsq(design, np.asarray(targets, dtype=np.float64), rcond=None)
    counts = pd.DataFrame([{"train_exams": len(features), "feature_dim": features.shape[1],
                            "target_dim": targets.shape[1], "design_rank": int(rank),
                            "design_columns": design.shape[1]}])
    return {"weights": weights}, counts


def apply_least_squares(head: dict, features: np.ndarray) -> tuple[np.ndarray, int]:
    design = np.hstack([np.asarray(features, dtype=np.float64), np.ones((len(features), 1))])
    mapped = design @ head["weights"]
    norms = np.linalg.norm(mapped, axis=1, keepdims=True)
    zero = (norms[:, 0] == 0)
    out = np.zeros_like(mapped)
    out[~zero] = mapped[~zero] / norms[~zero]
    return out, int(zero.sum())


class TrainEmbedder(TestEmbedder):

    step = "embed-train-exams"

    def __init__(self, config: Config, run: str):
        super().__init__(config, run)
        self.head = head_settings(config, self.configuration)
        self.capture_norm = self.head["features"] == "norm"
        settings = config.get("controls.replacement_heads")
        self.out_dir = config.path("paths.controls_dir") / settings["features_folder"] / run

    def path(self) -> Path:
        return self.out_dir / f"{FIT_SPLIT}.npz"

    def step_settings(self) -> dict:
        return {**self.base_settings(self.config.get("controls.validation.query_inputs")), "split": FIT_SPLIT,
                "features": self.head["features"]}

    def data(self) -> FitData:
        return FitData(self.config, self.configuration, FIT_SPLIT)

    def run(self) -> None:
        if not self.chosen.exists():
            raise FileNotFoundError(f"[{self.run_name}] no chosen weights; train the run first")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        data = self.data()
        ids = data.ids()
        unit = Unit(FIT_SPLIT, ids, data.contents(FileDigests(self.config.get("provenance.digest_workers"))),
                    [self.path()])
        records = ProvenanceRecords(self.out_dir, self.step, self.step_settings())
        if records.pending([unit], f"{self.run_name} {unit.name}"):
            device = compute_device(self.settings["device"])
            model = self.load_model(device)
            outputs = self.embed_outputs(model, data, device)
            partial = self.path().with_name(self.path().stem + ".partial.npz")
            np.savez(partial, exam_keys=np.array(ids, dtype=str), **outputs)
            os.replace(partial, self.path())
            records.write(unit)
            del model
            torch.cuda.empty_cache()


class FitReplacementHeads:

    step = "fit-replacement-heads"

    def __init__(self, config: Config, run: str):
        self.config = config
        self.run_name = run
        self.train = TrainEmbedder(config, run)
        self.test = TestEmbedder(config, run)
        self.configuration = self.train.configuration
        self.head = self.train.head
        self.fit = self.head["fit"]
        if self.fit not in REPLACES:
            raise ValueError(f"{self.configuration}: unknown fit {self.fit}")
        settings = config.get("controls.replacement_heads")
        self.out_dir = config.path("paths.controls_dir") / settings["folder"] / run
        self.fit_settings = settings[LOGISTIC] if self.fit == LOGISTIC else {}

    @property
    def head_path(self) -> Path:
        return self.out_dir / "head.joblib"

    @property
    def counts_path(self) -> Path:
        return self.out_dir / "fit_counts.csv"

    def applied_path(self, corpus: str, side: str, query_inputs: str) -> Path:
        return self.out_dir / f"{self.test.unit_name(corpus, side, query_inputs)}.npz"

    def fit_records(self) -> ProvenanceRecords:
        return ProvenanceRecords(self.out_dir, self.step, {
            "fit": self.fit, "features": self.head["features"], "settings": self.fit_settings,
            "train_features_sha256": sha256_file(self.train.path()),
            "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__}})

    def apply_records(self) -> ProvenanceRecords:
        return ProvenanceRecords(self.out_dir, f"{self.step} apply", {"head_sha256": sha256_file(self.head_path)})

    def train_inputs(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        data = self.train.data()
        ids = data.ids()
        with np.load(self.train.path()) as saved:
            if saved["exam_keys"].tolist() != ids:
                raise ValueError(f"{self.train.path().name} does not hold the train fit exams in their order")
            features = saved["alignment"] if self.head["features"] == "alignment" else saved["features"]
        return ids, features, (data.labels if self.fit == LOGISTIC else data.reports)

    def fit_head(self) -> None:
        ids, features, targets = self.train_inputs()
        contents = [f"{f}|{t}" for f, t in zip(row_digests(features), row_digests(targets))]
        unit = Unit("head", ids, contents, [self.head_path, self.counts_path])
        records = self.fit_records()
        if records.pending([unit], f"{self.run_name} head"):
            head, counts = (fit_logistic(features, targets, self.fit_settings) if self.fit == LOGISTIC
                            else fit_least_squares(features, targets))
            counts.to_csv(self.counts_path, index=False)
            if self.fit == LOGISTIC and not counts["fitted"].all():
                raise RuntimeError(f"[{self.run_name}] labels with fewer than 2 classes in their train cells: "
                                   f"{counts.loc[~counts['fitted'], 'label'].tolist()}; see {self.counts_path.name}")
            partial = self.head_path.with_name(self.head_path.name + ".partial")
            joblib.dump({"exam_keys": ids, **head}, partial)
            os.replace(partial, self.head_path)
            records.write(unit)

    def load_head(self) -> dict:
        return joblib.load(self.head_path)

    def apply(self) -> None:
        head = self.load_head()
        records = self.apply_records()
        for corpus in self.test.corpora():
            for side, query_inputs in self.test.units_of(corpus):
                source = self.test.path(corpus, side, query_inputs)
                with np.load(source) as saved:
                    exam_ids = pd.Series(saved["exam_ids"])
                if self.fit == LOGISTIC:
                    inputs = load_outputs(source, exam_ids)[0]
                else:
                    inputs = load_features(source, exam_ids)
                out = self.applied_path(corpus, side, query_inputs)
                unit = Unit(out.stem, exam_ids.tolist(), row_digests(inputs), [out])
                zero = 0
                if records.pending([unit], f"{self.run_name} {unit.name}"):
                    if self.fit == LOGISTIC:
                        replaced = apply_logistic(head, inputs)
                    else:
                        replaced, zero = apply_least_squares(head, inputs)
                    partial = out.with_name(out.stem + ".partial.npz")
                    np.savez(partial, exam_ids=exam_ids.to_numpy(dtype=str), zero_rows=zero,
                             **{REPLACES[self.fit]: replaced})
                    os.replace(partial, out)
                    records.write(unit)

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fit_head()
        self.apply()


def replaced_representation(config: Config, run: str, corpus: str, side: str, query_inputs: str,
                            exam_ids: pd.Series) -> tuple[str, np.ndarray]:
    heads = FitReplacementHeads(config, run)
    path = heads.applied_path(corpus, side, query_inputs)
    part = REPLACES[heads.fit]
    with np.load(path) as saved:
        if saved["exam_ids"].tolist() != exam_ids.tolist():
            raise ValueError(f"{path.name} does not hold these exams in this order")
        return part, saved[part]
