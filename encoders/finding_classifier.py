import hashlib
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from lib.exam_encoder import LABELS, N_AGE_BINS, SEX_IDS, age_bin
from config import Config
from encoders.embedding_store import EmbeddingStore
from encoders.exam_images import ExamImages
from provenance import ProvenanceRecords, Unit, sha256_file

SUMMARY_COLUMNS = ["label", "n_train", "n_positive", "n_iter", "converged"]


class ExamFeatures:

    def __init__(self, config: Config, corpus_name: str):
        self.corpus_name = corpus_name
        self.encoder = config.get("retrieval.classifier_encoder")
        self.dim = config.get(f"retrieval.encoders.{self.encoder}.dim")
        self.exam_images = ExamImages(config, corpus_name)
        self.store = EmbeddingStore(config, corpus_name)

    @property
    def n_features(self) -> int:
        return 3 * self.dim + 2 + N_AGE_BINS + len(SEX_IDS)

    def build(self, exams: pd.DataFrame) -> np.ndarray:
        images = self.exam_images.selection()
        images = images[images["exam_id"].isin(exams["exam_id"])]
        embedded = self.store.images(self.encoder)
        missing = int((~images["image_id"].isin(embedded.index)).sum())
        if missing:
            raise ValueError(f"{self.corpus_name}: {missing} images have no {self.encoder} embedding")
        position = pd.Series(np.arange(len(exams)), index=exams["exam_id"].to_numpy())
        features = np.zeros((len(exams), self.n_features))
        for offset, view in [(0, "frontal"), (self.dim, "lateral")]:
            rows = images[images["view"] == view]
            features[position[rows["exam_id"]].to_numpy(), offset:offset + self.dim] = \
                embedded.loc[rows["image_id"]].to_numpy()
        has_lateral = exams["exam_id"].isin(images.loc[images["view"] == "lateral", "exam_id"]).to_numpy()
        if not exams["exam_id"].isin(images.loc[images["view"] == "frontal", "exam_id"]).all():
            raise ValueError(f"{self.corpus_name}: an exam has no frontal embedding")
        features[:, 2 * self.dim] = has_lateral
        context_offset = 2 * self.dim + 1
        with_context = (exams["has_context_text"] == "1").to_numpy()
        if with_context.any():
            ids, context = self.store.load(self.store.text_file(self.encoder, "context"))
            context = pd.DataFrame(context, index=ids)
            wanted = exams.loc[with_context, "exam_id"]
            if not wanted.isin(context.index).all():
                raise ValueError(f"{self.corpus_name}: an exam with context text has no context embedding")
            features[with_context, context_offset:context_offset + self.dim] = context.loc[wanted].to_numpy()
        features[:, context_offset + self.dim] = with_context
        age_offset = context_offset + self.dim + 1
        ages = [age_bin(float(age)) if age != "" else age_bin(None) for age in exams["age"]]
        features[np.arange(len(exams)), age_offset + np.array(ages)] = 1.0
        sex_offset = age_offset + N_AGE_BINS
        sexes = [SEX_IDS.get(sex, SEX_IDS["unknown"]) for sex in exams["sex"]]
        features[np.arange(len(exams)), sex_offset + np.array(sexes)] = 1.0
        return features

    def describe(self, features: np.ndarray) -> str:
        age_offset = 3 * self.dim + 2
        sex_offset = age_offset + N_AGE_BINS
        return (f"exams {len(features)} | with lateral {int(features[:, 2 * self.dim].sum())} | with context "
                f"{int(features[:, 3 * self.dim + 1].sum())} | age known "
                f"{int(features[:, age_offset:sex_offset - 1].sum())} | sex known "
                f"{int(features[:, sex_offset:sex_offset + SEX_IDS['unknown']].sum())}")


def row_digests(features: np.ndarray) -> list[str]:
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in np.ascontiguousarray(features, dtype=np.float64)]


def fit_label(features: np.ndarray, cells: np.ndarray, settings: dict, path) -> dict:
    used = np.isin(cells, ["0", "1"])
    y = (cells[used] == "1").astype(int)
    model = LogisticRegression(C=settings["C"], max_iter=settings["max_iter"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(features[used], y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    summary = {"n_train": int(used.sum()), "n_positive": int(y.sum()), "n_iter": int(model.n_iter_[0]),
               "converged": converged}
    partial = path.with_name(path.name + ".partial")
    joblib.dump({"model": model, "summary": summary}, partial)
    os.replace(partial, path)
    return summary


class FindingClassifier:

    step = "finding-classifier"

    def __init__(self, config: Config):
        self.config = config
        self.settings = config.get("retrieval.classifier")
        self.out_dir = config.path("paths.retrieval_dir") / "classifier"
        self.records = ProvenanceRecords(self.out_dir, self.step, {
            "classifier": self.settings, "encoder": config.get("retrieval.classifier_encoder"),
            "train_corpora": config.get("retrieval.classifier_train_corpora")})

    def model_path(self, label: str):
        return self.out_dir / f"model_{label}.joblib"

    def model_units(self, features: np.ndarray, cells: pd.DataFrame, exam_keys: list[str]) -> dict[str, Unit]:
        digests = row_digests(features)
        return {label: Unit(f"model_{label}", exam_keys, [f"{d}|{c}" for d, c in zip(digests, cells[label])],
                            [self.model_path(label)]) for label in LABELS}

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        features, cells, exam_keys = self.train_set()
        units = self.model_units(features, cells, exam_keys)
        todo = [unit.name.removeprefix("model_") for unit in self.records.pending(list(units.values()), "train")]
        if todo:
            Parallel(n_jobs=self.settings["n_jobs"])(
                delayed(fit_label)(features, cells[label].to_numpy(), self.settings, self.model_path(label))
                for label in todo)
            for label in todo:
                self.records.write(units[label])
        summary = pd.DataFrame([dict(label=label, **joblib.load(self.model_path(label))["summary"])
                                for label in LABELS], columns=SUMMARY_COLUMNS)
        summary.to_csv(self.out_dir / "fit_summary.csv", index=False)
        models = [joblib.load(self.model_path(label))["model"] for label in LABELS]
        model_digests = {label: sha256_file(self.model_path(label)) for label in LABELS}
        for corpus in self.config.get("cohort.corpora"):
            self.predict(corpus, models, model_digests)

    def train_set(self) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
        parts, cells, keys = [], [], []
        for corpus in self.config.get("retrieval.classifier_train_corpora"):
            builder = ExamFeatures(self.config, corpus)
            exams = builder.exam_images.exams()
            exams = exams[exams["split"] == "train"].reset_index(drop=True)
            parts.append(builder.build(exams))
            cells.append(exams[LABELS])
            keys += [f"{corpus}/{exam_id}" for exam_id in exams["exam_id"]]
        return np.concatenate(parts), pd.concat(cells, ignore_index=True), keys

    def test_set(self, corpus: str) -> tuple[pd.Series, np.ndarray]:
        builder = ExamFeatures(self.config, corpus)
        exams = builder.exam_images.exams()
        exams = exams[exams["split"] == "test"].reset_index(drop=True)
        features = builder.build(exams)
        return exams["exam_id"], features

    def probability_unit(self, corpus: str, exam_ids: pd.Series, features: np.ndarray,
                         model_digests: dict[str, str]) -> tuple[Unit, ProvenanceRecords]:
        corpus_dir = self.config.path("paths.retrieval_dir") / corpus
        records = ProvenanceRecords(corpus_dir, f"{self.step} probabilities", {"models": model_digests})
        return Unit("classifier_probabilities", exam_ids, row_digests(features),
                    [corpus_dir / "classifier_probabilities.npz"]), records

    def predict(self, corpus: str, models: list, model_digests: dict[str, str]) -> None:
        exam_ids, features = self.test_set(corpus)
        unit, records = self.probability_unit(corpus, exam_ids, features, model_digests)
        path = unit.outputs[0]
        if records.pending([unit], corpus):
            path.parent.mkdir(parents=True, exist_ok=True)
            probabilities = np.column_stack([model.predict_proba(features)[:, 1] for model in models])
            partial = path.with_name(path.stem + ".partial.npz")
            np.savez(partial, exam_ids=exam_ids.to_numpy().astype(str), labels=np.array(LABELS),
                     probabilities=probabilities)
            os.replace(partial, path)
            records.write(unit)
        with np.load(path) as data:
            probabilities = data["probabilities"]
