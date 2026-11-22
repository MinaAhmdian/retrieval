import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import Config

ASSIGNED_COLUMNS = ["image_id", "view_class", "view_label", "p_lateral"]


class ViewClassifier:

    def __init__(self, config: Config):
        self.settings = config.get("cohort.view_classifier")
        self.cohort_dir = config.path("paths.cohort_dir")
        self.out_dir = self.cohort_dir / "view_classifier"

    def embeddings(self, corpus: str) -> pd.DataFrame:
        parts = [np.load(path) for path in sorted((self.cohort_dir / corpus / "image_pass").glob("embeddings_*.npz"))]
        if not parts:
            raise FileNotFoundError(f"no embeddings for {corpus}; run image-pass first")
        table = pd.DataFrame(np.concatenate([part["embedding"] for part in parts]),
                             index=np.concatenate([part["image_id"] for part in parts]))
        if table.index.duplicated().any():
            raise ValueError(f"{corpus} has more than one embedding for an image")
        return table

    def training_data(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        corpus = self.settings["train_corpus"]
        images = pd.read_csv(self.cohort_dir / corpus / "all_images.csv", dtype=str, keep_default_na=False)
        images = images[images["view_class"].isin(["frontal", "lateral"])].set_index("image_id")
        embeddings = self.embeddings(corpus)
        data = {}
        for split in (self.settings["train_split"], self.settings["eval_split"]):
            ids = images.index[images["split"] == split]
            missing = ids.difference(embeddings.index)
            if len(missing):
                raise ValueError(f"{len(missing)} {corpus} {split} images with a recorded view have no embedding")
            data[split] = (embeddings.loc[ids].to_numpy(), (images.loc[ids, "view_class"] == "lateral").to_numpy())
        return data

    def run(self) -> None:
        data = self.training_data()
        train_x, train_y = data[self.settings["train_split"]]
        eval_x, eval_y = data[self.settings["eval_split"]]
        model = LogisticRegression(C=self.settings["C"], max_iter=self.settings["max_iter"])
        model.fit(train_x, train_y)
        accuracy = float((model.predict(eval_x) == eval_y).mean())
        summary = {
            "train_corpus": self.settings["train_corpus"], "C": self.settings["C"],
            "n_train": int(len(train_y)), "n_train_lateral": int(train_y.sum()),
            "n_eval": int(len(eval_y)), "n_eval_lateral": int(eval_y.sum()),
            "n_iter": int(model.n_iter_[0]), "converged": bool(model.n_iter_[0] < self.settings["max_iter"]),
            "eval_accuracy": accuracy, "min_accuracy": self.settings["min_accuracy"], "assigned": {},
        }
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if accuracy < self.settings["min_accuracy"]:
            self.write_summary(summary)
            raise RuntimeError(f"valid accuracy {accuracy:.4f} is below {self.settings['min_accuracy']}; stop and report")
        for corpus in self.settings["apply_to"]:
            summary["assigned"][corpus] = self.assign(model, corpus)
        self.write_summary(summary)

    def assign(self, model: LogisticRegression, corpus: str) -> dict[str, int]:
        images = pd.read_csv(self.cohort_dir / corpus / "all_images.csv", dtype=str, keep_default_na=False)
        unknown = pd.Index(images.loc[images["view_class"] == "unknown", "image_id"])
        embeddings = self.embeddings(corpus)
        missing = unknown.difference(embeddings.index)
        if len(missing):
            raise ValueError(f"{len(missing)} {corpus} images without a recorded view have no embedding")
        p_lateral = model.predict_proba(embeddings.loc[unknown].to_numpy())[:, list(model.classes_).index(True)]
        is_lateral = p_lateral >= self.settings["lateral_threshold"]
        low, high = self.settings["uncertain_band"]
        labels = self.settings["assigned_labels"]
        assigned = pd.DataFrame({
            "image_id": unknown,
            "view_class": np.where(is_lateral, "lateral", "frontal"),
            "view_label": np.where(is_lateral, labels["lateral"], labels["frontal"]),
            "p_lateral": p_lateral,
        })
        assigned.to_csv(self.cohort_dir / corpus / "assigned_views.csv", index=False)
        return {"n_images": len(assigned), "n_frontal_assigned": int((~is_lateral).sum()),
                "n_lateral_assigned": int(is_lateral.sum()),
                f"n_uncertain_{low}_{high}": int(((p_lateral > low) & (p_lateral < high)).sum())}

    def write_summary(self, summary: dict) -> None:
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
