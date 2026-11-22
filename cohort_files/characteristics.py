import json

import numpy as np
import pandas as pd

from config import Config
from corpora import CORPORA
from lib.exam_encoder import LABELS

RESULT_COLUMNS = ["corpus", "split", "quantity", "value"]
SPLITS = ["train", "valid", "test"]


class Characteristics:

    def __init__(self, config: Config):
        self.config = config
        self.cohort_dir = config.path("paths.cohort_dir")
        self.out_dir = config.path("paths.cohort_files_dir")
        self.corpora = config.get("cohort.corpora")
        self.view_classifier = config.get("cohort.view_classifier")
        self.agreement_corpora = config.get("cohort.agreement_corpora")

    def run(self) -> None:
        evidence = self.read_evidence(self.out_dir / "cohort__evidence.csv")
        rows = []
        for corpus in self.corpora:
            exams = evidence[evidence["corpus"] == corpus]
            hashes = self.kept_image_hashes(corpus, exams)
            for split in [*SPLITS, "all"]:
                part = exams if split == "all" else exams[exams["split"] == split]
                if len(part):
                    rows += [(corpus, split, quantity, value)
                             for quantity, value in self.split_rows(exams, part, hashes, split)]
        rows.append((self.view_classifier["train_corpus"], self.view_classifier["eval_split"],
                     "view_classifier_accuracy", self.view_classifier_accuracy()))
        for corpus in self.agreement_corpora:
            rows += [(corpus, "all", f"chexbert_release_agreement_{label}", value)
                     for label, value in self.agreement(corpus, evidence[evidence["corpus"] == corpus]).items()]
        result = pd.DataFrame(rows, columns=RESULT_COLUMNS, dtype=object)
        path = self.out_dir / "cohort__result.csv"
        result.to_csv(path, index=False)

    @staticmethod
    def read_evidence(path) -> pd.DataFrame:
        evidence = pd.read_csv(path, dtype={"exam_id": str, "patient_id": str, "sex": str, "query_or_database": str},
                               keep_default_na=False, na_values={"age": [""], "n_entities": [""],
                                                                  **{label: [""] for label in LABELS}})
        for column in ["n_entities", *LABELS]:
            evidence[column] = evidence[column].astype("Int64")
        return evidence

    def kept_image_hashes(self, corpus: str, exams: pd.DataFrame) -> pd.DataFrame:
        corpus_dir = self.cohort_dir / corpus
        kept = pd.read_csv(corpus_dir / "kept_images.csv", dtype=str, keep_default_na=False,
                           usecols=["exam_id", "image_id"])
        kept = kept[kept["exam_id"].isin(exams["exam_id"])]
        hashes = pd.concat([pd.read_csv(path, dtype=str, keep_default_na=False, usecols=["image_id", "pixel_hash"])
                            for path in sorted((corpus_dir / "image_pass").glob("hashes_*.csv"))])
        kept = kept.merge(hashes, on="image_id", how="left", validate="one_to_one")
        if kept["pixel_hash"].isna().any() or (kept["pixel_hash"] == "").any():
            raise ValueError(f"{corpus}: kept images without a pixel hash")
        return kept

    def split_rows(self, exams: pd.DataFrame, part: pd.DataFrame, hashes: pd.DataFrame, split: str) -> list:
        splits_per_patient = exams.groupby("patient_id")["split"].nunique()
        part_hashes = hashes.loc[hashes["exam_id"].isin(part["exam_id"]), "pixel_hash"]
        rows = [
            ("n_exams", len(part)),
            ("n_patients", part["patient_id"].nunique()),
            ("n_images", int((part["n_frontal"] + part["n_lateral"]).sum())),
            ("n_with_lateral", int((part["n_lateral"] > 0).sum())),
            ("n_with_context_text", int(part["has_context_text"].sum())),
            ("n_with_report", int(part["has_report"].sum())),
            ("n_duplicate_image_hashes", int(len(part_hashes) - part_hashes.nunique())),
            ("n_patients_in_two_splits", int((splits_per_patient.loc[part["patient_id"].unique()] > 1).sum())),
        ]
        for label in LABELS:
            cells = part[label]
            labelled = cells.isin([0, 1]).sum()
            rows.append((f"prevalence_{label}", float((cells == 1).sum() / labelled) if labelled else np.nan))
        if split == "test":
            rows += [("n_queries", int((part["query_or_database"] == "query").sum())),
                     ("n_database", int((part["query_or_database"] == "database").sum()))]
        return rows

    def view_classifier_accuracy(self) -> float:
        with open(self.cohort_dir / "view_classifier" / "summary.json") as handle:
            return json.load(handle)["eval_accuracy"]

    def agreement(self, corpus: str, exams: pd.DataFrame) -> dict[str, float]:
        positive = self.agreement_positives(corpus, exams)
        values = {}
        for label in LABELS:
            n_shipped, n_both = int(positive[f"{label}_shipped"].sum()), int(positive[f"{label}_both"].sum())
            values[label] = float(n_both / n_shipped) if n_shipped else np.nan
        return values

    def agreement_positives(self, corpus: str, exams: pd.DataFrame) -> pd.DataFrame:
        shipped = CORPORA[corpus](self.config).shipped_labels()
        table = exams[["exam_id", *LABELS]].merge(shipped, on="exam_id", how="left", suffixes=("", "_shipped"),
                                                  validate="one_to_one", indicator=True)
        positive = pd.DataFrame({"exam_id": table["exam_id"]})
        for label in LABELS:
            positive[f"{label}_shipped"] = table[f"{label}_shipped"].eq(1).fillna(False).astype(bool)
            positive[f"{label}_both"] = positive[f"{label}_shipped"] & table[label].eq(1).fillna(False).astype(bool)
        return positive
