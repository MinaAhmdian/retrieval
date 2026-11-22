import pandas as pd

from config import Config
from trained_encoder.slot_images import SlotImages


class FitExams:

    def __init__(self, config: Config, corpus_name: str, before_filter: bool = False):
        self.corpus_name = corpus_name
        self.slot_images = SlotImages(config, corpus_name, before_filter=before_filter)
        self.texts_csv = config.path("paths.cohort_dir") / corpus_name / "report_texts.csv.gz"
        self.fit_splits = config.get("trained_encoder.fit_splits")
        self.is_training_corpus = corpus_name in config.get("trained_encoder.train_corpora")
        self.cached = None

    def candidates(self) -> pd.DataFrame:
        if self.cached is None:
            if not self.is_training_corpus:
                raise ValueError(f"{self.corpus_name} is not a training corpus")
            exams = self.slot_images.exams()
            exams = exams[exams["split"].isin(self.fit_splits) & (exams["has_report"] == "1")]
            texts = pd.read_csv(self.texts_csv, dtype=str, keep_default_na=False)
            joined = exams.merge(texts, on="exam_id", how="left", validate="one_to_one")
            if joined["label_text"].isna().any():
                raise ValueError(f"{self.corpus_name}: {int(joined['label_text'].isna().sum())} exams with "
                                 "has_report = 1 have no report text row")
            joined["eligible"] = joined["label_text"].str.strip() != ""
            self.cached = joined.reset_index(drop=True)
        return self.cached.copy()

    def eligible(self) -> pd.DataFrame:
        candidates = self.candidates()
        return candidates[candidates["eligible"]].drop(columns="eligible").reset_index(drop=True)

    def counts(self) -> pd.DataFrame:
        candidates = self.candidates()
        counts = candidates.groupby("split")["eligible"].agg(has_report="size", retained="sum")
        counts["excluded_empty_text"] = counts["has_report"] - counts["retained"]
        counts = counts.reindex(self.fit_splits, fill_value=0).astype(int)
        return counts[["has_report", "excluded_empty_text", "retained"]].rename_axis("split").reset_index()
