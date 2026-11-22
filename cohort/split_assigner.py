import pandas as pd

from lib.metrics import split_patients

SPLITS = {"train", "valid", "test"}


class SplitAssigner:

    def __init__(self, source: str, seed: int):
        if source not in ("release", "split_patients"):
            raise ValueError(f"unknown split source {source}")
        self.source = source
        self.seed = seed

    def assign(self, exams: pd.DataFrame) -> pd.DataFrame:
        exams = exams.copy()
        if self.source == "split_patients":
            if (exams["split"] != "").any():
                raise ValueError("the corpus has a release split; its config should say split_source: release")
            exams["split"] = split_patients(exams["patient_id"].tolist(), seed=self.seed)
        unexpected = set(exams["split"]) - SPLITS
        if unexpected:
            raise ValueError(f"split values outside {sorted(SPLITS)}: {sorted(unexpected)}")
        n_leaking = self.patients_in_two_splits(exams)
        if n_leaking:
            raise ValueError(f"{n_leaking} patients appear in more than one split")
        return exams

    @staticmethod
    def patients_in_two_splits(exams: pd.DataFrame) -> int:
        return int((exams.groupby("patient_id")["split"].nunique() > 1).sum())
