import re
from abc import ABC, abstractmethod

import pandas as pd

from config import Config
from lib.exam_encoder import LABELS

IMAGE_COLUMNS = ["exam_id", "patient_id", "image_id", "image_path", "view_class", "view_label",
                 "split", "age", "sex"]
REPORT_COLUMNS = ["exam_id", "label_text", "context_text"]
LABEL_COLUMNS = ["exam_id", *LABELS, "label_source"]
DICOM_AGE = re.compile(r"(\d+)([A-Z])")


class Corpus(ABC):

    name: str

    def __init__(self, config: Config):
        self.config = config

    def setting(self, key: str):
        return self.config.get(f"corpora.{self.name}.{key}")

    @abstractmethod
    def images(self) -> pd.DataFrame:
        pass

    def reports(self) -> pd.DataFrame | None:
        return None

    def labels(self) -> pd.DataFrame | None:
        return None

    def dicom_age_in_years(self, age: str | None) -> float:
        match = DICOM_AGE.fullmatch(str(age))
        if not match:
            return float("nan")
        return int(match.group(1)) / self.config.get("cohort.dicom_age_units_per_year")[match.group(2)]

    def shipped_labels(self) -> pd.DataFrame | None:
        return None

    def shipped_label_table(self, table: pd.DataFrame) -> pd.DataFrame:
        names = self.config.get("cohort.chexpert_label_names")
        labels = table[list(names)].apply(pd.to_numeric).astype("Int64").rename(columns=names)
        return pd.concat([table[["exam_id"]].reset_index(drop=True), labels.reset_index(drop=True)], axis=1)[
            ["exam_id", *LABELS]]

    @staticmethod
    def labels_from_findings(exam_ids: pd.Series, findings: pd.Series, label_map: dict[str, str]) -> pd.DataFrame:
        table = pd.DataFrame({"exam_id": exam_ids.to_numpy()})
        for label in LABELS:
            sources = {finding for finding, target in label_map.items() if target == label}
            if sources:
                table[label] = findings.map(lambda present: int(bool(sources & present))).to_numpy()
            else:
                table[label] = pd.NA
            table[label] = table[label].astype("Int64")
        return table

    @staticmethod
    def check_columns(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        missing = [column for column in columns if column not in table.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        return table[columns]
