import pandas as pd

from corpora.base import IMAGE_COLUMNS, LABEL_COLUMNS, Corpus
from lib.exam_encoder import LABELS


class TaixCorpus(Corpus):

    name = "taix"

    def __init__(self, config):
        super().__init__(config)
        self.annotation = pd.read_csv(self.setting("annotation"), dtype=str, keep_default_na=False)

    def images(self) -> pd.DataFrame:
        table = self.annotation.copy()
        table["split"] = table["Split"].map(self.setting("split_map"))
        if table["split"].isna().any():
            raise ValueError("annotation.csv holds a Split value outside split_map")
        table["exam_id"] = table["UID"]
        table["patient_id"] = table["PatientID"]
        table["image_id"] = table["UID"]
        table["image_path"] = str(self.setting("images_dir")) + "/" + table["UID"] + ".png"
        table["view_class"] = "frontal"
        table["view_label"] = self.setting("view_label")
        table["age"] = table["Age"].astype(float) / self.setting("age_days_per_year")
        table["sex"] = table["Sex"].where(table["Sex"].isin(["F", "M"]), "")
        return self.check_columns(table, IMAGE_COLUMNS)

    def labels(self) -> pd.DataFrame:
        table = pd.DataFrame({"exam_id": self.annotation["UID"]})
        rules = self.setting("binarize")
        for label in LABELS:
            if label not in rules:
                table[label] = pd.Series(pd.NA, index=table.index, dtype="Int64")
                continue
            grades = self.annotation[rules[label]["columns"]].astype(int)
            table[label] = grades.isin(rules[label]["positive_grades"]).any(axis=1).astype("Int64")
        table["label_source"] = "release_binarized"
        return self.check_columns(table, LABEL_COLUMNS)
