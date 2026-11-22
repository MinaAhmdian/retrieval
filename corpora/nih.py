from pathlib import Path

import pandas as pd

from corpora.base import IMAGE_COLUMNS, LABEL_COLUMNS, Corpus


class NihCorpus(Corpus):

    name = "nih"

    def __init__(self, config):
        super().__init__(config)
        self.entries = pd.read_csv(self.setting("data_entry"), dtype=str, keep_default_na=False)
        self.entries["exam_id"] = self.entries["Image Index"].str.removesuffix(".png")

    def images(self) -> pd.DataFrame:
        table = self.entries.copy()
        test_images = set(Path(self.setting("test_list")).read_text().split())
        unknown = test_images - set(table["Image Index"])
        if unknown:
            raise ValueError(f"{len(unknown)} test_list images are not in the data entry table")
        table["split"] = table["Image Index"].isin(test_images).map({True: "test", False: "train"})
        view_map = self.setting("view_map")
        views = table["View Position"].map(lambda value: view_map.get(value, ["frontal", "other"]))
        table["view_class"] = views.str[0]
        table["view_label"] = views.str[1]
        table["patient_id"] = table["Patient ID"]
        table["image_id"] = table["exam_id"]
        table["image_path"] = str(self.setting("images_dir")) + "/" + table["Image Index"]
        table["age"] = table["Patient Age"].astype(float)
        table["sex"] = table["Patient Sex"].where(table["Patient Sex"].isin(["F", "M"]), "")
        return self.check_columns(table, IMAGE_COLUMNS)

    def labels(self) -> pd.DataFrame:
        findings = self.entries["Finding Labels"].map(lambda value: set(value.split("|")))
        table = self.labels_from_findings(self.entries["exam_id"], findings, self.setting("label_map"))
        table["label_source"] = "release"
        return self.check_columns(table, LABEL_COLUMNS)
