from pathlib import Path, PurePosixPath

import pandas as pd

from corpora.base import IMAGE_COLUMNS, LABEL_COLUMNS, Corpus
from lib.exam_encoder import LABELS


class BraxCorpus(Corpus):

    name = "brax"

    def __init__(self, config):
        super().__init__(config)
        self.release = pd.read_csv(self.setting("spreadsheet"), dtype=str, keep_default_na=False)
        self.release["exam_id"] = self.release["PngPath"].map(lambda path: PurePosixPath(path).parts[2])

    def images(self) -> pd.DataFrame:
        table = self.release
        images_dir = Path(self.setting("images_dir"))
        view_map = self.setting("view_map")
        views = table["ViewPosition"].map(lambda value: view_map.get(value, ["other", "other"]))
        on_disk = table["PngPath"].map(lambda path: (images_dir / path).is_file())
        images = pd.DataFrame({
            "exam_id": table["exam_id"],
            "patient_id": table["PatientID"],
            "image_id": table["PngPath"].str.replace(r"\.png$", "", regex=True),
            "image_path": [str(images_dir / path) for path in table["PngPath"]],
            "view_class": views.str[0].where(on_disk, "unavailable"),
            "view_label": views.str[1].where(on_disk, "unavailable"),
            "split": "",
            "age": table["PatientAge"].map(self.setting("age_map")).fillna(table["PatientAge"]).astype(float),
            "sex": table["PatientSex"].where(table["PatientSex"].isin(["F", "M"]), ""),
        })
        return self.check_columns(images, IMAGE_COLUMNS)

    def labels(self) -> pd.DataFrame:
        columns = self.config.get("cohort.chexpert_label_names")
        if sorted(columns.values()) != sorted(LABELS):
            raise ValueError("chexpert_label_names must map onto the 14 labels")
        studies = self.release.groupby("exam_id")[list(columns)]
        if (studies.nunique() > 1).any().any():
            raise ValueError("a study has more than one value of a release label")
        table = studies.first().replace("", "0").astype(float).astype("Int64").rename(columns=columns)
        table = table.reset_index()
        table["label_source"] = "release"
        return self.check_columns(table, LABEL_COLUMNS)
