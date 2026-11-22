from pathlib import Path

import pandas as pd

from corpora.base import IMAGE_COLUMNS, LABEL_COLUMNS, Corpus
from corpora.vindr_headers import VinDrHeaders
from image_reader import ImageReader


class VinDrCorpus(Corpus):

    name = "vindr"

    def __init__(self, config):
        super().__init__(config)
        annotations = Path(self.setting("annotations_dir"))
        self.release = {split: pd.read_csv(annotations / f"image_labels_{split}.csv") for split in ("train", "test")}

    def images(self) -> pd.DataFrame:
        table = pd.DataFrame([(image_id, split) for split, labels in self.release.items()
                              for image_id in labels["image_id"].unique()], columns=["image_id", "split"])
        zip_file = self.setting("images_zip")
        table["image_path"] = [ImageReader.zip_path(zip_file, f"{self.setting('zip_member_prefix')}{split}/{image_id}.jpg")
                               for image_id, split in zip(table["image_id"], table["split"])]
        table["exam_id"] = table["image_id"]
        table["patient_id"] = table["image_id"]
        table["view_class"] = "frontal"
        table["view_label"] = self.setting("view_label")
        table = table.merge(self.release_headers(), on=["image_id", "split"], how="left", validate="one_to_one",
                            indicator=True)
        if (table["_merge"] != "both").any():
            raise ValueError(f"{int((table['_merge'] != 'both').sum())} release images have no fetched DICOM header")
        return self.check_columns(table, IMAGE_COLUMNS)

    def release_headers(self) -> pd.DataFrame:
        headers = VinDrHeaders(self.config).table()
        return pd.DataFrame({
            "image_id": headers["image_id"],
            "split": headers["split"],
            "age": headers["PatientAge"].map(self.dicom_age_in_years),
            "sex": headers["PatientSex"].where(headers["PatientSex"].isin(["F", "M"]), ""),
        })

    def labels(self) -> pd.DataFrame:
        findings = list(self.setting("label_map"))
        train = self.release["train"].groupby("image_id")[findings].sum() >= self.setting("train_min_votes")
        test = self.release["test"].set_index("image_id")[findings] == 1
        present = pd.concat([train, test])
        if present.index.duplicated().any():
            raise ValueError("an image is in both release label files")
        finding_sets = present.apply(lambda row: {finding for finding in findings if row[finding]}, axis=1)
        table = self.labels_from_findings(present.index.to_series(), finding_sets, self.setting("label_map"))
        table["label_source"] = "release"
        return self.check_columns(table, LABEL_COLUMNS)
