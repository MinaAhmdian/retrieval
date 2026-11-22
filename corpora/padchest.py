import ast
from pathlib import Path

import pandas as pd

from corpora.base import IMAGE_COLUMNS, LABEL_COLUMNS, Corpus


class PadChestCorpus(Corpus):

    name = "padchest"

    def __init__(self, config):
        super().__init__(config)
        self.release = pd.read_csv(self.setting("labels_csv"), dtype=str, keep_default_na=False)
        self.release["patient_id"] = self.merged_patient_ids(self.release)
        self.release["findings"] = self.release["Labels"].map(self.parse_labels)

    @staticmethod
    def merged_patient_ids(release: pd.DataFrame) -> pd.Series:
        parent = {patient: patient for patient in release["PatientID"]}

        def root(patient):
            while parent[patient] != patient:
                parent[patient] = parent[parent[patient]]
                patient = parent[patient]
            return patient

        for patients in release.groupby("StudyID")["PatientID"].unique():
            roots = sorted(root(patient) for patient in patients)
            for other in roots[1:]:
                parent[other] = roots[0]
        return release["PatientID"].map(root)

    @staticmethod
    def parse_labels(value: str) -> frozenset | None:
        if value == "nan":
            return None
        return frozenset(label.strip().lower() for label in ast.literal_eval(value))

    def images(self) -> pd.DataFrame:
        table = self.release.copy()
        images_dir = Path(self.setting("images_dir"))
        table["image_path"] = [str(images_dir / image_dir / image_id)
                               for image_dir, image_id in zip(table["ImageDir"], table["ImageID"])]
        view_map = self.setting("view_map")
        views = table["Projection"].map(lambda value: view_map.get(value, ["other", "other"]))
        table["view_class"] = views.str[0]
        table["view_label"] = views.str[1]
        on_disk = table["image_path"].map(lambda path: Path(path).is_file())
        table.loc[~on_disk, ["view_class", "view_label"]] = "unavailable"
        table["exam_id"] = table["StudyID"]
        table["image_id"] = table["ImageID"]
        table["split"] = ""
        birth_year = pd.to_numeric(table["PatientBirth"], errors="coerce")
        table["age"] = table["StudyDate_DICOM"].str[:4].astype(float) - birth_year
        table["sex"] = table["PatientSex_DICOM"].where(table["PatientSex_DICOM"].isin(["F", "M"]), "")
        return self.check_columns(table, IMAGE_COLUMNS)

    def labels(self) -> pd.DataFrame:
        if (self.release.groupby("StudyID")["findings"].nunique(dropna=False) > 1).any():
            raise ValueError("a study has more than one release label list")
        studies = self.release.drop_duplicates("StudyID")
        has_labels = studies["findings"].notna()
        findings = studies["findings"].map(lambda value: value if value is not None else frozenset())
        table = self.labels_from_findings(studies["StudyID"], findings, self.setting("label_map"))
        label_columns = table.columns.drop("exam_id")
        table.loc[~has_labels.to_numpy(), label_columns] = pd.NA
        table["label_source"] = "release"
        return self.check_columns(table, LABEL_COLUMNS)
