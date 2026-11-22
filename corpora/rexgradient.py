import json
from pathlib import Path, PurePosixPath

import pandas as pd

from corpora.base import IMAGE_COLUMNS, REPORT_COLUMNS, Corpus


class RexGradientCorpus(Corpus):

    name = "rexgradient"

    def __init__(self, config):
        super().__init__(config)
        self.records = self.read_records()

    def read_records(self) -> list[dict]:
        records = []
        for release_split, split in self.setting("split_files").items():
            path = Path(self.setting("meta_dir")) / f"{release_split}_metadata_view_position.json"
            with open(path) as handle:
                for study_key, record in json.load(handle).items():
                    records.append({**record, "study_key": study_key, "split": split})
        return records

    def images(self) -> pd.DataFrame:
        rows = []
        for record in self.records:
            images = zip(record["ImagePath"], record["ImageViewPosition"], record["ImageModality"], strict=True)
            for image_path, view, modality in images:
                view_class, view_label = self.view(view, modality)
                rows.append({
                    "exam_id": record["study_key"],
                    "patient_id": record["PatientID"],
                    "image_id": PurePosixPath(image_path).stem,
                    "image_path": str(Path(self.setting("images_dir")) / PurePosixPath(image_path).relative_to("..")),
                    "view_class": view_class,
                    "view_label": view_label,
                    "split": record["split"],
                    "age": self.dicom_age_in_years(record["PatientAge"]),
                    "sex": record["PatientSex"] if record["PatientSex"] in ("F", "M") else "",
                })
        return self.check_columns(pd.DataFrame(rows), IMAGE_COLUMNS)

    def view(self, view: str | None, modality: str) -> tuple[str, str]:
        if modality not in self.setting("kept_modalities"):
            return "other", "other"
        if view is None or view in self.setting("unknown_views"):
            return "unknown", "unknown"
        return tuple(self.setting("view_map").get(view, ["other", "other"]))

    def reports(self) -> pd.DataFrame:
        rows = [{
            "exam_id": record["study_key"],
            "label_text": " ".join(part for part in (record["Findings"], record["Impression"]) if part),
            "context_text": record["Indication"] or "",
        } for record in self.records]
        return self.check_columns(pd.DataFrame(rows), REPORT_COLUMNS)
