import zipfile
from pathlib import PurePosixPath

import pandas as pd

from corpora.base import IMAGE_COLUMNS, REPORT_COLUMNS, Corpus
from corpora.report_sections import ReportSections


class MimicCorpus(Corpus):

    name = "mimic"

    def images(self) -> pd.DataFrame:
        metadata = pd.read_csv(self.setting("metadata"), dtype=str, keep_default_na=False)
        split = pd.read_csv(self.setting("split"), dtype=str, keep_default_na=False)
        table = metadata.merge(split[["dicom_id", "split"]], on="dicom_id", how="left", validate="one_to_one")
        if table["split"].eq("").any() or table["split"].isna().any():
            raise ValueError("some metadata images have no row in the split file")
        table["split"] = table["split"].map(self.setting("split_map"))
        if table["split"].isna().any():
            raise ValueError("split file holds a value outside split_map")
        view_map = self.setting("view_map")
        views = table["ViewPosition"].map(lambda value: view_map.get(value, ["other", "other"]))
        table["view_class"] = views.str[0]
        table["view_label"] = views.str[1]
        table["exam_id"] = table["study_id"]
        table["patient_id"] = table["subject_id"]
        table["image_id"] = table["dicom_id"]
        table["image_path"] = self.image_paths(table)
        table["age"] = float("nan")
        table["sex"] = ""
        return self.check_columns(table, IMAGE_COLUMNS)

    def image_paths(self, table: pd.DataFrame) -> pd.Series:
        images_dir = str(self.setting("images_dir"))
        return (images_dir + "/p" + table["subject_id"].str[:2] + "/p" + table["subject_id"]
                + "/s" + table["study_id"] + "/" + table["dicom_id"] + ".jpg")

    def shipped_labels(self) -> pd.DataFrame:
        shipped = pd.read_csv(self.setting("shipped_labels"), dtype=str, keep_default_na=False)
        if shipped["study_id"].duplicated().any():
            raise ValueError("more than one shipped label row for a study")
        return self.shipped_label_table(shipped.rename(columns={"study_id": "exam_id"}).replace("", pd.NA))

    def report_sections(self):
        with zipfile.ZipFile(self.setting("reports_zip")) as archive:
            for name in archive.namelist():
                if name.endswith(".txt"):
                    yield PurePosixPath(name).stem.removeprefix("s"), ReportSections(archive.read(name).decode("utf-8"))

    def reports(self) -> pd.DataFrame:
        label_headers = self.setting("label_sections")
        context_headers = self.setting("context_sections")
        rows = [{"exam_id": exam_id, "label_text": sections.joined(label_headers),
                 "context_text": sections.first(context_headers)} for exam_id, sections in self.report_sections()]
        return self.check_columns(pd.DataFrame(rows), REPORT_COLUMNS)

    def label_section_presence(self) -> pd.DataFrame:
        headers = self.setting("label_sections")
        return pd.DataFrame([{"exam_id": exam_id, **{header: bool(sections.joined([header])) for header in headers}}
                             for exam_id, sections in self.report_sections()])
