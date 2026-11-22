import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

from corpora.base import IMAGE_COLUMNS, REPORT_COLUMNS, Corpus
from image_reader import ImageReader


class CheXpertCorpus(Corpus):

    name = "chexpert"

    def __init__(self, config):
        super().__init__(config)
        self.metadata = pd.read_csv(self.setting("metadata"), dtype=str, keep_default_na=False)
        folders = self.metadata["path_to_image"].str.split("/")
        self.metadata["exam_id"] = folders.str[1] + "/" + folders.str[2]
        self.frontal_rank = {label: rank for rank, label in enumerate(config.get("cohort.frontal_order"))}

    def zip_members(self) -> dict[str, Path]:
        members = {}
        zip_files = sorted((path for folder in self.setting("zips_dirs") for path in Path(folder).glob("*.zip")),
                           key=lambda path: path.name)
        names = [path.name for path in zip_files]
        if len(set(names)) != len(names):
            raise ValueError("a release zip is in more than one of zips_dirs")
        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file) as archive:
                members.update({name: zip_file for name in archive.namelist() if name.endswith(".png")})
        return members

    def images(self) -> pd.DataFrame:
        table = self.metadata
        members = self.zip_members()
        member = self.setting("zip_member_prefix") + table["path_to_image"].str.replace(r"\.jpg$", ".png", regex=True)
        views = [self.view(side, projection) for side, projection in zip(table["frontal_lateral"], table["ap_pa"])]
        images = pd.DataFrame({
            "exam_id": table["exam_id"],
            "patient_id": table["deid_patient_id"],
            "image_id": table["path_to_image"].str.replace(r"\.jpg$", "", regex=True),
            "image_path": [ImageReader.zip_path(members[name], name) if name in members else "" for name in member],
            "view_class": [view_class for view_class, _ in views],
            "view_label": [view_label for _, view_label in views],
            "split": "",
            "age": pd.to_numeric(table["age"], errors="coerce"),
            "sex": table["sex"].map(self.setting("sex_map")).fillna(""),
        })
        images.loc[images["image_path"] == "", ["view_class", "view_label"]] = "unavailable"
        corrupt = images["image_id"].isin(self.setting("unreadable_images"))
        images.loc[corrupt, ["view_class", "view_label"]] = "unreadable"
        return self.check_columns(images, IMAGE_COLUMNS)

    def view(self, side: str, projection: str) -> tuple[str, str]:
        view_class = self.setting("side_map").get(side, "other")
        if view_class == "frontal":
            return "frontal", self.setting("frontal_projections").get(projection, "other")
        return view_class, view_class

    def lead_images(self) -> pd.DataFrame:
        table = self.metadata.copy()
        views = [self.view(side, projection) for side, projection in zip(table["frontal_lateral"], table["ap_pa"])]
        corrupt = table["path_to_image"].str.replace(r"\.jpg$", "", regex=True).isin(self.setting("unreadable_images"))
        table["not_frontal"] = [view_class != "frontal" for view_class, _ in views] | corrupt.to_numpy()
        table["rank"] = [self.frontal_rank.get(view_label, len(self.frontal_rank)) for _, view_label in views]
        table["file_name"] = table["path_to_image"].map(lambda path: PurePosixPath(path).name)
        return table.sort_values(["exam_id", "not_frontal", "rank", "file_name"]).groupby("exam_id").head(1)

    def reports(self) -> pd.DataFrame:
        lead = self.lead_images()
        reports = pd.DataFrame({
            "exam_id": lead["exam_id"],
            "label_text": self.first_or_joined(lead, self.setting("label_fields"), join=True),
            "context_text": self.first_or_joined(lead, self.setting("context_fields"), join=False),
        })
        return self.check_columns(reports.reset_index(drop=True), REPORT_COLUMNS)

    def shipped_labels(self) -> pd.DataFrame:
        shipped = pd.read_json(self.setting("shipped_labels"), lines=True, dtype=False)
        lead = self.lead_images()[["exam_id", "path_to_image"]]
        table = lead.merge(shipped, on="path_to_image", how="left", validate="one_to_one", indicator=True)
        if (table["_merge"] != "both").any():
            raise ValueError(f"{int((table['_merge'] != 'both').sum())} lead images have no shipped label row")
        return self.shipped_label_table(table)

    @staticmethod
    def first_or_joined(table: pd.DataFrame, fields: list[str], join: bool) -> pd.Series:
        parts = table[fields].apply(lambda column: column.str.strip())
        if join:
            return parts.apply(lambda row: " ".join(part for part in row if part), axis=1)
        return parts.apply(lambda row: next((part for part in row if part), ""), axis=1)
