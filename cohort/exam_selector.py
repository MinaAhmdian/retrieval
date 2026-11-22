from pathlib import PurePath

import pandas as pd

EXAM_COLUMNS = ["exam_id", "patient_id", "split", "n_frontal", "n_lateral", "view_list", "age", "sex"]


class ExamSelector:

    def __init__(self, max_frontal: int, max_lateral: int, frontal_order: list[str],
                 rank_aliases: dict[str, str] | None = None):
        self.max_frontal = max_frontal
        self.max_lateral = max_lateral
        self.frontal_rank = {label: rank for rank, label in enumerate(frontal_order)}
        self.frontal_rank.update({alias: self.frontal_rank[target] for alias, target in (rank_aliases or {}).items()})

    def kept_images(self, images: pd.DataFrame) -> pd.DataFrame:
        table = images.copy()
        table["file_name"] = table["image_path"].map(lambda path: PurePath(path).name)
        frontal = table[table["view_class"] == "frontal"].copy()
        unranked = set(frontal["view_label"]) - set(self.frontal_rank)
        if unranked:
            raise ValueError(f"frontal view labels without a rank: {sorted(unranked)}")
        frontal["rank"] = frontal["view_label"].map(self.frontal_rank)
        frontal = frontal.sort_values(["exam_id", "rank", "file_name"]).groupby("exam_id").head(self.max_frontal)
        lateral = table[table["view_class"] == "lateral"].sort_values(["exam_id", "file_name"])
        lateral = lateral[lateral["exam_id"].isin(frontal["exam_id"])].groupby("exam_id").head(self.max_lateral)
        kept = pd.concat([frontal, lateral.assign(rank=len(self.frontal_rank))])
        kept["is_lateral"] = kept["view_class"] == "lateral"
        kept = kept.sort_values(["exam_id", "is_lateral", "rank", "file_name"])
        kept["slot"] = kept.groupby("exam_id").cumcount()
        return kept.drop(columns=["rank", "is_lateral"]).reset_index(drop=True)

    def exams(self, kept: pd.DataFrame) -> pd.DataFrame:
        self.check_constant_per_exam(kept, ["patient_id", "split", "age", "sex"])
        grouped = kept.groupby("exam_id", sort=True)
        exams = grouped[["patient_id", "split", "age", "sex"]].first()
        exams["n_frontal"] = grouped["view_class"].apply(lambda views: int((views == "frontal").sum()))
        exams["n_lateral"] = grouped["view_class"].apply(lambda views: int((views == "lateral").sum()))
        exams["view_list"] = grouped["view_label"].agg(";".join)
        return exams.reset_index()[EXAM_COLUMNS]

    @staticmethod
    def check_constant_per_exam(kept: pd.DataFrame, columns: list[str]) -> None:
        for column in columns:
            n_values = kept.groupby("exam_id")[column].nunique(dropna=False)
            if (n_values > 1).any():
                raise ValueError(f"{int((n_values > 1).sum())} exams have more than one {column}")
