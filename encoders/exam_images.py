import pandas as pd

from config import Config
from image_reader import ZIP_SEPARATOR, ImageReader

VIEWS = ["frontal", "lateral"]


class ExamImages:

    def __init__(self, config: Config, corpus_name: str, before_filter: bool = False):
        self.corpus_name = corpus_name
        if before_filter:
            self.cohort_csv = (config.path("paths.cohort_dir") / config.get("cohort.duplicate_filter.folder")
                               / config.get("cohort.duplicate_filter.evidence_before"))
        else:
            self.cohort_csv = config.path("paths.cohort_files_dir") / "cohort__evidence.csv"
        self.kept_images_csv = config.path("paths.cohort_dir") / corpus_name / "kept_images.csv"
        self.corpus_dir = config.path("paths.retrieval_dir") / corpus_name
        self.resized_dir = self.corpus_dir / "images_224"
        self.splits = ["test"]
        if corpus_name in config.get("retrieval.classifier_train_corpora"):
            self.splits.append("train")
        self.dry_run_exams = config.get("dry_run.exams_per_group")
        self.cached_exams = None

    def exams(self) -> pd.DataFrame:
        if self.cached_exams is None:
            cohort = pd.read_csv(self.cohort_csv, dtype=str, keep_default_na=False)
            exams = cohort[(cohort["corpus"] == self.corpus_name) & cohort["split"].isin(self.splits)]
            exams = exams.sort_values("exam_id")
            if self.dry_run_exams is not None:
                exams = exams.groupby(["split", "query_or_database"], sort=False).head(self.dry_run_exams)
            self.cached_exams = exams.reset_index(drop=True)
        return self.cached_exams.copy()

    def selection(self) -> pd.DataFrame:
        exams = self.exams()[["exam_id", "split"]]
        kept = pd.read_csv(self.kept_images_csv, dtype=str, keep_default_na=False,
                           usecols=["exam_id", "image_id", "image_path", "view_class", "slot"])
        kept = kept[kept["exam_id"].isin(exams["exam_id"]) & kept["view_class"].isin(VIEWS)]
        kept["slot"] = kept["slot"].astype(int)
        first = kept.loc[kept.groupby(["exam_id", "view_class"])["slot"].idxmin()]
        first = first.rename(columns={"view_class": "view"})[["exam_id", "view", "image_id", "image_path"]]
        selected = exams.merge(first, on="exam_id", how="left")
        n_frontal = int((selected["view"] == "frontal").sum())
        if n_frontal != len(exams):
            raise ValueError(f"{len(exams) - n_frontal} exams of {self.corpus_name} have no kept frontal image")
        selected["view_order"] = selected["view"].map({view: i for i, view in enumerate(VIEWS)})
        selected = selected.sort_values(["exam_id", "view_order"]).drop(columns="view_order")
        return selected.reset_index(drop=True)

    def shard_index(self, csv_path) -> pd.DataFrame:
        index = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        if "member" not in index:
            index["member"] = index.pop("png_path").str.split(ZIP_SEPARATOR, n=1).str[1]
        index["png_path"] = [ImageReader.zip_path(csv_path.with_suffix(".zip"), member) for member in index["member"]]
        return index[["image_id", "member", "png_path"]]

    def resized(self) -> pd.DataFrame:
        shards = sorted(self.resized_dir.glob("shard_*.csv"))
        if not shards:
            raise FileNotFoundError(f"no resized images for {self.corpus_name}; run resize-images first")
        return pd.concat([self.shard_index(path) for path in shards], ignore_index=True)
