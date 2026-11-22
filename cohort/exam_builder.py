import pandas as pd

from cohort.exam_selector import ExamSelector
from cohort.split_assigner import SplitAssigner
from config import Config
from corpora import CORPORA
from image_reader import ImageReader


class ExamBuilder:

    def __init__(self, config: Config, corpus_name: str):
        self.corpus = CORPORA[corpus_name](config)
        self.selector = ExamSelector(config.get("cohort.max_frontal"), config.get("cohort.max_lateral"),
                                     config.get("cohort.frontal_order"), config.get("cohort.frontal_rank_aliases"))
        self.splitter = SplitAssigner(config.get(f"corpora.{corpus_name}.split_source"),
                                      config.get("cohort.split_seed"))
        self.out_dir = config.path("paths.cohort_dir") / corpus_name
        self.check_files = config.get("cohort.check_image_files")

    def run(self) -> None:
        images = self.corpus.images()
        kept = self.selector.kept_images(self.with_assigned_views(images))
        exams = self.splitter.assign(self.selector.exams(kept))
        kept["split"] = kept["exam_id"].map(exams.set_index("exam_id")["split"])
        reports = self.corpus.reports()
        exams = self.add_report_fields(exams, reports)
        labels = self.kept_labels(exams, self.corpus.labels())
        if self.check_files:
            self.check_image_files(kept)
        self.write(images, kept, exams, reports, labels)

    def with_assigned_views(self, images: pd.DataFrame) -> pd.DataFrame:
        unknown = images["view_class"] == "unknown"
        path = self.out_dir / "assigned_views.csv"
        if not unknown.any():
            return images
        if not path.exists():
            return images
        assigned = pd.read_csv(path, dtype=str, keep_default_na=False).set_index("image_id")
        ids = images.loc[unknown, "image_id"]
        if not ids.isin(assigned.index).all():
            raise ValueError(f"{int((~ids.isin(assigned.index)).sum())} unknown images have no assigned view")
        images = images.copy()
        images.loc[unknown, "view_class"] = ids.map(assigned["view_class"]).to_numpy()
        images.loc[unknown, "view_label"] = ids.map(assigned["view_label"]).to_numpy()
        return images

    @staticmethod
    def kept_labels(exams: pd.DataFrame, labels: pd.DataFrame | None) -> pd.DataFrame | None:
        if labels is None:
            return None
        if labels["exam_id"].duplicated().any():
            raise ValueError("more than one label row for an exam")
        missing = ~exams["exam_id"].isin(labels["exam_id"])
        if missing.any():
            raise ValueError(f"{int(missing.sum())} kept exams have no label row")
        return labels[labels["exam_id"].isin(exams["exam_id"])].reset_index(drop=True)

    @staticmethod
    def add_report_fields(exams: pd.DataFrame, reports: pd.DataFrame | None) -> pd.DataFrame:
        if reports is None:
            return exams.assign(has_report=0, has_context_text=0)
        if reports["exam_id"].duplicated().any():
            raise ValueError("more than one report for an exam")
        with_report = set(reports["exam_id"])
        with_context = set(reports.loc[reports["context_text"] != "", "exam_id"])
        return exams.assign(has_report=exams["exam_id"].isin(with_report).astype(int),
                            has_context_text=exams["exam_id"].isin(with_context).astype(int))

    @staticmethod
    def check_image_files(kept: pd.DataFrame) -> None:
        reader = ImageReader()
        missing = sum(not reader.exists(path) for path in kept["image_path"])
        if missing:
            raise FileNotFoundError(f"{missing} kept images have no file")

    def write(self, images: pd.DataFrame, kept: pd.DataFrame, exams: pd.DataFrame,
              reports: pd.DataFrame | None, labels: pd.DataFrame | None) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        images.to_csv(self.out_dir / "all_images.csv", index=False)
        kept.to_csv(self.out_dir / "kept_images.csv", index=False)
        exams.to_csv(self.out_dir / "exams.csv", index=False)
        if reports is not None:
            reports.to_csv(self.out_dir / "report_texts.csv.gz", index=False)
        if labels is not None:
            labels.to_csv(self.out_dir / "release_labels.csv", index=False)
