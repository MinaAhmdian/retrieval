import pandas as pd

from config import Config


class ReportSelection:

    def __init__(self, config: Config, corpus_name: str):
        self.corpus_dir = config.path("paths.cohort_dir") / corpus_name
        self.split_source = config.get(f"corpora.{corpus_name}.split_source")

    def all_reports(self) -> pd.DataFrame:
        path = self.corpus_dir / "report_texts.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run build-exams first")
        return pd.read_csv(path, dtype=str, keep_default_na=False)[["exam_id", "label_text"]]

    def test_reports(self) -> pd.DataFrame:
        reports = self.all_reports()
        if self.split_source == "release":
            images = pd.read_csv(self.corpus_dir / "all_images.csv", dtype=str, keep_default_na=False,
                                 usecols=["exam_id", "split"])
            test_exams = set(images.loc[images["split"] == "test", "exam_id"])
        else:
            exams = pd.read_csv(self.corpus_dir / "exams.csv", dtype=str, keep_default_na=False)
            test_exams = set(exams.loc[exams["split"] == "test", "exam_id"])
        if not test_exams:
            raise ValueError(f"no test exams in {self.corpus_dir}; build the final exam table first")
        return reports[reports["exam_id"].isin(test_exams)].reset_index(drop=True)
