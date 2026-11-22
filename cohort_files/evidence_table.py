import pandas as pd

from cohort_files.duplicate_filter import DuplicateFilter
from config import Config
from lib.exam_encoder import LABELS
from lib.metrics import split_queries

EVIDENCE_COLUMNS = ["corpus", "exam_id", "patient_id", "split", "query_or_database", "n_frontal", "n_lateral",
                    "view_list", "age", "sex", "has_context_text", "has_report", "n_entities", *LABELS,
                    "label_source"]
CHEXBERT_CODES = {"1.0": 1, "0.0": 0, "-1.0": -1, "": 0}


class EvidenceTable:

    def __init__(self, config: Config):
        self.cohort_dir = config.path("paths.cohort_dir")
        self.out_dir = config.path("paths.cohort_files_dir")
        self.corpora = config.get("cohort.corpora")
        self.report_corpora = config.get("cohort.chexbert.corpora")
        self.seed = config.get("cohort.query_seed")
        self.duplicate_filter = DuplicateFilter(config)

    @property
    def path(self):
        return self.out_dir / "cohort__evidence.csv"

    def run(self) -> None:
        table = pd.concat([self.corpus_rows(corpus) for corpus in self.corpora], ignore_index=True)
        self.duplicate_filter.out_dir.mkdir(parents=True, exist_ok=True)
        table[EVIDENCE_COLUMNS].to_csv(self.duplicate_filter.evidence_before, index=False)
        table = self.duplicate_filter.apply(table)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        table[EVIDENCE_COLUMNS].to_csv(self.path, index=False)

    def corpus_rows(self, corpus: str) -> pd.DataFrame:
        corpus_dir = self.cohort_dir / corpus
        exams = pd.read_csv(corpus_dir / "exams.csv", dtype=str, keep_default_na=False)
        exams["age"] = pd.to_numeric(exams["age"])
        exams.insert(0, "corpus", corpus)
        if corpus in self.report_corpora:
            labels = self.chexbert_labels(corpus_dir)
            counts = pd.read_csv(corpus_dir / "radgraph_counts.csv", dtype={"exam_id": str})
        else:
            labels = pd.read_csv(corpus_dir / "release_labels.csv", dtype={"exam_id": str})
            counts = pd.DataFrame(columns=["exam_id", "n_entities"])
        table = exams.merge(labels, on="exam_id", how="left", validate="one_to_one", indicator=True)
        if (table["_merge"] != "both").any():
            raise ValueError(f"{corpus}: {int((table['_merge'] != 'both').sum())} kept exams have no label row")
        table[LABELS] = table[LABELS].astype("Int64")
        table["n_entities"] = self.test_entities(corpus, table, counts)
        table["query_or_database"] = self.query_roles(table)
        return table.drop(columns="_merge")

    @staticmethod
    def chexbert_labels(corpus_dir) -> pd.DataFrame:
        raw = pd.read_csv(corpus_dir / "chexbert_raw.csv", dtype=str, keep_default_na=False)
        unexpected = set(raw[LABELS].to_numpy().ravel()) - set(CHEXBERT_CODES)
        if unexpected:
            raise ValueError(f"CheXbert values outside {sorted(CHEXBERT_CODES)}: {sorted(unexpected)}")
        labels = raw[["exam_id"]].join(raw[LABELS].replace(CHEXBERT_CODES).astype("Int64"))
        labels["label_source"] = "chexbert"
        return labels

    @staticmethod
    def test_entities(corpus: str, table: pd.DataFrame, counts: pd.DataFrame) -> pd.Series:
        wanted = (table["split"] == "test") & (table["has_report"] == "1")
        values = table["exam_id"].map(counts.set_index("exam_id")["n_entities"]).where(wanted).astype("Int64")
        if values[wanted].isna().any():
            raise ValueError(f"{corpus}: {int(values[wanted].isna().sum())} test exams with a report lack n_entities")
        return values

    def query_roles(self, table: pd.DataFrame) -> pd.Series:
        test = table["split"] == "test"
        roles = pd.Series("", index=table.index)
        if test.any():
            labels = table.loc[test, LABELS].fillna(0).astype(int).to_numpy()
            is_query = split_queries(table.loc[test, "patient_id"].tolist(), labels, seed=self.seed)
            roles[test] = ["query" if query else "database" for query in is_query]
        return roles
