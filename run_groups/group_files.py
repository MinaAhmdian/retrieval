import pandas as pd

from lib.exam_encoder import LABELS
from config import Config
from controls.controls_files import ControlsFiles, ResultRows
from run_groups.group_retrieval import TransferRetrieval, group_dir
from trained_encoder.trained_retrieval import RULES


class GroupResultRows(ResultRows):

    def __init__(self, config: Config, group: str):
        super().__init__(config)
        self.group_runs = config.get(f"{group}.runs")
        self.single_seed = {c for cs in self.group_runs["configurations"].values() for c in cs}

    def ordered(self, result: pd.DataFrame) -> pd.DataFrame:
        evaluation = self.config.get("retrieval.evaluation")
        precision = sorted(m for m in result["metric"].unique() if m.startswith("p") and "_f1_" in m)
        order = {"split": ["test"],
                 "configuration": [c for cs in self.group_runs["configurations"].values() for c in cs],
                 "backbone": list(self.group_runs["configurations"]), "seeds": list(map(str, self.group_runs["seeds"])),
                 "test_corpus": self.config.get("cohort.corpora"), "archive": ["documented", "undocumented"],
                 "rule": RULES, "k": sorted(set(result["k"])),
                 "query_inputs": list(self.config.get("trained_encoder.query_inputs")), "level": ["finding", "report"],
                 "finding": [*LABELS, self.retrieval_files["macro_name"], self.retrieval_files["core_name"], "pooled"],
                 "metric": ["ap", self.retrieval_files["macro_metric"], *[f"hit{k}" for k in evaluation["hit_ks"]],
                            *[f"ndcg{k}" for k in evaluation["ndcg_ks"]], *precision]}
        keys = pd.DataFrame({column: pd.Categorical(result[column], categories=list(dict.fromkeys(values)),
                                                    ordered=True) for column, values in order.items()})
        if keys.isna().any().any():
            raise ValueError("a result row has a value outside the expected categories")
        return result.loc[keys.sort_values(list(order)).index].reset_index(drop=True)


class GroupFiles(ControlsFiles):

    group = ""
    retrieval_job = None

    def __init__(self, config: Config):
        self.config = config
        root = group_dir(config, self.group)
        self.files_dir = root / config.get(f"{self.group}.files.folder")
        self.units_csv = root / "evidence_units.csv"
        self.left_out_csv = root / "combinations_left_out.csv"
        self.omitted_csv = root / "omitted_summaries.csv"
        self.replicates_csv = root / "bootstrap_replicates.csv"

    @property
    def evidence_path(self):
        return self.files_dir / f"{self.group}__evidence.csv.gz"

    @property
    def result_path(self):
        return self.files_dir / f"{self.group}__result.csv"

    def jobs(self) -> list:
        return [self.retrieval_job(self.config, corpus)
                for corpus in self.config.get(f"{self.group}.evaluation.test_corpora")]

    def result_rows(self) -> ResultRows:
        return GroupResultRows(self.config, self.group)


class TransferFiles(GroupFiles):
    group = "transfer"
    retrieval_job = TransferRetrieval
