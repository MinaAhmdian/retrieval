import pandas as pd

from config import Config
from trained_encoder.context_permutations import ContextPermutations
from trained_encoder.exam_inputs import ExamInputs
from trained_encoder.fit_exams import FitExams
from trained_encoder.slot_images import SlotImages


class LateralPermutations(ContextPermutations):

    step = "lateral-permutations"
    settings_key = "controls.lateral_permutation"
    root_key = "paths.controls_dir"
    fields_name = "lateral_slots"

    def __init__(self, config: Config):
        super().__init__(config)
        self.inputs = ExamInputs(config)
        self.fields = self.inputs.lateral_columns

    def lateral_rows(self, exams: pd.DataFrame, slot_images: SlotImages) -> pd.DataFrame:
        slots = slot_images.slots()
        table = self.inputs.base(exams, slots[slots["exam_id"].isin(exams["exam_id"])])
        return table[self.inputs.has_lateral(table).to_numpy()]

    def donors_of(self, corpus: str, split: str, table: pd.DataFrame) -> pd.Series:
        if not self.inputs.has_lateral(table).any():
            return pd.Series(dtype=str, name="donor_exam_id")
        return self.donors(corpus, split)

    def group_rows(self) -> dict[tuple[str, str], pd.DataFrame]:
        groups = {}
        for corpus in self.config.get("cohort.corpora"):
            slot_images = SlotImages(self.config, corpus)
            exams = slot_images.exams()
            groups[(corpus, "test")] = self.lateral_rows(exams[exams["split"] == "test"].reset_index(drop=True),
                                                         slot_images)
            fit_exams = FitExams(self.config, corpus)
            if fit_exams.is_training_corpus:
                eligible = fit_exams.eligible()
                for split in self.fit_splits:
                    part = eligible[eligible["split"] == split].reset_index(drop=True)
                    groups[(corpus, split)] = self.lateral_rows(part, fit_exams.slot_images)
        return groups


class ReportPermutations(ContextPermutations):

    step = "report-permutations"
    settings_key = "controls.report_permutation"
    root_key = "paths.controls_dir"
    fields_name = "moved_fields"

    def __init__(self, config: Config):
        super().__init__(config)
        self.fields = []

    def group_rows(self) -> dict[tuple[str, str], pd.DataFrame]:
        groups = {}
        for corpus in self.config.get("retrieval.documented_corpora"):
            exams = SlotImages(self.config, corpus).exams()
            groups[(corpus, "test")] = exams[(exams["split"] == "test")
                                             & (exams["query_or_database"] == "database")]
        return groups
