import os
from collections import Counter

import numpy as np
import pandas as pd

from config import Config
from provenance import ProvenanceRecords, Unit
from trained_encoder.fit_exams import FitExams
from trained_encoder.slot_images import SlotImages

MAPPING_COLUMNS = ["recipient_exam_id", "donor_exam_id"]
CONTEXT_FIELDS = ["has_context_text", "age", "sex"]


def derangement(n: int, seed: int) -> tuple[np.ndarray, int]:
    if n == 1:
        raise ValueError("a group of one exam has no donor other than itself")
    rng = np.random.default_rng(seed)
    draws = 0
    while True:
        draws += 1
        order = rng.permutation(n)
        if not (order == np.arange(n)).any():
            return order, draws


def check_mapping(mapping: pd.DataFrame, group: pd.DataFrame, fields_moved: list[str] = CONTEXT_FIELDS,
                  fields_name: str = "context_fields") -> dict:
    recipients, donors = mapping["recipient_exam_id"], mapping["donor_exam_id"]
    members = set(group["exam_id"])
    fields = group.set_index("exam_id")[fields_moved]
    own = Counter(map(tuple, fields.loc[recipients].to_numpy()))
    moved = Counter(map(tuple, fields.loc[donors].to_numpy())) if set(donors) <= members else Counter()
    return {"recipients_are_the_group": recipients.is_unique and set(recipients) == members,
            "donors_are_the_group": donors.is_unique and set(donors) == members,
            "no_self_assignment": bool((recipients.to_numpy() != donors.to_numpy()).all()),
            f"{fields_name}_preserved": own == moved}


class ContextPermutations:

    step = "context-permutations"
    settings_key = "trained_encoder.context_permutation"
    root_key = "paths.trained_dir"
    fields_name = "context_fields"

    def __init__(self, config: Config):
        self.config = config
        settings = config.get(self.settings_key)
        self.seed = settings["seed"]
        self.folder = config.path(self.root_key) / settings["folder"]
        self.records = ProvenanceRecords(self.folder, self.step, {"seed": self.seed})
        self.fit_splits = config.get("trained_encoder.fit_splits")
        self.fields = CONTEXT_FIELDS

    def groups(self) -> dict[tuple[str, str], pd.DataFrame]:
        return {key: group[["exam_id", *self.fields]].sort_values("exam_id").reset_index(drop=True)
                for key, group in self.group_rows().items()}

    def group_rows(self) -> dict[tuple[str, str], pd.DataFrame]:
        groups = {}
        for corpus in self.config.get("cohort.corpora"):
            exams = SlotImages(self.config, corpus).exams()
            groups[(corpus, "test")] = exams[exams["split"] == "test"]
            fit_exams = FitExams(self.config, corpus)
            if fit_exams.is_training_corpus:
                eligible = fit_exams.eligible()
                for split in self.fit_splits:
                    groups[(corpus, split)] = eligible[eligible["split"] == split]
        return groups

    def path(self, corpus: str, split: str):
        return self.folder / f"{corpus}__{split}.csv"

    def unit(self, corpus: str, split: str, group: pd.DataFrame) -> Unit:
        ids = group["exam_id"].tolist()
        return Unit(f"{corpus}__{split}", ids, ids, [self.path(corpus, split)])

    def donors(self, corpus: str, split: str) -> pd.Series:
        mapping = pd.read_csv(self.path(corpus, split), dtype=str, keep_default_na=False)
        return mapping.set_index("recipient_exam_id")["donor_exam_id"]

    def run(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        groups = self.groups()
        units = {key: self.unit(*key, group) for key, group in groups.items() if len(group)}
        todo = {unit.name for unit in self.records.pending(list(units.values()), self.step)}
        failed = []
        for (corpus, split), group in groups.items():
            if not len(group):
                continue
            unit = units[(corpus, split)]
            draws = "reused"
            if unit.name in todo:
                order, draws = derangement(len(group), self.seed)
                ids = group["exam_id"].to_numpy()
                mapping = pd.DataFrame({"recipient_exam_id": ids, "donor_exam_id": ids[order]})
                partial = unit.outputs[0].with_suffix(".partial")
                mapping.to_csv(partial, index=False)
                os.replace(partial, unit.outputs[0])
                self.records.write(unit)
            mapping = pd.read_csv(unit.outputs[0], dtype=str, keep_default_na=False)
            checks = check_mapping(mapping, group, self.fields, self.fields_name)
            failed += [f"{corpus} {split} {name}" for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(f"{self.step} checks failed: {failed}")
