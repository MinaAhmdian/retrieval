import numpy as np
import pandas as pd

from lib.exam_encoder import LABELS, N_SLOTS, SEX_IDS, VIEW_IDS, age_bin
from config import Config

SLOT_COLUMNS = [f"image_{slot}" for slot in range(N_SLOTS)]
VIEW_COLUMNS = [f"view_{slot}" for slot in range(N_SLOTS)]
CONTEXT_COLUMNS = ["context_exam", "age_id", "sex_id"]
INPUT_COLUMNS = ["exam_id", *SLOT_COLUMNS, *VIEW_COLUMNS, *CONTEXT_COLUMNS]
UNKNOWN_AGE = age_bin(None)


class ExamInputs:

    def __init__(self, config: Config):
        settings = config.get("trained_encoder")
        self.view_keys = settings["view_keys"]
        self.sex_keys = settings["sex_keys"]
        self.empty_view = VIEW_IDS[settings["empty_slot_view"]]
        self.configurations = settings["input_configurations"]
        self.query_inputs = settings["query_inputs"]
        lateral = settings["slot_positions"]["lateral"]
        self.lateral_columns = [SLOT_COLUMNS[slot] for slot in lateral] + [VIEW_COLUMNS[slot] for slot in lateral]
        self.lateral_slot_columns = [SLOT_COLUMNS[slot] for slot in lateral]

    def base(self, exams: pd.DataFrame, slots: pd.DataFrame) -> pd.DataFrame:
        unknown_views = set(slots["view_label"]) - set(self.view_keys)
        if unknown_views:
            raise ValueError(f"view labels without a view key: {sorted(unknown_views)}")
        unknown_sex = set(exams["sex"]) - set(self.sex_keys)
        if unknown_sex:
            raise ValueError(f"sex values without a key: {sorted(unknown_sex)}")
        table = pd.DataFrame({"exam_id": exams["exam_id"].to_numpy()})
        placed = slots.assign(view_id=slots["view_label"].map(lambda label: VIEW_IDS[self.view_keys[label]]))
        images = placed.pivot(index="exam_id", columns="position", values="image_id")
        views = placed.pivot(index="exam_id", columns="position", values="view_id")
        for slot in range(N_SLOTS):
            image = images[slot] if slot in images else pd.Series(dtype=str)
            view = views[slot] if slot in views else pd.Series(dtype=float)
            table[SLOT_COLUMNS[slot]] = table["exam_id"].map(image).fillna("").to_numpy()
            table[VIEW_COLUMNS[slot]] = table["exam_id"].map(view).fillna(self.empty_view).astype(int).to_numpy()
        if (table[SLOT_COLUMNS[0]] == "").any():
            raise ValueError(f"{int((table[SLOT_COLUMNS[0]] == '').sum())} exams have no first frontal image")
        has_context = exams["has_context_text"].to_numpy() == "1"
        table["context_exam"] = np.where(has_context, exams["exam_id"].to_numpy(), "")
        ages = pd.to_numeric(exams["age"].replace("", None), errors="raise").to_numpy(dtype=float)
        table["age_id"] = [age_bin(None if np.isnan(age) else age) for age in ages]
        table["sex_id"] = [SEX_IDS[self.sex_keys[sex]] for sex in exams["sex"]]
        return table[INPUT_COLUMNS]

    @staticmethod
    def labels(exams: pd.DataFrame) -> np.ndarray:
        return exams[LABELS].replace("", "-1").astype(int).to_numpy()

    def has_lateral(self, table: pd.DataFrame) -> pd.Series:
        return (table[self.lateral_slot_columns] != "").any(axis=1)

    def apply(self, table: pd.DataFrame, operations: dict, donors: pd.Series | None = None,
              lateral_donors: pd.Series | None = None) -> pd.DataFrame:
        out = table.copy()
        if "copy_first_frontal_to" in operations:
            slot = operations["copy_first_frontal_to"]
            out[SLOT_COLUMNS[slot]] = out[SLOT_COLUMNS[0]]
            out[VIEW_COLUMNS[slot]] = out[VIEW_COLUMNS[0]]
        if operations.get("lateral_from_donor"):
            if lateral_donors is None:
                raise ValueError("lateral_from_donor needs the lateral permutation")
            in_group = out["exam_id"].isin(lateral_donors.index).to_numpy()
            has_lateral = self.has_lateral(out).to_numpy()
            if (has_lateral != in_group).any():
                raise ValueError(f"{int((has_lateral != in_group).sum())} exams differ between having a lateral image "
                                 "and being in the lateral permutation")
            donor_ids = out.loc[in_group, "exam_id"].map(lateral_donors)
            own = out.set_index("exam_id")[self.lateral_columns]
            if not donor_ids.isin(own.index).all():
                raise ValueError("a lateral donor is not among the exams of the table")
            for column in self.lateral_columns:
                out.loc[in_group, column] = own.loc[donor_ids, column].to_numpy()
        for slot in operations.get("empty_slots", []):
            out[SLOT_COLUMNS[slot]] = ""
            out[VIEW_COLUMNS[slot]] = self.empty_view
        if operations.get("context_from_donor"):
            if donors is None:
                raise ValueError("context_from_donor needs the context permutation")
            donor_ids = out["exam_id"].map(donors)
            if donor_ids.isna().any():
                raise ValueError(f"{int(donor_ids.isna().sum())} exams have no donor in the context permutation")
            own = table.set_index("exam_id")[CONTEXT_COLUMNS]
            if not donor_ids.isin(own.index).all():
                raise ValueError("a donor is not among the exams of the table")
            out[CONTEXT_COLUMNS] = own.loc[donor_ids].to_numpy()
            out[["age_id", "sex_id"]] = out[["age_id", "sex_id"]].astype(int)
        if operations.get("drop_text"):
            out["context_exam"] = ""
        if operations.get("unknown_age_sex"):
            out["age_id"] = UNKNOWN_AGE
            out["sex_id"] = SEX_IDS[self.sex_keys[""]]
        if operations.get("drop_context"):
            out["context_exam"] = ""
            out["age_id"] = UNKNOWN_AGE
            out["sex_id"] = SEX_IDS[self.sex_keys[""]]
        return out

    def configured(self, table: pd.DataFrame, configuration: str, query_inputs: str = "all",
                   donors: pd.Series | None = None, lateral_donors: pd.Series | None = None) -> pd.DataFrame:
        out = self.apply(table, self.configurations[configuration], donors, lateral_donors)
        return self.apply(out, self.query_inputs[query_inputs])
