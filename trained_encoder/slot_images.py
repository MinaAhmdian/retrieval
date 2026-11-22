import pandas as pd

from config import Config
from encoders.exam_images import ExamImages


class SlotImages:

    def __init__(self, config: Config, corpus_name: str, before_filter: bool = False):
        settings = config.get("trained_encoder")
        self.corpus_name = corpus_name
        self.first_images = ExamImages(config, corpus_name, before_filter=before_filter)
        self.extra_dir = self.first_images.corpus_dir / settings["slot_images"]["folder"]
        self.splits = ["test"] + (settings["fit_splits"] if corpus_name in settings["train_corpora"] else [])
        self.slot_positions = settings["slot_positions"]
        self.dry_run_exams = config.get("dry_run.exams_per_group")
        self.cached = {}

    def exams(self) -> pd.DataFrame:
        if "exams" not in self.cached:
            cohort = pd.read_csv(self.first_images.cohort_csv, dtype=str, keep_default_na=False)
            exams = cohort[(cohort["corpus"] == self.corpus_name) & cohort["split"].isin(self.splits)]
            exams = exams.sort_values("exam_id")
            if self.dry_run_exams is not None:
                exams = exams.groupby(["split", "query_or_database"], sort=False).head(self.dry_run_exams)
            self.cached["exams"] = exams.reset_index(drop=True)
        return self.cached["exams"].copy()

    def slots(self) -> pd.DataFrame:
        if "slots" not in self.cached:
            exams = self.exams()[["exam_id", "split"]]
            kept = pd.read_csv(self.first_images.kept_images_csv, dtype=str, keep_default_na=False,
                               usecols=["exam_id", "image_id", "image_path", "view_class", "view_label", "slot"])
            kept = kept[kept["exam_id"].isin(exams["exam_id"]) & kept["view_class"].isin(self.slot_positions)]
            kept = kept.assign(slot=kept["slot"].astype(int)).sort_values(["exam_id", "slot"])
            rank = kept.groupby(["exam_id", "view_class"]).cumcount()
            capacity = kept["view_class"].map({view: len(positions) for view, positions in self.slot_positions.items()})
            if (rank >= capacity).any():
                raise ValueError(f"{int((rank >= capacity).sum())} images of {self.corpus_name} exceed the slots of "
                                 "their view")
            kept["position"] = [self.slot_positions[view][r] for view, r in zip(kept["view_class"], rank)]
            slots = exams.merge(kept.rename(columns={"view_class": "view"}), on="exam_id", how="inner")
            columns = ["exam_id", "split", "view", "view_label", "image_id", "image_path", "position"]
            self.cached["slots"] = slots.sort_values(["exam_id", "position"])[columns].reset_index(drop=True)
        return self.cached["slots"].copy()

    def extra(self) -> pd.DataFrame:
        if "extra" not in self.cached:
            slots = self.slots()
            converted = self.first_images.selection()["image_id"]
            self.cached["extra"] = slots[~slots["image_id"].isin(converted)].reset_index(drop=True)
        return self.cached["extra"].copy()

    def resized(self) -> pd.DataFrame:
        parts = [self.first_images.resized()]
        parts += [self.first_images.shard_index(path) for path in sorted(self.extra_dir.glob("shard_*.csv"))]
        index = pd.concat(parts, ignore_index=True)
        if index["image_id"].duplicated().any():
            raise ValueError(f"{int(index['image_id'].duplicated().sum())} images of {self.corpus_name} are resized "
                             "twice")
        return index


class ExtraImages(ExamImages):

    def __init__(self, config: Config, corpus_name: str):
        super().__init__(config, corpus_name, before_filter=True)
        self.slot_images = SlotImages(config, corpus_name, before_filter=True)
        self.resized_dir = self.slot_images.extra_dir

    def selection(self) -> pd.DataFrame:
        return self.slot_images.extra()
