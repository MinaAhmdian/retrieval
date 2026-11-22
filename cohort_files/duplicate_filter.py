import pandas as pd

from cohort_files.characteristics import Characteristics
from config import Config

IMAGE_COLUMNS = ["corpus", "exam_id", "patient_id", "image_id", "split", "query_or_database", "group", "pixel_hash"]
EXCLUDED_COLUMNS = ["corpus", "exam_id", "group", "image_id", "pixel_hash", "retained_corpus", "retained_exam_id",
                    "retained_group", "retained_image_id"]
CROSSING_COLUMNS = ["pixel_hash", "corpus", "exam_id", "patient_id", "image_id", "split", "query_or_database",
                    "exam_excluded"]


def exam_groups(evidence: pd.DataFrame) -> pd.Series:
    return evidence["split"].where(evidence["split"] != "test", "test_" + evidence["query_or_database"])


def exclusions(images: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    unknown = sorted(set(images["group"]) - set(order))
    if unknown:
        raise ValueError(f"exams outside the group order: {unknown}")
    retained = images.iloc[:0]
    matches = []
    for group in order:
        part = images[images["group"] == group]
        found = part.merge(retained, on="pixel_hash", suffixes=("", "_retained"))
        excluded = set(zip(found["corpus"], found["exam_id"]))
        matches.append(found)
        keep = [key not in excluded for key in zip(part["corpus"], part["exam_id"])]
        retained = pd.concat([retained, part[keep]], ignore_index=True)
    found = pd.concat(matches, ignore_index=True).rename(columns={
        "corpus_retained": "retained_corpus", "exam_id_retained": "retained_exam_id",
        "group_retained": "retained_group", "image_id_retained": "retained_image_id"})
    return found[EXCLUDED_COLUMNS].sort_values(EXCLUDED_COLUMNS).reset_index(drop=True)


def crossing_images(images: pd.DataFrame, excluded: pd.DataFrame) -> pd.DataFrame:
    groups = images.groupby("pixel_hash")["group"].nunique()
    rows = images[images["pixel_hash"].isin(groups.index[groups > 1])].copy()
    excluded_keys = set(zip(excluded["corpus"], excluded["exam_id"]))
    rows["exam_excluded"] = [int(key in excluded_keys) for key in zip(rows["corpus"], rows["exam_id"])]
    return rows[CROSSING_COLUMNS].sort_values(["pixel_hash", "corpus", "exam_id", "image_id"]).reset_index(drop=True)


def group_overlaps(images: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    hashes = {group: set(images.loc[images["group"] == group, "pixel_hash"]) for group in order}
    return pd.DataFrame([{"group_a": a, "group_b": b, "shared_hashes": len(hashes[a] & hashes[b])}
                         for i, a in enumerate(order) for b in order[i + 1:]])


class DuplicateFilter:

    def __init__(self, config: Config):
        self.config = config
        self.settings = config.get("cohort.duplicate_filter")
        self.order = self.settings["group_order"]
        self.out_dir = config.path("paths.cohort_dir") / self.settings["folder"]
        self.characteristics = Characteristics(config)

    @property
    def evidence_before(self):
        return self.out_dir / self.settings["evidence_before"]

    @property
    def excluded_csv(self):
        return self.out_dir / "excluded_exams.csv"

    @property
    def crossing_csv(self):
        return self.out_dir / "crossing_images.csv"

    def image_hashes(self, evidence: pd.DataFrame) -> pd.DataFrame:
        evidence = evidence.assign(group=exam_groups(evidence))
        parts = []
        for corpus, exams in evidence.groupby("corpus", sort=False):
            hashes = self.characteristics.kept_image_hashes(corpus, exams)
            parts.append(hashes.merge(exams[["exam_id", "corpus", "patient_id", "split", "query_or_database", "group"]],
                                      on="exam_id", how="left", validate="many_to_one"))
        return pd.concat(parts, ignore_index=True)[IMAGE_COLUMNS]

    def apply(self, evidence: pd.DataFrame) -> pd.DataFrame:
        images = self.image_hashes(evidence)
        excluded = exclusions(images, self.order)
        crossing = crossing_images(images, excluded)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        excluded.to_csv(self.excluded_csv, index=False)
        crossing.to_csv(self.crossing_csv, index=False)
        excluded_keys = set(zip(excluded["corpus"], excluded["exam_id"]))
        keep = [key not in excluded_keys for key in zip(evidence["corpus"], evidence["exam_id"])]
        return evidence[keep].reset_index(drop=True)
