import os
from pathlib import Path

import numpy as np
import pandas as pd

from lib.exam_encoder import LABELS
from lib.metrics import split_queries
from config import Config
from cohort_files.characteristics import Characteristics
from encoders.embedding_store import EmbeddingStore
from encoders.retrieval import RetrievalInputs
from encoders.text_embedder import TextEmbedder
from labels.radgraph_annotator import RadGraphAnnotator
from provenance import ProvenanceRecords, Unit, sha256_file
from trained_encoder.fit_exams import FitExams
from trained_encoder.slot_images import SlotImages
from trained_encoder.test_outputs import TestEmbedder, TestExams
from trained_encoder.trainer import LIB_DIR

QUERY, DATABASE, REMOVED = "query", "database", "removed_query"
ASSIGNMENT_COLUMNS = ["exam_id", "role"]


def validation_dir(config: Config) -> Path:
    return config.path("paths.controls_dir") / config.get("controls.validation.folder")


def assign_roles(exams: pd.DataFrame, image_hashes: pd.DataFrame, seed: int, fraction: float) -> pd.Series:
    labels = (exams[LABELS] == "1").astype(int).to_numpy()
    is_query = split_queries(exams["patient_id"].tolist(), labels, seed=seed, fraction=fraction)
    database_ids = set(exams.loc[~is_query, "exam_id"])
    database_hashes = set(image_hashes.loc[image_hashes["exam_id"].isin(database_ids), "pixel_hash"])
    sharing = set(image_hashes.loc[image_hashes["pixel_hash"].isin(database_hashes), "exam_id"])
    removed = is_query & exams["exam_id"].isin(sharing).to_numpy()
    roles = np.where(~is_query, DATABASE, np.where(removed, REMOVED, QUERY))
    return pd.Series(roles, index=exams.index, name="role")


class ValidationAssignments:

    step = "validation-assignments"

    def __init__(self, config: Config):
        self.config = config
        self.settings = config.get("controls.validation")
        self.folder = validation_dir(config) / "assignments"
        self.records = ProvenanceRecords(self.folder, self.step, {
            "seed": self.settings["split_seed"], "fraction": self.settings["query_fraction"],
            "metrics_lib_sha256": sha256_file(LIB_DIR / "metrics.py")})

    def path(self, corpus: str) -> Path:
        return self.folder / f"{corpus}.csv"

    def valid_exams(self, corpus: str) -> pd.DataFrame:
        exams = FitExams(self.config, corpus).eligible()
        return exams[exams["split"] == "valid"].sort_values("exam_id").reset_index(drop=True)

    def roles(self, corpus: str) -> pd.Series:
        saved = pd.read_csv(self.path(corpus), dtype=str, keep_default_na=False)
        return saved.set_index("exam_id")["role"]

    def run(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        characteristics = Characteristics(self.config)
        for corpus in self.config.get("trained_encoder.train_corpora"):
            exams = self.valid_exams(corpus)
            hashes = characteristics.kept_image_hashes(corpus, exams)
            per_exam = hashes.sort_values(["exam_id", "pixel_hash"]).groupby("exam_id")["pixel_hash"].agg(",".join)
            contents = [f"{patient}|{','.join(cells)}|{per_exam.get(exam, '')}" for exam, patient, cells in
                        zip(exams["exam_id"], exams["patient_id"], exams[LABELS].to_numpy().tolist())]
            unit = Unit(corpus, exams["exam_id"].tolist(), contents, [self.path(corpus)])
            if self.records.pending([unit], f"{self.step} {corpus}"):
                roles = assign_roles(exams, hashes, self.settings["split_seed"], self.settings["query_fraction"])
                partial = unit.outputs[0].with_suffix(".partial")
                pd.DataFrame({"exam_id": exams["exam_id"], "role": roles})[ASSIGNMENT_COLUMNS].to_csv(partial,
                                                                                                     index=False)
                os.replace(partial, unit.outputs[0])
                self.records.write(unit)


class ValidExams(TestExams):

    split = "valid"

    def split_exams(self, slot_images: SlotImages) -> pd.DataFrame:
        exams = FitExams(self.config, self.corpus).eligible()
        return exams[exams["split"] == self.split].sort_values("exam_id").reset_index(drop=True)

    def side_mask(self, exams: pd.DataFrame, side: str) -> np.ndarray:
        roles = exams["exam_id"].map(ValidationAssignments(self.config).roles(self.corpus))
        if roles.isna().any():
            raise ValueError(f"{self.corpus}: {int(roles.isna().sum())} valid exams have no assignment; run "
                             "validation-assignments")
        return (roles == side).to_numpy()


class ValidEmbedder(TestEmbedder):

    step = "embed-valid-exams"

    def __init__(self, config: Config, run: str):
        settings = config.get("controls.validation")
        if run not in settings["weight_runs"]:
            raise ValueError(f"{run} is not in controls.validation.weight_runs: {settings['weight_runs']}")
        super().__init__(config, run)
        self.query_inputs = [settings["query_inputs"]]
        self.reuse = {}
        self.out_dir = validation_dir(config) / "outputs" / run

    def corpora(self) -> list[str]:
        return self.config.get("trained_encoder.train_corpora")

    def units_of(self, corpus: str) -> list[tuple[str, str]]:
        return [("query", name) for name in self.query_inputs]

    def exams(self, corpus: str, side: str, query_inputs: str) -> ValidExams:
        return ValidExams(self.config, corpus, self.configuration, query_inputs, side)


class ValidRadGraph(RadGraphAnnotator):

    step = "radgraph-valid"

    def __init__(self, config: Config, corpus_name: str):
        if corpus_name not in config.get("trained_encoder.train_corpora"):
            raise ValueError(f"{corpus_name} is not a training corpus")
        self.config = config
        self.corpus_name = corpus_name
        self.out_dir = validation_dir(config) / "radgraph" / corpus_name / "entities"
        self.settings = config.get("cohort.radgraph")
        self.records = ProvenanceRecords(self.out_dir, self.step, self.step_settings(config, corpus_name))

    def reports(self) -> pd.DataFrame:
        exams = FitExams(self.config, self.corpus_name).eligible()
        valid = exams[exams["split"] == "valid"].sort_values("exam_id")
        return valid[["exam_id", "label_text"]].reset_index(drop=True)


class ValidReportEmbedder(TextEmbedder):

    step = "embed-valid-reports"

    def __init__(self, config: Config, corpus_name: str):
        if corpus_name not in config.get("trained_encoder.train_corpora"):
            raise ValueError(f"{corpus_name} is not a training corpus")
        self.config = config
        self.corpus_name = corpus_name
        self.store = EmbeddingStore(config, corpus_name)
        self.settings = config.get("retrieval.embedding")
        self.encoder = config.get("trained_encoder.retrieval.documented_report_encoder")
        self.folder = validation_dir(config) / config.get("controls.validation.reports_folder") / corpus_name

    def path(self) -> Path:
        return self.folder / f"{self.encoder}.npz"

    def texts(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        exams = ValidationAssignments(self.config).valid_exams(self.corpus_name)
        roles = exams["exam_id"].map(ValidationAssignments(self.config).roles(self.corpus_name))
        if roles.isna().any():
            raise ValueError(f"{self.corpus_name}: {int(roles.isna().sum())} valid exams have no assignment")
        database = exams[(roles == DATABASE).to_numpy()]
        reports = database.rename(columns={"label_text": "text"})[["exam_id", "text"]].reset_index(drop=True)
        return reports, pd.DataFrame(columns=["exam_id", "text"])

    def jobs(self) -> list[tuple[str, str, pd.DataFrame, int | None]]:
        reports, _ = self.texts()
        return [(self.encoder, "reports", reports, None)]

    def unit(self, encoder: str, kind: str, texts: pd.DataFrame, max_tokens: int | None) -> tuple[Unit, ProvenanceRecords]:
        records = ProvenanceRecords(self.folder, self.step, self.step_settings(self.config, encoder, max_tokens))
        return Unit(kind, texts["exam_id"], texts["text"], [self.path()]), records


class ValidRetrievalInputs(RetrievalInputs):

    def __init__(self, config: Config, corpus_name: str):
        if corpus_name not in config.get("trained_encoder.train_corpora"):
            raise ValueError(f"{corpus_name} is not a training corpus")
        self.config = config
        self.corpus_name = corpus_name
        assignments = ValidationAssignments(config)
        exams = assignments.valid_exams(corpus_name)
        roles = exams["exam_id"].map(assignments.roles(corpus_name))
        if roles.isna().any():
            raise ValueError(f"{corpus_name}: {int(roles.isna().sum())} valid exams have no assignment")
        self.queries = exams[(roles == QUERY).to_numpy()].reset_index(drop=True)
        self.database = exams[(roles == DATABASE).to_numpy()].reset_index(drop=True)
        self.entities_dir = validation_dir(config) / "radgraph" / corpus_name / "entities"
        self.reports_path = ValidReportEmbedder(config, corpus_name).path()
        self.empty_code = config.get("retrieval.evaluation.empty_label_code")
        self.cache = {}

    def reports(self, encoder: str) -> np.ndarray:
        if encoder != self.config.get("trained_encoder.retrieval.documented_report_encoder"):
            raise ValueError(f"embed-valid-reports embeds with the documented report encoder only, not {encoder}")
        ids, embeddings = EmbeddingStore.load(self.reports_path)
        frame = pd.DataFrame(embeddings, index=ids)
        missing = ~self.database["exam_id"].isin(frame.index)
        if missing.any():
            raise ValueError(f"{self.corpus_name}: {int(missing.sum())} valid database exams have no report embedding")
        return frame.loc[self.database["exam_id"]].to_numpy()

    def images(self, encoder: str, side: str, two_view: bool) -> np.ndarray:
        raise NotImplementedError("the validation retrieval scores the trained encoder's outputs only")

    def probabilities(self, side: str) -> np.ndarray:
        raise NotImplementedError("the validation retrieval scores the trained encoder's outputs only")
