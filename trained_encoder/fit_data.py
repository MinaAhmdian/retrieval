import hashlib

import numpy as np
import pandas as pd

from config import Config
from controls.permutations import LateralPermutations
from encoders.embedding_store import EmbeddingStore
from provenance import FileDigests
from trained_encoder.context_permutations import ContextPermutations
from trained_encoder.exam_inputs import SLOT_COLUMNS, VIEW_COLUMNS, ExamInputs
from trained_encoder.fit_exams import FitExams


def digest(row: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(row, dtype=np.float32).tobytes()).hexdigest()[:16]


class ContextEmbeddings:

    def __init__(self, config: Config, corpus_name: str):
        store = EmbeddingStore(config, corpus_name)
        encoder = config.get("trained_encoder.text_encoder")
        frames = []
        for kind in config.get("trained_encoder.context_files"):
            path = store.text_file(encoder, kind)
            if path.exists():
                ids, embeddings = store.load(path)
                frames.append(pd.DataFrame(embeddings, index=ids))
        table = pd.concat(frames) if frames else pd.DataFrame()
        repeated = table.index.duplicated(keep=False)
        if repeated.any():
            pairs = table[repeated].groupby(level=0).agg(lambda column: column.max() - column.min())
            if (pairs.to_numpy() > 1e-5).any():
                raise ValueError(f"{corpus_name}: an exam has different context embeddings in two files")
        self.table = table[~table.index.duplicated()]
        self.dim = config.get(f"retrieval.encoders.{encoder}.dim")

    def rows(self, exam_ids: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        mask = (exam_ids != "").to_numpy()
        out = np.zeros((len(exam_ids), self.dim), dtype=np.float32)
        named = exam_ids[mask]
        missing = ~named.isin(self.table.index)
        if missing.any():
            raise ValueError(f"{int(missing.sum())} exams have a context text but no context embedding")
        if mask.any():
            out[mask] = self.table.loc[named].to_numpy()
        return out, mask


def training_corpora(config: Config, configuration: str) -> list[str]:
    corpora = config.get("trained_encoder.train_corpora")
    chosen = config.get("controls.configuration_training").get(configuration, {}).get("train_corpora", corpora)
    if not chosen or not set(chosen) <= set(corpora) or len(set(chosen)) != len(chosen):
        raise ValueError(f"{configuration}: train_corpora must be distinct corpora of {corpora}")
    return [corpus for corpus in corpora if corpus in chosen]


class FitData:

    def __init__(self, config: Config, configuration: str, split: str):
        self.split = split
        self.configuration = configuration
        inputs = ExamInputs(config)
        operations = config.get(f"trained_encoder.input_configurations.{configuration}")
        permutations = ContextPermutations(config) if operations.get("context_from_donor") else None
        lateral_permutations = LateralPermutations(config) if operations.get("lateral_from_donor") else None
        encoder = config.get("trained_encoder.text_encoder")
        report_kind = config.get("trained_encoder.training_report_file")
        parts = []
        self.corpora = training_corpora(config, configuration)
        for corpus in self.corpora:
            fit_exams = FitExams(config, corpus)
            exams = fit_exams.eligible()
            exams = exams[exams["split"] == split].reset_index(drop=True)
            slots = fit_exams.slot_images.slots()
            table = inputs.base(exams, slots[slots["exam_id"].isin(exams["exam_id"])])
            donors = permutations.donors(corpus, split) if permutations else None
            lateral_donors = (lateral_permutations.donors_of(corpus, split, table) if lateral_permutations
                              else None)
            table = inputs.configured(table, configuration, donors=donors, lateral_donors=lateral_donors)
            store = EmbeddingStore(config, corpus)
            ids, reports = store.load(store.text_file(encoder, report_kind))
            reports = pd.DataFrame(reports, index=ids)
            if not exams["exam_id"].isin(reports.index).all():
                raise ValueError(f"{corpus} {split}: a fit exam has no training report embedding")
            context, context_mask = ContextEmbeddings(config, corpus).rows(table["context_exam"])
            resized = fit_exams.slot_images.resized().set_index("image_id")["png_path"]
            parts.append({"table": table.assign(corpus=corpus), "labels": inputs.labels(exams),
                          "reports": reports.loc[exams["exam_id"]].to_numpy(dtype=np.float32),
                          "report_keys": exams["label_text"].tolist(), "context": context,
                          "context_mask": context_mask, "png": resized})
        self.table = pd.concat([part["table"] for part in parts], ignore_index=True)
        self.labels = np.concatenate([part["labels"] for part in parts])
        self.reports = np.concatenate([part["reports"] for part in parts])
        self.report_keys = sum((part["report_keys"] for part in parts), [])
        self.context = np.concatenate([part["context"] for part in parts])
        self.context_mask = np.concatenate([part["context_mask"] for part in parts])
        png = pd.concat([part["png"] for part in parts])
        self.slot_paths = np.full((len(self.table), len(SLOT_COLUMNS)), "", dtype=object)
        for slot, column in enumerate(SLOT_COLUMNS):
            filled = (self.table[column] != "").to_numpy()
            if not self.table.loc[filled, column].isin(png.index).all():
                raise ValueError(f"{split}: a slot image has no resized PNG")
            self.slot_paths[filled, slot] = png.loc[self.table.loc[filled, column]].to_numpy()

    def __len__(self) -> int:
        return len(self.table)

    def ids(self) -> list[str]:
        return (self.table["corpus"] + ":" + self.table["exam_id"]).tolist()

    def contents(self, digests: FileDigests) -> list[str]:
        paths = [path for path in self.slot_paths.ravel() if path]
        checksums = dict(zip(paths, digests.digests(paths)))
        rows = []
        for i, row in enumerate(self.table.itertuples(index=False)):
            slots = ",".join(checksums.get(path, "") for path in self.slot_paths[i])
            views = ",".join(str(getattr(row, column)) for column in VIEW_COLUMNS)
            context = digest(self.context[i]) if self.context_mask[i] else ""
            key = hashlib.sha256(self.report_keys[i].encode()).hexdigest()[:16]
            rows.append(f"{slots}|{views}|{context}|{row.age_id},{row.sex_id}|"
                        f"{','.join(map(str, self.labels[i]))}|{digest(self.reports[i])}|{key}")
        return rows

    def counts(self) -> dict:
        return {"exams": len(self.table), "slots": [int((self.table[c] != "").sum()) for c in SLOT_COLUMNS],
                "context": int(self.context_mask.sum())}
