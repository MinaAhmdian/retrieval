import numpy as np
import pandas as pd
import torch

from config import Config
from encoders.embedding_store import EmbeddingStore
from encoders.exam_images import ExamImages
from encoders.models import compute_device, load_encoder
from provenance import ProvenanceRecords, Unit


class TextEmbedder:

    step = "embed-texts"

    def __init__(self, config: Config, corpus_name: str):
        self.config = config
        self.corpus_name = corpus_name
        self.exam_images = ExamImages(config, corpus_name, before_filter=True)
        self.store = EmbeddingStore(config, corpus_name)
        self.settings = config.get("retrieval.embedding")
        self.texts_csv = config.path("paths.cohort_dir") / corpus_name / "report_texts.csv.gz"

    @staticmethod
    def step_settings(config: Config, encoder: str, max_tokens: int | None) -> dict:
        return {"encoder": config.get(f"retrieval.encoders.{encoder}"), "max_tokens": max_tokens}

    def jobs(self) -> list[tuple[str, str, pd.DataFrame, int | None]]:
        reports, context = self.texts()
        jobs = []
        if len(reports):
            jobs += [(name, "reports", reports, None) for name, settings in self.config.get("retrieval.encoders").items()
                     if "text_max_tokens" in settings]
        if len(context):
            jobs.append((self.config.get("retrieval.classifier_encoder"), "context", context,
                         self.config.get("retrieval.context_max_tokens")))
        return jobs

    def unit(self, encoder: str, kind: str, texts: pd.DataFrame, max_tokens: int | None) -> tuple[Unit, ProvenanceRecords]:
        records = ProvenanceRecords(self.store.folder(encoder), f"{self.step} {kind}",
                                    self.step_settings(self.config, encoder, max_tokens))
        return Unit(kind, texts["exam_id"], texts["text"], [self.store.text_file(encoder, kind)]), records


    def texts(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.corpus_name not in self.config.get("retrieval.documented_corpora"):
            return pd.DataFrame(columns=["exam_id", "text"]), pd.DataFrame(columns=["exam_id", "text"])
        exams = self.exam_images.exams()[["exam_id", "split", "has_report", "has_context_text"]]
        texts = pd.read_csv(self.texts_csv, dtype=str, keep_default_na=False)
        exams = exams.merge(texts, on="exam_id", how="left")
        reports = exams[(exams["split"] == "test") & (exams["has_report"] == "1")]
        if reports["label_text"].isna().any():
            raise ValueError(f"{int(reports['label_text'].isna().sum())} test exams have no report text row")
        context = exams[exams["has_context_text"] == "1"]
        if context["context_text"].fillna("").str.strip().eq("").any():
            raise ValueError("an exam with has_context_text = 1 has an empty context text")
        return (reports.rename(columns={"label_text": "text"})[["exam_id", "text"]],
                context.rename(columns={"context_text": "text"})[["exam_id", "text"]])

    def run(self) -> None:
        reports, context = self.texts()
        failed = []
        for name, kind, texts, max_tokens in self.jobs():
            unit, records = self.unit(name, kind, texts, max_tokens)
            path = unit.outputs[0]
            if records.pending([unit], f"{self.corpus_name} {name}"):
                device = compute_device(self.settings["device"])
                try:
                    encoder = load_encoder(self.config, name, device)
                except Exception:
                    failed.append(name)
                    continue
                self.store.save(path, texts["exam_id"].tolist(), self.embed(encoder, texts["text"].tolist(),
                                                                            max_tokens))
                records.write(unit)
                del encoder
                torch.cuda.empty_cache()
        if failed:
            raise RuntimeError(f"encoders that did not load: {failed}")

    def embed(self, encoder, texts: list[str], max_tokens: int | None) -> np.ndarray:
        order = np.argsort([len(text) for text in texts], kind="stable")
        size = self.settings["text_batch_size"]
        out = np.zeros((len(texts), encoder.dim), dtype=np.float32)
        for start in range(0, len(texts), size):
            batch = order[start:start + size]
            batch_texts = [texts[i] for i in batch]
            out[batch] = (encoder.embed_texts(batch_texts) if max_tokens is None
                          else encoder.embed_texts(batch_texts, max_tokens=max_tokens))
        return out
