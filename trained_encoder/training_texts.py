import pandas as pd

from config import Config
from encoders.models import compute_device, load_encoder
from encoders.text_embedder import TextEmbedder
from trained_encoder.fit_exams import FitExams

FIT_EXAMS = "fit_exams"


class TrainingTexts(TextEmbedder):

    step = "embed-training-texts"

    def __init__(self, config: Config, corpus_name: str):
        super().__init__(config, corpus_name)
        self.fit_exams = FitExams(config, corpus_name, before_filter=True)
        self.encoder_name = config.get("trained_encoder.text_encoder")
        self.text_sets = config.get("trained_encoder.texts")
        self.text_device = config.get("trained_encoder.text_device")

    def set_exams(self, kind: str, text_set: dict) -> pd.DataFrame:
        if text_set["exams"] == FIT_EXAMS:
            exams = self.fit_exams.eligible()
        else:
            exams = self.fit_exams.slot_images.exams()
            exams = exams[exams[text_set["exams"]] == "1"].merge(
                pd.read_csv(self.texts_csv, dtype=str, keep_default_na=False), on="exam_id", how="left",
                validate="one_to_one")
            if exams[text_set["field"]].isna().any():
                raise ValueError(f"{kind}: {int(exams[text_set['field']].isna().sum())} exams have no text row")
            if exams[text_set["field"]].str.strip().eq("").any():
                raise ValueError(f"{kind}: an exam with {text_set['exams']} = 1 has an empty text")
        exams = exams[exams["split"].isin(text_set["splits"])]
        return exams.rename(columns={text_set["field"]: "text"})[["exam_id", "text"]].reset_index(drop=True)

    def jobs(self) -> list[tuple[str, str, pd.DataFrame, int]]:
        if not self.fit_exams.is_training_corpus:
            return []
        return [(self.encoder_name, kind, self.set_exams(kind, text_set), self.config.get(text_set["max_tokens"]))
                for kind, text_set in self.text_sets.items()]

    def load_text_encoder(self):
        return load_encoder(self.config, self.encoder_name, compute_device(self.text_device))

    def run(self) -> None:
        jobs = self.jobs()
        if not jobs:
            return
        encoder = None
        for name, kind, texts, max_tokens in jobs:
            unit, records = self.unit(name, kind, texts, max_tokens)
            path = unit.outputs[0]
            if records.pending([unit], f"{self.corpus_name} {kind}"):
                encoder = self.load_text_encoder() if encoder is None else encoder
                self.store.save(path, texts["exam_id"].tolist(), self.embed(encoder, texts["text"].tolist(),
                                                                            max_tokens))
                records.write(unit)
