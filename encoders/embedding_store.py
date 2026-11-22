import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config


class EmbeddingStore:

    def __init__(self, config: Config, corpus_name: str):
        self.root = config.path("paths.retrieval_dir") / corpus_name / "embeddings"

    def folder(self, encoder: str) -> Path:
        return self.root / encoder

    def image_chunk(self, encoder: str, chunk: int) -> Path:
        return self.folder(encoder) / f"images_{chunk:05d}.npz"

    def text_file(self, encoder: str, kind: str) -> Path:
        return self.folder(encoder) / f"{kind}.npz"

    @staticmethod
    def save(path: Path, ids: list[str], embeddings: np.ndarray) -> None:
        if len(ids) != len(embeddings):
            raise ValueError(f"{len(ids)} ids for {len(embeddings)} embeddings")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.stem + ".partial.npz")
        np.savez(partial, ids=np.array(ids, dtype=str), embeddings=embeddings.astype(np.float32))
        os.replace(partial, path)


    @staticmethod
    def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
        with np.load(path) as data:
            return data["ids"], data["embeddings"]

    def images(self, encoder: str) -> pd.DataFrame:
        parts = [self.load(path) for path in sorted(self.folder(encoder).glob("images_[0-9][0-9][0-9][0-9][0-9].npz"))]
        if not parts:
            raise FileNotFoundError(f"no image embeddings for {encoder} in {self.root}")
        return pd.DataFrame(np.concatenate([e for _, e in parts]), index=np.concatenate([i for i, _ in parts]))
