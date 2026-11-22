import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import Config
from encoders.embedding_store import EmbeddingStore
from encoders.exam_images import ExamImages
from encoders.models import compute_device, load_encoder
from image_reader import ImageReader
from provenance import FileDigests, ProvenanceRecords, Unit, chunked


class ResizedImages(Dataset):

    def __init__(self, png_paths: list[str], transform):
        self.png_paths = png_paths
        self.transform = transform
        self.reader = ImageReader()

    def __len__(self) -> int:
        return len(self.png_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.transform(self.reader.open(self.png_paths[index]))


class ImageEmbedder:

    step = "embed-images"

    def __init__(self, config: Config, corpus_name: str):
        self.config = config
        self.corpus_name = corpus_name
        self.exam_images = ExamImages(config, corpus_name, before_filter=True)
        self.store = EmbeddingStore(config, corpus_name)
        self.settings = config.get("retrieval.embedding")
        self.classifier_encoder = config.get("retrieval.classifier_encoder")
        self.digests = FileDigests(config.get("provenance.digest_workers"))

    @staticmethod
    def step_settings(config: Config, encoder: str) -> dict:
        return {"encoder": config.get(f"retrieval.encoders.{encoder}"),
                "chunk_size": config.get("retrieval.embedding.chunk_size")}

    def records(self, encoder: str) -> ProvenanceRecords:
        return ProvenanceRecords(self.store.folder(encoder), self.step, self.step_settings(self.config, encoder))

    def images(self) -> pd.DataFrame:
        images = self.exam_images.selection().merge(self.exam_images.resized(), on="image_id", how="left")
        if images["png_path"].isna().any():
            raise ValueError(f"{int(images['png_path'].isna().sum())} images are not resized; run resize-images")
        return images.assign(png_digest=self.digests.digests(images["png_path"].tolist()))

    def images_for(self, encoder: str, images: pd.DataFrame) -> pd.DataFrame:
        part = images if encoder == self.classifier_encoder else images[images["split"] == "test"]
        return part.reset_index(drop=True)

    def units(self, encoder: str, images: pd.DataFrame | None = None) -> list[Unit]:
        part = self.images_for(encoder, self.images() if images is None else images)
        return [Unit(f"images_{chunk:05d}", rows["image_id"], rows["png_digest"],
                     [self.store.image_chunk(encoder, chunk)]) for chunk, rows in chunked(part, self.settings["chunk_size"])]


    def run(self) -> None:
        images = self.images()
        failed = []
        for name in self.config.get("retrieval.encoders"):
            part = self.images_for(name, images)
            records = self.records(name)
            records.folder.mkdir(parents=True, exist_ok=True)
            todo = records.pending(self.units(name, images), f"{self.corpus_name} {name}")
            if todo:
                device = compute_device(self.settings["device"])
                try:
                    encoder = load_encoder(self.config, name, device)
                except Exception:
                    failed.append(name)
                    continue
                chunks = dict(chunked(part, self.settings["chunk_size"]))
                for unit in todo:
                    self.embed_chunk(encoder, unit, chunks[int(unit.name.split("_")[-1])])
                    records.write(unit)
                del encoder
                torch.cuda.empty_cache()
        if failed:
            raise RuntimeError(f"encoders that did not load: {failed}")


    def embed_chunk(self, encoder, unit: Unit, part: pd.DataFrame) -> None:
        loader = DataLoader(ResizedImages(part["png_path"].tolist(), encoder.transform),
                            batch_size=self.settings["image_batch_size"], num_workers=self.settings["num_workers"])
        embeddings = np.concatenate([encoder.embed_images(batch) for batch in loader])
        if embeddings.shape != (len(part), encoder.dim):
            raise RuntimeError(f"{encoder.name} {unit.name}: shape {embeddings.shape}, expected "
                               f"({len(part)}, {encoder.dim})")
        self.store.save(unit.outputs[0], part["image_id"].tolist(), embeddings)
