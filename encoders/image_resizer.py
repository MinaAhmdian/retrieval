import io
import os
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import Config
from encoders.exam_images import ExamImages
from image_reader import ImageReader
from provenance import FileDigests, ProvenanceRecords, Unit, chunked

GRAY_MODES = ("L", "I;16", "I;16B", "I;16L", "I", "F")


class ImageConverter:

    def __init__(self, size: int):
        self.size = size
        self.reader = ImageReader()

    @staticmethod
    def gray_values(image: Image.Image) -> np.ndarray:
        if image.mode not in GRAY_MODES:
            image = image.convert("L")
        return np.asarray(image).astype(np.float32)

    @staticmethod
    def equalize(values: np.ndarray) -> np.ndarray:
        distinct, inverse, counts = np.unique(values.ravel(), return_inverse=True, return_counts=True)
        cdf = np.cumsum(counts)
        if cdf[-1] == cdf[0]:
            return np.zeros(values.shape, dtype=np.uint8)
        levels = np.round((cdf - cdf[0]) * (255.0 / (cdf[-1] - cdf[0])))
        return levels[inverse].reshape(values.shape).astype(np.uint8)

    def png(self, image_path: str) -> bytes:
        values = self.gray_values(self.reader.open(image_path))
        resized = Image.fromarray(values, mode="F").resize((self.size, self.size), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        Image.fromarray(self.equalize(np.asarray(resized)), mode="L").save(buffer, format="PNG")
        return buffer.getvalue()


class ResizeFiles(Dataset):

    def __init__(self, paths: list[str], size: int):
        self.paths = paths
        self.converter = ImageConverter(size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[int, bytes]:
        return index, self.converter.png(self.paths[index])


class ImageResizer:

    step = "resize-images"

    def __init__(self, config: Config, corpus_name: str):
        self.corpus_name = corpus_name
        self.exam_images = ExamImages(config, corpus_name, before_filter=True)
        self.settings = config.get("retrieval.images")
        self.digests = FileDigests(config.get("provenance.digest_workers"))
        self.records = ProvenanceRecords(self.exam_images.resized_dir, self.step,
                                         self.step_settings(config, corpus_name))

    @staticmethod
    def step_settings(config: Config, corpus_name: str) -> dict:
        return {key: config.get(f"retrieval.images.{key}") for key in ("size", "shard_size")}

    def units(self, selection: pd.DataFrame | None = None) -> list[Unit]:
        selection = self.exam_images.selection() if selection is None else selection
        contents = [f"{path}|{digest}" for path, digest in
                    zip(selection["image_path"], self.digests.digests(selection["image_path"].tolist()))]
        out_dir = self.exam_images.resized_dir
        return [Unit(f"shard_{shard:05d}", part["image_id"], contents[part.index[0]:part.index[-1] + 1],
                     [out_dir / f"shard_{shard:05d}.zip", out_dir / f"shard_{shard:05d}.csv"])
                for shard, part in chunked(selection, self.settings["shard_size"])]

    @staticmethod
    def log_key(unit: Unit) -> str:
        return str(int(unit.name.split("_")[-1]) + 1)

    def run(self) -> None:
        selection = self.exam_images.selection()
        self.exam_images.resized_dir.mkdir(parents=True, exist_ok=True)
        units = self.units(selection)
        todo = self.records.pending(units, self.corpus_name)
        parts = dict(chunked(selection, self.settings["shard_size"]))
        for unit in todo:
            self.write_shard(unit, parts[int(self.log_key(unit)) - 1])
            self.records.write(unit)

    def write_shard(self, unit: Unit, part: pd.DataFrame) -> None:
        zip_path, csv_path = unit.outputs
        loader = DataLoader(ResizeFiles(part["image_path"].tolist(), self.settings["size"]),
                            batch_size=self.settings["pool_chunk"], num_workers=self.settings["workers"],
                            collate_fn=list)
        image_ids, members = part["image_id"].tolist(), []
        first_position = part.index[0]
        partial = zip_path.with_suffix(".zip.partial")
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as archive:
            for batch in loader:
                for index, data in batch:
                    member = f"{first_position + index:07d}.png"
                    archive.writestr(member, data)
                    members.append((image_ids[index], member))
        if len(members) != len(part):
            raise RuntimeError(f"{unit.name} wrote {len(members)} of {len(part)} images")
        os.replace(partial, zip_path)
        pd.DataFrame(members, columns=["image_id", "member"]).to_csv(csv_path, index=False)
