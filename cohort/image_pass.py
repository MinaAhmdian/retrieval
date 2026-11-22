import hashlib
import tempfile
import zipfile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import Config
from image_reader import ImageReader
from provenance import FileDigests, ProvenanceRecords, Unit, chunked

HASH_COLUMNS = ["image_id", "pixel_hash", "mode", "height", "width"]
UNREADABLE = "unreadable"


def pixel_hash(image: Image.Image) -> tuple[str, str, int, int, np.ndarray]:
    pixels = np.asarray(image)
    digest = hashlib.sha256(f"{image.mode}|{pixels.shape}|".encode() + pixels.tobytes()).hexdigest()
    return digest, image.mode, pixels.shape[0], pixels.shape[1], pixels


class ImageFiles(Dataset):

    def __init__(self, table: pd.DataFrame, transform):
        self.paths = table["image_path"].tolist()
        self.embed = table["embed"].tolist()
        self.transform = transform
        self.reader = ImageReader()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        try:
            image = self.reader.open(self.paths[index])
        except (OSError, zipfile.BadZipFile):
            return index, "", UNREADABLE, 0, 0, torch.zeros(0)
        digest, mode, height, width, pixels = pixel_hash(image)
        tensor = self.transform(self.grayscale(image, pixels)) if self.embed[index] else torch.zeros(0)
        return index, digest, mode, height, width, tensor

    @staticmethod
    def grayscale(image: Image.Image, pixels: np.ndarray) -> Image.Image:
        if image.mode not in ("I;16", "I;16B", "I;16L", "I", "F"):
            return image.convert("L")
        values = pixels.astype(np.float64)
        low, high = values.min(), values.max()
        scaled = (values - low) * (255.0 / (high - low)) if high > low else np.zeros_like(values)
        return Image.fromarray(np.round(scaled).astype(np.uint8), mode="L")


def collate(batch):
    indices, digests, modes, heights, widths, tensors = zip(*batch)
    to_embed = [i for i, tensor in enumerate(tensors) if tensor.numel()]
    stacked = torch.stack([tensors[i] for i in to_embed]) if to_embed else None
    return list(indices), list(digests), list(modes), list(heights), list(widths), to_embed, stacked


class ImagePass:

    step = "image-pass"

    def __init__(self, config: Config, corpus_name: str):
        self.config = config
        self.corpus_name = corpus_name
        corpus_dir = config.path("paths.cohort_dir") / corpus_name
        self.images_csv = corpus_dir / "all_images.csv"
        self.out_dir = corpus_dir / "image_pass"
        self.settings = config.get("cohort.image_pass")
        self.digests = FileDigests(config.get("provenance.digest_workers"))
        self.records = ProvenanceRecords(self.out_dir, self.step, self.step_settings(config, corpus_name))
        self.cached_table = None

    @staticmethod
    def step_settings(config: Config, corpus_name: str) -> dict:
        settings = config.get("cohort.image_pass")
        return {"hash_view_classes": settings["hash_view_classes"], "embed_rule": settings["embed_rules"].get(corpus_name),
                "resize": settings["resize"], "center_crop": settings["center_crop"],
                "weights_dir": config.get("tools.biovil_t.weights_dir"), "chunk_size": settings["chunk_size"]}

    def table(self) -> pd.DataFrame:
        if self.cached_table is None:
            images = pd.read_csv(self.images_csv, dtype=str, keep_default_na=False)
            images = images[images["view_class"].isin(self.settings["hash_view_classes"])].reset_index(drop=True)
            rule = self.settings["embed_rules"].get(self.corpus_name)
            images["embed"] = False
            if rule is not None:
                selected = images["view_class"].isin(rule["view_class"])
                if "split" in rule:
                    selected &= images["split"].isin(rule["split"])
                images["embed"] = selected
            self.cached_table = images
        return self.cached_table.copy()

    def units(self, images: pd.DataFrame | None = None) -> list[Unit]:
        images = self.table() if images is None else images
        contents = [f"{path}|{digest}|{embed}" for path, digest, embed in
                    zip(images["image_path"], self.digests.digests(images["image_path"].tolist()), images["embed"])]
        images = images.assign(content=contents)
        units = []
        for chunk, part in chunked(images, self.settings["chunk_size"]):
            outputs = [self.out_dir / f"hashes_{chunk:05d}.csv"]
            if part["embed"].any():
                outputs.append(self.out_dir / f"embeddings_{chunk:05d}.npz")
            units.append(Unit(f"hashes_{chunk:05d}", part["image_id"], part["content"], outputs))
        return units

    @staticmethod
    def log_key(unit: Unit) -> str:
        return str(int(unit.name.split("_")[-1]) + 1)

    def run(self) -> None:
        images = self.table()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        units = self.units(images)
        todo = self.records.pending(units, self.corpus_name)
        model, transform, device = self.model() if any(len(unit.outputs) > 1 for unit in todo) else (None, None, None)
        parts = dict(chunked(images, self.settings["chunk_size"]))
        for unit in todo:
            self.process_chunk(unit, parts[int(self.log_key(unit)) - 1], model, transform, device)
            self.records.write(unit)

    def model(self):
        from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
        from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
        device = torch.device(self.settings["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("config asks for cuda but no GPU is visible")
        tempfile.tempdir = str(self.config.path("tools.biovil_t.weights_dir"))
        model = get_biovil_t_image_encoder()
        tempfile.tempdir = None
        model = model.requires_grad_(False).eval().to(device)
        transform = create_chest_xray_transform_for_inference(resize=self.settings["resize"],
                                                              center_crop_size=self.settings["center_crop"])
        return model, transform, device

    def process_chunk(self, unit: Unit, part: pd.DataFrame, model, transform, device) -> None:
        loader = DataLoader(ImageFiles(part, transform), batch_size=self.settings["batch_size"],
                            num_workers=self.settings["num_workers"], collate_fn=collate)
        rows, embedded_ids, embeddings = [], [], []
        image_ids = part["image_id"].tolist()
        for indices, digests, modes, heights, widths, to_embed, stacked in loader:
            rows += [{"image_id": image_ids[i], "pixel_hash": d, "mode": m, "height": h, "width": w}
                     for i, d, m, h, w in zip(indices, digests, modes, heights, widths)]
            if stacked is not None:
                with torch.no_grad():
                    output = model(stacked.to(device))
                    embeddings.append(output.projected_global_embedding.float().cpu().numpy())
                embedded_ids += [image_ids[indices[i]] for i in to_embed]
        if embeddings:
            np.savez(unit.outputs[1], image_id=np.array(embedded_ids), embedding=np.concatenate(embeddings))
        pd.DataFrame(rows, columns=HASH_COLUMNS).to_csv(unit.outputs[0], index=False)
