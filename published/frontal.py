import os
import zipfile

import numpy as np
import pandas as pd

from lib.exam_encoder import LABELS, N_SLOTS, SEX_IDS, VIEW_IDS
from config import Config
from corpora.mimic import MimicCorpus
from encoders.image_resizer import ImageConverter
from encoders.models import compute_device
from image_reader import ImageReader
from provenance import FileDigests, ProvenanceRecords, Unit, sha256_file
from published.population import published_dir
from trained_encoder.backbones import backbone_settings
from trained_encoder.exam_inputs import INPUT_COLUMNS, SLOT_COLUMNS, UNKNOWN_AGE, VIEW_COLUMNS, ExamInputs
from trained_encoder.slot_images import SlotImages
from trained_encoder.test_outputs import TestEmbedder
from trained_encoder.trainer import LIB_DIR

CORPUS = "mimic"


def population_images(config: Config) -> pd.DataFrame:
    from published.population import PublishedPopulation
    population = PublishedPopulation(config)
    if population.records.stale_reason(population.unit()) is not None:
        raise RuntimeError("run published-population first: its outputs are missing or out of date")
    images = pd.read_csv(population.outputs["images.csv"], dtype=str, keep_default_na=False)
    images = images[images["split"] == config.get("published.frontal.split")]
    return images.sort_values("image_id").reset_index(drop=True)


class FrontalImages:

    step = "published-frontal-images"

    def __init__(self, config: Config):
        self.config = config
        self.settings = config.get("published.frontal")
        self.folder = published_dir(config) / self.settings["folder"]
        self.outputs = {name: self.folder / name for name in ("images.csv", "converted.zip")}
        self.size = config.get("retrieval.images.size")
        self.digests = FileDigests(config.get("provenance.digest_workers"))
        self.records = ProvenanceRecords(self.folder, self.step, {"split": self.settings["split"], "size": self.size,
                                                                  "rule": "resize-images (ImageConverter)"})

    def plan(self) -> pd.DataFrame:
        images = population_images(self.config)
        slot_pngs = SlotImages(self.config, CORPUS).resized().set_index("image_id")["png_path"]
        plan = images[["image_id"]].copy()
        found = plan["image_id"].isin(slot_pngs.index)
        plan["source"] = np.where(found, "slot_image", "converted")
        copies = MimicCorpus(self.config).image_paths(images.rename(
            columns={"patient_id": "subject_id", "exam_id": "study_id", "image_id": "dicom_id"}))
        plan["read_path"] = np.where(found, plan["image_id"].map(slot_pngs).fillna(""), copies)
        return plan

    def unit(self, plan: pd.DataFrame) -> Unit:
        digests = self.digests.digests(plan["read_path"].tolist())
        contents = [f"{source}|{path}|{digest}" for source, path, digest in
                    zip(plan["source"], plan["read_path"], digests)]
        return Unit("frontal_images", plan["image_id"], contents, list(self.outputs.values()))

    def run(self) -> None:
        plan = self.plan()
        self.folder.mkdir(parents=True, exist_ok=True)
        unit = self.unit(plan)
        if self.records.pending([unit], CORPUS):
            converter = ImageConverter(self.size)
            partial = self.outputs["converted.zip"].with_suffix(".zip.partial")
            png = plan["read_path"].copy()
            with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as archive:
                for i in np.flatnonzero(plan["source"].to_numpy() == "converted"):
                    member = f"{plan['image_id'].iloc[i]}.png"
                    archive.writestr(member, converter.png(plan["read_path"].iloc[i]))
                    png.iloc[i] = ImageReader.zip_path(self.outputs["converted.zip"], member)
            os.replace(partial, self.outputs["converted.zip"])
            plan.assign(png_path=png)[["image_id", "source", "png_path"]].to_csv(self.outputs["images.csv"],
                                                                              index=False)
            self.records.write(unit)

    def index(self) -> pd.DataFrame:
        plan = self.plan()
        if self.records.stale_reason(self.unit(plan)) is not None:
            raise RuntimeError("run published-frontal-images first: its outputs are missing or out of date")
        return pd.read_csv(self.outputs["images.csv"], dtype=str, keep_default_na=False)


class SingleImages:

    def __init__(self, config: Config, images: pd.DataFrame, png: pd.Series, operations: dict):
        inputs = ExamInputs(config)
        sex_unknown = SEX_IDS[config.get("trained_encoder.sex_keys")[""]]
        table = pd.DataFrame({"exam_id": images["image_id"].to_numpy()})
        for slot in range(N_SLOTS):
            table[SLOT_COLUMNS[slot]] = images["image_id"].to_numpy() if slot == 0 else ""
            table[VIEW_COLUMNS[slot]] = (images["view"].map(lambda v: VIEW_IDS[inputs.view_keys[v]]).to_numpy()
                                         if slot == 0 else inputs.empty_view)
        table["context_exam"], table["age_id"], table["sex_id"] = "", UNKNOWN_AGE, sex_unknown
        self.table = inputs.apply(table[INPUT_COLUMNS], operations)
        if (self.table[SLOT_COLUMNS[1:]] != "").any().any() or (self.table["context_exam"] != "").any():
            raise ValueError("the operations leave an image beside the first frontal or a context text")
        self.slot_paths = np.full((len(table), N_SLOTS), "", dtype=object)
        self.slot_paths[:, 0] = png.loc[self.table[SLOT_COLUMNS[0]]].to_numpy()
        dim = config.get(f"retrieval.encoders.{config.get('trained_encoder.text_encoder')}.dim")
        self.context = np.zeros((len(table), dim), dtype=np.float32)
        self.context_mask = np.zeros(len(table), dtype=bool)
        self.labels = np.zeros((len(table), len(LABELS)), dtype=np.int64)
        self.reports, self.report_keys = None, None

    def __len__(self) -> int:
        return len(self.table)


class FrontalEmbedder:

    step = "embed-published-frontal"

    def __init__(self, config: Config, run: str):
        self.settings = config.get("published.encoder_frontal")
        if run not in self.settings["runs"]:
            raise ValueError(f"{run} is not a run of published.encoder_frontal.runs: {self.settings['runs']}")
        self.config, self.run_name = config, run
        self.embedder = TestEmbedder(config, run)
        self.folder = published_dir(config) / self.settings["folder"]
        self.output = self.folder / f"{run}.npz"

    def step_settings(self) -> dict:
        backbone, configuration, _ = self.run_name.split("__")
        return {"run": self.run_name, "chosen_sha256": sha256_file(self.embedder.chosen),
                "operations": self.settings["operations"],
                "backbone": backbone_settings(self.config, backbone, configuration),
                "encoder": self.config.get("trained_encoder.encoder"),
                "mixed_precision": self.config.get("trained_encoder.training.mixed_precision"),
                "lib_sha256": sha256_file(LIB_DIR / "exam_encoder.py")}

    def data(self) -> SingleImages:
        images = population_images(self.config)
        png = FrontalImages(self.config).index().set_index("image_id")["png_path"]
        return SingleImages(self.config, images, png, self.settings["operations"])

    def unit(self, data: SingleImages) -> Unit:
        digests = FileDigests(self.config.get("provenance.digest_workers")).digests(data.slot_paths[:, 0].tolist())
        contents = [f"{digest}|{view}" for digest, view in zip(digests, data.table[VIEW_COLUMNS[0]])]
        return Unit(self.run_name, data.table["exam_id"], contents, [self.output])

    def run(self) -> None:
        if not self.embedder.chosen.exists():
            raise FileNotFoundError(f"[{self.run_name}] no chosen weights")
        data = self.data()
        self.folder.mkdir(parents=True, exist_ok=True)
        unit = self.unit(data)
        records = ProvenanceRecords(self.folder, self.step, self.step_settings())
        if records.pending([unit], self.run_name):
            device = compute_device(self.embedder.settings["device"])
            outputs = self.embedder.embed_outputs(self.embedder.load_model(device), data, device)
            partial = self.output.with_name(self.output.stem + ".partial.npz")
            np.savez(partial, image_ids=data.table["exam_id"].to_numpy(dtype=str), alignment=outputs["alignment"],
                     logits=outputs["logits"])
            os.replace(partial, self.output)
            records.write(unit)
