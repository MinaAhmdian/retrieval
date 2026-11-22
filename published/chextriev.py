import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from provenance import ProvenanceRecords, Unit, sha256_file, sha256_json
from published.crops import PublishedCrops
from published.population import PublishedPopulation, published_dir
from published.source_lists import environment_versions

REPO_DIR = Path(__file__).resolve().parents[1]
SPLITS = {"train": "train", "valid": "validate", "test": "test"}
RESUME_NAME = "resume"
VERSION_MODULES = ["torch", "torchvision", "pytorch_lightning", "torchmetrics", "faiss", "einops",
                   "rotary_embedding_torch", "numpy", "pandas", "PIL"]


def chextriev_dir(config: Config) -> Path:
    return published_dir(config) / config.get("published.chextriev.folder")


def dependency_versions() -> dict:
    import importlib
    from importlib.metadata import version
    out = {}
    for name in VERSION_MODULES:
        module = importlib.import_module(name)
        out[name] = getattr(module, "__version__", None) or version(name)
    return out


def set_torch_home(config: Config) -> Path:
    home = Path(config.get("published.chextriev.torch_home"))
    os.environ["TORCH_HOME"] = str(home)
    return home


class AdaptedSource:

    def __init__(self, config: Config):
        self.config = config
        self.source = config.get("published.source")
        self.settings = config.get("published.chextriev")
        self.root = chextriev_dir(config)
        self.folder = self.root / "source"
        self.patch = REPO_DIR / self.settings["adaptation"]
        self.manifest = self.root / "source_manifest.json"
        self.records = ProvenanceRecords(self.root, "chextriev-source", {
            "commit": self.source["commit"], "excluded": self.settings["excluded_source_folders"]})

    def unit(self) -> Unit:
        return Unit("source", ["commit", "patch"], [self.source["commit"], sha256_file(self.patch)], [self.manifest])

    def prepare(self) -> dict:
        unit = self.unit()
        if self.records.stale_reason(unit) is None and self.folder.is_dir():
            return json.loads(self.manifest.read_text())
        self.records.discard(unit)
        shutil.rmtree(self.folder, ignore_errors=True)
        self.folder.mkdir(parents=True)
        checkout, commit = self.source["checkout"], self.source["commit"]
        paths = [f":(exclude){name}" for name in self.settings["excluded_source_folders"]]
        archive = subprocess.run(["git", "-C", checkout, "archive", commit, "--", ".", *paths], check=True,
                                 capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(self.folder)], input=archive, check=True)
        patched = [line[6:].strip() for line in self.patch.read_text().splitlines() if line.startswith("+++ b/")]
        before = {name: sha256_file(self.folder / name) for name in patched}
        subprocess.run(["git", "apply", "-p1", str(self.patch)], cwd=self.folder, check=True)
        after = {name: sha256_file(self.folder / name) for name in patched}
        files = {str(p.relative_to(self.folder)): sha256_file(p) for p in sorted(self.folder.rglob("*")) if p.is_file()}
        manifest = {"url": self.source["url"], "commit": commit, "adaptation_patch": str(self.settings["adaptation"]),
                    "adaptation_patch_sha256": sha256_file(self.patch),
                    "patched_files": {name: {"before": before[name], "after": after[name]} for name in patched},
                    "files": files}
        self.manifest.write_text(json.dumps(manifest, indent=1))
        self.records.write(unit)
        return manifest

    def import_path(self) -> None:
        if str(self.folder) not in sys.path:
            sys.path.insert(0, str(self.folder))


class CropItems:

    def __init__(self, images: pd.DataFrame, region_labels: np.ndarray, crops_dir: Path, transform, regions: list,
                 members: dict):
        self.image_ids = images["image_id"].tolist()
        self.chunks = images["chunk"].astype(int).tolist()
        self.region_labels = region_labels.astype(np.float32)
        self.crops_dir, self.transform, self.regions, self.members = crops_dir, transform, regions, members
        self.handles = {}

    def __len__(self) -> int:
        return len(self.image_ids)

    def archive(self, chunk: int) -> zipfile.ZipFile:
        if chunk not in self.handles:
            self.handles[chunk] = zipfile.ZipFile(self.crops_dir / f"crops_{chunk:03d}.zip")
        return self.handles[chunk]

    def read(self, chunk: int, member: str):
        import io
        from PIL import Image
        with Image.open(io.BytesIO(self.archive(chunk).read(member))) as image:
            image.load()
        return image

    def __getitem__(self, idx: int) -> dict:
        import torch
        image_id, chunk = self.image_ids[idx], self.chunks[idx]
        node_feat = torch.stack([self.transform(self.read(chunk, self.members["region"].format(
            image_id=image_id, region=region))) for region in range(len(self.regions))])
        global_feat = self.transform(self.read(chunk, self.members["image"].format(image_id=image_id)))
        return {"idx": idx, "global_feat": global_feat, "node_feat": node_feat,
                "y": torch.from_numpy(self.region_labels[idx]), "anatomy_name": list(self.regions),
                "complete_labels": torch.zeros(14), "masked_img": []}


def as_dataset(items: CropItems):
    from torch.utils.data import Dataset

    class Items(Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, idx):
            return items[idx]
    return Items()


class ChextrievData:

    def __init__(self, config: Config, limit: int | None = None):
        self.config = config
        population = PublishedPopulation(config)
        if population.records.stale_reason(population.unit()) is not None:
            raise RuntimeError("run published-population first: its outputs are missing or out of date")
        images = pd.read_csv(population.outputs["images.csv"], dtype=str, keep_default_na=False)
        with np.load(population.outputs["regions.npz"]) as saved:
            if saved["image_id"].tolist() != images["image_id"].tolist():
                raise ValueError("regions.npz and images.csv list different images")
            boxes, labels = saved["boxes"], saved["labels"]
        images["chunk"] = images["chunk"].astype(int)
        crops = PublishedCrops(config)
        units = [unit for unit, _, _ in crops.units(images, boxes)]
        if limit is not None:
            keep = images.groupby("split", sort=False).head(limit).index
            images, labels = images.loc[keep], labels[keep]
            used = {f"chunk_{chunk:03d}" for chunk in images["chunk"].unique()}
            units = [unit for unit in units if unit.name in used]
        incomplete = [unit.name for unit in units if crops.records.stale_reason(unit) is not None]
        if incomplete:
            raise RuntimeError(f"{len(incomplete)} crop chunks are not complete: {incomplete}")
        self.crops = crops
        self.images, self.labels = images, labels
        self.zip_sha = {int(unit.name.split("_")[1]): crops.records.read(unit)["outputs"][unit.outputs[0].name]
                        for unit in units}

    def part(self, name: str, order_by_id: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
        mask = (self.images["split"] == name).to_numpy()
        part, labels = self.images[mask], self.labels[mask]
        if order_by_id:
            order = np.argsort(part["image_id"].to_numpy(), kind="stable")
            part, labels = part.iloc[order], labels[order]
        return part.reset_index(drop=True), labels

    def fingerprint(self) -> dict:
        out = {}
        for name in SPLITS:
            part, labels = self.part(name)
            out[name] = {"images": len(part), "ids_chunks_sha256": sha256_json([part["image_id"].tolist(),
                                                                               part["chunk"].tolist()]),
                         "labels_sha256": hashlib.sha256(np.ascontiguousarray(labels, dtype=np.int8)).hexdigest()}
        out["crop_zips_sha256"] = {f"{chunk:03d}": sha for chunk, sha in self.zip_sha.items()}
        return out


def source_transform(source: AdaptedSource):
    source.import_path()
    from torchvision import transforms
    from dataloader.anaxnet import CustomDataset
    full = CustomDataset(pd.DataFrame({"split": []}), split="train").transform
    first = full.transforms[0]
    if not (isinstance(first, transforms.Resize) and tuple(first.size) == (224, 224) and first.antialias):
        raise RuntimeError("the source transform does not start with Resize((224, 224), antialias=True)")
    return transforms.Compose(full.transforms[1:])


def seed_everything(seed: int) -> None:
    import random
    import torch
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class FetchWeights:

    step = "fetch-chextriev-weights"

    def __init__(self, config: Config):
        self.config = config
        self.settings = config.get("published.chextriev")

    def path(self) -> Path:
        return set_torch_home(self.config) / "hub" / "checkpoints" / self.settings["weights"]["file"]

    def run(self) -> None:
        path = self.path()
        if not path.exists():
            import torch
            from torchvision.models import ResNet50_Weights
            url = ResNet50_Weights.DEFAULT.url
            if not url.endswith("/" + self.settings["weights"]["file"]):
                raise RuntimeError(f"torchvision's ResNet50_Weights.DEFAULT is not {self.settings['weights']['file']}")
            torch.hub.load_state_dict_from_url(url, progress=False, check_hash=True)
        self.check()

    def check(self) -> None:
        path = self.path()
        if not path.exists() or sha256_file(path) != self.settings["weights"]["sha256"]:
            raise RuntimeError(f"{path} is missing or its SHA-256 differs from published.chextriev.weights; run "
                               "fetch-chextriev-weights on the login node")


class ChextrievRun:

    def __init__(self, config: Config):
        self.config = config
        self.settings = config.get("published.chextriev")
        self.versions = environment_versions()
        FetchWeights(config).check()
        self.source = AdaptedSource(config)
        self.manifest = self.source.prepare()
        self.source.import_path()
        self.run_dir = chextriev_dir(config) / f"seed_{self.settings['seed']}"
        self.checkpoints = self.run_dir / "checkpoints"
        self.data = ChextrievData(config, self.settings["limit_images"])
        self.regions = config.get("published.regions")
        self.members = config.get("published.crops.members")
        self.transform = source_transform(self.source)

    def items(self, split: str, order_by_id: bool = False):
        part, labels = self.data.part(split, order_by_id)
        return part, as_dataset(CropItems(part, labels, self.data.crops.folder, self.transform, self.regions,
                                          self.members))

    def model_config(self, train_batches: int) -> dict:
        settings = self.settings
        devices, microbatch = settings["devices"], settings["microbatch"]
        if settings["effective_batch"] % (microbatch * devices):
            raise ValueError("effective_batch must be a multiple of microbatch x devices")
        config = dict(settings["model"])
        config.update({"batch_size": microbatch, "grad_accum": settings["effective_batch"] // (microbatch * devices),
                       "gpu_ids": list(range(devices)), "save_dir": str(self.run_dir), "train": True})
        config["num_cpus"] = len(os.sched_getaffinity(0))
        config["num_gpus"] = devices
        config["num_workers"] = min(settings["num_workers"], config["num_cpus"] // config["num_gpus"])
        batches_per_gpu = -(-train_batches // devices)
        config["num_steps"] = -(-batches_per_gpu // config["grad_accum"]) * config["epochs"]
        return config

    def settings_record(self, model_config: dict) -> dict:
        keep = {k: v for k, v in model_config.items() if k not in ("num_cpus", "num_workers", "save_dir")}
        return {"model": keep, "seed": self.settings["seed"], "effective_batch": self.settings["effective_batch"],
                "versions": self.versions, "dependencies": dependency_versions(),
                "adaptation_patch_sha256": self.manifest["adaptation_patch_sha256"],
                "weights": self.settings["weights"], "transform": repr(self.transform)}


class ChextrievTrainer:

    step = "train-chextriev"

    def __init__(self, config: Config):
        self.config = config
        self.started = time.time()

    def run(self) -> None:
        import torch
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
        from pytorch_lightning.loggers import CSVLogger
        from torch.utils.data import DataLoader

        run = ChextrievRun(self.config)
        settings = run.settings
        from model.graph_transformer import CustomModel

        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision("medium")
        seed_everything(settings["seed"])
        _, train_set = run.items("train")
        _, valid_set = run.items("valid")
        _, test_set = run.items("test")
        microbatch = settings["microbatch"]
        workers = min(settings["num_workers"], len(os.sched_getaffinity(0)) // settings["devices"])
        train_loader = DataLoader(train_set, batch_size=microbatch, shuffle=True, pin_memory=False, num_workers=workers)
        valid_loader = DataLoader(valid_set, batch_size=microbatch, shuffle=False, pin_memory=False, num_workers=workers)
        test_loader = DataLoader(test_set, batch_size=microbatch, shuffle=False, pin_memory=False, num_workers=workers)
        model_config = run.model_config(len(train_loader))
        record = {"inputs": run.data.fingerprint(), "settings": run.settings_record(model_config)}
        run.run_dir.mkdir(parents=True, exist_ok=True)
        inputs_file = run.run_dir / "inputs.json"
        if inputs_file.exists():
            if json.loads(inputs_file.read_text()) != record:
                raise RuntimeError(f"{inputs_file} differs from this run's inputs or settings; it is not continued. "
                                   "Remove the run folder to train again from the start")
        else:
            inputs_file.write_text(json.dumps(record, indent=1))
        summary = run.run_dir / "summary.json"
        records = ProvenanceRecords(run.run_dir, self.step, record["settings"])
        unit = Unit("summary", list(record["inputs"]), [json.dumps(v, sort_keys=True) for v in record["inputs"].values()],
                    [summary])
        if records.stale_reason(unit) is None:
            return

        model = CustomModel(model_config, num_classes=model_config["num_classes"], num_nodes=model_config["num_nodes"])
        folder = str(run.checkpoints)
        loss_callback = ModelCheckpoint(save_top_k=1, save_last=True, monitor="val_loss", mode="min", dirpath=folder,
                                        filename="model-{epoch:02d}-{val_loss:.5f}")
        loss_callback.CHECKPOINT_NAME_LAST = "{epoch:02d}-last"
        auc_callback = ModelCheckpoint(save_top_k=1, monitor="val_auc", mode="max", dirpath=folder,
                                       filename="model-{epoch:02d}-{val_auc:.5f}")
        map_callback = ModelCheckpoint(save_top_k=1, monitor="val_mAP", mode="max", dirpath=folder,
                                       filename="model-{epoch:02d}-{val_mAP:.5f}")
        early_stopping = EarlyStopping(monitor="val_auc", mode="max", patience=4)
        resume_callback = ModelCheckpoint(dirpath=folder, filename=RESUME_NAME, save_top_k=1, monitor=None,
                                          save_on_train_epoch_end=True)
        budget = settings["time_budget_hours"] * 3600
        started = self.started

        class TimeBudget(Callback):
            stopped = False
            epoch_started = time.time()
            longest = 0.0

            def on_train_epoch_start(self, trainer, pl_module):
                TimeBudget.epoch_started = time.time()

            def on_train_epoch_end(self, trainer, pl_module):
                TimeBudget.longest = max(TimeBudget.longest, time.time() - TimeBudget.epoch_started)
                if time.time() - started + TimeBudget.longest > budget:
                    TimeBudget.stopped = True
                    trainer.should_stop = True

        job = os.environ.get("SLURM_JOB_ID") or f"local_{int(self.started)}"
        devices = model_config["gpu_ids"] if settings["accelerator"] == "gpu" else settings["devices"]
        trainer = pl.Trainer(accelerator=settings["accelerator"], devices=devices,
                             strategy="ddp_find_unused_parameters_true", max_epochs=model_config["epochs"],
                             logger=CSVLogger(str(run.run_dir), name="logs", version=f"job_{job}"),
                             precision="bf16-mixed" if model_config["set_precision"] else "32-true",
                             gradient_clip_val=0.5, gradient_clip_algorithm="norm",
                             accumulate_grad_batches=model_config["grad_accum"],
                             callbacks=[loss_callback, auc_callback, early_stopping, map_callback, resume_callback,
                                        TimeBudget()],
                             num_sanity_val_steps=0, val_check_interval=0.5, enable_progress_bar=False)
        resume_file = run.checkpoints / f"{RESUME_NAME}.ckpt"
        resume = str(resume_file) if resume_file.exists() else None
        trainer.fit(model, train_loader, valid_loader, ckpt_path=resume)
        if TimeBudget.stopped and trainer.current_epoch < model_config["epochs"] and not early_stopping.stopped_epoch:
            return
        trainer.test(model, dataloaders=test_loader)
        best = Path(auc_callback.best_model_path)
        validations = []
        for metrics in sorted((run.run_dir / "logs").glob("job_*/metrics.csv")):
            table = pd.read_csv(metrics)
            if "val_auc" in table:
                rows = table.dropna(subset=["val_auc"])
                validations += rows[[c for c in ("epoch", "step", "val_auc", "val_loss", "val_mAP") if c in rows]
                                    ].to_dict("records")
        stopped = "early_stopping" if early_stopping.stopped_epoch else "max_epochs"
        summary.write_text(json.dumps({
            "selected_checkpoint": best.name, "selected_sha256": sha256_file(best),
            "selected_val_auc": float(auc_callback.best_model_score), "stopped": stopped,
            "epochs": int(trainer.current_epoch), "validations": validations,
            "effective_batch": settings["effective_batch"], "grad_accum": model_config["grad_accum"]}, indent=1))
        records.write(unit)


class ChextrievEmbedder:

    step = "embed-chextriev"

    def __init__(self, config: Config):
        self.config = config

    def run(self) -> None:
        import torch
        from torch.utils.data import DataLoader
        run = ChextrievRun(self.config)
        from model.graph_transformer import CustomModel
        summary = json.loads((run.run_dir / "summary.json").read_text())
        checkpoint = run.checkpoints / summary["selected_checkpoint"]
        if sha256_file(checkpoint) != summary["selected_sha256"]:
            raise RuntimeError(f"{checkpoint.name} differs from the checkpoint summary.json selected")
        folder = published_dir(self.config) / run.settings["embeddings_folder"]
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / f"chextriev_rerun__{run.settings['seed']}.npz"
        images, dataset = run.items("test", order_by_id=True)
        records = ProvenanceRecords(folder, self.step, {"checkpoint_sha256": summary["selected_sha256"],
                                                        "transform": repr(run.transform)})
        unit = Unit(output.stem, images["image_id"], [f"{chunk}" for chunk in images["chunk"]], [output])
        if records.pending([unit], "chextriev"):
            model_config = json.loads((run.run_dir / "inputs.json").read_text())["settings"]["model"]
            model_config.update({"save_dir": str(run.run_dir), "num_workers": run.settings["num_workers"]})
            device = "cuda" if run.settings["accelerator"] == "gpu" else "cpu"
            model = CustomModel.load_from_checkpoint(str(checkpoint), config=model_config, map_location=device).eval()
            loader = DataLoader(dataset, batch_size=run.settings["microbatch"], shuffle=False,
                                num_workers=min(run.settings["num_workers"], len(os.sched_getaffinity(0))))
            parts = []
            with torch.no_grad():
                for batch in loader:
                    parts.append(model.retrieval_pass(batch["node_feat"].to(device), batch["global_feat"].to(device))
                                 .float().cpu().numpy())
            embedding = np.concatenate(parts)
            partial = output.with_name(output.stem + ".partial.npz")
            np.savez(partial, image_ids=images["image_id"].to_numpy(dtype=str), embedding=embedding)
            os.replace(partial, output)
            records.write(unit)
