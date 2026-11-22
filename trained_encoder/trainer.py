import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler

from lib.losses import alignment_loss, profile_loss
from config import Config
from encoders.models import compute_device
from provenance import FileDigests, ProvenanceRecords, Unit, sha256_file
from trained_encoder.backbones import backbone_settings
from trained_encoder.exam_batches import ExamItems, collate
from trained_encoder.exam_inputs import SLOT_COLUMNS
from trained_encoder.exam_model import ExamModel, build_model, configuration_settings
from trained_encoder.fit_data import FitData
from trained_encoder.runs import all_run_names, run_root

LIB_DIR = Path(__file__).resolve().parents[1] / "lib"
RUNTIME_SETTINGS = ("num_workers", "device", "time_budget_hours")
FIRST_EPOCH = 1
LOSS_TERMS = ("alignment", "profile")


def should_stop(valid_losses: list[float], patience: int) -> bool:
    return len(valid_losses) - 1 - int(np.argmin(valid_losses)) >= patience


class Trainer:

    step = "train-model"

    def __init__(self, config: Config, run: str, model: ExamModel | None = None, data: dict | None = None):
        if run not in all_run_names(config):
            raise ValueError(f"{run} is not a run: {all_run_names(config)}")
        self.config = config
        self.run_name = run
        self.backbone_name, self.configuration, seed = run.split("__")
        self.seed = int(seed)
        self.overrides = config.get("controls.configuration_training").get(self.configuration, {})
        unknown = set(self.overrides) - {"dropout_probs", "loss_terms", "train_corpora"}
        if unknown:
            raise ValueError(f"{self.configuration}: unknown training settings {sorted(unknown)}")
        self.settings = {**config.get("trained_encoder.training"),
                         **{key: value for key, value in self.overrides.items()
                            if key not in ("loss_terms", "train_corpora")}}
        self.loss_terms = tuple(self.overrides.get("loss_terms", LOSS_TERMS))
        if not self.loss_terms or not set(self.loss_terms) <= set(LOSS_TERMS):
            raise ValueError(f"{self.configuration}: loss_terms must be among {LOSS_TERMS}")
        self.run_dir = run_root(config, run) / self.settings["folder"] / run
        self.records = ProvenanceRecords(self.run_dir, self.step, self.step_settings())
        self.started = time.time()
        self.model_override, self.data_override = model, data

    def step_settings(self) -> dict:
        settings = {key: value for key, value in self.settings.items() if key not in RUNTIME_SETTINGS}
        out = {"run": self.run_name, "backbone": backbone_settings(self.config, self.backbone_name, self.configuration),
               "lora": self.config.get("trained_encoder.lora"), "encoder": self.config.get("trained_encoder.encoder"),
               "training": settings, "seed": self.seed,
               "configuration": self.config.get(f"trained_encoder.input_configurations.{self.configuration}"),
               "context_permutation_seed": self.config.get("trained_encoder.context_permutation.seed"),
               "lib_sha256": {name: sha256_file(LIB_DIR / name) for name in ("exam_encoder.py", "losses.py")}}
        if self.overrides:
            out["configuration_training"] = self.overrides
        return {**out, **configuration_settings(self.config, self.configuration)}


    def load_data(self) -> dict:
        if self.data_override is not None:
            return self.data_override
        return {split: FitData(self.config, self.configuration, split) for split in ("train", "valid")}

    def build_model(self) -> ExamModel:
        torch.manual_seed(self.seed)
        if self.model_override is not None:
            return self.model_override()
        return build_model(self.config, self.backbone_name, self.configuration)

    def fingerprint(self, data: dict) -> tuple[list[str], list[str]]:
        digests = FileDigests(self.config.get("provenance.digest_workers"))
        ids, contents = [], []
        for split in ("train", "valid"):
            ids += [f"{split}:{exam}" for exam in data[split].ids()]
            contents += data[split].contents(digests)
        return ids, contents

    def write_population(self, data: dict, fingerprint: tuple[list[str], list[str]]) -> None:
        ids, contents = fingerprint
        unit = Unit("population", ids, contents, [self.run_dir / "population.csv.gz", self.run_dir / "settings.json"])
        if self.records.stale_reason(unit) is None:
            return
        self.records.discard(unit)
        frames = []
        for split in ("train", "valid"):
            table = data[split].table[["corpus", "exam_id", *SLOT_COLUMNS]].copy()
            table.insert(0, "split", split)
            for slot, column in enumerate(SLOT_COLUMNS):
                table[f"{column}_png"] = data[split].slot_paths[:, slot]
            frames.append(table)
        pd.concat(frames, ignore_index=True).to_csv(unit.outputs[0], index=False)
        unit.outputs[1].write_text(json.dumps(self.step_settings(), indent=1))
        self.records.write(unit)

    def checkpoint(self, epoch: int) -> Path:
        return self.run_dir / f"epoch_{epoch:02d}.pt"

    def epoch_unit(self, epoch: int, fingerprint: tuple[list[str], list[str]]) -> Unit:
        previous = self.checkpoint(epoch - 1)
        previous_sha = sha256_file(previous) if epoch > FIRST_EPOCH and previous.exists() else "none"
        ids, contents = fingerprint
        return Unit(f"epoch_{epoch:02d}", ["previous_epoch", *ids], [previous_sha, *contents], [self.checkpoint(epoch)])

    def reusable_epochs(self, fingerprint) -> int:
        epoch = FIRST_EPOCH
        while epoch <= self.settings["max_epochs"]:
            unit = self.epoch_unit(epoch, fingerprint)
            reason = self.records.stale_reason(unit)
            if reason == "outputs without a provenance record":
                raise RuntimeError(f"[{self.run_name}] {unit.name} has no provenance record; it is not reused or "
                                   "removed unchecked")
            if reason is not None:
                break
            epoch += 1
        for later in range(epoch, self.settings["max_epochs"] + 1):
            self.records.discard(self.epoch_unit(later, fingerprint))
        return epoch - FIRST_EPOCH


    def loader(self, data, transform, shuffle_epoch: int | None) -> DataLoader:
        items = ExamItems(data, transform)
        sampler = None
        if shuffle_epoch is not None:
            generator = torch.Generator().manual_seed(self.seed * 1000 + shuffle_epoch)
            sampler = RandomSampler(items, generator=generator)
        return DataLoader(items, batch_size=self.settings["batch_size"], sampler=sampler, shuffle=False,
                          num_workers=self.settings["num_workers"], collate_fn=collate,
                          pin_memory=self.device.type == "cuda")

    def to_device(self, batch: dict) -> dict:
        return {key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()}

    def autocast(self):
        dtype = self.settings["mixed_precision"]
        return torch.autocast(self.device.type, dtype=getattr(torch, dtype) if dtype else None, enabled=bool(dtype))

    def losses(self, model: ExamModel, batch: dict, dropout_probs: dict | None) -> tuple[torch.Tensor, torch.Tensor]:
        with self.autocast():
            alignment, logits = model(batch, dropout_probs)
        aligned = alignment_loss(alignment.float(), batch["report"], batch["report_keys"],
                                 temperature=self.settings["temperature"])
        profiled = profile_loss(logits.float(), batch["labels"])
        if not (torch.isfinite(aligned) and torch.isfinite(profiled)):
            raise RuntimeError(f"[{self.run_name}] a loss is not finite: alignment {aligned.item()}, profile "
                               f"{profiled.item()}")
        return aligned, profiled

    def combined(self, aligned, profiled):
        terms = {"alignment": aligned, "profile": profiled}
        total = terms[self.loss_terms[0]]
        for name in self.loss_terms[1:]:
            total = total + terms[name]
        return total

    def train_epoch(self, model, optimizer, data, epoch: int) -> float:
        model.train()
        total, batches = 0.0, 0
        for batch in self.loader(data, model.transform, shuffle_epoch=epoch):
            batch = self.to_device(batch)
            aligned, profiled = self.losses(model, batch, self.settings["dropout_probs"])
            loss = self.combined(aligned, profiled)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable(), self.settings["grad_clip_norm"])
            optimizer.step()
            total, batches = total + loss.item(), batches + 1
        return total / batches

    @torch.no_grad()
    def validate(self, model, data) -> dict:
        model.eval()
        sums = {"alignment": 0.0, "profile": 0.0}
        batches = 0
        for batch in self.loader(data, model.transform, shuffle_epoch=None):
            aligned, profiled = self.losses(model, self.to_device(batch), None)
            sums["alignment"] += aligned.item()
            sums["profile"] += profiled.item()
            batches += 1
        means = {key: value / batches for key, value in sums.items()}
        return {"valid_loss": self.combined(means["alignment"], means["profile"]), "valid_alignment": means["alignment"],
                "valid_profile": means["profile"]}

    def rng_state(self) -> dict:
        state = {"torch": torch.get_rng_state()}
        if self.device.type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def set_rng_state(self, state: dict) -> None:
        torch.set_rng_state(state["torch"].cpu())
        if "cuda" in state:
            torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])

    def save_epoch(self, unit: Unit, model, optimizer, history: list[dict]) -> None:
        partial = unit.outputs[0].with_suffix(".partial")
        torch.save({"epoch": history[-1]["epoch"], "weights": model.weights(), "optimizer": optimizer.state_dict(),
                    "rng": self.rng_state(), "history": history}, partial)
        os.replace(partial, unit.outputs[0])
        self.records.write(unit)

    def finish(self, history: list[dict], fingerprint, stopped: str) -> None:
        losses = [entry["valid_loss"] for entry in history]
        best = int(np.argmin(losses))
        chosen_epoch = history[best]["epoch"]
        epoch_units = [self.epoch_unit(entry["epoch"], fingerprint) for entry in history]
        unit = Unit("chosen", [u.name for u in epoch_units], [sha256_file(u.outputs[0]) for u in epoch_units],
                    [self.run_dir / "chosen.pt", self.run_dir / "summary.json"])
        if self.records.stale_reason(unit) is None:
            return
        self.records.discard(unit)
        state = torch.load(self.checkpoint(chosen_epoch), map_location="cpu", weights_only=False)
        torch.save({"epoch": chosen_epoch, "weights": state["weights"]}, unit.outputs[0])
        summary = {"run": self.run_name, "epochs": history, "chosen_epoch": chosen_epoch,
                   "chosen_valid_loss": losses[best], "stopped": stopped,
                   "valid_loss_below_first_epoch": bool(min(losses[1:], default=np.inf) < losses[0])}
        unit.outputs[1].write_text(json.dumps(summary, indent=1))
        self.records.write(unit)

    def run(self, epoch_limit: int | None = None) -> None:
        self.device = compute_device(self.settings["device"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        data = self.load_data()
        fingerprint = self.fingerprint(data)
        self.write_population(data, fingerprint)
        model = self.build_model().to(self.device)
        optimizer = torch.optim.AdamW(model.trainable(), lr=self.settings["learning_rate"],
                                      weight_decay=self.settings["weight_decay"])
        reused = self.reusable_epochs(fingerprint)
        history = []
        if reused:
            state = torch.load(self.checkpoint(reused), map_location=self.device, weights_only=False)
            model.load_weights(state["weights"])
            optimizer.load_state_dict(state["optimizer"])
            self.set_rng_state(state["rng"])
            history = state["history"]
        longest = max((entry["seconds"] for entry in history), default=0.0)
        while True:
            losses = [entry["valid_loss"] for entry in history]
            if losses and should_stop(losses, self.settings["patience"]):
                return self.finish(history, fingerprint, "early_stopping")
            if len(history) >= self.settings["max_epochs"]:
                return self.finish(history, fingerprint, "max_epochs")
            if epoch_limit is not None and len(history) >= epoch_limit:
                return
            if time.time() - self.started + longest > self.settings["time_budget_hours"] * 3600:
                return
            epoch = len(history) + FIRST_EPOCH
            unit = self.epoch_unit(epoch, fingerprint)
            start = time.time()
            train_loss = self.train_epoch(model, optimizer, data["train"], epoch)
            entry = {"epoch": epoch, "train_loss": train_loss, **self.validate(model, data["valid"]),
                     "seconds": time.time() - start}
            history.append(entry)
            longest = max(longest, entry["seconds"])
            self.save_epoch(unit, model, optimizer, history)

    def peak_memory(self) -> str:
        if self.device.type != "cuda":
            return ""
        return f" | peak GPU memory {torch.cuda.max_memory_allocated(self.device) / 2 ** 30:.1f} GiB"
