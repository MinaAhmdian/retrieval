from pathlib import Path

import torch
from torch import nn

from lib.exam_encoder import ExamEncoder
from config import Config
from provenance import sha256_file
from trained_encoder.backbones import load_backbone
from trained_encoder.exam_batches import place_tokens

LIB_DIR = Path(__file__).resolve().parents[1] / "lib"


def configuration_model(config: Config, configuration: str) -> str | None:
    return config.get("controls.configuration_models").get(configuration)


def build_model(config: Config, backbone_name: str, configuration: str) -> nn.Module:
    model = configuration_model(config, configuration)
    if model is None:
        return ExamModel(config, backbone_name, configuration=configuration)
    if model == "fusion":
        from controls.fusion_model import FusionModel
        return FusionModel(config, backbone_name)
    raise ValueError(f"{configuration}: unknown model {model}")


def configuration_settings(config: Config, configuration: str) -> dict:
    settings = {}
    if configuration_model(config, configuration) is not None:
        settings["model"] = configuration_model(config, configuration)
        settings["fusion"] = config.get("controls.fusion")
        settings["fusion_lib_sha256"] = sha256_file(LIB_DIR / "fusion.py")
    if config.get(f"trained_encoder.input_configurations.{configuration}").get("lateral_from_donor"):
        settings["lateral_permutation_seed"] = config.get("controls.lateral_permutation.seed")
    return settings


class ExamModel(nn.Module):

    def __init__(self, config: Config, backbone_name: str, backbone=None, configuration: str | None = None):
        super().__init__()
        self.backbone = load_backbone(config, backbone_name, configuration) if backbone is None else backbone
        self.encoder = ExamEncoder(self.backbone.image_dim, **config.get("trained_encoder.encoder"))

    def trainable(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    @property
    def transform(self):
        return self.backbone.transform

    def forward(self, batch: dict, dropout_probs: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        image_tokens, slot_mask = place_tokens(self.backbone(batch["images"]), batch["slot_index"])
        return self.encoder(image_tokens, slot_mask, batch["view_ids"], batch["text_emb"], batch["text_mask"],
                            batch["age_ids"], batch["sex_ids"], dropout_probs=dropout_probs)

    def weights(self) -> dict:
        return {"encoder": self.encoder.state_dict(), "adapters": self.backbone.adapter_state()}

    def load_weights(self, weights: dict) -> None:
        self.encoder.load_state_dict(weights["encoder"])
        missing, unexpected = self.backbone.model.load_state_dict(weights["adapters"], strict=False)
        missing_adapters = [name for name in missing if "lora_" in name]
        if unexpected or missing_adapters:
            raise RuntimeError(f"adapter weights do not match: {len(unexpected)} unexpected, "
                               f"{len(missing_adapters)} missing")
