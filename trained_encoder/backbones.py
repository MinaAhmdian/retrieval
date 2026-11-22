import re
import tempfile

import torch
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
from torch import nn

from config import Config
from encoders.models import snapshot_dir

ADAPTER_MARK = "lora_"


def backbone_settings(config: Config, name: str, configuration: str | None = None) -> dict:
    settings = dict(config.get(f"trained_encoder.backbones.{name}"))
    return settings


class Backbone(nn.Module):

    def __init__(self, config: Config, name: str, configuration: str | None = None):
        super().__init__()
        self.name = name
        self.settings = backbone_settings(config, name, configuration)
        self.lora = config.get("trained_encoder.lora")
        self.image_dim = self.settings["image_dim"]
        self.tokens = self.settings["tokens"]
        self.model: nn.Module = None

    def adapted_modules(self) -> list[str]:
        pattern = re.compile(self.settings["lora_modules"]["pattern"])
        types = tuple(getattr(nn, kind) for kind in self.settings["lora_modules"]["types"])
        names = [name for name, module in self.model.named_modules()
                 if pattern.search(name) and isinstance(module, types)]
        if not names:
            raise ValueError(f"{self.name}: no module matches the LoRA modules")
        return names

    def add_adapters(self) -> None:
        self.model.requires_grad_(False)
        self.adapter_targets = self.adapted_modules()
        lora = LoraConfig(r=self.lora["rank"], lora_alpha=self.lora["alpha"], lora_dropout=self.lora["dropout"],
                          target_modules=self.adapter_targets)
        self.model = inject_adapter_in_model(lora, self.model)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(ADAPTER_MARK in name)
        self.model.eval()

    def train(self, mode: bool = True) -> "Backbone":
        super().train(mode)
        self.model.eval()
        return self

    def adapter_state(self) -> dict[str, torch.Tensor]:
        return {name: value for name, value in self.model.state_dict().items() if ADAPTER_MARK in name}

    def counts(self) -> dict[str, int]:
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        return {"adapted_modules": len(self.adapter_targets), "trainable": trainable, "frozen": frozen}

    def transform(self, image: Image.Image) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def stage3(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{self.name} has no stage-3 map")

    def stage4_pooled(self, maps: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{self.name} has no stage 4")

    def check_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tuple(tokens.shape[1:]) != (self.tokens, self.image_dim):
            raise RuntimeError(f"{self.name}: token map {tuple(tokens.shape)}, expected [n, {self.tokens}, "
                               f"{self.image_dim}]")
        return tokens


class BioViLTBackbone(Backbone):

    def __init__(self, config: Config, name: str, configuration: str | None = None):
        super().__init__(config, name, configuration)
        from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
        from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
        tempfile.tempdir = str(config.path("tools.biovil_t.weights_dir"))
        image_model = get_biovil_t_image_encoder()
        tempfile.tempdir = None
        self.model = image_model.encoder.encoder
        self.add_adapters()
        size = self.settings["input_size"]
        self.image_transform = create_chest_xray_transform_for_inference(resize=size, center_crop_size=size)

    def transform(self, image: Image.Image) -> torch.Tensor:
        return self.image_transform(image.convert("L"))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.model(images)
        return self.check_tokens(feature_map.flatten(2).transpose(1, 2))

    def stage3(self, images: torch.Tensor) -> torch.Tensor:
        model = self.model
        x = model.maxpool(model.relu(model.bn1(model.conv1(images))))
        return model.layer3(model.layer2(model.layer1(x)))

    def stage4_pooled(self, maps: torch.Tensor) -> torch.Tensor:
        return self.model.layer4(maps).mean(dim=(2, 3))


class RadDinoBackbone(Backbone):

    def __init__(self, config: Config, name: str, configuration: str | None = None):
        super().__init__(config, name, configuration)
        from transformers import AutoImageProcessor, AutoModel
        hub = config.get(f"retrieval.encoders.{self.settings['weights_from']}.hub")[0]
        model_dir = snapshot_dir(hub)
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir)
        self.add_adapters()
        self.input_size = self.settings["input_size"]

    def transform(self, image: Image.Image) -> torch.Tensor:
        if image.size != (self.input_size, self.input_size):
            raise ValueError(f"{self.name} expects {self.input_size} x {self.input_size} images, got {image.size}")
        return self.processor(images=image, do_resize=False, do_center_crop=False,
                              return_tensors="pt")["pixel_values"][0]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.check_tokens(self.model(pixel_values=images).last_hidden_state[:, 1:])


def load_backbone(config: Config, name: str, configuration: str | None = None) -> Backbone:
    return globals()[config.get(f"trained_encoder.backbones.{name}.class")](config, name, configuration)
