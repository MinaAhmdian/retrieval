import torch
from torch import nn

from lib.exam_encoder import ExamEncoder
from lib.fusion import FusionHeads, MidlevelFusion
from config import Config
from trained_encoder.backbones import load_backbone

FRONTAL_SLOT = 0


class FusionModel(nn.Module):

    def __init__(self, config: Config, backbone_name: str, streams: tuple | None = None):
        super().__init__()
        settings = config.get("controls.fusion")
        if streams is None:
            streams = (load_backbone(config, backbone_name), load_backbone(config, backbone_name))
        self.frontal_stream, self.lateral_stream = streams
        self.fusion = MidlevelFusion(**settings["midlevel"])
        self.heads = FusionHeads(**settings["heads"])
        self.stage3_map = tuple(settings["stage3_map"])
        self.lateral_slot = config.get("trained_encoder.slot_positions.lateral")[0]

    def trainable(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    @property
    def transform(self):
        return self.frontal_stream.transform

    def check_map(self, maps: torch.Tensor) -> torch.Tensor:
        if tuple(maps.shape[1:]) != self.stage3_map:
            raise RuntimeError(f"stage-3 map {tuple(maps.shape)}, expected [n, {', '.join(map(str, self.stage3_map))}]")
        return maps

    def forward(self, batch: dict, dropout_probs: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        slot_index = batch["slot_index"]
        used = [FRONTAL_SLOT, self.lateral_slot]
        others = [slot for slot in range(slot_index.shape[1]) if slot not in used]
        if (slot_index[:, others] >= 0).any():
            raise ValueError("the fusion model takes only the first frontal and the first lateral image")
        slot_mask = slot_index >= 0
        if not slot_mask[:, FRONTAL_SLOT].all():
            raise ValueError("an exam has no first frontal image")
        text_mask, age_ids, sex_ids = batch["text_mask"], batch["age_ids"], batch["sex_ids"]
        if dropout_probs is not None and self.training:
            slot_mask, text_mask, age_ids, sex_ids = ExamEncoder.apply_evidence_dropout(
                slot_mask, batch["view_ids"], text_mask, age_ids, sex_ids, dropout_probs)
        lateral_present = slot_mask[:, self.lateral_slot]
        images = batch["images"]
        frontal_maps = self.check_map(self.frontal_stream.stage3(images[slot_index[:, FRONTAL_SLOT]]))
        lateral_maps = frontal_maps.new_zeros(frontal_maps.shape)
        if lateral_present.any():
            lateral_images = images[slot_index[lateral_present, self.lateral_slot]]
            lateral_maps[lateral_present] = self.check_map(self.lateral_stream.stage3(lateral_images))
        n, channels, height, width = frontal_maps.shape
        frontal_tokens, lateral_tokens = self.fusion(frontal_maps.flatten(2).transpose(1, 2),
                                                     lateral_maps.flatten(2).transpose(1, 2), lateral_present)

        def as_maps(tokens: torch.Tensor) -> torch.Tensor:
            return tokens.transpose(1, 2).reshape(n, channels, height, width)

        frontal_vectors = self.frontal_stream.stage4_pooled(as_maps(frontal_tokens))
        lateral_vectors = frontal_vectors.new_zeros(frontal_vectors.shape)
        if lateral_present.any():
            lateral_vectors[lateral_present] = self.lateral_stream.stage4_pooled(
                as_maps(lateral_tokens)[lateral_present])
        return self.heads(frontal_vectors, lateral_vectors, lateral_present, batch["text_emb"], text_mask, age_ids,
                          sex_ids)

    def weights(self) -> dict:
        return {"fusion": self.fusion.state_dict(), "heads": self.heads.state_dict(),
                "frontal_adapters": self.frontal_stream.adapter_state(),
                "lateral_adapters": self.lateral_stream.adapter_state()}

    def load_weights(self, weights: dict) -> None:
        self.fusion.load_state_dict(weights["fusion"])
        self.heads.load_state_dict(weights["heads"])
        for stream, key in ((self.frontal_stream, "frontal_adapters"), (self.lateral_stream, "lateral_adapters")):
            missing, unexpected = stream.model.load_state_dict(weights[key], strict=False)
            missing_adapters = [name for name in missing if "lora_" in name]
            if unexpected or missing_adapters:
                raise RuntimeError(f"{key} do not match: {len(unexpected)} unexpected, {len(missing_adapters)} missing")
