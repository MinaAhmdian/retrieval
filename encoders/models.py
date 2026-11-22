import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image

from config import Config


def snapshot_dir(hub: dict) -> Path:
    return Path(snapshot_download(hub["repo"], revision=hub["revision"], allow_patterns=hub["files"],
                                  local_files_only=True))


def l2_normalize(embeddings: torch.Tensor) -> np.ndarray:
    embeddings = embeddings.float()
    return torch.nn.functional.normalize(embeddings, dim=-1).cpu().numpy()


class Encoder:

    def __init__(self, config: Config, name: str, device: torch.device):
        self.config = config
        self.name = name
        self.settings = config.get(f"retrieval.encoders.{name}")
        self.device = device
        self.dim = self.settings["dim"]
        self.text_max_tokens = self.settings.get("text_max_tokens")

    @property
    def has_text(self) -> bool:
        return self.text_max_tokens is not None

    def transform(self, image: Image.Image) -> torch.Tensor:
        raise NotImplementedError

    def embed_images(self, batch: torch.Tensor) -> np.ndarray:
        raise NotImplementedError

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError(f"{self.name} has no text tower")


class BioViLT(Encoder):

    def __init__(self, config: Config, name: str, device: torch.device):
        super().__init__(config, name, device)
        from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
        from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
        from transformers import AutoModel, AutoTokenizer
        tempfile.tempdir = str(config.path("tools.biovil_t.weights_dir"))
        image_model = get_biovil_t_image_encoder()
        tempfile.tempdir = None
        self.image_model = image_model.requires_grad_(False).eval().to(device)
        self.image_transform = create_chest_xray_transform_for_inference(resize=self.settings["resize"],
                                                                          center_crop_size=self.settings["center_crop"])
        text_dir = snapshot_dir(self.settings["hub"][0])
        self.tokenizer = AutoTokenizer.from_pretrained(text_dir, trust_remote_code=True)
        self.text_model = AutoModel.from_pretrained(text_dir, trust_remote_code=True,
                                                    use_safetensors=False).eval().to(device)

    def transform(self, image: Image.Image) -> torch.Tensor:
        return self.image_transform(image.convert("L"))

    @torch.no_grad()
    def embed_images(self, batch: torch.Tensor) -> np.ndarray:
        return l2_normalize(self.image_model(batch.to(self.device)).projected_global_embedding)

    @torch.no_grad()
    def embed_texts(self, texts: list[str], max_tokens: int | None = None) -> np.ndarray:
        tokens = self.tokenizer(texts, padding="longest", truncation=True,
                                max_length=max_tokens or self.text_max_tokens, return_tensors="pt").to(self.device)
        return l2_normalize(self.text_model.get_projected_text_embeddings(input_ids=tokens.input_ids,
                                                                          attention_mask=tokens.attention_mask))


class RadDino(Encoder):

    def __init__(self, config: Config, name: str, device: torch.device):
        super().__init__(config, name, device)
        from transformers import AutoImageProcessor, AutoModel
        model_dir = snapshot_dir(self.settings["hub"][0])
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).eval().to(device)

    def transform(self, image: Image.Image) -> torch.Tensor:
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def embed_images(self, batch: torch.Tensor) -> np.ndarray:
        return l2_normalize(self.model(pixel_values=batch.to(self.device)).last_hidden_state[:, 0])


class BiomedCLIP(Encoder):

    def __init__(self, config: Config, name: str, device: torch.device):
        super().__init__(config, name, device)
        import open_clip
        model_dir = snapshot_dir(self.settings["hub"][0])
        model_ref = f"local-dir:{model_dir}"
        with open(model_dir / "open_clip_config.json") as handle:
            text_cfg = json.load(handle)["model_cfg"]["text_cfg"]
        text_cfg["hf_model_name"] = str(snapshot_dir(self.settings["hub"][1]))
        text_cfg["hf_model_pretrained"] = False
        model, self.preprocess = open_clip.create_model_from_pretrained(model_ref, text_cfg=text_cfg)
        self.model = model.eval().to(device)
        self.tokenizer = open_clip.get_tokenizer(model_ref, context_length=self.text_max_tokens)

    def transform(self, image: Image.Image) -> torch.Tensor:
        return self.preprocess(image)

    @torch.no_grad()
    def embed_images(self, batch: torch.Tensor) -> np.ndarray:
        return l2_normalize(self.model.encode_image(batch.to(self.device)))

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(self.model.encode_text(self.tokenizer(texts).to(self.device)))


class MedSigLIP(Encoder):

    def __init__(self, config: Config, name: str, device: torch.device):
        super().__init__(config, name, device)
        from transformers import AutoModel, AutoProcessor
        model_dir = snapshot_dir(self.settings["hub"][0])
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).eval().to(device)

    def transform(self, image: Image.Image) -> torch.Tensor:
        return self.processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def embed_images(self, batch: torch.Tensor) -> np.ndarray:
        return l2_normalize(self.model.get_image_features(pixel_values=batch.to(self.device)))

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        tokens = self.processor(text=texts, padding="max_length", truncation=True, max_length=self.text_max_tokens,
                                return_tensors="pt").to(self.device)
        return l2_normalize(self.model.get_text_features(input_ids=tokens["input_ids"]))


class CXRClip(Encoder):

    def __init__(self, config: Config, name: str, device: torch.device):
        super().__init__(config, name, device)
        sys.path.insert(0, self.settings["git"][0]["dir"])
        from cxrclip.data.data_utils import load_tokenizer, load_transform, transform_image
        from cxrclip.model import build_model
        checkpoint = torch.load(self.settings["checkpoints"][0]["path"], map_location="cpu", weights_only=False)
        ckpt_config = checkpoint["config"]
        text_dir = str(snapshot_dir(self.settings["hub"][0]))
        tokenizer_config = dict(ckpt_config["tokenizer"], pretrained_model_name_or_path=text_dir, cache_dir=text_dir)
        self.tokenizer = load_tokenizer(**tokenizer_config)
        model_config = dict(ckpt_config["model"])
        model_config["text_encoder"] = dict(model_config["text_encoder"], name=text_dir, cache_dir=text_dir)
        model = build_model(model_config=model_config, loss_config=ckpt_config["loss"], tokenizer=self.tokenizer)
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        if missing:
            raise RuntimeError(f"{len(missing)} model weights are not in the checkpoint, e.g. {missing[:3]}")
        self.n_unused_checkpoint_keys = len(unexpected)
        self.model = model.eval().to(device)
        self.image_transforms = load_transform(split=self.settings["transform_split"],
                                               transform_config=ckpt_config["transform"])
        self.normalize = "imagenet" if model_config["image_encoder"]["name"] == "resnet" else "huggingface"
        self.transform_image = transform_image

    def transform(self, image: Image.Image) -> torch.Tensor:
        return self.transform_image(self.image_transforms, image.convert("RGB"), normalize=self.normalize)

    @torch.no_grad()
    def embed_images(self, batch: torch.Tensor) -> np.ndarray:
        features = self.model.encode_image(batch.to(self.device))
        return l2_normalize(self.model.image_projection(features) if self.model.projection else features)

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(texts, padding="longest", truncation=True, max_length=self.text_max_tokens,
                                return_tensors="pt").to(self.device)
        features = self.model.encode_text(tokens)
        return l2_normalize(self.model.text_projection(features) if self.model.projection else features)


class MedImageInsight(Encoder):

    def __init__(self, config: Config, name: str, device: torch.device):
        super().__init__(config, name, device)
        repo_dir = snapshot_dir(self.settings["hub"][0])
        sys.path.insert(0, str(repo_dir))
        from medimageinsightmodel import MedImageInsight as Loader
        loader = Loader(model_dir=str(repo_dir / self.settings["model_subdir"]),
                        vision_model_name=self.settings["vision_model"], language_model_name="language_model")
        loader.load_model()
        self.preprocess = loader.preprocess
        self.model = loader.model.eval().to(device)

    def transform(self, image: Image.Image) -> torch.Tensor:
        return self.preprocess(image.convert("RGB"))

    @torch.no_grad()
    def embed_images(self, batch: torch.Tensor) -> np.ndarray:
        return l2_normalize(self.model.encode_image(batch.to(self.device)))


def compute_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config asks for cuda but no GPU is visible")
    return device


def load_encoder(config: Config, name: str, device: torch.device) -> Encoder:
    return globals()[config.get(f"retrieval.encoders.{name}.class")](config, name, device)
