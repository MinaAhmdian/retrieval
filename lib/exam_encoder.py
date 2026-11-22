"""
lib/exam_encoder.py
Created on October 17, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_SLOTS = 4
N_VIEWS = 4
N_AGE_BINS = 11
N_SEX = 3
N_LABELS = 14

VIEW_IDS = {"PA": 0, "AP": 1, "lateral": 2, "other": 3}
SEX_IDS = {"F": 0, "M": 1, "unknown": 2}
LABELS = ["atelectasis", "cardiomegaly", "consolidation", "edema", "enlarged_cardiomediastinum", "fracture",
          "lung_lesion", "lung_opacity", "no_finding", "pleural_effusion", "pleural_other", "pneumonia",
          "pneumothorax", "support_devices"]


def age_bin(age):
    if age is None or age != age or age < 0 or age >= 100:
        return N_AGE_BINS - 1
    return int(age // 10)


class ExamEncoder(nn.Module):

    def __init__(self, image_dim, text_dim=128, width=256, depth=4, heads=8, align_dim=128, ff=1024, drop=0.1):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, width)
        self.text_proj = nn.Linear(text_dim, width)
        self.view_emb = nn.Embedding(N_VIEWS, width)
        self.slot_emb = nn.Embedding(N_SLOTS, width)
        self.age_emb = nn.Embedding(N_AGE_BINS, width)
        self.sex_emb = nn.Embedding(N_SEX, width)
        self.text_null = nn.Parameter(torch.zeros(width))
        self.type_emb = nn.Embedding(5, width)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=ff, dropout=drop, batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(width)
        self.align_head = nn.Linear(width, align_dim)
        self.profile_head = nn.Linear(width, N_LABELS)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.text_null, std=0.02)

    @staticmethod
    def apply_evidence_dropout(slot_mask, view_ids, text_mask, age_ids, sex_ids, probs, generator=None):
        B = slot_mask.shape[0]
        dev = slot_mask.device
        def draw(p):
            return torch.rand(B, device=dev, generator=generator) < p
        slot_mask = slot_mask.clone()
        is_lateral = view_ids == VIEW_IDS["lateral"]
        drop_lat = draw(probs.get("lateral", 0.0))
        slot_mask[:, 2:] = slot_mask[:, 2:] & ~(drop_lat[:, None] & is_lateral[:, 2:])
        drop_second = draw(probs.get("second_image", 0.0))
        slot_mask[:, 1] = slot_mask[:, 1] & ~drop_second
        slot_mask[:, 3] = slot_mask[:, 3] & ~drop_second
        text_mask = text_mask & ~draw(probs.get("text", 0.0))
        age_ids = torch.where(draw(probs.get("age", 0.0)), torch.full_like(age_ids, N_AGE_BINS - 1), age_ids)
        sex_ids = torch.where(draw(probs.get("sex", 0.0)), torch.full_like(sex_ids, N_SEX - 1), sex_ids)
        return slot_mask, text_mask, age_ids, sex_ids

    def forward(self, image_tokens, slot_mask, view_ids, text_emb, text_mask, age_ids, sex_ids, dropout_probs=None):
        B, S, T, _ = image_tokens.shape
        if dropout_probs is not None and self.training:
            slot_mask, text_mask, age_ids, sex_ids = self.apply_evidence_dropout(slot_mask, view_ids, text_mask, age_ids, sex_ids, dropout_probs)
        img = self.image_proj(image_tokens)
        img = img + self.view_emb(view_ids)[:, :, None, :] + self.slot_emb(torch.arange(S, device=img.device))[None, :, None, :]
        img = img + self.type_emb.weight[1]
        img = img.reshape(B, S * T, -1)
        img_pad = ~slot_mask[:, :, None].expand(B, S, T).reshape(B, S * T)
        txt = torch.where(text_mask[:, None], self.text_proj(text_emb), self.text_null[None, :].expand(B, -1)) + self.type_emb.weight[2]
        age = self.age_emb(age_ids) + self.type_emb.weight[3]
        sex = self.sex_emb(sex_ids) + self.type_emb.weight[4]
        cls = self.cls.expand(B, -1, -1) + self.type_emb.weight[0]
        x = torch.cat([cls, img, txt[:, None, :], age[:, None, :], sex[:, None, :]], dim=1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), img_pad, torch.zeros(B, 3, dtype=torch.bool, device=x.device)], dim=1)
        h = self.norm(self.transformer(x, src_key_padding_mask=pad)[:, 0])
        return F.normalize(self.align_head(h), dim=-1), self.profile_head(h)
