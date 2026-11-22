"""
lib/fusion.py
Created on November 11, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MidlevelFusion(nn.Module):

    def __init__(self, channels=1024, heads=8, dropout=0.1):
        super().__init__()
        self.front_norm = nn.LayerNorm(channels)
        self.lateral_norm = nn.LayerNorm(channels)
        self.front_attention = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.lateral_attention = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.front_gate = nn.Parameter(torch.zeros(()))
        self.lateral_gate = nn.Parameter(torch.zeros(()))

    def forward(self, frontal, lateral, lateral_present):
        if frontal.shape != lateral.shape or frontal.ndim != 3:
            raise ValueError("The token arrays must have the same [B, T, C] shape.")
        if lateral_present.dtype != torch.bool or lateral_present.shape != frontal.shape[:1]:
            raise ValueError("The lateral mask must be bool [B].")
        front_out = frontal.clone()
        lateral_out = torch.zeros_like(lateral)
        if lateral_present.any():
            front = frontal[lateral_present]
            side = lateral[lateral_present]
            front_norm = self.front_norm(front)
            side_norm = self.lateral_norm(side)
            front_update = self.front_attention(front_norm, side_norm, side_norm, need_weights=False)[0]
            side_update = self.lateral_attention(side_norm, front_norm, front_norm, need_weights=False)[0]
            front_out[lateral_present] = front + torch.tanh(self.front_gate) * front_update
            lateral_out[lateral_present] = side + torch.tanh(self.lateral_gate) * side_update
        return front_out, lateral_out


class FusionHeads(nn.Module):

    def __init__(self, image_dim=2048, text_dim=128, width=256):
        super().__init__()
        self.image_proj = nn.Linear(2 * image_dim, width)
        self.text_proj = nn.Linear(text_dim, width)
        self.text_null = nn.Parameter(torch.zeros(width))
        self.age_emb = nn.Embedding(11, width)
        self.sex_emb = nn.Embedding(3, width)
        self.combine = nn.Sequential(nn.LayerNorm(4 * width), nn.Linear(4 * width, width), nn.GELU())
        self.align_head = nn.Linear(width, 128)
        self.profile_head = nn.Linear(width, 14)
        nn.init.trunc_normal_(self.text_null, std=0.02)

    def forward(self, frontal, lateral, lateral_present, text_emb, text_mask, age_ids, sex_ids):
        lateral = torch.where(lateral_present[:, None], lateral, torch.zeros_like(lateral))
        image = self.image_proj(torch.cat([frontal, lateral], dim=1))
        text = torch.where(text_mask[:, None], self.text_proj(text_emb), self.text_null[None, :])
        features = self.combine(torch.cat([image, text, self.age_emb(age_ids), self.sex_emb(sex_ids)], dim=1))
        return F.normalize(self.align_head(features), dim=-1), self.profile_head(features)


def select_depth(ndcg, macro_ap, profile_ap, depths=(10, 20, 50, 100, 200)):
    ndcg, macro_ap, profile_ap = map(lambda x: np.asarray(x, dtype=np.float64), (ndcg, macro_ap, profile_ap))
    depths = np.asarray(depths, dtype=int)
    if ndcg.ndim != 2 or ndcg.shape[0] < 1 or ndcg.shape[1] != len(depths) or macro_ap.shape != ndcg.shape or profile_ap.shape != (ndcg.shape[0],):
        raise ValueError("Use one row per corpus and one column per depth.")
    if not all(np.isfinite(x).all() for x in (ndcg, macro_ap, profile_ap)):
        raise ValueError("Every validation mean must be finite.")
    if not all(((0 <= x) & (x <= 1)).all() for x in (ndcg, macro_ap, profile_ap)):
        raise ValueError("Use validation means on the original 0–1 scale.")
    eligible = np.flatnonzero(macro_ap.mean(axis=0) >= profile_ap.mean() - 0.01)
    if not len(eligible):
        return None
    order = sorted(eligible, key=lambda j: (-ndcg[:, j].mean(), depths[j]))
    return int(depths[order[0]])
