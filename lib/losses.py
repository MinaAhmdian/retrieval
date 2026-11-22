"""
lib/losses.py
Created on October 17, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import torch
import torch.nn.functional as F


def alignment_loss(z, r, report_keys, temperature=0.07):
    logits = z @ r.T / temperature
    keys = report_keys
    same = torch.tensor([[a == b for b in keys] for a in keys], device=z.device)
    eye = torch.eye(len(keys), dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(same & ~eye, float("-inf"))
    target = torch.arange(len(keys), device=z.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def profile_loss(logits, labels):
    labels = labels.float()
    keep = labels >= 0
    if keep.sum() == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[keep], labels[keep])
