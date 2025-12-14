from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def make_mlp(input_dim: int, hidden: List[int], dropout: float = 0.1) -> nn.Sequential:
    layers = []
    dims = [input_dim] + hidden
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(dims[-1], 1))
    return nn.Sequential(*layers)


class TabularNet(nn.Module):
    def __init__(self, cat_cardinalities: List[int], embed_dim: int, num_features: int, hidden: List[int]):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(card, embed_dim) for card in cat_cardinalities])
        self.num_bn = nn.BatchNorm1d(num_features) if num_features > 0 else nn.Identity()
        input_dim = embed_dim * len(cat_cardinalities) + num_features
        self.mlp = make_mlp(input_dim, hidden)

    def forward(self, cats: torch.Tensor, nums: torch.Tensor):
        embed_outs = [emb(cats[:, i]) for i, emb in enumerate(self.embeds)]
        if embed_outs:
            cat_vec = torch.cat(embed_outs, dim=1)
        else:
            cat_vec = torch.zeros((nums.size(0), 0), device=nums.device)
        num_vec = self.num_bn(nums) if nums.numel() > 0 else torch.zeros((cats.size(0), 0), device=cats.device)
        x = torch.cat([cat_vec, num_vec], dim=1)
        return self.mlp(x).squeeze(-1)


@dataclass
class TorchArtifacts:
    model: TabularNet
    cat_cols: List[str]
    num_cols: List[str]
    cat_maps: Dict[str, Dict[str, int]]
    num_mean: Dict[str, float]
    num_std: Dict[str, float]
