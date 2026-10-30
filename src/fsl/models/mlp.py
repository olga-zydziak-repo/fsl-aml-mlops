"""MLP encoder — the tabular counterpart of Conv4.

Honours the same contract as every other backbone in `fsl/models/registry.py`:

    forward(x) -> (B, D)   flat embedding

with one accommodation: tabular loaders hand over `(B, F)` feature vectors
rather than `(B, C, H, W)` images, so `forward` flattens everything after the
batch dimension. A `(B, F)` tensor passes through untouched; a `(B, F, 1, 1)`
tensor (should anything ever reshape to the image convention) collapses to the
same thing.

Why so small: on tabular inputs the encoder is not there to extract features —
the features are the input, already engineered at freeze time. Its only job is
to learn a metric in which windows of the same typology sit close together. A
deep network has nothing extra to see and, on classes with a few hundred
windows, plenty to overfit. The floor to beat is a RANDOM projection of these
same features (Johnson-Lindenstrauss largely preserves the input geometry), so
depth that does not buy separation buys nothing at all.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Two hidden layers -> `hid`-dimensional embedding.

    Args:
        in_features: width of the input vector (n_features from the manifest).
        hid: width of the hidden layers and of the embedding.
        depth: number of hidden layers (2 by default).
        dropout: dropout probability between hidden layers; 0 disables it.
    """

    def __init__(self, in_features: int, hid: int = 64, depth: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        if in_features < 1:
            raise ValueError(f"in_features must be >= 1, got {in_features}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        layers: list[nn.Module] = []
        width = in_features
        for _ in range(depth):
            layers += [nn.Linear(width, hid), nn.LayerNorm(hid), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = hid
        layers.append(nn.Linear(width, hid))   # embedding head, no activation
        self.net = nn.Sequential(*layers)
        self.in_features = in_features
        self.embedding_dim = hid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))
