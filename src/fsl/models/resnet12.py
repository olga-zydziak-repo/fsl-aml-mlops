"""ResNet-12: the standard heavier backbone in few-shot literature.

Four residual stages (64 -> 160 -> 320 -> 640 channels), each three 3x3 convs
with a 1x1 projection shortcut, LeakyReLU(0.1), and a 2x2 max-pool at the end
of the stage. A global average pool turns the final map into a 640-d embedding,
so the embedding width does NOT depend on the input size - unlike Conv4, whose
flattened output scales with image_size and blows up on large inputs.

Same contract as Conv4: `forward(x) -> (B, D)` flat embedding, so `fast_adapt`
and everything downstream is unchanged.

Why this one: Conv4 is the classic minimal baseline; ResNet-12 is what most
few-shot papers use when they want a real number rather than a fast one. It is
trained from scratch here - no pretrained weights, so the "learn a comparison
metric from the base classes alone" claim stays intact.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    """Three 3x3 convs + projection shortcut + LeakyReLU, then 2x2 max-pool."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv3 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.relu = nn.LeakyReLU(0.1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.relu(out + residual)
        return self.pool(out)


class ResNet12(nn.Module):
    """ResNet-12 encoder -> 640-d embedding (global average pooled).

    Args:
        in_c: input channels (1 grayscale, 3 RGB).
        widths: channel width per stage. The default is the literature standard;
            `hid` is deliberately NOT a knob here - the width schedule is part of
            the architecture's identity.
    """

    def __init__(self, in_c: int = 3,
                 widths: tuple[int, int, int, int] = (64, 160, 320, 640)) -> None:
        super().__init__()
        stages = []
        prev = in_c
        for width in widths:
            stages.append(_ResBlock(prev, width))
            prev = width
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = widths[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.stages(x)).flatten(1)
