"""Prototypical Network: the simplest meta-learner.

Conv4 encoder -> embedding; class prototype = mean support embedding;
classify query by nearest prototype. The episode-split logic in `fast_adapt`
(sort by label, index-arithmetic split) is verified in tests/test_protonet.py.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class Conv4(nn.Module):
    """Standard 4-block convolutional few-shot encoder -> flat embedding."""

    def __init__(self, in_c: int = 1, hid: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(in_c, hid),
            conv_block(hid, hid),
            conv_block(hid, hid),
            conv_block(hid, hid),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


def fast_adapt(model, batch, ways: int, shot: int, query: int, device):
    """Process one episode; return (loss, accuracy).

    Sorts samples by label so each class is contiguous, then takes the first
    `shot` of each class block as support and the rest as query. Prototypes are
    the per-class mean of support embeddings; query is classified by nearest
    prototype (negative euclidean distance as logits).
    """
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    sort = torch.sort(labels, stable=True)
    data = data[sort.indices]
    labels = labels[sort.indices]

    emb = model(data)

    support_mask = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query)
    for offset in range(shot):
        support_mask[selection + offset] = True
    support_mask_t = torch.from_numpy(support_mask).to(device)
    query_mask_t = torch.from_numpy(~support_mask).to(device)

    prototypes = emb[support_mask_t].reshape(ways, shot, -1).mean(dim=1)
    q = emb[query_mask_t]
    q_labels = labels[query_mask_t].long()

    logits = -torch.cdist(q, prototypes)
    loss = F.cross_entropy(logits, q_labels)
    acc = (logits.argmax(1) == q_labels).float().mean()
    return loss, acc
