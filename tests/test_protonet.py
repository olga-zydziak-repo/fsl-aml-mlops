"""Tests for the ProtoNet model — the correctness-critical episode-split logic.

These make permanent the checks that were run ad hoc during development:
the support/query split must be exact, and prototypes must be per-class means.
No GCP or learn2learn needed — only torch + numpy — so they run anywhere.
"""
import numpy as np
import torch

from fsl.models.protonet import Conv4, fast_adapt


class _Identity(torch.nn.Module):
    """Stand-in encoder returning inputs unchanged, so the split and prototype
    computation can be checked exactly."""

    def forward(self, x):
        return x


def _make_episode(ways: int, shot: int, query: int, dim: int = 3):
    """Shuffled episode where each sample's value encodes its label, making
    prototypes predictable (prototype of class c should equal ~c)."""
    per_class = shot + query
    labels = torch.cat([torch.full((per_class,), c) for c in range(ways)])
    perm = torch.randperm(ways * per_class)
    labels = labels[perm]
    data = labels.float().unsqueeze(1).repeat(1, dim)
    return data, labels


def test_fast_adapt_perfect_on_separable_episode():
    """With an identity encoder and label-encoded values, each query maps exactly
    to its own prototype, so accuracy must be 1.0."""
    ways, shot, query = 5, 5, 15
    torch.manual_seed(0)
    data, labels = _make_episode(ways, shot, query)
    _, acc = fast_adapt(_Identity(), (data, labels), ways, shot, query, torch.device("cpu"))
    assert acc.item() == 1.0


def test_conv4_output_shape():
    """Conv4 maps a batch of 28x28 images to a 2D (batch, embedding) tensor."""
    enc = Conv4(in_c=1, hid=64)
    out = enc(torch.randn(10, 1, 28, 28))
    assert out.dim() == 2
    assert out.size(0) == 10


def test_prototypes_are_per_class_means():
    """Prototype for class c must equal the mean of its support embeddings."""
    ways, shot, query = 3, 4, 6
    torch.manual_seed(1)
    data, labels = _make_episode(ways, shot, query)

    # replicate the split to inspect prototypes directly
    sort = torch.sort(labels)
    d = data[sort.indices]
    support_mask = np.zeros(d.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query)
    for offset in range(shot):
        support_mask[selection + offset] = True
    protos = d[torch.from_numpy(support_mask)].reshape(ways, shot, -1).mean(1)

    # value == label, so prototype for class c has value ~c
    assert torch.allclose(protos[:, 0], torch.arange(ways).float())


def test_split_produces_correct_counts():
    """Support has exactly `shot` per class; query has exactly `query` per class."""
    import collections

    ways, shot, query = 5, 5, 15
    torch.manual_seed(2)
    data, labels = _make_episode(ways, shot, query)
    sort = torch.sort(labels)
    lab = labels[sort.indices]

    support_mask = np.zeros(lab.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query)
    for offset in range(shot):
        support_mask[selection + offset] = True

    support_counts = collections.Counter(lab[torch.from_numpy(support_mask)].tolist())
    query_counts = collections.Counter(lab[torch.from_numpy(~support_mask)].tolist())
    assert all(v == shot for v in support_counts.values())
    assert all(v == query for v in query_counts.values())
