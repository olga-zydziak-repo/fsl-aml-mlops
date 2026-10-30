"""Encoder dispatch — one place that knows which backbone a run should build.

Same pattern as `fsl/data/registry.py`: the training loop asks, the registry
answers based on `cfg.encoder`. Adding a backbone is one module plus one entry;
nothing else in the system changes.

Every encoder must honour the Conv4 contract:

    forward(x: (B, C, H, W)) -> (B, D)   flat embedding

`fast_adapt` (prototypes = mean of support embeddings, nearest-prototype
classification) is architecture-agnostic and needs nothing else. Tabular
loaders hand over `(B, F)` instead of `(B, C, H, W)`; `mlp` flattens after the
batch dimension, so both shapes work and the loop stays unaware.

INPUT SIZE. Encoders disagree about what they want to be fed, so the size is a
property of the (dataset, encoder) PAIR, not of the dataset alone:
  - conv4 / resnet12: whatever the dataset's manifest says (84 for RESISC45)
  - resnet18_pretrained: 224, the size its ImageNet weights were trained at
  - mlp: 1 — a feature vector has no spatial extent; the tabular geometry
    convention is (n_features, 1), so nothing must try to resize it
`preferred_image_size` resolves this; an explicit `cfg.image_size` from the
config always wins, so you can trade accuracy for speed deliberately.
"""
from __future__ import annotations

# encoder -> input size it wants, or None = "use the dataset's manifest size".
_PREFERRED_SIZE: dict[str, int | None] = {
    "conv4": None,
    "resnet12": None,
    "resnet18_pretrained": 224,
    "mlp": 1,
}


def known_encoders() -> list[str]:
    """Names accepted by `get_encoder`, for error messages and validation."""
    return sorted(_PREFERRED_SIZE)


def preferred_image_size(encoder: str, dataset_size: int) -> int:
    """Input side length for this encoder, given the dataset's own size.

    Args:
        encoder: encoder name.
        dataset_size: `train_size` from the frozen manifest (1 for tabular).

    Returns:
        The side length to resize to.
    """
    want = _PREFERRED_SIZE.get(encoder)
    return dataset_size if want is None else want


def get_encoder(cfg):
    """Build the encoder named by `cfg.encoder`.

    Args:
        cfg: a TrainConfig — reads `encoder`, `in_channels`, `embedding_hid`,
            and, for `mlp`, the optional `mlp_depth` / `mlp_dropout`.

    Returns:
        An `nn.Module` mapping (B, C, H, W) -> (B, D), or (B, F) -> (B, D) for
        tabular encoders.

    Raises:
        ValueError: if `cfg.encoder` is not registered. The message lists what is.
    """
    name = str(getattr(cfg, "encoder", "conv4")).strip().lower()

    if name == "conv4":
        from fsl.models.protonet import Conv4

        return Conv4(in_c=cfg.in_channels, hid=cfg.embedding_hid)

    if name == "resnet12":
        from fsl.models.resnet12 import ResNet12

        # embedding_hid is intentionally ignored: ResNet-12's width schedule is
        # part of the architecture, not a tunable knob.
        return ResNet12(in_c=cfg.in_channels)

    if name == "resnet18_pretrained":
        return _resnet18_pretrained(cfg.in_channels)

    if name == "mlp":
        from fsl.models.mlp import MLP

        # in_channels carries n_features under the tabular geometry convention
        # (see fsl/data/manifest.py): (n_features, 1).
        return MLP(
            in_features=cfg.in_channels,
            hid=cfg.embedding_hid,
            depth=int(getattr(cfg, "mlp_depth", 2)),
            dropout=float(getattr(cfg, "mlp_dropout", 0.0)),
        )

    raise ValueError(
        f"no encoder registered under {name!r}. Known: {', '.join(known_encoders())}. "
        "To add one, write fsl/models/<name>.py exposing an nn.Module with "
        "forward(x) -> (B, D), then register it in fsl/models/registry.py."
    )


def _resnet18_pretrained(in_c: int):
    """ImageNet-pretrained ResNet-18 with the classifier removed -> 512-d.

    The whole network is fine-tuned (the training loop optimises
    `model.parameters()`), so a lower learning rate than a from-scratch run is
    usually the right call — pretrained features are easy to destroy.

    Grayscale input: the pretrained first conv expects 3 channels, so its
    weights are SUMMED across the channel dimension into a 1-channel kernel.
    That keeps the pretrained filters meaningful instead of reinitialising the
    first layer at random.

    NOTE: the weights are downloaded from download.pytorch.org on first use and
    cached in TORCH_HOME. Inside a training container that download happens on
    every run unless the image pre-caches them at build time or the cache dir
    is mounted — worth doing before this becomes a routine backbone.
    """
    import torch
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    if in_c != 3:
        old = model.conv1
        new = nn.Conv2d(in_c, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=False)
        with torch.no_grad():
            summed = old.weight.sum(dim=1, keepdim=True)          # (64,1,7,7)
            new.weight.copy_(summed.repeat(1, in_c, 1, 1) / in_c)
        model.conv1 = new

    model.fc = nn.Identity()      # (B, 512) embedding instead of 1000 logits
    return model