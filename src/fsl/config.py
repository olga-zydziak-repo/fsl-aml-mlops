"""Training configuration as a single typed object.

Replaces the notebook's config cell: instead of editing module-level constants,
you construct a TrainConfig (or build one from CLI args). Every run is fully
described by its config.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # --- GCP / data ---
    project: str = ""
    region: str = "us-central1"
    bucket: str = ""                       # derived from project if left empty
    dataset: str = "omniglot"
    cache_dir: str = ""                    # derived from dataset if left empty

    # --- input geometry (the encoder and the loader must agree) ---
    # IMAGE domains: Conv4 has four max-pools, so image_size decides the
    # embedding width: 28 -> 1x1 (Omniglot), 84 -> 5x5 (mini-ImageNet
    # convention, used for RESISC45), 256 -> 16x16. in_channels is 1 for
    # grayscale, 3 for RGB.
    # TABULAR domains: the same two fields carry (n_features, 1) — the feature
    # vector is the input and there is no spatial extent. SAML-D freezes 26
    # window features, so (26, 1). Both values come from the frozen manifest
    # via fsl/data/manifest.get_geometry; these defaults only matter for CLI
    # runs that skip the manifest.
    in_channels: int = 1
    image_size: int = 28

    # --- class fold (tabular domains only) ---
    # SAML-D has 27 classes, not enough for one fixed 25/10/10 split, so the
    # freeze defines 5 class-disjoint folds (17 train / 5 val / 5 test, each
    # pool 3 suspicious + 2 normal). The loader reads this; image domains
    # ignore it. It is logged so a run's split is never ambiguous afterwards.
    fold: int = 0

    # --- encoder (see fsl/models/registry.py) ---
    # "conv4" (baseline, embedding_hid applies) | "resnet12" (from scratch,
    # width schedule fixed) | "resnet18_pretrained" (ImageNet, wants 224px) |
    # "mlp" (tabular: in_channels = n_features, embedding_hid = width)
    encoder: str = "conv4"

    # --- mlp encoder knobs (ignored by the image backbones) ---
    mlp_depth: int = 2
    mlp_dropout: float = 0.0

    # --- data augmentation policy (TRAIN split only; val/test stay clean) ---
    # "none" | "flip" (H+V mirror) | "flip_rot90" (mirrors + 90-degree turns -
    # the natural symmetries of overhead imagery) | "trivial" (H-flip +
    # TrivialAugmentWide, the off-the-shelf benchmark policy)
    # Tabular runs keep "none": there is no symmetry of a feature vector that
    # leaves its typology unchanged.
    augmentation: str = "none"

    # --- episode (few-shot is defined by these) ---
    # query=15 is the image-benchmark convention. Tabular SAML-D uses query=5:
    # with account-disjoint sampling a class needs ~2*(k_shot+query) windows on
    # separate accounts, and the smallest typologies hold ~25 windows, so 15
    # would drop them from the pool and a 5-class pool could not form an
    # episode at all.
    n_way: int = 5
    k_shot: int = 5
    query: int = 15

    # --- training ---
    seed: int = 0
    train_iters: int = 300
    eval_every: int = 50
    val_episodes: int = 200
    test_episodes: int = 1000
    lr: float = 1e-3
    embedding_hid: int = 64

    # --- experiment tracking ---
    experiment_name: str = "fsl-omniglot-protonet"
    log_to_vertex: bool = True

    def __post_init__(self) -> None:
        if not self.bucket and self.project:
            self.bucket = f"{self.project}-fsl-data"
        if not self.cache_dir:
            self.cache_dir = os.path.expanduser(f"~/data_cache/{self.dataset}")

    def as_params(self) -> dict:
        """Flat parameter dict for experiment logging (provenance added by caller).

        Encoder-specific knobs are included only for the encoder that uses
        them, so a run's logged parameters stay comparable with earlier runs of
        the same backbone.
        """
        params = {
            "model": "protonet",
            "embedding_hid": self.embedding_hid,
            "distance": "euclidean",
            "n_way": self.n_way,
            "k_shot": self.k_shot,
            "query": self.query,
            "seed": self.seed,
            "train_iters": self.train_iters,
            "lr": self.lr,
            "optimizer": "adam",
            "eval_episodes": self.test_episodes,
            "dataset": self.dataset,
            "fold": self.fold,
            "in_channels": self.in_channels,
            "image_size": self.image_size,
            "encoder": self.encoder,
            "augmentation": self.augmentation,
        }
        if str(self.encoder).strip().lower() == "mlp":
            params["mlp_depth"] = self.mlp_depth
            params["mlp_dropout"] = self.mlp_dropout
        return params