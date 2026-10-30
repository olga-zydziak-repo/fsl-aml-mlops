#!/usr/bin/env python3
"""Entry point: train ProtoNet on a frozen dataset, logging to Vertex Experiments.

This is the inner loop - the fast, local way to answer "does this actually
train?" without going anywhere near the pipeline. It is also the cheapest place
to smoke-test a NEW domain right after its loader is written.

Usage:
    python scripts/train.py --project MY-PROJECT
    python scripts/train.py --project MY-PROJECT --dataset resisc45
    python scripts/train.py --project MY-PROJECT --no-vertex          # local, no logging

    # quick smoke test of a new loader (minutes, not hours):
    python scripts/train.py --project MY-PROJECT --dataset resisc45 \
        --train-iters 60 --eval-every 20 --val-episodes 40 --test-episodes 100 \
        --no-vertex

Input geometry (--in-channels / --image-size) defaults per dataset, because the
encoder and the loader must agree on it: Conv4 has four max-pools, so the image
size decides the embedding width (28 -> 1x1 for Omniglot, 84 -> 5x5 for RESISC45,
following the mini-ImageNet convention). Override the defaults only if you know
why.
"""
import argparse

from fsl.config import TrainConfig
from fsl.training.loop import train

# dataset -> (in_channels, image_size). One entry per registered loader; keep in
# sync with fsl/data/registry.py. A dataset absent from here just means you must
# pass --in-channels / --image-size explicitly.
GEOMETRY_DEFAULTS = {
    "omniglot": (1, 28),
    "resisc45": (3, 84),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ProtoNet on a frozen dataset.")
    ap.add_argument("--project", required=True, help="GCP project ID")
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--dataset", default="omniglot",
                    help="which frozen dataset to episode over (see fsl/data/registry.py)")
    ap.add_argument("--n-way", type=int, default=5)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--query", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=None,
                    help="learning rate (default: TrainConfig's)")
    ap.add_argument("--embedding-hid", type=int, default=None,
                    help="encoder width (default: TrainConfig's)")
    # episode budgets - lower them for a fast smoke test
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--val-episodes", type=int, default=None)
    ap.add_argument("--test-episodes", type=int, default=None)
    # input geometry - defaults per dataset, override only deliberately
    ap.add_argument("--in-channels", type=int, default=None,
                    help="1 for grayscale, 3 for RGB (default: per dataset)")
    ap.add_argument("--image-size", type=int, default=None,
                    help="side length fed to the encoder (default: per dataset)")
    ap.add_argument("--encoder", default="conv4",
                    help="backbone (see fsl/models/registry.py): conv4 | resnet12 | "
                         "resnet18_pretrained")
    ap.add_argument("--augmentation", default="none",
                    choices=["none", "flip", "flip_rot90", "trivial"],
                    help="train-split augmentation policy (val/test always clean)")
    ap.add_argument("--no-vertex", action="store_true",
                    help="Disable Vertex Experiments logging (local run).")
    args = ap.parse_args()

    from fsl.models.registry import preferred_image_size

    default_c, default_size = GEOMETRY_DEFAULTS.get(args.dataset, (None, None))
    in_channels = args.in_channels if args.in_channels is not None else default_c
    # input size is a property of the (dataset, encoder) PAIR: the dataset
    # proposes, the encoder may override (pretrained weights want their own
    # size), and an explicit --image-size always wins.
    if args.image_size is not None:
        image_size = args.image_size
    elif default_size is not None:
        image_size = preferred_image_size(args.encoder, default_size)
    else:
        image_size = None
    if in_channels is None or image_size is None:
        ap.error(
            f"no geometry default for dataset {args.dataset!r} - "
            "pass --in-channels and --image-size explicitly"
        )

    # Only override TrainConfig's defaults for flags actually given, so the
    # dataclass stays the single source of truth for everything unspecified.
    optional = {
        "lr": args.lr,
        "embedding_hid": args.embedding_hid,
        "eval_every": args.eval_every,
        "val_episodes": args.val_episodes,
        "test_episodes": args.test_episodes,
    }
    cfg = TrainConfig(
        project=args.project,
        region=args.region,
        dataset=args.dataset,
        n_way=args.n_way,
        k_shot=args.k_shot,
        query=args.query,
        seed=args.seed,
        train_iters=args.train_iters,
        augmentation=args.augmentation,
        encoder=args.encoder,
        in_channels=in_channels,
        image_size=image_size,
        log_to_vertex=not args.no_vertex,
        **{k: v for k, v in optional.items() if v is not None},
    )
    print(f"dataset={cfg.dataset} | encoder={cfg.encoder} | "
          f"{cfg.in_channels}x{cfg.image_size}x{cfg.image_size} "
          f"| {cfg.n_way}-way {cfg.k_shot}-shot | aug={cfg.augmentation} "
          f"| {cfg.train_iters} iters")
    train(cfg)


if __name__ == "__main__":
    main()
