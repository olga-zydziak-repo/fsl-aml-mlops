#!/usr/bin/env python3
"""Frozen-encoder baseline: what does an UNTRAINED Conv4 score on this domain?

Why this exists. The gate needs an `accuracy_threshold`, and setting it by
looking at the trained model's numbers would be fitting the exam to the
student. The honest anchor is a floor computed independently of any training:
the SAME architecture, randomly initialised, frozen, evaluated on the SAME
episodes. A ProtoNet with a random encoder beats chance comfortably (random
projections preserve some structure), so this floor is a much stricter
opponent than 1/n_way — and "the trained model must clear the frozen floor by
a stated margin" is a rule you can defend before seeing any test number.

Methodology guard: this evaluates on the VALIDATION split, never on test.
Validation is already burned (we watched it during training); test stays
untouched until the pipeline's gate. The floor transfers: frozen-encoder
accuracy differs little between two disjoint class sets drawn from the same
distribution, and the margin dominates anyway.

Usage:
    python scripts/baseline_frozen.py --project MY-PROJECT --dataset resisc45
    # multiple seeds for a stabler floor (recommended):
    python scripts/baseline_frozen.py --project MY-PROJECT --dataset resisc45 --seeds 0 1 2

Prints per-seed accuracy, the pooled floor, and a PROPOSED threshold =
floor + margin. The margin (default 10pp) is deliberately a priori — decided
before any test number exists — and recorded in the config next to the value.
"""
import argparse

import numpy as np
import torch

from fsl.config import TrainConfig
from fsl.data.registry import get_loaders
from fsl.models.protonet import Conv4
from fsl.training.loop import evaluate, set_seed

GEOMETRY_DEFAULTS = {
    "omniglot": (1, 28),
    "resisc45": (3, 84),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen (untrained) encoder floor.")
    ap.add_argument("--project", required=True)
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--dataset", default="resisc45")
    ap.add_argument("--n-way", type=int, default=5)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--query", type=int, default=15)
    ap.add_argument("--embedding-hid", type=int, default=64)
    ap.add_argument("--episodes", type=int, default=400,
                    help="validation episodes per seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="one frozen encoder per seed; the floor pools them")
    ap.add_argument("--margin", type=float, default=0.10,
                    help="a-priori margin added to the floor for the proposed "
                         "accuracy_threshold (fraction, not pp)")
    ap.add_argument("--in-channels", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=None)
    args = ap.parse_args()

    default_c, default_size = GEOMETRY_DEFAULTS.get(args.dataset, (None, None))
    in_channels = args.in_channels if args.in_channels is not None else default_c
    image_size = args.image_size if args.image_size is not None else default_size
    if in_channels is None or image_size is None:
        ap.error(f"no geometry default for {args.dataset!r} - pass --in-channels/--image-size")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"frozen-encoder floor | dataset={args.dataset} | "
          f"{in_channels}x{image_size}x{image_size} | {args.n_way}-way {args.k_shot}-shot | "
          f"{args.episodes} val episodes x {len(args.seeds)} seeds | device={device.type}")

    per_seed = []
    all_accs = []
    for seed in args.seeds:
        cfg = TrainConfig(
            project=args.project, region=args.region, dataset=args.dataset,
            n_way=args.n_way, k_shot=args.k_shot, query=args.query, seed=seed,
            in_channels=in_channels, image_size=image_size,
            embedding_hid=args.embedding_hid, log_to_vertex=False,
        )
        set_seed(seed)
        load_frozen, build_task_samplers = get_loaders(cfg)
        bundle, _ = load_frozen(cfg)
        _, val_tasks, _ = build_task_samplers(bundle, cfg)

        model = Conv4(in_c=cfg.in_channels, hid=cfg.embedding_hid).to(device)
        # frozen = evaluated exactly as initialised; evaluate() sets eval mode
        accs = evaluate(model, val_tasks, cfg, args.episodes, device)
        per_seed.append((seed, float(np.mean(accs)), float(np.std(accs))))
        all_accs.append(accs)
        print(f"  seed {seed}: val acc {per_seed[-1][1]:.4f} "
              f"(std between episodes {per_seed[-1][2]:.4f})")

    pooled = np.concatenate(all_accs)
    floor_mean = float(np.mean(pooled))
    floor_std = float(np.std(pooled))
    n = len(pooled)
    ci95 = 1.96 * floor_std / np.sqrt(n)
    proposed = floor_mean + args.margin

    print()
    print(f"FROZEN FLOOR: {floor_mean:.4f} +/- {ci95:.4f} (ci95, {n} episodes pooled)")
    print(f"  episode std at the floor: {floor_std:.4f}")
    print(f"  spread across seeds: "
          f"{max(m for _, m, _ in per_seed) - min(m for _, m, _ in per_seed):.4f}")
    print()
    print(f"PROPOSED accuracy_threshold = floor + margin({args.margin:.2f}) "
          f"= {proposed:.4f}")
    print("Config lines (evaluation section):")
    print(f'  "accuracy_threshold": {round(proposed, 3)},')
    print('  "_comment": "threshold = frozen-encoder floor '
          f'({floor_mean:.3f} on val, seeds {args.seeds}) + a-priori margin '
          f'{args.margin:.2f}; set before any test number was seen"')


if __name__ == "__main__":
    main()
