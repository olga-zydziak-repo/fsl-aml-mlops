#!/usr/bin/env python3
"""Encoder screening by successive halving — cheap comparison before the pipeline.

The question: which backbone is worth a full pipeline run? Training every
candidate to convergence answers it and wastes most of the compute on the ones
that were never going to win. Successive halving instead gives everyone a small
budget, drops the worst half, doubles the budget for the survivors, repeats.

    round 1: N candidates x B iters
    round 2: N/2 survivors x 2B iters
    round 3: N/4 survivors x 4B iters   ... until one remains

This is the same idea as Vizier's median-stopping, done explicitly over a
handful of named architectures instead of a continuous space. It leans on a
measured fact about this project: on RESISC45 the ranking is visible early
(val was already ~0.61 at 100-300 iterations and only crept to 0.68 by 1400),
so a short budget is a usable predictor of the long one.

WHY LOCAL. ResNet-12 has ~110x the parameters of Conv4, and a full CPU pipeline
run of Conv4 already takes ~2.75h. Screening belongs on the Workbench GPU; only
the WINNER goes through the pipeline, where it gets a from-scratch run, the
gates, and a registry version. Screening is exploration, the pipeline is
judgement — keeping them apart is the same split as HPO vs the gauntlet.

Selection uses VALIDATION accuracy. Test stays untouched until the pipeline.

Usage:
    python scripts/screen_encoders.py --project MY-PROJECT --dataset resisc45 \
        --augmentation flip_rot90 --candidates conv4 resnet12 --base-iters 300

    # include the pretrained backbone (downloads ImageNet weights on first use):
    python scripts/screen_encoders.py --project MY-PROJECT --dataset resisc45 \
        --augmentation flip_rot90 \
        --candidates conv4 resnet12 resnet18_pretrained --base-iters 300
"""
import argparse
import json
import time
from pathlib import Path


GEOMETRY_DEFAULTS = {
    "omniglot": (1, 28),
    "resisc45": (3, 84),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Successive-halving encoder screening.")
    ap.add_argument("--project", required=True)
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--dataset", default="resisc45")
    ap.add_argument("--candidates", nargs="+",
                    default=["conv4", "resnet12"],
                    help="encoder names to screen (see fsl/models/registry.py)")
    ap.add_argument("--base-iters", type=int, default=300,
                    help="training iterations in round 1; doubles each round")
    ap.add_argument("--val-episodes", type=int, default=300,
                    help="validation episodes used to rank each round")
    ap.add_argument("--augmentation", default="none",
                    choices=["none", "flip", "flip_rot90", "trivial"],
                    help="keep this FIXED at the winning policy — screening one "
                         "variable at a time is the whole point")
    ap.add_argument("--n-way", type=int, default=5)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--query", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embedding-hid", type=int, default=64,
                    help="applies to conv4 only; other backbones fix their width")
    ap.add_argument("--out", default="screening_result.json",
                    help="where to write the ranking (local file)")
    args = ap.parse_args()

    import numpy as np
    import torch

    from fsl.config import TrainConfig
    from fsl.data.registry import get_loaders
    from fsl.models.registry import known_encoders, preferred_image_size
    from fsl.training.loop import evaluate, set_seed, train

    for name in args.candidates:
        if name not in known_encoders():
            ap.error(f"unknown encoder {name!r}; known: {', '.join(known_encoders())}")

    in_channels, dataset_size = GEOMETRY_DEFAULTS.get(args.dataset, (None, None))
    if in_channels is None:
        ap.error(f"no geometry default for dataset {args.dataset!r}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"screening {len(args.candidates)} encoders on {args.dataset} "
          f"| aug={args.augmentation} | device={device.type}")
    if device.type == "cpu":
        print("WARNING: no GPU visible. ResNet-12 has ~110x Conv4's parameters — "
              "this will be slow. Consider running where a GPU is available.")

    survivors = list(args.candidates)
    iters = args.base_iters
    history: list[dict] = []
    rnd = 0

    while len(survivors) > 1:
        rnd += 1
        print(f"\n--- round {rnd}: {len(survivors)} candidates x {iters} iters ---")
        scored = []
        for name in survivors:
            image_size = preferred_image_size(name, dataset_size)
            cfg = TrainConfig(
                project=args.project, region=args.region, dataset=args.dataset,
                n_way=args.n_way, k_shot=args.k_shot, query=args.query,
                seed=args.seed, train_iters=iters, lr=args.lr,
                embedding_hid=args.embedding_hid, augmentation=args.augmentation,
                encoder=name, in_channels=in_channels, image_size=image_size,
                log_to_vertex=False,
            )
            set_seed(args.seed)
            started = time.time()
            result = train(cfg)
            model = result["model"]
            model.eval()

            load_frozen, build_task_samplers = get_loaders(cfg)
            bundle, _ = load_frozen(cfg)
            _, val_tasks, _ = build_task_samplers(bundle, cfg)
            accs = evaluate(model, val_tasks, cfg, args.val_episodes, device)
            val = float(np.mean(accs))
            elapsed = time.time() - started
            n_par = sum(p.numel() for p in model.parameters())
            scored.append((val, name))
            history.append({"round": rnd, "encoder": name, "iters": iters,
                            "image_size": image_size, "val_accuracy": val,
                            "seconds": round(elapsed, 1), "parameters": n_par})
            print(f"  {name:20s} val={val:.4f}  ({elapsed/60:.1f} min, "
                  f"{n_par/1e6:.2f}M param, {image_size}px)")

        scored.sort(reverse=True)
        # round the surviving half UP: with 3 candidates, floor division would
        # keep 1 and end the screening at the shortest budget - the survivors
        # would never be confirmed at a longer horizon, which is the point.
        keep = max(1, (len(scored) + 1) // 2) if len(scored) > 2 else 1
        dropped = [n for _, n in scored[keep:]]
        survivors = [n for _, n in scored[:keep]]
        if dropped:
            print(f"  dropped: {', '.join(dropped)}")
        iters *= 2

    winner = survivors[0]
    best = max((h for h in history if h["encoder"] == winner),
               key=lambda h: h["iters"])
    print(f"\nWINNER: {winner} "
          f"(val={best['val_accuracy']:.4f} at {best['iters']} iters, "
          f"{best['image_size']}px)")
    print("Config lines (training section):")
    print(f'  "encoder": "{winner}",')
    if winner != "conv4":
        print("  # note: embedding_hid is ignored by this encoder")
    print("\nNext: a FULL pipeline run with this encoder — the gates judge it on "
          "test, and only then does it earn a registry version.")

    Path(args.out).write_text(json.dumps(
        {"dataset": args.dataset, "augmentation": args.augmentation,
         "winner": winner, "history": history}, indent=2))
    print(f"ranking written to {args.out}")


if __name__ == "__main__":
    main()
