#!/usr/bin/env python3
"""HPO (Vizier) entry point - one trial = one run of this script.

Vertex AI Hyperparameter Tuning launches this container once per trial,
passing the trial's hyperparameters as CLI args (the arg names below MUST
match the keys of parameter_spec in the tuning job). The script:

  1. trains with the trial's hyperparameters (same fsl core as everything else),
  2. computes the FINAL VALIDATION accuracy on the returned model
     (same seed -> build_task_samplers rebuilds the same val split;
      no changes to fsl needed - the wrapper evaluates the returned model),
  3. reports val_accuracy to Vizier via cloudml-hypertune.

METHODOLOGY (why validation, not test): Vizier optimizes what we report. If it
saw test accuracy, hyperparameters would be selected ON the test set and the
test would stop being a test (selection leakage). The winning config goes
through the normal pipeline afterwards, where the gates judge it on episodes
the tuning never optimized against.

Requires `cloudml-hypertune` in the image (added to pyproject.toml).
"""
import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description="One HPO trial: train + report val accuracy.")
    # fixed context
    ap.add_argument("--project", required=True)
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--bucket", default="")
    ap.add_argument("--dataset", default="omniglot")
    ap.add_argument("--n-way", type=int, default=5)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--query", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augmentation", default="none",
                    choices=["none", "flip", "flip_rot90", "trivial"])
    ap.add_argument("--val-episodes", type=int, default=400,
                    help="final validation episodes (Vizier's signal; more = less noise)")
    # --- tuned hyperparameters ---
    # UWAGA: Vizier przekazuje je jako --<parameter_id>=<value>, gdzie parameter_id
    # pochodzi z parameter_spec (podkreslniki!). Aliasy przyjmuja OBA formaty.
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--embedding-hid", "--embedding_hid", dest="embedding_hid",
                    type=int, required=True)
    ap.add_argument("--train-iters", "--train_iters", dest="train_iters",
                    type=int, required=True)
    args = ap.parse_args()

    import numpy as np
    import torch
    import hypertune

    from fsl.config import TrainConfig
    from fsl.data.manifest import fetch_manifest, get_geometry
    from fsl.data.registry import get_loaders
    from fsl.training.loop import evaluate, train

    bucket = args.bucket or f"{args.project}-fsl-data"
    manifest = fetch_manifest(args.project, bucket, args.dataset)
    in_channels, image_size = get_geometry(manifest, args.dataset)
    cfg = TrainConfig(
        project=args.project, region=args.region, bucket=bucket, dataset=args.dataset,
        n_way=args.n_way, k_shot=args.k_shot, query=args.query, seed=args.seed,
        lr=args.lr, embedding_hid=args.embedding_hid, train_iters=args.train_iters,
        augmentation=args.augmentation,
        in_channels=in_channels, image_size=image_size,
        log_to_vertex=False,   # Vizier tracks trials; Experiments logging stays off here
    )

    result = train(cfg)
    model = result["model"]
    model.eval()
    device = next(model.parameters()).device

    # final validation on the SAME split (same seed -> same class partition)
    load_frozen, build_task_samplers = get_loaders(cfg)
    dataset, _ = load_frozen(cfg)
    _, val_tasks, _ = build_task_samplers(dataset, cfg)
    val_accs = evaluate(model, val_tasks, cfg, args.val_episodes, device)
    val_mean = float(np.mean(val_accs))
    print(f"TRIAL RESULT: val_accuracy={val_mean:.4f} "
          f"(lr={args.lr}, hid={args.embedding_hid}, iters={args.train_iters})")

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="val_accuracy",
        metric_value=val_mean,
        global_step=args.train_iters,
    )
    print("Reported to Vizier.")


if __name__ == "__main__":
    main()
