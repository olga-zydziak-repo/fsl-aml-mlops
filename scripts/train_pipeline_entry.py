#!/usr/bin/env python3
"""Pipeline entry point for training inside a container component.

This is the CONTAINER-mode counterpart to scripts/train.py. Both import the
SAME core (`fsl.training.loop.train`) - zero duplicated model logic. The
difference is how results leave:
  - scripts/train.py (inner loop): train() logs to Vertex Experiments, done.
  - this script (pipeline): train() returns a dict; we save the model to GCS
    and write metric values to the KFP output paths so the gate/register
    components downstream can consume them.

Order matters here: the frozen MANIFEST is fetched FIRST, because it is the
single source of truth for the data's input geometry (channels/train size for
images, n_features for tabular domains) - TrainConfig needs those before the
model is built. The same fetch also yields the archive sha for provenance, so
nothing is read twice.

architecture.json is the CONTRACT between this run and everything downstream
that rebuilds the model (heavy eval, explainability, the final HPO validation).
Whatever those need to reconstruct the exact network has to be written here:
the backbone name included. A hardcoded architecture label is worse than none,
because it reads as provenance while being a guess.

Usage (KFP fills the --*-output-path args via OutputPath):
    python scripts/train_pipeline_entry.py \
        --project P --bucket B --dataset resisc45 --encoder conv4 \
        --n-way 5 --k-shot 5 --query 15 --seed 0 --train-iters 1400 \
        --embedding-hid 64 --lr 0.001 \
        --accuracy-mean-output-path /path --accuracy-ci95-output-path /path \
        --accuracy-std-output-path /path --train-time-output-path /path \
        --model-dir-output-path /path --data-sha-output-path /path
"""
import argparse
import json
import time
import uuid
from pathlib import Path


def _write(path: str, value) -> None:
    """Write a single output value to a KFP output path (creates parent dirs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(value))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ProtoNet inside a pipeline container.")
    # training params (mirror scripts/train.py, forwarded into TrainConfig)
    ap.add_argument("--project", required=True)
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--bucket", default="")
    ap.add_argument("--dataset", default="omniglot")
    ap.add_argument("--encoder", default="conv4",
                    help="backbone name; see fsl/models/registry.py "
                         "(conv4 | resnet12 | resnet18_pretrained | mlp)")
    ap.add_argument("--mlp-depth", type=int, default=2,
                    help="hidden layers, mlp encoder only")
    ap.add_argument("--mlp-dropout", type=float, default=0.0,
                    help="dropout between hidden layers, mlp encoder only")
    ap.add_argument("--fold", type=int, default=0,
                    help="class fold for tabular datasets; image domains ignore it")
    ap.add_argument("--n-way", type=int, default=5)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--query", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-iters", type=int, default=300)
    ap.add_argument("--embedding-hid", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--augmentation", default="none",
                    choices=["none", "flip", "flip_rot90", "trivial"])
    # in-pipeline runs don't log to Experiments from here (a dedicated log
    # component does that downstream); keep training pure.
    ap.add_argument("--no-vertex", action="store_true", default=True)
    # KFP output paths (filled by OutputPath in the container component)
    ap.add_argument("--accuracy-mean-output-path", required=True)
    ap.add_argument("--accuracy-ci95-output-path", required=True)
    ap.add_argument("--accuracy-std-output-path", required=True)
    ap.add_argument("--train-time-output-path", required=True)
    ap.add_argument("--model-dir-output-path", required=True)
    ap.add_argument("--data-sha-output-path", required=True)
    args = ap.parse_args()

    # import the SAME core the inner loop uses - no duplicated model logic
    import torch
    from google.cloud import storage

    from fsl.config import TrainConfig
    from fsl.data.manifest import get_geometry
    from fsl.errors import explain_failure
    from fsl.models.registry import known_encoders
    from fsl.training.loop import train

    encoder = str(args.encoder).strip().lower()
    if encoder not in known_encoders():
        raise SystemExit(
            f"unknown encoder {args.encoder!r}. Known: {', '.join(known_encoders())}."
        )

    bucket = args.bucket or f"{args.project}-fsl-data"

    # --- manifest FIRST: geometry (for the model) + sha (for provenance) ---
    client = storage.Client(project=args.project)
    gcs = client.bucket(bucket)
    with explain_failure("reading dataset manifest from GCS", bucket=bucket):
        manifest = json.loads(
            gcs.blob(f"raw/{args.dataset}/MANIFEST.json").download_as_bytes().decode("utf-8")
        )
    archive_sha = manifest["archive_sha256"]
    # geometry flows from the frozen manifest - the single source of truth
    # about the data's shape (config/CLI no longer carry it for the pipeline).
    # Tabular domains return (n_features, 1); nothing here needs to know which.
    in_channels, image_size = get_geometry(manifest, args.dataset)
    print(f"geometry from manifest: in_channels={in_channels}, image_size={image_size}")

    cfg = TrainConfig(
        project=args.project, region=args.region, bucket=bucket, dataset=args.dataset,
        n_way=args.n_way, k_shot=args.k_shot, query=args.query, seed=args.seed,
        train_iters=args.train_iters, embedding_hid=args.embedding_hid, lr=args.lr,
        augmentation=args.augmentation,
        encoder=encoder, mlp_depth=args.mlp_depth, mlp_dropout=args.mlp_dropout,
        fold=args.fold,
        in_channels=in_channels, image_size=image_size,
        log_to_vertex=False,   # pipeline logs via a separate component, not here
    )
    print(f"encoder={cfg.encoder} | dataset={cfg.dataset} fold={cfg.fold} | "
          f"{cfg.n_way}-way {cfg.k_shot}-shot, query={cfg.query}")

    # train() returns metrics + the model object (see fsl/training/loop.py)
    result = train(cfg)

    # --- persist the model to GCS (the piece loop.py doesn't do) ---
    import tempfile
    local_model = Path(tempfile.mkdtemp()) / "model"
    local_model.mkdir(parents=True, exist_ok=True)
    torch.save(result["model"].state_dict(), local_model / "model.pt")

    arch = {
        # `encoder` is what downstream rebuilds from; `architecture` stays as a
        # human-readable label and now tells the truth instead of saying Conv4
        # for every run.
        "encoder": cfg.encoder,
        "architecture": cfg.encoder,
        "embedding_hid": cfg.embedding_hid,
        "in_channels": cfg.in_channels,
        "image_size": cfg.image_size,
        "augmentation": cfg.augmentation,
        "dataset": cfg.dataset,
        "n_way": cfg.n_way,
        "k_shot": cfg.k_shot,
        "query": cfg.query,     # episode shape, needed to re-evaluate identically
        "seed": cfg.seed,       # so evaluation can reconstruct the same class split
    }
    if cfg.encoder == "mlp":
        arch["mlp_depth"] = cfg.mlp_depth
        arch["mlp_dropout"] = cfg.mlp_dropout
    if cfg.image_size == 1:
        # Tabular domains (samld windows, samld-case cases, any future one) are
        # split into class folds. Keyed on the geometry, not on dataset names:
        # a name list silently skipped samld-case, and evaluate/explain would
        # then have re-scored a fold-3 model on fold 0.
        arch["fold"] = cfg.fold   # which class split this artifact was trained on
    (local_model / "architecture.json").write_text(json.dumps(arch, indent=2, sort_keys=True))

    # Unique per artifact. The old name (seed + Unix seconds) collided when two folds of a
    # campaign finished in the same second: the later upload overwrote the earlier model,
    # and every step reading that model_dir (explain, active learning, registry) got the
    # wrong fold. Fold, timestamp and a random suffix make a collision impossible.
    fold_tag = f"-f{cfg.fold}" if cfg.image_size == 1 else ""
    model_prefix = (f"models/{args.dataset}/protonet{fold_tag}-seed{cfg.seed}-"
                    f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}")
    with explain_failure("saving model to GCS", bucket=bucket):
        for fpath in local_model.iterdir():
            gcs.blob(f"{model_prefix}/{fpath.name}").upload_from_filename(str(fpath))
    model_gcs_dir = f"gs://{bucket}/{model_prefix}"
    print(f"Model saved to {model_gcs_dir}")

    # --- write outputs for downstream KFP components ---
    _write(args.accuracy_mean_output_path, result["test_accuracy_mean"])
    _write(args.accuracy_ci95_output_path, result["test_accuracy_ci95"])
    _write(args.accuracy_std_output_path, result["test_accuracy_std"])
    _write(args.train_time_output_path, result["train_time_seconds"])
    _write(args.model_dir_output_path, model_gcs_dir)
    _write(args.data_sha_output_path, archive_sha)
    print("Wrote all KFP outputs.")


if __name__ == "__main__":
    main()