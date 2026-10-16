"""Fail-fast validation for the FSL pipeline.

Two layers, split by what they need:

  validate_config(cfg)  - OFFLINE, zero-cost. Checks the config dict alone:
      placeholders, missing keys, cross-field consistency. Runs anywhere
      (no GCP, no Docker). Call this at the TOP of every notebook, before
      compiling or submitting anything.

  validate_remote(cfg)  - needs GCP. Checks the image (do the wrappers accept
      the args the pipeline will pass?), that frozen data exists in GCS, and
      that the caller can reach the project. Optional but recommended before a
      costly run. Skips gracefully if google-cloud libs aren't importable.

Design: every problem found is COLLECTED, not raised one-at-a-time, so you see
the full list in one shot instead of fixing-rerunning-discovering the next.
Raises ConfigError with all problems at once, or prints an all-clear.

This exists because the real failures in this project were config-shaped and
discovered LATE: a placeholder project id caught only mid-run, a wrapper
missing --lr surfacing 40 minutes into training, an accelerator set with no
quota leaving trials hung. All of those are catchable in the first second.
"""
from __future__ import annotations

from typing import Any


class ConfigError(Exception):
    """Raised when validation finds one or more problems. Message lists them all."""


# The args each wrapper's argparse defines (from scripts/*.py). validate_remote
# checks the image's actual --help against these; validate_config uses the keys
# to know what the pipeline must supply. Keep in sync if a wrapper gains a flag.
WRAPPER_REQUIRED_ARGS = {
    "train_pipeline_entry.py": [
        "--project", "--bucket", "--dataset", "--n-way", "--k-shot",
        "--query", "--seed", "--train-iters", "--embedding-hid", "--lr",
        "--encoder", "--fold", "--mlp-depth", "--mlp-dropout",
    ],
    "hpo_train_entry.py": [
        "--project", "--bucket", "--dataset", "--seed",
        "--val-episodes", "--lr", "--embedding-hid", "--train-iters",
    ],
    "evaluate_pipeline_entry.py": [
        "--project", "--bucket", "--dataset", "--model-dir",
        "--accuracy-threshold", "--max-std-threshold", "--test-episodes",
    ],
    "explain_pipeline_entry.py": [
        "--project", "--bucket", "--dataset", "--model-dir", "--n-episodes",
    ],
}

# Top-level keys every run needs, with the type each must have.
_REQUIRED_TOP = {
    "project": str, "region": str, "bucket": str, "dataset": str,
    "training_image_uri": str, "training": dict, "evaluation": dict,
    "compute": dict,
}
# Keys inside training that the pipeline threads through (lr included - the
# omission that bit us).
_REQUIRED_TRAINING = [
    "n_way", "k_shot", "query", "seed", "train_iters",
    "eval_every", "val_episodes", "test_episodes", "lr", "embedding_hid",
]
_REQUIRED_EVAL = ["accuracy_threshold", "max_std_threshold"]

# Substrings that betray an untouched template value.
_PLACEHOLDER_MARKERS = ("YOUR-", "YOUR_", "<", "changeme", "TODO", "xxx", "PLACEHOLDER")


def _known_encoders() -> list[str]:
    """Encoder names from the registry; never a local copy."""
    from fsl.models.registry import known_encoders

    return known_encoders()


def _looks_like_placeholder(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    upper = value.upper()
    return any(m.upper() in upper for m in _PLACEHOLDER_MARKERS)


def validate_config(cfg: dict, *, require_hpo: bool = False) -> None:
    """Offline structural + consistency checks on the config dict.

    Collects every problem and raises ConfigError listing all of them, so one
    run surfaces the whole set. Pass require_hpo=True when submitting the HPO
    notebook to also validate the hpo section.

    Raises:
        ConfigError: if any problem is found. Message enumerates them.
    """
    problems: list[str] = []

    # 1. required top-level keys present and correctly typed
    for key, expected_type in _REQUIRED_TOP.items():
        if key not in cfg:
            problems.append(f"missing top-level key: '{key}'")
        elif not isinstance(cfg[key], expected_type):
            problems.append(
                f"'{key}' should be {expected_type.__name__}, "
                f"got {type(cfg[key]).__name__}"
            )

    # 2. no placeholders left in string fields (recursive)
    def _scan(prefix: str, d: dict) -> None:
        for k, v in d.items():
            if k == "_comment":
                continue
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _scan(path, v)
            elif _looks_like_placeholder(v):
                problems.append(f"placeholder not filled in: '{path}' = {v!r}")

    _scan("", cfg)

    # 3. training sub-keys
    training = cfg.get("training", {})
    if isinstance(training, dict):
        for key in _REQUIRED_TRAINING:
            if key not in training:
                problems.append(f"missing training.{key}")
        # sanity on values that have obvious valid ranges
        for key in ("n_way", "k_shot", "query", "train_iters",
                    "val_episodes", "test_episodes", "embedding_hid"):
            if key in training and isinstance(training[key], int) and training[key] <= 0:
                problems.append(f"training.{key} must be > 0, got {training[key]}")
        if "lr" in training and not (0 < training.get("lr", 0) < 1):
            problems.append(f"training.lr looks off: {training.get('lr')} (expected 0 < lr < 1)")
        # The allow-list comes from the encoder registry itself. A hardcoded
        # copy here drifted the moment `mlp` was registered: one truth about the
        # system kept in two places is the same bug class as the hardcoded
        # Omniglot imports. The registry module imports torch only inside its
        # builders, so asking it is as cheap as the old literal tuple.
        known = _known_encoders()
        enc = str(training.get("encoder", "conv4")).strip().lower()
        if enc not in known:
            problems.append(
                f"training.encoder must be one of {'|'.join(known)}, got {enc!r}"
            )
        fold = training.get("fold", 0)
        if not isinstance(fold, int) or fold < 0:
            problems.append(f"training.fold must be an int >= 0, got {fold!r}")
        if enc == "mlp":
            depth = training.get("mlp_depth", 2)
            if not isinstance(depth, int) or depth < 1:
                problems.append(f"training.mlp_depth must be an int >= 1, got {depth!r}")
            drop = training.get("mlp_dropout", 0.0)
            if not isinstance(drop, (int, float)) or not (0 <= drop < 1):
                problems.append(f"training.mlp_dropout must be in [0, 1), got {drop!r}")
        aug = training.get("augmentation", "none")
        if aug not in ("none", "flip", "flip_rot90", "trivial"):
            problems.append(
                f"training.augmentation must be one of none|flip|flip_rot90|trivial, got {aug!r}"
            )

    # 4. evaluation sub-keys and ranges
    evaluation = cfg.get("evaluation", {})
    if isinstance(evaluation, dict):
        for key in _REQUIRED_EVAL:
            if key not in evaluation:
                problems.append(f"missing evaluation.{key}")
        thr = evaluation.get("accuracy_threshold")
        if thr is not None and not (0 <= thr <= 1):
            problems.append(f"evaluation.accuracy_threshold must be in [0,1], got {thr}")
        mstd = evaluation.get("max_std_threshold")
        if mstd is not None and mstd < 0:
            problems.append(f"evaluation.max_std_threshold must be >= 0, got {mstd}")

    # 5. image URI shape (Artifact Registry, has a tag)
    img = cfg.get("training_image_uri", "")
    if isinstance(img, str) and img and not _looks_like_placeholder(img):
        if ":" not in img.rsplit("/", 1)[-1]:
            problems.append(
                f"training_image_uri has no tag (expected '...:tag'): {img}"
            )
        if "-docker.pkg.dev/" not in img:
            problems.append(
                f"training_image_uri doesn't look like Artifact Registry: {img}"
            )

    # 6. HPO section (only when this run will tune)
    hpo = cfg.get("hpo", {})
    if require_hpo or (isinstance(hpo, dict) and hpo.get("enabled")):
        if not isinstance(hpo, dict):
            problems.append("hpo section missing but HPO requested")
        else:
            problems.extend(_validate_hpo(hpo, training))

    if problems:
        bullet = "\n".join(f"  - {p}" for p in problems)
        raise ConfigError(
            f"Config validation failed ({len(problems)} problem(s)):\n{bullet}\n\n"
            "Fix these in the config file this notebook loaded and re-run this cell."
        )
    print(f"Config OK — {_summary(cfg)}")


def _validate_hpo(hpo: dict, training: dict) -> list[str]:
    """HPO-specific consistency. The accelerator/quota trap lives here."""
    out: list[str] = []

    max_trials = hpo.get("max_trials")
    parallel = hpo.get("parallel")
    if isinstance(max_trials, int) and max_trials <= 0:
        out.append(f"hpo.max_trials must be > 0, got {max_trials}")
    if isinstance(parallel, int) and parallel <= 0:
        out.append(f"hpo.parallel must be > 0, got {parallel}")
    if isinstance(max_trials, int) and isinstance(parallel, int) and parallel > max_trials:
        out.append(
            f"hpo.parallel ({parallel}) > hpo.max_trials ({max_trials}) — "
            "parallelism can't exceed the trial budget"
        )

    # the accelerator consistency that cost us hours: count and type must agree
    acc_type = hpo.get("accelerator_type", "")
    acc_count = hpo.get("accelerator_count", 0)
    if acc_type and acc_count <= 0:
        out.append(
            f"hpo.accelerator_type is '{acc_type}' but accelerator_count is "
            f"{acc_count} — set count >= 1 or clear the type"
        )
    if not acc_type and acc_count > 0:
        out.append(
            f"hpo.accelerator_count is {acc_count} but accelerator_type is empty "
            "— set a type (e.g. NVIDIA_TESLA_T4) or set count to 0"
        )
    if acc_type and "GPU" not in acc_type.upper() and "TPU" not in acc_type.upper() \
            and not acc_type.startswith("NVIDIA"):
        out.append(f"hpo.accelerator_type '{acc_type}' doesn't look like a valid accelerator")

    # a GPU request is the thing that silently hangs on missing Vertex-Training
    # quota — not an error, but worth a visible heads-up from the validator
    if acc_type:
        print(
            f"NOTE: HPO requests {acc_count}x {acc_type}. This needs "
            "'Custom model training' GPU quota in Vertex Training (separate from "
            "Workbench/Compute Engine quota). If trials hang in QUEUED, that quota "
            "is the first thing to check."
        )

    ps = hpo.get("params_source", "auto")
    if ps not in ("auto", "config"):
        out.append(f"hpo.params_source must be 'auto' or 'config', got {ps!r}")

    return out


def _summary(cfg: dict) -> str:
    t = cfg.get("training", {})
    hpo = cfg.get("hpo", {})
    bits = [
        f"project={cfg.get('project')}",
        f"dataset={cfg.get('dataset')}",
        f"encoder={t.get('encoder', 'conv4')}",
        f"{t.get('n_way')}-way {t.get('k_shot')}-shot, query={t.get('query')}",
    ]
    if "fold" in t:
        bits.append(f"fold={t.get('fold')}")
    if hpo.get("enabled"):
        acc = hpo.get("accelerator_type") or "CPU"
        bits.append(f"HPO on ({hpo.get('max_trials')} trials, {acc})")
    else:
        bits.append("HPO off")
    return ", ".join(bits)


def validate_remote(cfg: dict, *, check_image: bool = True,
                    check_data: bool = True) -> None:
    """GCP-dependent checks: image args, frozen data in GCS, project reachable.

    Optional. Skips (with a printed note) if google-cloud libs aren't available,
    so importing this module never hard-fails in a bare environment. Run before
    a costly submission when you want certainty the image and data are right.

    NOTE: verifying the image requires Docker (to run `<img> --help`) and is NOT
    checkable in every environment. Where Docker is absent, image checking is
    skipped with a note — the offline validate_config already caught the
    config-side of the --lr class of bug.
    """
    problems: list[str] = []
    project = cfg.get("project")
    bucket = cfg.get("bucket")
    dataset = cfg.get("dataset")

    try:
        from google.cloud import storage  # noqa: F401
    except Exception:
        print("validate_remote: google-cloud-storage not importable — skipping "
              "remote checks (run this on the Workbench).")
        return

    if check_data:
        try:
            from google.cloud import storage
            client = storage.Client(project=project)
            gcs = client.bucket(bucket)
            manifest_blob = gcs.blob(f"raw/{dataset}/MANIFEST.json")
            if not manifest_blob.exists():
                problems.append(
                    f"frozen data manifest missing: "
                    f"gs://{bucket}/raw/{dataset}/MANIFEST.json — "
                    "has the dataset been frozen (rung 1)?"
                )
            archive_blob = gcs.blob(f"raw/{dataset}/{dataset}.tar.gz")
            if not archive_blob.exists():
                problems.append(
                    f"frozen archive missing: gs://{bucket}/raw/{dataset}/{dataset}.tar.gz"
                )
        except Exception as e:
            problems.append(f"couldn't reach GCS bucket '{bucket}': {type(e).__name__}: {e}")

    if check_image:
        print("validate_remote: image arg-checking requires Docker; if this "
              "environment has it, verify manually:\n"
              f"  docker run --rm --entrypoint python {cfg.get('training_image_uri')} "
              "scripts/train_pipeline_entry.py --help | grep -- --lr")

    if problems:
        bullet = "\n".join(f"  - {p}" for p in problems)
        raise ConfigError(
            f"Remote validation failed ({len(problems)} problem(s)):\n{bullet}"
        )
    print("Remote checks OK (data present in GCS).")