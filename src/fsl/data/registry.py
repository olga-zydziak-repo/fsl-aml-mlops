"""Dataset dispatch — one place that knows which loader serves which domain.

Before this, `loop.py` imported `fsl.data.omniglot` directly, which meant the
training loop was welded to one dataset. Swapping domains would have required
editing the loop (or copying it). Now the loop asks the registry and the
registry answers based on `cfg.dataset`.

Adding a domain is: write `fsl/data/<name>.py` exposing the same two functions,
add one entry below. Nothing else in the system changes — the wrappers, the
gates, the registry of models, and HPO all stay as they are.

Every loader must honour the same contract:

    load_frozen_<name>(cfg) -> (bundle, archive_sha256)
    build_task_samplers(bundle, cfg) -> (train_tasks, val_tasks, test_tasks)

where `bundle` is whatever that loader's `build_task_samplers` accepts (the
loop only passes it through), and `archive_sha256` is the provenance hash the
loop records for the run.
"""
from __future__ import annotations

from typing import Callable

# Loader modules are imported lazily inside the branches: a broken or
# dependency-heavy loader for one domain must not stop another from working.
_ALIASES = {
    "omniglot": "omniglot",
    "resisc45": "resisc45",
    "nwpu-resisc45": "resisc45",
    "nwpu_resisc45": "resisc45",
    "samld": "samld",
    "saml-d": "samld",
    "saml_d": "samld",
    # SAML-D at case level: one example = one flagged case (accounts linked by
    # suspicious transactions), not one account-week. Hyphen in the canonical
    # name on purpose: it becomes part of the Vertex experiment name
    # (fsl-<dataset>), and experiment names reject underscores.
    "samld-case": "samld-case",
    "samld_case": "samld-case",
    "samldcase": "samld-case",
}

# Input geometry per domain: (in_channels, image_size).
#
# These are DEFAULTS for starting a run, not truth about a trained artifact.
# Once a model is trained, `architecture.json` records what it actually saw, and
# evaluation/explainability reconstruct from that file — never from this table.
# Otherwise retagging a domain's defaults would silently invalidate old models.
#
# Tabular domains reuse the same two-field contract: (n_features, 1). The
# feature vector IS the input and there is no spatial extent, so `image_size`
# is 1 by convention. Threading a third field through TrainConfig, the pipeline
# parameters and every wrapper would cost more than this convention does.
DOMAIN_GEOMETRY = {
    "omniglot": (1, 28),   # grayscale characters
    "resisc45": (3, 84),   # RGB aerial scenes, native 256 resized to 84
    "samld": (26, 1),      # tabular: 26 account-x-week window features
    "samld-case": (31, 1), # tabular: 31 case features (size, roles, structure, flow, ...)
}


def canonical_name(dataset: str) -> str:
    """Canonical dataset key (the name used under gs://<bucket>/raw/<name>/).

    Raises:
        ValueError: for an unknown name; the message lists the known ones.
    """
    key = _ALIASES.get(str(dataset).strip().lower())
    if key is None:
        raise ValueError(f"unknown dataset {dataset!r}. Known: {', '.join(sorted(set(_ALIASES)))}.")
    return key


def get_input_geometry(dataset: str) -> tuple[int, int]:
    """Return (in_channels, image_size) for a dataset name.

    Args:
        dataset: dataset name or alias, e.g. "resisc45" or "NWPU-RESISC45".

    Returns:
        The domain's default input geometry.

    Raises:
        ValueError: if the dataset has no registered geometry.
    """
    key = _ALIASES.get(str(dataset).strip().lower())
    if key not in DOMAIN_GEOMETRY:
        raise ValueError(
            f"no input geometry registered for dataset {dataset!r}. "
            f"Known: {', '.join(sorted(DOMAIN_GEOMETRY))}."
        )
    return DOMAIN_GEOMETRY[key]


def get_loaders(cfg) -> tuple[Callable, Callable]:
    """Return (load_frozen_fn, build_task_samplers_fn) for `cfg.dataset`.

    Args:
        cfg: a TrainConfig; only `cfg.dataset` is read.

    Returns:
        The two functions the training loop calls, in that order.

    Raises:
        ValueError: if `cfg.dataset` has no registered loader. The message
            lists what IS registered, so a typo is obvious.
    """
    key = _ALIASES.get(str(cfg.dataset).strip().lower())

    if key == "omniglot":
        from fsl.data.omniglot import build_task_samplers, load_frozen_omniglot

        return load_frozen_omniglot, build_task_samplers

    if key == "resisc45":
        from fsl.data.resisc45 import build_task_samplers, load_frozen_resisc45

        return load_frozen_resisc45, build_task_samplers

    if key in ("samld", "samld-case"):
        # Same loader for both units: the freeze writes the same contract
        # (feature table + schema + folds + per-fold stats); schema.json says
        # which table and which grouping column to use.
        from fsl.data.samld import build_task_samplers, load_frozen_samld

        return load_frozen_samld, build_task_samplers

    known = sorted(set(_ALIASES))
    raise ValueError(
        f"no loader registered for dataset {cfg.dataset!r}. "
        f"Known: {', '.join(known)}. "
        "To add one, create fsl/data/<name>.py with load_frozen_<name>() and "
        "build_task_samplers(), then register it in fsl/data/registry.py."
    )
