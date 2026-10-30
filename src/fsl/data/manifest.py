"""Frozen-manifest access — the single source of truth about a dataset's shape.

The freeze step writes MANIFEST.json next to the archive. Image datasets carry
an `image` section ({"channels": 3, "train_size": 84, ...}); tabular datasets
carry a `geometry` section ({"kind": "tabular", "n_features": 26, ...}). The
encoder's input geometry is a property OF THE DATA, so it should flow from that
manifest — not be duplicated in configs, CLI defaults, and pipeline parameters
that can drift apart. Wrappers call `fetch_manifest` once (they need it for the
provenance SHA anyway) and `get_geometry` to size the model.

Backward compatibility: older manifests (Omniglot's predates the `image`
section) fall back to per-dataset defaults; an unknown dataset with neither
section is a hard, readable error rather than a silently wrong tensor shape.
"""
from __future__ import annotations

import json

# Fallbacks for manifests frozen before the `image` section existed.
_LEGACY_GEOMETRY = {
    "omniglot": (1, 28),
}


def fetch_manifest(project: str, bucket: str, dataset: str) -> dict:
    """Download and parse `raw/{dataset}/MANIFEST.json` from GCS.

    Args:
        project: GCP project id.
        bucket: GCS bucket name.
        dataset: dataset key (e.g. "resisc45").

    Returns:
        The manifest as a dict.
    """
    from google.cloud import storage

    blob = storage.Client(project=project).bucket(bucket).blob(
        f"raw/{dataset}/MANIFEST.json"
    )
    return json.loads(blob.download_as_bytes().decode("utf-8"))


def get_geometry(manifest: dict, dataset: str) -> tuple[int, int]:
    """Return (in_channels, image_size) for the encoder, from the manifest.

    For image datasets this is (channels, TRAINING size) — what the loader
    resizes to and the encoder is built for, not the native size on disk.

    For tabular datasets the same two-field contract carries (n_features, 1):
    the feature vector is the input and there is no spatial extent. Mapping
    onto the existing pair is deliberate — TrainConfig, architecture.json and
    every wrapper already thread these two ints end to end, and a third field
    would have to be plumbed through all of them.

    Args:
        manifest: dict returned by `fetch_manifest`.
        dataset: dataset key, used for legacy fallback and error text.

    Returns:
        (in_channels, image_size)

    Raises:
        ValueError: if the manifest has neither a usable `geometry` nor `image`
            section and the dataset has no legacy fallback — better a named
            error than a wrong model shape.
    """
    geom = manifest.get("geometry")
    if geom and geom.get("kind") == "tabular":
        n_features = int(geom["n_features"])
        if n_features < 1:
            raise ValueError(
                f"manifest for {dataset!r} declares n_features={n_features}; "
                "re-freeze the dataset."
            )
        return n_features, 1

    image = manifest.get("image")
    if image and "channels" in image and "train_size" in image:
        return int(image["channels"]), int(image["train_size"])
    if dataset in _LEGACY_GEOMETRY:
        return _LEGACY_GEOMETRY[dataset]
    raise ValueError(
        f"manifest for {dataset!r} has no usable geometry: expected either an "
        f"'image' section (with 'channels' and 'train_size') or a 'geometry' "
        f"section (kind='tabular' with 'n_features'), and no legacy fallback "
        "exists. Re-freeze the dataset with the current freeze script, or add "
        "a fallback in fsl/data/manifest.py."
    )