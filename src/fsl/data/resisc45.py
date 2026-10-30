"""NWPU-RESISC45 loader — the domain-swap counterpart of `omniglot.py`.

Same two-function contract, so `loop.py` doesn't care which domain it is:

    load_frozen_resisc45(cfg) -> (bundle, archive_sha256)
    build_task_samplers(bundle, cfg) -> (train_tasks, val_tasks, test_tasks)

Two things differ from Omniglot and both are deliberate:

1. **The class split comes from the frozen MANIFEST, not from code.** Few-shot
   requires disjoint classes across train/val/test; that partition is part of
   the dataset definition, so it was frozen alongside the bytes (rung 10). Two
   runs are only comparable if they episode over the same class partition.

2. **Images are RGB 256x256, resized to 84.** Conv4 has four max-pools, so the
   input size decides the embedding width: 28 -> 1x1 (Omniglot), 84 -> 5x5
   (the mini-ImageNet convention this follows), 256 -> 16x16 (needlessly large).
   `cfg.image_size` and `cfg.in_channels` carry this downstream so the encoder
   and the artifact metadata agree.

The bookkeeping maps are built from the filesystem and handed to
`MetaDataset(labels_to_indices=...)`, which skips its default full-dataset
iteration — otherwise constructing it would decode all 31,500 JPEGs.
"""
from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

# Extensions we treat as images. Anything else in a class directory is ignored.
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Per-channel normalisation. ImageNet statistics are used as a reasonable
# default for RGB aerial imagery; swap for dataset-specific statistics if you
# ever want to squeeze the last fraction of a point (and record it in the
# manifest if you do, since it changes what the model sees).
_NORM_MEAN = (0.485, 0.456, 0.406)
_NORM_STD = (0.229, 0.224, 0.225)


@dataclass
class Resisc45Bundle:
    """What `load_frozen_*` hands to `build_task_samplers`.

    Holds the extracted root and the frozen class partition, not the images —
    decoding happens lazily, per episode.
    """

    root: Path
    class_split: dict[str, list[str]]
    manifest: dict = field(default_factory=dict)


def index_split(root: Path, classes: list[str]) -> tuple[list[Path], dict[int, list[int]]]:
    """Build the file list and label->indices map for one split.

    Pure filesystem work: no image is opened. Labels are remapped to a dense
    0..n-1 range local to this split, which is what the episodic transforms
    expect (a split's classes are its whole world).

    Args:
        root: directory holding one subdirectory per class.
        classes: the class names belonging to this split.

    Returns:
        (files, labels_to_indices) where files[i] is the path of sample i and
        labels_to_indices[label] lists the sample indices of that class.

    Raises:
        FileNotFoundError: if a class directory named in the split is missing.
        ValueError: if a class directory contains no images.
    """
    files: list[Path] = []
    labels_to_indices: dict[int, list[int]] = {}

    for label, class_name in enumerate(sorted(classes)):
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"class directory missing from the extracted archive: {class_dir}. "
                "Does the frozen manifest match the archive?"
            )
        members = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not members:
            raise ValueError(f"no images found in class directory: {class_dir}")
        labels_to_indices[label] = list(range(len(files), len(files) + len(members)))
        files.extend(members)

    return files, labels_to_indices


def _find_class_root(base: Path, expected_classes: list[str]) -> Path | None:
    """Locate the directory that actually holds the class subdirectories.

    Handles archives packed flat (classes at the top) and archives with one
    wrapping directory. Returns None if neither layout matches.
    """
    probe = expected_classes[0]
    if (base / probe).is_dir():
        return base
    for child in sorted(p for p in base.iterdir() if p.is_dir()):
        if (child / probe).is_dir():
            return child
    return None


def _ensure_extracted(archive: Path, extracted: Path,
                      expected_classes: list[str]) -> Path:
    """Extract the archive if needed and return the directory holding the classes.

    Completion is tracked with a marker file written AFTER extraction finishes —
    the directory merely existing proves nothing, since an interrupted
    `extractall` leaves a directory full of nothing. If the marker is absent,
    any partial extraction is discarded and redone from scratch.

    Raises:
        FileNotFoundError: if, after extracting, the class directories still
            can't be located — with a listing of what IS there.
    """
    import shutil

    marker = extracted / ".extraction_complete"

    if marker.exists():
        root = _find_class_root(extracted, expected_classes)
        if root is not None:
            return root
        print("extraction marker present but classes not found — redoing")

    if extracted.exists():
        shutil.rmtree(extracted)
    extracted.mkdir(parents=True)
    print(f"extracting {archive.name} ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extracted)

    root = _find_class_root(extracted, expected_classes)
    if root is None:
        found = sorted(p.name for p in extracted.iterdir())[:10]
        raise FileNotFoundError(
            f"extracted the archive but couldn't find class directories under "
            f"{extracted}.\n  expected a directory named {expected_classes[0]!r} "
            f"(from the frozen manifest)\n  found at top level: {found}\n"
            "The archive layout doesn't match the manifest — re-run the freeze "
            "notebook, or adjust _find_class_root for the actual layout."
        )
    marker.write_text(f"extracted from {archive.name}\n")
    return root


def load_frozen_resisc45(cfg) -> tuple[Resisc45Bundle, str]:
    """Fetch the frozen archive from GCS (once), extract, return it with its hash.

    Mirrors the Omniglot path: the manifest is the source of truth for the
    archive's SHA256 and for the class split, and the returned hash is what the
    training loop records as data provenance.

    Downloads are cached under `cfg.cache_dir`; re-runs reuse the extraction.

    Args:
        cfg: a TrainConfig — uses project, bucket, dataset, cache_dir.

    Returns:
        (bundle, archive_sha256)

    Raises:
        ValueError: if the downloaded archive's hash doesn't match the manifest.
    """
    import hashlib

    from google.cloud import storage

    cache = Path(cfg.cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"{cfg.dataset}.tar.gz"
    # Deliberately NOT "extracted": the freeze notebook unzips the publisher's
    # archive there, and that tree is nested one level deeper (zip root folder).
    # This directory holds OUR tar, whose members are class dirs at top level.
    extracted = cache / "frozen"
    prefix = f"raw/{cfg.dataset}"

    bucket = storage.Client(project=cfg.project).bucket(cfg.bucket)
    manifest = json.loads(
        bucket.blob(f"{prefix}/MANIFEST.json").download_as_bytes().decode("utf-8")
    )
    expected_sha = manifest["archive_sha256"]

    if not archive.exists():
        print(f"downloading gs://{cfg.bucket}/{prefix}/{cfg.dataset}.tar.gz ...")
        bucket.blob(f"{prefix}/{cfg.dataset}.tar.gz").download_to_filename(str(archive))

    h = hashlib.sha256()
    with open(archive, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual_sha = h.hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(
            "frozen archive hash mismatch — the local cache is stale or corrupt.\n"
            f"  manifest: {expected_sha}\n  local:    {actual_sha}\n"
            f"Delete {archive} and re-run to re-download."
        )

    class_split = manifest["class_split"]
    all_classes = sorted(c for names in class_split.values() for c in names)
    root = _ensure_extracted(archive, extracted, all_classes)

    bundle = Resisc45Bundle(
        root=root,
        class_split=class_split,
        manifest=manifest,
    )
    print(
        f"resisc45 ready: {manifest['n_classes']} classes, {manifest['n_images']} images; "
        f"split {{{', '.join(f'{k}: {len(v)}' for k, v in bundle.class_split.items())}}}"
    )
    return bundle, actual_sha


def build_task_samplers(bundle: Resisc45Bundle, cfg):
    """Build the three episodic samplers over disjoint class sets.

    Each split gets its own dataset restricted to its classes, so an episode can
    never mix a training class into a test task. Samplers are infinite
    (`num_tasks=-1`): the loop draws as many episodes as it wants.

    Args:
        bundle: what `load_frozen_resisc45` returned.
        cfg: a TrainConfig — uses n_way, k_shot, query, image_size.

    Returns:
        (train_tasks, val_tasks, test_tasks)
    """
    import random as _random

    import learn2learn as l2l
    import torch
    from learn2learn.data.transforms import (
        ConsecutiveLabels,
        KShots,
        LoadData,
        NWays,
        RemapLabels,
    )
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision import transforms

    image_size = getattr(cfg, "image_size", 84)
    augmentation = getattr(cfg, "augmentation", "none")

    class _RandomRot90:
        """Random multiple-of-90-degree turn on a CHW tensor. Lossless (pure
        transpose, no interpolation) - the right rotation for overhead imagery,
        unlike RandomRotation's arbitrary angles with black corners."""

        def __call__(self, t):
            return torch.rot90(t, k=_random.randint(0, 3), dims=(1, 2))

    def _build_transform(train: bool):
        """TRAIN gets the augmentation policy; val/test are ALWAYS clean so
        episodes stay comparable across runs and policies."""
        resize = transforms.Resize((image_size, image_size))
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(_NORM_MEAN, _NORM_STD)
        if not train or augmentation == "none":
            return transforms.Compose([resize, to_tensor, normalize])
        if augmentation == "flip":
            return transforms.Compose([
                resize, to_tensor,
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                normalize,
            ])
        if augmentation == "flip_rot90":
            return transforms.Compose([
                resize, to_tensor,
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                _RandomRot90(),
                normalize,
            ])
        if augmentation == "trivial":
            # literature-standard recipe: random H-flip + TrivialAugmentWide;
            # TAW needs PIL/uint8, so it sits BEFORE ToTensor
            return transforms.Compose([
                resize,
                transforms.RandomHorizontalFlip(0.5),
                transforms.TrivialAugmentWide(),
                to_tensor, normalize,
            ])
        raise ValueError(
            f"unknown augmentation policy {augmentation!r} "
            "(expected none|flip|flip_rot90|trivial)"
        )

    class _SplitDataset(Dataset):
        """Lazy image dataset for one split; decodes on access, not on init."""

        def __init__(self, files: list[Path], tfm) -> None:
            self.files = files
            self.tfm = tfm
            self.labels: list[int] = []

        def __len__(self) -> int:
            return len(self.files)

        def __getitem__(self, idx: int):
            img = Image.open(self.files[idx]).convert("RGB")
            return self.tfm(img), self.labels[idx]

    samplers = []
    for split_name in ("train", "val", "test"):
        classes = bundle.class_split[split_name]
        files, labels_to_indices = index_split(bundle.root, classes)

        ds = _SplitDataset(files, _build_transform(train=(split_name == "train")))
        # dense label per sample, in the same order as `files`
        labels = [0] * len(files)
        for label, indices in labels_to_indices.items():
            for i in indices:
                labels[i] = label
        ds.labels = labels

        # passing the maps skips MetaDataset's default full-dataset scan
        meta = l2l.data.MetaDataset(ds, labels_to_indices=labels_to_indices)
        task_transforms = [
            NWays(meta, cfg.n_way),
            KShots(meta, cfg.k_shot + cfg.query),
            LoadData(meta),
            RemapLabels(meta),
            ConsecutiveLabels(meta),
        ]
        samplers.append(l2l.data.TaskDataset(meta, task_transforms, num_tasks=-1))

    return tuple(samplers)
