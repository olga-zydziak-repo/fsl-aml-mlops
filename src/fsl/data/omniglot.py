"""Load the frozen Omniglot from GCS and build episodic task samplers.

Reads the frozen archive from GCS (never the internet), extracts it locally
once, and builds N-way K-shot task samplers with the standard 1100/100/423
meta-split. Mirrors sections 3-4 of the rung-2 notebook.
"""
from __future__ import annotations

import gzip
import json
import random
import tarfile
from pathlib import Path

import learn2learn as l2l
from torchvision import transforms

from fsl.config import TrainConfig


def load_frozen_omniglot(cfg: TrainConfig):
    """Download + extract the frozen archive from GCS; return (MetaDataset, sha256).

    The returned sha256 is the manifest's archive checksum — log it as a
    parameter to tie a run to its exact data version.
    """
    from google.cloud import storage

    cache = Path(cfg.cache_dir)
    archive_blob = f"raw/{cfg.dataset}/{cfg.dataset}.tar.gz"
    manifest_blob = f"raw/{cfg.dataset}/MANIFEST.json"

    client = storage.Client(project=cfg.project)
    bucket = client.bucket(cfg.bucket)

    manifest = json.loads(bucket.blob(manifest_blob).download_as_bytes().decode("utf-8"))
    archive_sha = manifest["archive_sha256"]

    if not (cache / "omniglot-py").exists():
        cache.mkdir(parents=True, exist_ok=True)
        archive = cache / f"{cfg.dataset}.tar.gz"
        bucket.blob(archive_blob).download_to_filename(str(archive))
        with open(archive, "rb") as raw:
            with gzip.GzipFile(fileobj=raw) as gz:
                with tarfile.open(fileobj=gz) as tar:
                    tar.extractall(cache)  # our own trusted archive

    try:
        from torchvision.transforms import InterpolationMode
        resize = transforms.Resize(28, interpolation=InterpolationMode.LANCZOS)
    except Exception:
        resize = transforms.Resize(28)

    omniglot = l2l.vision.datasets.FullOmniglot(
        root=str(cache),
        download=False,
        transform=transforms.Compose([
            resize,
            transforms.ToTensor(),
            lambda x: 1.0 - x,  # invert: white character on black
        ]),
    )
    return l2l.data.MetaDataset(omniglot), archive_sha


def build_task_samplers(dataset, cfg: TrainConfig):
    """Return (train_tasks, val_tasks, test_tasks) using the standard Omniglot
    1100/100/423 meta-split (seeded shuffle). Test classes are never seen in
    training."""
    classes = list(range(1623))
    random.Random(cfg.seed).shuffle(classes)
    split = {
        "train": classes[:1100],
        "val": classes[1100:1200],
        "test": classes[1200:],
    }

    def make(subset, num_tasks, augment=False):
        T = l2l.data.transforms
        tfs = [
            T.FilterLabels(dataset, subset),
            T.NWays(dataset, cfg.n_way),
            T.KShots(dataset, cfg.k_shot + cfg.query),  # support + query per class
            T.LoadData(dataset),
            T.RemapLabels(dataset),
            T.ConsecutiveLabels(dataset),  # group by class (needed for the split)
        ]
        if augment:
            tfs.append(
                l2l.vision.transforms.RandomClassRotation(
                    dataset, [0.0, 90.0, 180.0, 270.0]
                )
            )
        return l2l.data.TaskDataset(dataset, task_transforms=tfs, num_tasks=num_tasks)

    return (
        make(split["train"], 20000, augment=True),
        make(split["val"], cfg.val_episodes),
        make(split["test"], cfg.test_episodes),
    )
