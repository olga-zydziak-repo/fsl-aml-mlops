"""SAML-D loader — AML typologies as few-shot classes over tabular windows.

Same contract as the image loaders (see `fsl/data/registry.py`):

    load_frozen_samld(cfg) -> (bundle, archive_sha256)
    build_task_samplers(bundle, cfg) -> (train_tasks, val_tasks, test_tasks)

What differs from the image domains, and why:

* A sample is an (account x week) window: a ~26-dim feature vector, not an
  image. `fast_adapt` never looks at the shape, so nothing in the loop or the
  model changes — only what `.sample()` puts in the tensor.

* Standardisation is data, not code. mu/sigma live in the frozen artifact,
  one set per fold, computed on that fold's TRAIN classes only. Computing them
  here over the whole table would leak test-class information into the input
  scaling, and it would lift train and test symmetrically — no gate downstream
  could see it.

* Episodes are grouped by account. Support and query rows for a class come
  from DISJOINT account sets, because in production the accounts behind a new
  typology are unknown. Ungrouped sampling would let the encoder recognise the
  account instead of the behaviour: four of the sixteen suspicious classes
  average 3-6 windows per account, so the same account can otherwise appear on
  both sides of an episode.

  This relies on `fast_adapt` sorting labels stably (`torch.sort(labels,
  stable=True)`): the sampler emits support rows before query rows within each
  class block, and an unstable sort would silently reshuffle them, mixing the
  account sets back together.

* Class splits are never recomputed at load time: a split recomputed in every
  step would drift with the numpy version between images and quietly change what
  "frozen" means. They come either from `folds.json` in the frozen artifact or,
  when `cfg.split_uri` is set, from the split.json the pipeline's prepare_experiment
  step wrote once for the whole run (content-addressed, tied to the data SHA).

* One loader, two units. `samld` freezes account x week windows; `samld-case`
  freezes flagged cases (accounts linked by suspicious transactions). Both
  write the same contract, and schema.json names the table (`table`), the
  grouping column for support/query disjointness (`group_col`) and the unit.
  The dataset name comes from `cfg.dataset`, so the manifest and the local
  cache are per unit.
"""
from __future__ import annotations

import hashlib
import json
import logging
import tarfile
from pathlib import Path

import numpy as np
import torch

from fsl.data.manifest import fetch_manifest

LOG = logging.getLogger(__name__)

DATASET = "samld"          # default name, for log messages only


def _name(cfg) -> str:
    from fsl.data.registry import canonical_name
    return canonical_name(getattr(cfg, "dataset", DATASET))


# --------------------------------------------------------------------------- #
# Freeze retrieval
# --------------------------------------------------------------------------- #
def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _ensure_local(cfg, manifest: dict, name: str = DATASET) -> Path:
    """Download + verify + extract the frozen archive into `cfg.cache_dir`.

    Returns the directory holding windows.parquet / schema.json / folds.json /
    stats.json. Re-extraction is skipped when a marker for this exact SHA
    exists, so re-runs on a warm Workbench are free.
    """
    sha = manifest["archive_sha256"]
    root = Path(cfg.cache_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    data_dir = root / "extracted"
    marker = root / f".verified-{sha}"

    if marker.exists() and (data_dir / "windows.parquet").exists():
        LOG.info("Frozen %s already extracted at %s (sha %s...)", name, data_dir, sha[:12])
        return data_dir

    from google.cloud import storage

    gcs = manifest["gcs"]
    archive = root / f"{name}.tar.gz"
    LOG.info("Downloading gs://%s/%s ...", gcs["bucket"], gcs["archive"])
    (storage.Client(project=cfg.project)
            .bucket(gcs["bucket"])
            .blob(gcs["archive"])
            .download_to_filename(str(archive)))

    actual = _sha256_file(archive)
    if actual != sha:
        raise ValueError(
            f"{name}: archive checksum mismatch — manifest says {sha}, "
            f"downloaded bytes hash to {actual}. Refusing to train on unverified data."
        )

    if data_dir.exists():
        for p in sorted(data_dir.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
    data_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(data_dir)

    marker.write_text(sha)
    LOG.info("Verified + extracted %s (sha %s...)", name, sha[:12])
    return data_dir


def load_frozen_samld(cfg):
    """Fetch, verify and load a frozen SAML-D table (windows or cases).

    Args:
        cfg: TrainConfig; reads `project`, `bucket`, `cache_dir`, `dataset`.

    Returns:
        (bundle, archive_sha256) where bundle carries the example table (under
        "windows" for every unit, so downstream code has one key), the schema,
        the fold definitions and the per-fold standardisation stats.
    """
    import pandas as pd

    name = _name(cfg)
    manifest = fetch_manifest(cfg.project, cfg.bucket, name)
    data_dir = _ensure_local(cfg, manifest, name)

    schema = json.loads((data_dir / "schema.json").read_text())
    table = pd.read_parquet(data_dir / schema.get("table", "windows.parquet"))
    split_uri = str(getattr(cfg, "split_uri", "") or "")
    split_sha = ""
    if split_uri:
        # The experiment design comes from the pipeline's prepare_experiment step,
        # not from the freeze. Only accept it if it was computed on these exact bytes.
        from fsl.data.split import load_split

        doc = load_split(split_uri, cfg.project)
        if doc["data_sha"] != manifest["archive_sha256"] or doc["dataset"] != name:
            raise ValueError(
                f"{name}: split {split_uri} was computed on {doc['dataset']} "
                f"{doc['data_sha'][:12]}..., the frozen data is {name} "
                f"{manifest['archive_sha256'][:12]}... — refusing to mix them")
        folds, stats, split_sha = doc["folds"], doc["stats"], doc["split_sha"]
        schema = {**schema, "classes": doc["classes"], "log1p_features": doc["log1p_features"]}
        LOG.info("%s: split from %s (split_sha %s...)", name, split_uri, split_sha[:12])
    elif (data_dir / "folds.json").exists():
        folds = json.loads((data_dir / "folds.json").read_text())
        stats = json.loads((data_dir / "stats.json").read_text())
    else:
        # A case snapshot without an experiment design (the freeze reduced to data
        # engineering). Fine for prepare_experiment, which computes the design;
        # build_task_samplers refuses it.
        folds, stats = None, None

    LOG.info("%s: %d %ss, %d features, %d classes, %s folds",
             name, len(table), schema.get("unit", "window"), len(schema["feature_names"]),
             len(schema.get("classes", [])), "no" if folds is None else len(folds))

    bundle = {
        "windows": table,
        "schema": schema,
        "folds": folds,
        "stats": stats,
        "manifest": manifest,
        "name": name,
        "split_uri": split_uri,
        "split_sha": split_sha,      # empty = folds/stats of the freeze itself
    }
    return bundle, manifest["archive_sha256"]


# --------------------------------------------------------------------------- #
# Episode sampling
# --------------------------------------------------------------------------- #
def _feasible(counts: np.ndarray, k_shot: int, query: int) -> bool:
    """Can this class's accounts be split into support/query sides at all?

    `counts` is rows-per-account. Greedy check: fill the support side with the
    largest accounts until it holds `k_shot` rows; the rest must hold `query`.
    """
    if counts.size < 2:
        return False
    order = np.sort(counts)[::-1]
    taken = 0
    for i, c in enumerate(order):
        taken += int(c)
        if taken >= k_shot:
            return int(order[i + 1:].sum()) >= query
    return False


class GroupedEpisodeSampler:
    """Samples n-way k-shot episodes with account-disjoint support/query.

    `.sample()` returns `(data, labels)` shaped exactly as `fast_adapt`
    expects: `n_way * (k_shot + query)` rows, one contiguous block per class,
    support rows first inside each block, labels 0..n_way-1.
    """

    def __init__(self, X: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                 classes: list[str], cfg, split: str, rng: np.random.Generator,
                 name: str = DATASET):
        self.X = X
        self.name = name
        self.k_shot = int(cfg.k_shot)
        self.query = int(cfg.query)
        self.n_way = int(cfg.n_way)
        self.rng = rng
        self.split = split

        self.by_class: dict[str, dict] = {}
        dropped: list[str] = []
        for c in classes:
            m = labels == c
            rows = np.flatnonzero(m)
            accs = groups[m]
            uniq, inv = np.unique(accs, return_inverse=True)
            counts = np.bincount(inv, minlength=uniq.size)
            if not _feasible(counts, self.k_shot, self.query):
                dropped.append(f"{c}(rows={rows.size},accts={uniq.size})")
                continue
            self.by_class[c] = {
                "rows": rows,
                "acct_rows": [rows[inv == a] for a in range(uniq.size)],
            }

        if dropped:
            LOG.warning("%s/%s: %d class(es) cannot form a grouped %d-shot/%d-query "
                        "episode and are excluded: %s",
                        name, split, len(dropped), self.k_shot, self.query,
                        ", ".join(sorted(dropped)))
        if len(self.by_class) < self.n_way:
            raise ValueError(
                f"{name}/{split}: only {len(self.by_class)} usable classes for "
                f"n_way={self.n_way} (k_shot={self.k_shot}, query={self.query}). "
                f"Lower cfg.query, lower cfg.n_way, or re-freeze with a higher "
                f"min_windows so every class in a pool can form an episode."
            )
        self.classes = sorted(self.by_class)
        LOG.info("%s/%s: %d usable classes", name, split, len(self.classes))

    def _split_accounts(self, acct_rows: list[np.ndarray], tries: int = 8):
        """Random account partition whose sides hold enough rows; greedy fallback."""
        n = len(acct_rows)
        for _ in range(tries):
            order = self.rng.permutation(n)
            sup: list[int] = []
            taken = 0
            for a in order:
                sup.append(a)
                taken += acct_rows[a].size
                if taken >= self.k_shot:
                    break
            qry = [a for a in order if a not in sup]
            if sum(acct_rows[a].size for a in qry) >= self.query:
                return sup, qry
        order = sorted(range(n), key=lambda a: -acct_rows[a].size)
        sup, taken = [], 0
        for a in order:
            sup.append(a)
            taken += acct_rows[a].size
            if taken >= self.k_shot:
                break
        return sup, [a for a in order if a not in sup]

    def sample_indexed(self):
        """One episode plus its provenance.

        Returns:
            (data, labels, idx, names): `data`/`labels` exactly as `sample()`;
            `idx` are the window-table row numbers behind each row of `data`
            (same order), `names[j]` is the class name behind episode label j.

        Explainability needs this: a case study has to say WHICH windows
        (account, week) a decision was matched against, and show their raw
        values, not just standardised tensors.

        RNG consumption is identical to `sample()` (which delegates here), so
        a sampler seeded like evaluate's draws the very same test episodes.
        """
        pick = self.rng.choice(len(self.classes), self.n_way, replace=False)
        rows, labels = [], []
        for j, ci in enumerate(pick):
            info = self.by_class[self.classes[ci]]
            sup_a, qry_a = self._split_accounts(info["acct_rows"])
            sup_pool = np.concatenate([info["acct_rows"][a] for a in sup_a])
            qry_pool = np.concatenate([info["acct_rows"][a] for a in qry_a])
            rows.append(self.rng.choice(sup_pool, self.k_shot, replace=False))
            rows.append(self.rng.choice(qry_pool, self.query, replace=False))
            labels += [j] * (self.k_shot + self.query)
        idx = np.concatenate(rows)
        data = torch.from_numpy(self.X[idx]).float()
        names = [self.classes[ci] for ci in pick]
        return data, torch.from_numpy(np.asarray(labels, dtype=np.int64)), idx, names

    def sample(self):
        data, labels, _, _ = self.sample_indexed()
        return data, labels


def build_task_samplers(bundle: dict, cfg):
    """Build train/val/test episode samplers for one fold.

    The fold is read from `cfg.fold` when present, else 0 — so a single-fold
    run needs no config change, and the multi-fold campaign sets it per run.

    Args:
        bundle: as returned by `load_frozen_samld`.
        cfg: TrainConfig; reads n_way, k_shot, query, seed, and optional fold.

    Returns:
        (train_tasks, val_tasks, test_tasks)
    """
    schema, folds, stats = bundle["schema"], bundle["folds"], bundle["stats"]
    W = bundle["windows"]
    name = bundle.get("name", DATASET)
    if folds is None:
        raise ValueError(f"{name}: no folds — this snapshot carries no experiment design; "
                         "run the prepare_experiment step and pass its split_uri (cfg.split_uri)")

    fold = int(getattr(cfg, "fold", 0))
    if not 0 <= fold < len(folds):
        raise ValueError(f"{name}: fold {fold} out of range (0..{len(folds) - 1})")
    spec = folds[fold]
    st = stats[str(fold)]

    feats = schema["feature_names"]
    # `.to_numpy()` can hand back a read-only view onto pandas' block manager;
    # copy() so the in-place log1p below is legal.
    X = W[feats].astype("float64").to_numpy().copy()
    heavy = [feats.index(c) for c in schema["log1p_features"]]
    X[:, heavy] = np.log1p(X[:, heavy])
    X = (X - np.asarray(st["mu"])) / np.asarray(st["sigma"])
    X = X.astype("float32")

    labels = W[schema["label_col"]].to_numpy().astype(str)
    groups = W[schema["group_col"]].to_numpy()

    LOG.info("%s: fold %d — train %d / val %d / test %d classes",
             name, fold, len(spec["train"]), len(spec["val"]), len(spec["test"]))

    def _sampler(split: str, offset: int):
        rng = np.random.default_rng(int(cfg.seed) + offset)
        return GroupedEpisodeSampler(X, labels, groups, spec[split], cfg, split, rng, name=name)

    return _sampler("train", 0), _sampler("val", 1), _sampler("test", 2)
