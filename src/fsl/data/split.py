"""Experiment design for tabular few-shot domains: class folds, standardisation, floors.

The freeze answers "what are the cases" (linking, account groups, features). This
module answers "how is the experiment cut": which typologies can take part, which
are tested in which fold, how features are scaled per fold, and what accuracy a
random, untrained encoder already reaches (the floor the gate measures learning
against). It runs as the first step of the pipeline (scripts/prepare_experiment_entry.py)
and writes one document, split.json, that every later step reads.

Why a document and not a function every step calls: a split recomputed in each
step would drift with library versions between images and nobody would notice.
Computed once per run, content-addressed (split_sha) and recorded in the model's
architecture.json and in the analyst's stan.json, it is a fact the run can be
audited against.

The fold logic is a verbatim port of the freeze in scripts/bootstrap_fsl_data.py
(_build_samld_case). On the same cases it must give the same folds and the same
statistics; prepare_experiment_entry.py checks that against the frozen folds.json
and stats.json and stops the run if they differ.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SPLIT_VERSION = "split-v1"
THRESHOLD_RULE = "min(floor + 0.10, floor + 0.5 * (1 - floor))"


# --------------------------------------------------------------------------- #
# Typologies, folds, statistics
# --------------------------------------------------------------------------- #
def feasible(groups_of_class, k: int, q: int) -> bool:
    """Can one class's account groups be split into a support side holding k
    rows and a query side holding q rows? Same greedy test as the freeze and as
    fsl.data.samld._feasible."""
    _, counts = np.unique(groups_of_class, return_counts=True)
    if counts.size < 2:
        return False
    order = np.sort(counts)[::-1]
    taken = 0
    for i, c in enumerate(order):
        taken += int(c)
        if taken >= k:
            return int(order[i + 1:].sum()) >= q
    return False


def feasible_classes(labels: np.ndarray, groups: np.ndarray, k: int, q: int):
    """(kept, dropped): classes that can form a grouped k+q episode, sorted."""
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    kept = sorted(c for c in np.unique(labels) if feasible(groups[labels == c], k, q))
    dropped = sorted(set(np.unique(labels)) - set(kept))
    return [str(c) for c in kept], [str(c) for c in dropped]


def make_folds(kept: list[str], pool: int, seed: int):
    """Class-disjoint folds: every class tested at least once.

    A seeded permutation is cut into blocks of `pool` classes; fold k tests
    block k, validates on block k+1 and trains on the rest. Leftover classes
    (len(kept) not divisible by pool) get one extra fold, padded with the first
    classes of the same permutation. Returns (folds, leftover).
    """
    rng = np.random.default_rng(seed)
    perm = [str(c) for c in rng.permutation(kept)]
    blocks = [perm[i:i + pool] for i in range(0, len(perm) - len(perm) % pool, pool)]
    leftover = perm[len(blocks) * pool:]
    if len(blocks) < 3:
        raise ValueError(f"need >= 3 blocks of {pool} classes for test/val/train, "
                         f"have {len(kept)} classes")
    folds = []
    for k in range(len(blocks)):
        te, va = sorted(blocks[k]), sorted(blocks[(k + 1) % len(blocks)])
        tr = sorted(c for c in kept if c not in te and c not in va)
        folds.append({"fold": k, "train": tr, "val": va, "test": te})
    if leftover:
        pad = [c for c in perm if c not in leftover][:pool - len(leftover)]
        te = sorted(leftover + pad)
        va = sorted(next(b for b in blocks if not set(b) & set(te)))
        tr = sorted(c for c in kept if c not in te and c not in va)
        folds.append({"fold": len(folds), "train": tr, "val": va, "test": te})
    return folds, leftover


def log1p_features(table, features: list[str]) -> list[str]:
    """Heavy-tailed, non-negative features that are log-transformed before scaling."""
    X = table[features].astype("float64")
    return sorted(c for c in features if X[c].min() >= 0 and X[c].max() > 50)


def fold_stats(table, features: list[str], heavy: list[str], folds: list[dict],
               label_col: str) -> dict:
    """mu/sigma per fold, computed on that fold's TRAIN classes only."""
    X = table[features].astype("float64")
    X[heavy] = np.log1p(X[heavy])
    stats = {}
    for f in folds:
        m = table[label_col].isin(f["train"])
        mu, sd = X[m].mean(), X[m].std().replace(0, 1.0)
        stats[str(f["fold"])] = {"mu": [float(mu[c]) for c in features],
                                 "sigma": [float(sd[c]) for c in features]}
    return stats


def make_split(table, schema: dict, *, pool: int, k: int, q: int, seed: int) -> dict:
    """Typologies, folds, log1p list and per-fold statistics for one snapshot.

    `table` is the case snapshot (one row per case, with the schema's label and
    group columns and its feature columns). Classes that cannot form a grouped
    k+q episode are dropped first, exactly as the freeze does, so a snapshot
    that already went through the freeze gives the same folds again.
    """
    label_col, group_col = schema["label_col"], schema["group_col"]
    features = list(schema["feature_names"])
    kept, dropped = feasible_classes(table[label_col].to_numpy(), table[group_col].to_numpy(), k, q)
    table = table[table[label_col].isin(kept)].reset_index(drop=True)
    folds, leftover = make_folds(kept, pool, seed)
    heavy = log1p_features(table, features)
    stats = fold_stats(table, features, heavy, folds, label_col)
    return {
        "classes": kept,
        "dropped_classes": dropped,
        "leftover_tested_in_extra_fold": leftover,
        "folds": folds,
        "log1p_features": heavy,
        "stats": stats,
        "params": {"pool": pool, "feasibility_k_q": [k, q], "seed": seed},
    }


# --------------------------------------------------------------------------- #
# Floors and thresholds
# --------------------------------------------------------------------------- #
def threshold(floor: float) -> float:
    """Gate threshold from the floor (rule fixed before the floors were computed):
    ten points over the floor, or half of the floor's errors removed, whichever
    is lower. The second part keeps a high floor from asking for >= 1.0."""
    return round(min(floor + 0.10, floor + 0.5 * (1 - floor)), 4)


def compute_floors(bundle: dict, *, n_way: int, k_shot: int, query: int,
                   seeds=(0, 1, 2), episodes: int = 400, hid: int = 64) -> dict:
    """Accuracy of a random, never-trained MLP encoder on each fold's TEST classes.

    Same procedure as cell 3 of samld_case_freeze.ipynb: for every fold and
    seed, build the fold's samplers with that seed, initialise the encoder
    under torch.manual_seed(seed) and evaluate `episodes` test episodes; the
    floor is the mean over seeds, rounded to 4 places.
    """
    import torch

    from fsl.config import TrainConfig
    from fsl.data.samld import build_task_samplers
    from fsl.models.mlp import MLP
    from fsl.training.loop import evaluate

    features = bundle["schema"]["feature_names"]
    floors, per_seed = {}, {}
    for f in bundle["folds"]:
        k = int(f["fold"])
        accs = []
        for seed in seeds:
            cfg = TrainConfig(dataset=bundle.get("name", "samld-case"), seed=seed, fold=k,
                              n_way=n_way, k_shot=k_shot, query=query, log_to_vertex=False)
            _, _, te = build_task_samplers(bundle, cfg)
            torch.manual_seed(seed)
            model = MLP(in_features=len(features), hid=hid)
            accs.append(float(evaluate(model, te, cfg, episodes, torch.device("cpu")).mean()))
        floors[str(k)] = round(float(np.mean(accs)), 4)
        per_seed[str(k)] = [round(a, 4) for a in accs]
    return {"floors": floors, "thresholds": {k: threshold(v) for k, v in floors.items()},
            "floors_per_seed": per_seed,
            "floor_params": {"seeds": list(seeds), "episodes": episodes,
                             "encoder": {"name": "mlp", "hid": hid, "depth": 2, "trained": False},
                             "episode": [n_way, k_shot, query], "threshold_rule": THRESHOLD_RULE}}


# --------------------------------------------------------------------------- #
# The document
# --------------------------------------------------------------------------- #
_IDENTITY_KEYS = ("version", "dataset", "data_sha", "classes", "folds", "log1p_features",
                  "stats", "params", "floors", "thresholds", "floor_params")


def split_sha(doc: dict) -> str:
    """Content hash of everything that defines the experiment (not library versions)."""
    core = {k: doc[k] for k in _IDENTITY_KEYS if k in doc}
    blob = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def split_path(dataset: str, sha: str) -> str:
    """Content-addressed location inside the bucket: the same split always lands in the same place."""
    return f"splits/{dataset}/{sha[:16]}/split.json"


def load_split(uri: str, project: str = "") -> dict:
    """Read split.json from gs://... or a local path and check its own hash."""
    if uri.startswith("gs://"):
        from google.cloud import storage

        bucket, _, name = uri[len("gs://"):].partition("/")
        doc = json.loads(storage.Client(project=project or None).bucket(bucket).blob(name)
                         .download_as_bytes().decode("utf-8"))
    else:
        doc = json.loads(Path(uri).read_text())
    if doc.get("version") != SPLIT_VERSION:
        raise ValueError(f"{uri}: split version {doc.get('version')!r}, expected {SPLIT_VERSION!r}")
    if split_sha(doc) != doc.get("split_sha"):
        raise ValueError(f"{uri}: split_sha does not match the content — the file was edited after it was written")
    return doc


def compare_with_frozen(split: dict, frozen_folds: list, frozen_stats: dict,
                        frozen_log1p: list) -> dict:
    """How the computed split relates to the one the freeze wrote (acceptance check)."""
    same_folds = json.dumps(split["folds"], sort_keys=True) == json.dumps(frozen_folds, sort_keys=True)
    same_log1p = list(split["log1p_features"]) == list(frozen_log1p)
    max_diff = 0.0
    same_keys = set(split["stats"]) == set(frozen_stats)
    if same_keys:
        for k, v in split["stats"].items():
            for part in ("mu", "sigma"):
                a, b = np.asarray(v[part]), np.asarray(frozen_stats[k][part])
                if a.shape != b.shape:
                    same_keys = False
                    break
                max_diff = max(max_diff, float(np.max(np.abs(a - b))) if a.size else 0.0)
    return {"same_folds": bool(same_folds), "same_log1p": bool(same_log1p),
            "same_stat_keys": bool(same_keys), "stats_max_abs_diff": max_diff}
