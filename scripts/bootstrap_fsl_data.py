#!/usr/bin/env python3
"""
bootstrap_fsl_data.py — Download/build a few-shot dataset and freeze it
immutably in Google Cloud Storage with full provenance.

This is rung 1 of the MLOps ladder: data treated as a versioned, checksummed,
reproducible artifact — not a side effect of `download=True` in a training
script.

What it does (idempotently — safe to re-run):
  1. Creates the GCS bucket if it does not exist (regional, uniform access).
  2. Skips all work if the frozen artifact already exists (unless --force).
  3. Materialises the dataset into a local scratch dir (download for image
     benchmarks, feature aggregation for tabular ones).
  4. Computes a SHA256 for every file (the integrity anchor).
  5. Packs a *deterministic* tar.gz, so the archive checksum depends only on
     file contents + paths (sorted entries, zeroed mtime/uid/gid).
  6. Writes MANIFEST.json with full provenance: source, library versions,
     timestamp, per-file checksums, environment, and — for datasets that
     declare it — input geometry (n_features / in_channels / image_size).
  7. Uploads {archive, manifest, checksums} to gs://<bucket>/raw/<dataset>/.

Designed for Vertex AI Workbench, where Application Default Credentials are
already configured. No secrets are handled here.

Dependencies:
    pip install learn2learn google-cloud-storage          # image datasets
    pip install pandas pyarrow                            # tabular datasets

Usage:
    python bootstrap_fsl_data.py --project my-proj --dataset omniglot
    python bootstrap_fsl_data.py --project my-proj --dataset samld -v
    python bootstrap_fsl_data.py --project my-proj --dataset miniimagenet --force
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import platform
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NoReturn

LOG = logging.getLogger("bootstrap_fsl_data")


# --------------------------------------------------------------------------- #
# Dataset registry — add new few-shot datasets here; the rest is generic.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    download: Callable[[str], None]   # given a local root dir, materialise into it
    source: str                       # human-readable provenance note
    # Optional: derive input geometry from the materialised data root. Image
    # datasets leave this None (geometry is implied by the loader); tabular
    # datasets MUST declare it, because the encoder's input width is data-
    # dependent and has to survive into the manifest.
    geometry: Callable[[Path], dict] | None = None


# --------------------------------------------------------------------------- #
# Image datasets (learn2learn)
# --------------------------------------------------------------------------- #
def _download_omniglot(root: str) -> None:
    import learn2learn as l2l
    # FullOmniglot pulls the raw Omniglot data + builds its on-disk layout.
    # We freeze whatever it produces under `root` (raw freeze, intentionally).
    l2l.vision.datasets.FullOmniglot(root=root, download=True)


def _download_miniimagenet(root: str) -> None:
    import learn2learn as l2l
    # Pre-processed 84x84 caches (Ravi & Larochelle splits) — no ImageNet
    # licence / sign-up hurdle. One cache per meta-split.
    for mode in ("train", "validation", "test"):
        l2l.vision.datasets.MiniImagenet(root=root, mode=mode, download=True)


# ======================= SLOT: RESISC45 ==================================== #
# Paste your local `_download_resisc45` here (it is NOT in the GitHub snapshot
# this file was rebuilt from). Its registry entry goes into the SLOT marker
# inside DATASETS below.
# =========================================================================== #


# --------------------------------------------------------------------------- #
# Tabular dataset: SAML-D  (transactions -> account x week windows)
# --------------------------------------------------------------------------- #
SAMLD_CSV_DEFAULT = Path.home() / "data_cache/saml_d_source/SAML-D.csv"
SAMLD_MIN_SUSP = 2        # min. suspicious tx in a window to label it as that typology
SAMLD_MIN_WIN = 10        # classes with fewer windows are dropped
SAMLD_NORMAL_CAP = 5000   # subsample cap per NORMAL class (suspicious kept whole)
SAMLD_SEED = 0            # subsample + fold permutation seed (recorded in schema)
SAMLD_N_FOLDS = 5
SAMLD_POOL = 5            # classes in the val pool and in the test pool, per fold


def _build_samld(root: str) -> None:
    """Aggregate SAML-D transactions into (account x week) windows.

    Writes into `root`:
      windows.parquet — one row per window: acc, week, ~30 features, label
      schema.json     — feature order, log1p'd features, classes, build params
      folds.json      — explicit class-disjoint train/val/test splits per fold
      stats.json      — mu/sigma per fold, computed on TRAIN CLASSES ONLY

    The mu/sigma-per-fold rule is the one leak no downstream gate can catch:
    standardising on all classes lets test-class information into the encoder's
    input scaling, and it lifts train and test symmetrically.
    """
    import numpy as np
    import pandas as pd

    csv = Path(os.environ.get("SAMLD_CSV", str(SAMLD_CSV_DEFAULT))).expanduser()
    if not csv.exists():
        die(f"SAML-D CSV not found: {csv}. Set $SAMLD_CSV or copy it from "
            f"gs://<bucket>/sources/samld/SAML-D.csv")
    out = Path(root)

    dt = {
        "Payment_currency": "category", "Received_currency": "category",
        "Sender_bank_location": "category", "Receiver_bank_location": "category",
        "Payment_type": "category", "Laundering_type": "category",
        "Is_laundering": "uint8", "Amount": "float32",
        "Sender_account": "int64", "Receiver_account": "int64",
    }
    LOG.info("Reading %s ...", csv)
    df = pd.read_csv(csv, dtype=dt)
    LOG.info("SAML-D: %d transactions, %s .. %s", len(df), df["Date"].min(), df["Date"].max())

    # ---- transaction-level derived flags ---------------------------------- #
    d = pd.to_datetime(df["Date"])
    df["day"] = (d - d.min()).dt.days.astype("int32")
    df["week"] = (df["day"] // 7).astype("int32")
    hh = pd.to_numeric(df["Time"].str[:2], errors="coerce")
    df["night"] = (hh < 6).fillna(False).astype("uint8")

    def _diff(a: str, b: str):
        cs = df[a].cat.categories.union(df[b].cat.categories)
        return (df[a].cat.set_categories(cs).cat.codes
                != df[b].cat.set_categories(cs).cat.codes).astype("uint8")

    df["xborder"] = _diff("Sender_bank_location", "Receiver_bank_location")
    df["curm"] = _diff("Payment_currency", "Received_currency")
    df["cash"] = df["Payment_type"].astype(str).str.contains("cash", case=False).astype("uint8")
    df["r100"] = (df["Amount"] % 100 == 0).astype("uint8")
    df["s10k"] = df["Amount"].between(9000, 10000).astype("uint8")

    # Payment-type indicators: sorted by name (not by frequency) so the feature
    # order is stable across re-freezes even if counts shift.
    pt_cols: list[str] = []
    top_types = sorted(str(t) for t in df["Payment_type"].value_counts().index[:6])
    for t in top_types:
        k = "pt_" + t.lower().replace(" ", "_").replace("-", "_")
        df[k] = (df["Payment_type"].astype(str) == t).astype("uint8")
        pt_cols.append(k)

    # ---- long form: every transaction seen from both accounts -------------- #
    base = ["Amount", "week", "day", "night", "xborder", "curm", "cash", "r100", "s10k",
            "Is_laundering", "Laundering_type"] + pt_cols
    o = df[["Sender_account", "Receiver_account"] + base].rename(
        columns={"Sender_account": "acc", "Receiver_account": "cpty"})
    o["dir"] = np.uint8(1)
    i = df[["Receiver_account", "Sender_account"] + base].rename(
        columns={"Receiver_account": "acc", "Sender_account": "cpty"})
    i["dir"] = np.uint8(0)
    lf = pd.concat([o, i], ignore_index=True, copy=False)
    del o, i, df
    lf["amt_out"] = (lf["Amount"] * lf["dir"]).astype("float32")
    lf["amt_in"] = (lf["Amount"] * (1 - lf["dir"])).astype("float32")
    LOG.info("long-form rows: %d", len(lf))

    # ---- window aggregation ------------------------------------------------ #
    W = lf.groupby(["acc", "week"], observed=True, sort=False).agg(
        n=("dir", "size"), n_out=("dir", "sum"),
        amt_mean=("Amount", "mean"), amt_std=("Amount", "std"),
        amt_max=("Amount", "max"), amt_min=("Amount", "min"),
        amt_out_sum=("amt_out", "sum"), amt_in_sum=("amt_in", "sum"),
        cpty_u=("cpty", "nunique"), n_days=("day", "nunique"),
        xb=("xborder", "mean"), curm=("curm", "mean"), cash=("cash", "mean"),
        night=("night", "mean"), r100=("r100", "mean"), s10k=("s10k", "mean"),
        **{k: (k, "mean") for k in pt_cols})
    W["amt_std"] = W["amt_std"].fillna(0.0)
    W["out_ratio"] = W["n_out"] / W["n"]
    W["net"] = W["amt_out_sum"] - W["amt_in_sum"]
    W["flow_ratio"] = np.log1p(W["amt_out_sum"]) - np.log1p(W["amt_in_sum"])
    W["tx_per_day"] = W["n"] / W["n_days"]
    LOG.info("windows: %d", len(W))

    # ---- labels: dominant typology in the window --------------------------- #
    def _dominant(sub):
        c = (sub.groupby(["acc", "week", "Laundering_type"], observed=True)
                .size().reset_index(name="k"))
        return (c.sort_values("k")
                 .drop_duplicates(["acc", "week"], keep="last")
                 .set_index(["acc", "week"]))

    S = _dominant(lf[lf["Is_laundering"] == 1])
    N = _dominant(lf[lf["Is_laundering"] == 0])
    suspicious = set(S["Laundering_type"].astype(str).unique())
    del lf

    lab = __import__("pandas").Series(index=W.index, dtype=object)
    lab.update(N["Laundering_type"].astype(str))
    lab.update(S[S["k"] >= SAMLD_MIN_SUSP]["Laundering_type"].astype(str))
    W["label"] = lab
    W = W.dropna(subset=["label"])

    vc = W["label"].value_counts()
    keep = sorted(vc[vc >= SAMLD_MIN_WIN].index)
    W = W[W["label"].isin(keep)]

    # ---- subsample NORMAL classes (suspicious kept whole) ------------------ #
    rng = np.random.default_rng(SAMLD_SEED)
    parts = []
    for c in keep:
        sub = W[W["label"] == c]
        if c not in suspicious and len(sub) > SAMLD_NORMAL_CAP:
            sel = np.sort(rng.choice(len(sub), SAMLD_NORMAL_CAP, replace=False))
            sub = sub.iloc[sel]
        parts.append(sub)
    W = __import__("pandas").concat(parts).reset_index()   # acc, week become columns
    LOG.info("frozen windows: %d | classes: %d (suspicious: %d)",
             len(W), len(keep), sum(c in suspicious for c in keep))

    # ---- folds: class-disjoint, fixed size AND fixed composition ----------- #
    # Both the size (5/5) and the suspicious/normal mix of the val/test pools
    # are held constant across folds, so the spread across folds measures the
    # difficulty of the class set — not the size or the makeup of the pool.
    # Normal classes are more numerous and (per the feature gate) easier, so a
    # fold that happened to draw only normals would not be comparable.
    feats = [c for c in W.columns if c not in ("acc", "week", "label")]
    sus_p = [str(c) for c in rng.permutation([c for c in keep if c in suspicious])]
    nor_p = [str(c) for c in rng.permutation([c for c in keep if c not in suspicious])]
    k_sus = max(1, min(SAMLD_POOL - 1, round(SAMLD_POOL * len(sus_p) / len(keep))))
    k_nor = SAMLD_POOL - k_sus
    if len(sus_p) < 2 * k_sus or len(nor_p) < 2 * k_nor:
        die(f"not enough classes for folds: suspicious {len(sus_p)}, normal {len(nor_p)} "
            f"(need >= {2 * k_sus} / {2 * k_nor} for disjoint val+test pools)")

    folds = []
    for k in range(SAMLD_N_FOLDS):
        def _window(lst: list[str], per: int, offset: int) -> list[str]:
            return [lst[(k * per + offset + j) % len(lst)] for j in range(per)]
        s_te, n_te = _window(sus_p, k_sus, 0), _window(nor_p, k_nor, 0)
        s_va = [c for c in _window(sus_p, k_sus, k_sus) if c not in s_te]
        n_va = [c for c in _window(nor_p, k_nor, k_nor) if c not in n_te]
        te, va = sorted(s_te + n_te), sorted(s_va + n_va)
        tr = sorted(c for c in keep if c not in te and c not in va)
        if len(te) != SAMLD_POOL or len(va) != SAMLD_POOL:
            die(f"fold {k}: incomplete pools (test {len(te)}, val {len(va)}); "
                f"check k_sus={k_sus} / k_nor={k_nor} against the class counts")
        folds.append({"fold": k, "train": tr, "val": va, "test": te})
        LOG.info("fold %d: train %d / val %d / test %d (test: %d suspicious)",
                 k, len(tr), len(va), len(te), sum(c in suspicious for c in te))

    # ---- per-fold standardisation stats, TRAIN CLASSES ONLY ---------------- #
    X = W[feats].astype("float64")
    heavy = sorted(c for c in feats if X[c].min() >= 0 and X[c].max() > 50)
    X[heavy] = np.log1p(X[heavy])
    stats = {}
    for f in folds:
        m = W["label"].isin(f["train"])
        mu = X[m].mean()
        sd = X[m].std().replace(0, 1.0)
        stats[str(f["fold"])] = {"mu": [float(mu[c]) for c in feats],
                                 "sigma": [float(sd[c]) for c in feats]}

    # ---- write artifacts --------------------------------------------------- #
    W.to_parquet(out / "windows.parquet", index=False)
    (out / "schema.json").write_text(json.dumps({
        "feature_names": feats,
        "log1p_features": heavy,
        "label_col": "label",
        "group_col": "acc",
        "classes": keep,
        "suspicious_classes": sorted(c for c in keep if c in suspicious),
        "min_susp": SAMLD_MIN_SUSP,
        "min_windows": SAMLD_MIN_WIN,
        "normal_cap": SAMLD_NORMAL_CAP,
        "subsample_seed": SAMLD_SEED,
        "n_folds": SAMLD_N_FOLDS,
        "pool_size": SAMLD_POOL,
        "window": "account x week",
        "aggregation_version": "v0",
        "source_csv": csv.name,
    }, indent=2, sort_keys=True))
    (out / "folds.json").write_text(json.dumps(folds, indent=2))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))


def _geometry_samld(root: Path) -> dict:
    schema = json.loads((root / "schema.json").read_text())
    n = len(schema["feature_names"])
    return {
        "kind": "tabular",
        "n_features": n,
        # Mapped onto the existing two-field contract so TrainConfig /
        # architecture.json / the wrappers need no new plumbing.
        "in_channels": n,
        "image_size": 1,
        "feature_names": schema["feature_names"],
        "n_classes": len(schema["classes"]),
        "n_folds": schema["n_folds"],
    }


# --------------------------------------------------------------------------- #
# Tabular dataset: SAML-D at CASE level  (flagged transactions -> cases)
# --------------------------------------------------------------------------- #
# Why a second SAML-D unit: on account x week windows the typologies that
# differ by multi-account structure (Bipartite vs Stacked Bipartite, Fan_In vs
# Layered_Fan_In, Cycle) are indistinguishable: a single account's weekly
# summary does not contain the layering. The case-level prototype
# (samld_case_proto.ipynb, criteria fixed before the run) passed: structural
# pairs 2-way ProtoNet 0.64 -> 0.83, size-only control 0.64 vs 0.86.
#
# FRAMING, recorded in schema.json: case boundaries come from Is_laundering
# (which transactions are flagged), so the task is TYPOLOGY ASSIGNMENT FOR AN
# ALREADY DETECTED CASE, not detection. Laundering_type is used only as the
# label (dominant type), never to draw boundaries.
SAMLD_CASE_LINK_DAYS = 7     # flagged tx sharing an account within <= 7 days -> same case
SAMLD_CASE_MIN_PURITY = 0.5  # case kept only if its dominant typology holds > 50% of its tx
SAMLD_CASE_POOL = 4          # classes per val pool and per test pool (4-way episodes)
SAMLD_CASE_K, SAMLD_CASE_Q = 5, 5   # feasibility: every kept class must form a grouped 5+5 episode
SAMLD_CASE_FEATS = [
    "n_tx", "n_acc", "duration_days", "n_days", "tx_per_day", "mean_gap",
    "share_src", "share_sink", "share_mid", "max_out_share", "max_in_share", "deg_cv",
    "edge_repeat", "density", "depth", "has_cycle", "reciprocity", "is_bipartite",
    "amt_total", "amt_mean", "amt_cv", "amt_max_med", "r100", "s10k",
    "passthrough", "fwd_1d", "sink_conc", "src_conc", "cash", "xb", "curm",
]


def _samld_case_load_flagged(csv: Path, chunksize: int = 2_000_000):
    """Flagged transactions only; day counted from the minimum of the WHOLE file
    (same day definition as the window freeze)."""
    import pandas as pd

    cols = ["Date", "Sender_account", "Receiver_account", "Amount", "Payment_currency",
            "Received_currency", "Sender_bank_location", "Receiver_bank_location",
            "Payment_type", "Is_laundering", "Laundering_type"]
    dt = {"Sender_account": "int64", "Receiver_account": "int64", "Amount": "float64",
          "Is_laundering": "int8"}
    dmin, parts, n = None, [], 0
    for ch in pd.read_csv(csv, usecols=cols, dtype=dt, chunksize=chunksize):
        n += len(ch)
        d = pd.to_datetime(ch["Date"]).min()
        dmin = d if dmin is None else min(dmin, d)
        parts.append(ch[ch["Is_laundering"] == 1].copy())
    S = pd.concat(parts, ignore_index=True)
    S["day"] = (pd.to_datetime(S["Date"]) - dmin).dt.days.astype("int32")
    return S.reset_index(drop=True), n


def _samld_case_link(S, max_gap: int):
    """Connected components of transactions that share an account within max_gap days."""
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(S)
    t = np.r_[np.arange(n), np.arange(n)]
    acc = np.r_[S["Sender_account"].to_numpy(), S["Receiver_account"].to_numpy()]
    day = np.r_[S["day"].to_numpy(), S["day"].to_numpy()]
    o = np.lexsort((t, day, acc))
    acc, day, t = acc[o], day[o], t[o]
    link = (acc[1:] == acc[:-1]) & ((day[1:] - day[:-1]) <= max_gap)
    A = coo_matrix((np.ones(int(link.sum())), (t[:-1][link], t[1:][link])), shape=(n, n))
    return connected_components(A, directed=False)[1]


def _samld_case_features(g) -> dict:
    """31 features of one case: size, roles, structure, flow, amounts, timing, instruments."""
    from collections import defaultdict

    import networkx as nx
    import numpy as np
    import pandas as pd

    s, r = g["Sender_account"].to_numpy(), g["Receiver_account"].to_numpy()
    amt, day = g["Amount"].to_numpy(), g["day"].to_numpy()
    accs = np.unique(np.r_[s, r])
    n_acc, n_tx = len(accs), len(g)
    G = nx.DiGraph()
    G.add_nodes_from(accs.tolist())
    G.add_edges_from(zip(s.tolist(), r.tolist()))
    n_e = G.number_of_edges()
    outd = np.array([G.out_degree(a) for a in accs], float)
    ind = np.array([G.in_degree(a) for a in accs], float)
    src, sink, mid = (ind == 0) & (outd > 0), (outd == 0) & (ind > 0), (ind > 0) & (outd > 0)
    deg = outd + ind
    out_amt = pd.Series(amt).groupby(s).sum()
    in_amt = pd.Series(amt).groupby(r).sum()
    mids = accs[mid]
    if len(mids):
        i_, o_ = in_amt.reindex(mids).to_numpy(), out_amt.reindex(mids).to_numpy()
        passthrough = float(np.mean(np.minimum(i_, o_) / np.maximum(np.maximum(i_, o_), 1e-9)))
        in_days = defaultdict(set)
        for a, d in zip(r, day):
            in_days[a].add(int(d))
        mo = np.isin(s, mids)
        fwd = [(int(d) in in_days[a]) or (int(d) - 1 in in_days[a]) for a, d in zip(s[mo], day[mo])]
        fwd_1d = float(np.mean(fwd)) if fwd else 0.0
    else:
        passthrough, fwd_1d = 0.0, 0.0
    ds = np.sort(day)
    pt = g["Payment_type"].astype(str).str.lower()
    tot = float(amt.sum()) or 1.0
    return {
        "n_tx": n_tx, "n_acc": n_acc,
        "duration_days": int(ds[-1] - ds[0] + 1), "n_days": len(np.unique(ds)),
        "tx_per_day": n_tx / len(np.unique(ds)), "mean_gap": float(np.diff(ds).mean()) if n_tx > 1 else 0.0,
        "share_src": float(src.mean()), "share_sink": float(sink.mean()), "share_mid": float(mid.mean()),
        "max_out_share": float(outd.max() / max(n_acc - 1, 1)), "max_in_share": float(ind.max() / max(n_acc - 1, 1)),
        "deg_cv": float(deg.std() / max(deg.mean(), 1e-9)),
        "edge_repeat": n_tx / max(n_e, 1), "density": n_e / max(n_acc * (n_acc - 1), 1),
        "depth": float(nx.dag_longest_path_length(nx.condensation(G))),
        "has_cycle": float(any(len(c) > 1 for c in nx.strongly_connected_components(G))),
        "reciprocity": float(nx.reciprocity(G)) if n_e else 0.0,
        "is_bipartite": float(nx.is_bipartite(G.to_undirected())),
        "amt_total": tot, "amt_mean": float(amt.mean()),
        "amt_cv": float(amt.std() / max(amt.mean(), 1e-9)),
        "amt_max_med": float(amt.max() / max(np.median(amt), 1e-9)),
        "r100": float((amt % 100 == 0).mean()), "s10k": float(((amt >= 9000) & (amt <= 10000)).mean()),
        "passthrough": passthrough, "fwd_1d": fwd_1d,
        "sink_conc": float(in_amt.max() / tot), "src_conc": float(out_amt.max() / tot),
        "cash": float(pt.str.contains("cash").mean()),
        "xb": float((g["Sender_bank_location"].astype(str) != g["Receiver_bank_location"].astype(str)).mean()),
        "curm": float((g["Payment_currency"].astype(str) != g["Received_currency"].astype(str)).mean()),
    }


def _samld_case_feasible(groups_of_class, k: int, q: int) -> bool:
    """Same greedy test as fsl.data.samld._feasible: can the class's groups be
    split into a support side holding k rows and a query side holding q rows?"""
    import numpy as np

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


def _build_samld_case(root: str) -> None:
    """Build flagged cases and freeze them under the same contract as windows.

    Writes into `root`:
      cases.parquet — one row per case: case, grp, label, purity, 31 features
      schema.json   — table/unit/group column, features, log1p'd features,
                      classes, linking parameters, the task framing
      folds.json    — class-disjoint folds, every class tested exactly once
      stats.json    — mu/sigma per fold, TRAIN CLASSES ONLY

    `grp` joins cases that share any account over the whole period; support
    and query are drawn from disjoint groups, so a launderer reused across
    cases is never on both sides of an episode.
    """
    import numpy as np
    import pandas as pd

    csv = Path(os.environ.get("SAMLD_CSV", str(SAMLD_CSV_DEFAULT))).expanduser()
    if not csv.exists():
        die(f"SAML-D CSV not found: {csv}. Set $SAMLD_CSV or copy it from "
            f"gs://<bucket>/sources/samld/SAML-D.csv")
    out = Path(root)

    S, n_all = _samld_case_load_flagged(csv)
    LOG.info("SAML-D: %d transactions, %d flagged", n_all, len(S))
    S["case"] = _samld_case_link(S, SAMLD_CASE_LINK_DAYS)
    S["grp"] = _samld_case_link(S, 10 ** 9)
    if not (S.groupby("case")["grp"].nunique() == 1).all():
        die("a case spans two account groups — linking is broken")

    rows = []
    for cid, g in S.groupby("case", sort=True):
        vc = g["Laundering_type"].astype(str).value_counts()
        f = _samld_case_features(g)
        f.update(case=int(cid), grp=int(g["grp"].iloc[0]), label=str(vc.index[0]),
                 purity=float(vc.iloc[0] / len(g)))
        rows.append(f)
    C = pd.DataFrame(rows)
    big = int(C["n_tx"].max())
    LOG.info("cases: %d | largest: %d tx (%.1f%% of flagged)", len(C), big, 100 * big / len(S))
    C = C[C["purity"] > SAMLD_CASE_MIN_PURITY].reset_index(drop=True)

    labels = C["label"].to_numpy()
    groups = C["grp"].to_numpy()
    keep = sorted(c for c in np.unique(labels)
                  if _samld_case_feasible(groups[labels == c], SAMLD_CASE_K, SAMLD_CASE_Q))
    dropped = sorted(set(np.unique(labels)) - set(keep))
    if dropped:
        LOG.warning("classes that cannot form a grouped %d+%d episode, dropped: %s",
                    SAMLD_CASE_K, SAMLD_CASE_Q, dropped)
    C = C[C["label"].isin(keep)].reset_index(drop=True)
    C = C[["case", "grp", "label", "purity"] + SAMLD_CASE_FEATS]

    # ---- folds: every class tested exactly once ---------------------------- #
    rng = np.random.default_rng(SAMLD_SEED)
    perm = [str(c) for c in rng.permutation(keep)]
    P = SAMLD_CASE_POOL
    blocks = [perm[i:i + P] for i in range(0, len(perm) - len(perm) % P, P)]
    always_train = perm[len(blocks) * P:]
    if len(blocks) < 3:
        die(f"need >= 3 blocks of {P} classes for test/val/train, have {len(keep)} classes")
    folds = []
    for k in range(len(blocks)):
        te, va = sorted(blocks[k]), sorted(blocks[(k + 1) % len(blocks)])
        tr = sorted(c for c in keep if c not in te and c not in va)
        folds.append({"fold": k, "train": tr, "val": va, "test": te})
        LOG.info("fold %d: train %d / val %d / test %d", k, len(tr), len(va), len(te))
    if always_train:
        # Leftover classes (len(keep) not divisible by P) get one extra fold so
        # every class is tested at least once. The pool is padded with the
        # first classes of the SAME seeded permutation (no hand-picking); those
        # padding classes are therefore tested twice, each time against a
        # different training set. Val = the next block, disjoint from the test.
        pad = [c for c in perm if c not in always_train][:P - len(always_train)]
        te = sorted(always_train + pad)
        va_block = next(b for b in blocks if not set(b) & set(te))
        va = sorted(va_block)
        tr = sorted(c for c in keep if c not in te and c not in va)
        folds.append({"fold": len(folds), "train": tr, "val": va, "test": te})
        LOG.info("extra fold %d for leftover %s (padded with %s): train %d / val %d / test %d",
                 len(folds) - 1, always_train, pad, len(tr), len(va), len(te))

    # ---- per-fold standardisation stats, TRAIN CLASSES ONLY ---------------- #
    X = C[SAMLD_CASE_FEATS].astype("float64")
    heavy = sorted(c for c in SAMLD_CASE_FEATS if X[c].min() >= 0 and X[c].max() > 50)
    X[heavy] = np.log1p(X[heavy])
    stats = {}
    for f in folds:
        m = C["label"].isin(f["train"])
        mu, sd = X[m].mean(), X[m].std().replace(0, 1.0)
        stats[str(f["fold"])] = {"mu": [float(mu[c]) for c in SAMLD_CASE_FEATS],
                                 "sigma": [float(sd[c]) for c in SAMLD_CASE_FEATS]}

    C.to_parquet(out / "cases.parquet", index=False)
    (out / "schema.json").write_text(json.dumps({
        "table": "cases.parquet",
        "unit": "case",
        "id_col": "case",
        "feature_names": SAMLD_CASE_FEATS,
        "log1p_features": heavy,
        "label_col": "label",
        "group_col": "grp",
        "classes": keep,
        "suspicious_classes": keep,
        "dropped_classes": dropped,
        "leftover_tested_in_extra_fold": always_train,
        "link_days": SAMLD_CASE_LINK_DAYS,
        "largest_case": {"n_tx": big, "share_of_flagged": round(big / len(S), 4)},
        "n_flagged_tx": int(len(S)),
        "min_purity": SAMLD_CASE_MIN_PURITY,
        "feasibility_k_q": [SAMLD_CASE_K, SAMLD_CASE_Q],
        "subsample_seed": SAMLD_SEED,
        "n_folds": len(folds),
        "pool_size": P,
        "aggregation_version": "case-v0",
        "framing": ("typology assignment for an already flagged case: boundaries from "
                    "Is_laundering, label = dominant Laundering_type; not detection"),
        "source_csv": csv.name,
    }, indent=2, sort_keys=True))
    (out / "folds.json").write_text(json.dumps(folds, indent=2))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))


def _geometry_samld_case(root: Path) -> dict:
    schema = json.loads((root / "schema.json").read_text())
    n = len(schema["feature_names"])
    return {
        "kind": "tabular",
        "unit": "case",
        "n_features": n,
        "in_channels": n,
        "image_size": 1,
        "feature_names": schema["feature_names"],
        "n_classes": len(schema["classes"]),
        "n_folds": schema["n_folds"],
    }


DATASETS: dict[str, DatasetSpec] = {
    "omniglot": DatasetSpec(
        name="omniglot",
        download=_download_omniglot,
        source=(
            "learn2learn FullOmniglot (Lake et al. 2015): 1623 character classes "
            "x 20 examples, handwritten characters from 50 alphabets."
        ),
    ),
    "miniimagenet": DatasetSpec(
        name="miniimagenet",
        download=_download_miniimagenet,
        source=(
            "learn2learn MiniImagenet (Ravi & Larochelle 2017 splits): "
            "64/16/20 train/val/test classes x 600 examples, 84x84 ImageNet subset cache."
        ),
    ),
    # =================== SLOT: paste your "resisc45" entry here ============ #
    "samld": DatasetSpec(
        name="samld",
        download=_build_samld,
        source=(
            "SAML-D (Oztas et al. 2023, Kaggle): 9,504,852 synthetic transactions, "
            "28 laundering typologies, 2022-10-07..2023-08-23. Freeze v0: aggregated "
            "into account x week windows (~30 features); a window takes a suspicious "
            "typology label when >=2 of its transactions carry it, else the dominant "
            "normal typology; classes with <10 windows dropped; NORMAL classes "
            "subsampled to 5000 windows each (seed 0), suspicious kept whole; 5 "
            "class-disjoint folds with fixed 5-class val/test pools (3 suspicious + "
            "2 normal each); mu/sigma computed on "
            "train classes only, per fold. Feature-engineering provenance lives in "
            "schema.json (aggregation_version)."
        ),
        geometry=_geometry_samld,
    ),
    "samld-case": DatasetSpec(
        name="samld-case",
        download=_build_samld_case,
        source=(
            "SAML-D (Oztas et al. 2023, Kaggle), CASE level. Flagged transactions "
            "(Is_laundering=1) sharing an account within 7 days form one case; label = "
            "dominant Laundering_type (cases with purity <= 0.5 dropped); 31 case "
            "features (size, account roles, structure, flow, amounts, timing, "
            "instruments). Task framing: typology assignment for an already flagged "
            "case, NOT detection. Classes that cannot form a grouped 5+5 episode are "
            "dropped; 4-class val/test pools, every class tested exactly once; "
            "support/query disjoint by account group; mu/sigma on train classes per fold."
        ),
        geometry=_geometry_samld_case,
    ),
}


# --------------------------------------------------------------------------- #
# Integrity & packaging helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def checksum_tree(root: Path) -> dict[str, str]:
    """SHA256 of every file under `root`, keyed by sorted POSIX relative path."""
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[p.relative_to(root).as_posix()] = sha256_file(p)
    return out


def make_deterministic_tar(root: Path, archive_path: Path) -> str:
    """Pack `root` into a reproducible tar.gz; return the archive SHA256.

    Reproducible means: entries sorted by path, and mtime/uid/gid/owner/mode
    normalized, plus a gzip header with mtime=0. The resulting bytes therefore
    depend only on file contents and relative paths — so the same data yields
    the same archive checksum on any machine, any day.
    """
    files = sorted(p for p in root.rglob("*") if p.is_file())
    with open(archive_path, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:  # uncompressed tar into gzip
                for p in files:
                    info = tarfile.TarInfo(name=p.relative_to(root).as_posix())
                    info.size = p.stat().st_size
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    with p.open("rb") as fh:
                        tar.addfile(info, fh)
    return sha256_file(archive_path)


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# GCS helpers (Application Default Credentials on Workbench)
# --------------------------------------------------------------------------- #
def get_storage_client(project: str):
    try:
        from google.cloud import storage
    except ImportError as e:
        die("google-cloud-storage not installed. Run: pip install google-cloud-storage", e)
    return storage.Client(project=project)


def ensure_bucket(client, bucket_name: str, location: str):
    from google.cloud.exceptions import NotFound
    try:
        bucket = client.get_bucket(bucket_name)
        LOG.info("Bucket gs://%s already exists.", bucket_name)
        return bucket
    except NotFound:
        LOG.info("Creating bucket gs://%s in %s ...", bucket_name, location)
        bucket = client.bucket(bucket_name)
        # Professional defaults: uniform bucket-level access (no legacy ACLs).
        bucket.iam_configuration.uniform_bucket_level_access_enabled = True
        client.create_bucket(bucket, location=location)
        LOG.info("Created bucket gs://%s.", bucket_name)
        return client.get_bucket(bucket_name)


def blob_exists(bucket, name: str) -> bool:
    return bucket.blob(name).exists()


def upload(bucket, local: Path, name: str) -> None:
    LOG.info("Uploading %s -> gs://%s/%s", local.name, bucket.name, name)
    bucket.blob(name).upload_from_filename(str(local))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def die(msg: str, exc: Exception | None = None) -> NoReturn:
    LOG.error(msg)
    if exc is not None:
        LOG.debug("cause: %r", exc)
    sys.exit(1)


def resolve_project(arg: str | None) -> str:
    if arg:
        return arg
    env = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    if env:
        LOG.info("Using project from environment: %s", env)
        return env
    die("No --project given and GOOGLE_CLOUD_PROJECT is not set.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Freeze a few-shot dataset into GCS with provenance (MLOps rung 1).",
    )
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="omniglot")
    ap.add_argument("--project", default=None, help="GCP project (else $GOOGLE_CLOUD_PROJECT)")
    ap.add_argument("--bucket", default=None, help="Bucket name (default: <project>-fsl-data)")
    ap.add_argument("--location", default="europe-west4",
                    help="Bucket region — co-locate with your Workbench/compute.")
    ap.add_argument("--scratch-dir", default=None,
                    help="Local build dir (default: an auto-cleaned temp dir).")
    ap.add_argument("--force", action="store_true",
                    help="Rebuild and overwrite even if the frozen artifact exists.")
    ap.add_argument("--keep-scratch", action="store_true",
                    help="Do not delete the local scratch dir on success.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    project = resolve_project(args.project)
    bucket_name = args.bucket or f"{project}-fsl-data"
    spec = DATASETS[args.dataset]
    prefix = f"raw/{spec.name}"
    archive_blob = f"{prefix}/{spec.name}.tar.gz"
    manifest_blob = f"{prefix}/MANIFEST.json"
    checksums_blob = f"{prefix}/checksums.sha256"

    client = get_storage_client(project)
    bucket = ensure_bucket(client, bucket_name, args.location)

    # Idempotency: if already frozen, stop (unless forced).
    if blob_exists(bucket, archive_blob) and not args.force:
        LOG.info("Frozen artifact already present: gs://%s/%s", bucket_name, archive_blob)
        LOG.info("Nothing to do — use --force to refreeze. OK")
        return 0

    # Local scratch (auto-cleaned unless --keep-scratch / --scratch-dir).
    scratch_ctx = None
    if args.scratch_dir:
        scratch = Path(args.scratch_dir).expanduser().resolve()
        scratch.mkdir(parents=True, exist_ok=True)
    else:
        scratch_ctx = tempfile.TemporaryDirectory(prefix="fsl-")
        scratch = Path(scratch_ctx.name)

    data_root = scratch / spec.name
    data_root.mkdir(parents=True, exist_ok=True)

    # 1. Materialise (download for image sets, aggregate for tabular ones).
    LOG.info("Materialising '%s' into %s ...", spec.name, data_root)
    try:
        spec.download(str(data_root))
    except ImportError as e:
        die(f"Missing dependency for '{spec.name}': {e}. "
            f"Image sets need learn2learn; tabular sets need pandas + pyarrow.", e)
    except Exception as e:  # noqa: BLE001 - surface any build failure clearly
        die(f"Build failed for '{spec.name}': {e}", e)

    files = [p for p in data_root.rglob("*") if p.is_file()]
    if not files:
        die(f"No files were produced under {data_root}; aborting.")
    total_bytes = sum(p.stat().st_size for p in files)
    LOG.info("Produced %d files, %.1f MB.", len(files), total_bytes / 1e6)

    # 2. Per-file checksums (integrity anchor).
    LOG.info("Computing SHA256 for %d files ...", len(files))
    checksums = checksum_tree(data_root)

    # 3. Deterministic archive (stable checksum).
    archive_local = scratch / f"{spec.name}.tar.gz"
    LOG.info("Packing deterministic archive ...")
    archive_sha = make_deterministic_tar(data_root, archive_local)
    LOG.info("Archive SHA256: %s (%.1f MB)", archive_sha, archive_local.stat().st_size / 1e6)

    # 4. Manifest (provenance).
    manifest = {
        "dataset": spec.name,
        "source": spec.source,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "gcs": {
            "bucket": bucket_name,
            "archive": archive_blob,
            "manifest": manifest_blob,
            "checksums": checksums_blob,
        },
        "archive_sha256": archive_sha,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "learn2learn": _pkg_version("learn2learn"),
            "google_cloud_storage": _pkg_version("google-cloud-storage"),
            "pandas": _pkg_version("pandas"),
            "numpy": _pkg_version("numpy"),
            "pyarrow": _pkg_version("pyarrow"),
        },
        "files_sha256": checksums,
    }
    if spec.geometry is not None:
        manifest["geometry"] = spec.geometry(data_root)
        LOG.info("Geometry: %s", {k: v for k, v in manifest["geometry"].items()
                                  if k != "feature_names"})

    manifest_local = scratch / "MANIFEST.json"
    manifest_local.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    checksums_local = scratch / "checksums.sha256"
    checksums_local.write_text(
        "".join(f"{sha}  {rel}\n" for rel, sha in sorted(checksums.items()))
    )

    # 5. Upload {archive, checksums, manifest}.
    upload(bucket, archive_local, archive_blob)
    upload(bucket, checksums_local, checksums_blob)
    upload(bucket, manifest_local, manifest_blob)

    LOG.info("Frozen gs://%s/%s  OK", bucket_name, prefix)
    LOG.info("  archive   : gs://%s/%s", bucket_name, archive_blob)
    LOG.info("  manifest  : gs://%s/%s", bucket_name, manifest_blob)
    LOG.info("  checksums : gs://%s/%s", bucket_name, checksums_blob)

    if scratch_ctx and not args.keep_scratch:
        scratch_ctx.cleanup()
    elif args.keep_scratch:
        LOG.info("Scratch kept at %s", scratch)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
