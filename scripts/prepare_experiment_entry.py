#!/usr/bin/env python3
"""Pipeline step: the experiment design of one run (folds, standardisation, floors).

Runs first in the campaign, right after verify. Reads the frozen case snapshot,
decides which typologies can take part, cuts them into class-disjoint folds,
computes the per-fold standardisation on training typologies only and the floor
of every fold (a random, never-trained encoder on the fold's test typologies),
and writes all of it as ONE content-addressed document:

    gs://<bucket>/splits/<dataset>/<split_sha[:16]>/split.json

Outputs for the pipeline: split_uri (every later step reads the design from it),
split_sha, and fold_ids — the list the campaign loops over, so the number of
folds is decided here, at run time, from the data.

Acceptance check against the freeze: while the frozen archive still carries its
own folds.json/stats.json, the computed split is compared with them. Different
folds or statistics mean the port is wrong, and the step stops the run before
anything trains (--require-frozen-match, on by default). Floors are compared
with the Workbench floors file when the campaign passes it (--legacy-floors-json)
and only reported: a different torch build can move a random initialisation.

Usage (KFP fills the *-output-path args):
    python scripts/prepare_experiment_entry.py --project P --bucket B --dataset samld-case \
        --split-uri-output-path /x --split-sha-output-path /x --fold-ids-output-path /x \
        --report-html-path /x --summary-output-path /x
"""
from __future__ import annotations

import argparse
import html
import json
import platform
from pathlib import Path


def _write(path: str, value) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(value if isinstance(value, str) else json.dumps(value))


def _flag(s) -> bool:
    return str(s).lower() not in ("false", "0", "no", "")


def _store(doc: dict, split_root: str, dataset: str, project: str) -> str:
    """Write split.json under its content address; refuse to overwrite different content."""
    from fsl.data.split import split_path

    rel = split_path(dataset, doc["split_sha"])
    body = json.dumps(doc, indent=1, sort_keys=True)
    if split_root.startswith("gs://"):
        from google.cloud import storage

        bucket, _, prefix = split_root[len("gs://"):].partition("/")
        name = f"{prefix.rstrip('/')}/{rel}" if prefix else rel
        blob = storage.Client(project=project).bucket(bucket).blob(name)
        if blob.exists():
            old = json.loads(blob.download_as_bytes().decode("utf-8"))
            if old.get("split_sha") != doc["split_sha"]:
                raise SystemExit(f"gs://{bucket}/{name} holds a different split — refusing to overwrite")
            print(f"split already stored at gs://{bucket}/{name} (same content)")
        else:
            blob.upload_from_string(body, content_type="application/json")
        return f"gs://{bucket}/{name}"
    path = Path(split_root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return str(path)


def _report(doc: dict, cmp: dict | None, legacy: dict | None, uri: str) -> str:
    rows = []
    for f in doc["folds"]:
        k = str(f["fold"])
        leg = ""
        if legacy:
            lf = legacy.get("floors", {}).get(k)
            leg = f"<td>{lf if lf is not None else '-'}</td>"
        rows.append(f"<tr><td>{k}</td><td>{html.escape(', '.join(f['test']))}</td>"
                    f"<td>{len(f['train'])} / {len(f['val'])}</td>"
                    f"<td>{doc['floors'][k]:.4f}</td><td>{doc['thresholds'][k]:.4f}</td>{leg}</tr>")
    legh = "<th>floor (Workbench file)</th>" if legacy else ""
    if cmp is None:
        check = "<p>The frozen archive has no folds.json: nothing to compare with.</p>"
    else:
        ok = cmp["same_folds"] and cmp["same_log1p"] and cmp["same_stat_keys"] and cmp["stats_max_abs_diff"] <= 1e-9
        check = (f"<p class='{'ok' if ok else 'bad'}'>Comparison with the freeze: folds "
                 f"{'identical' if cmp['same_folds'] else 'DIFFERENT'}, log1p list "
                 f"{'identical' if cmp['same_log1p'] else 'DIFFERENT'}, standardisation max |diff| "
                 f"{cmp['stats_max_abs_diff']:.2e}.</p>")
    if legacy and cmp is not None:
        check += f"<p>Floors vs the Workbench file: max |diff| {cmp.get('floors_max_abs_diff', float('nan')):.4f}.</p>"
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{{font-family:-apple-system,Segoe UI,sans-serif;margin:24px;color:#1a1a1a;max-width:900px}}
table{{border-collapse:collapse;margin:12px 0}}td,th{{border:1px solid #ddd;padding:5px 9px;font-size:13px;text-align:left}}
.ok{{color:#1a7f37;font-weight:600}}.bad{{color:#c62828;font-weight:600}}.meta{{color:#777;font-size:12px}}
</style></head><body>
<h2>Experiment design: {html.escape(doc['dataset'])}</h2>
<p>{len(doc['folds'])} folds over {len(doc['classes'])} typologies (blocks of {doc['params']['pool']};
leftover tested in an extra fold: {html.escape(', '.join(doc['leftover_tested_in_extra_fold']) or 'none')};
dropped as infeasible for a grouped {doc['params']['feasibility_k_q'][0]}+{doc['params']['feasibility_k_q'][1]} episode:
{html.escape(', '.join(doc['dropped_classes']) or 'none')}).</p>
<table><tr><th>fold</th><th>test typologies</th><th>train / val typologies</th><th>floor</th><th>gate threshold</th>{legh}</tr>
{''.join(rows)}</table>
{check}
<p class="meta">Floor = random, never-trained MLP, seeds {doc['floor_params']['seeds']} x {doc['floor_params']['episodes']}
test episodes; threshold = {html.escape(doc['floor_params']['threshold_rule'])}.<br>
split_sha {doc['split_sha']} · data_sha {doc['data_sha'][:16]}… · {html.escape(uri)}<br>
numpy {doc['env']['numpy']} · pandas {doc['env']['pandas']} · torch {doc['env']['torch']}</p>
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment design step: folds, standardisation, floors.")
    ap.add_argument("--project", required=True)
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--bucket", default="")
    ap.add_argument("--dataset", default="samld-case")
    ap.add_argument("--split-root", default="", help="where split.json goes; default gs://<bucket> (a local dir works for tests)")
    ap.add_argument("--pool", type=int, default=4, help="typologies per test/val block = n_way of the task")
    ap.add_argument("--k", type=int, default=5, help="feasibility: support rows a class must offer")
    ap.add_argument("--q", type=int, default=5, help="feasibility: query rows a class must offer")
    ap.add_argument("--seed", type=int, default=0, help="seed of the typology permutation")
    ap.add_argument("--n-way", type=int, default=4)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--query", type=int, default=5)
    ap.add_argument("--floor-seeds", default="0,1,2")
    ap.add_argument("--floor-episodes", type=int, default=400)
    ap.add_argument("--legacy-floors-json", default="", help="floors file from the Workbench, for comparison only")
    ap.add_argument("--require-frozen-match", default="true", type=_flag,
                    help="stop the run if the frozen archive has folds/stats and they differ")
    ap.add_argument("--split-uri-output-path", required=True)
    ap.add_argument("--split-sha-output-path", required=True)
    ap.add_argument("--fold-ids-output-path", required=True)
    ap.add_argument("--report-html-path", required=True)
    ap.add_argument("--summary-output-path", required=True)
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import torch

    from fsl.config import TrainConfig
    from fsl.data.registry import get_loaders
    from fsl.data.split import (SPLIT_VERSION, compare_with_frozen, compute_floors, make_split,
                                split_sha)

    bucket = args.bucket or f"{args.project}-fsl-data"
    cfg = TrainConfig(project=args.project, region=args.region, bucket=bucket, dataset=args.dataset,
                      log_to_vertex=False)
    load_frozen, _ = get_loaders(cfg)
    frozen, data_sha = load_frozen(cfg)          # verifies the archive SHA-256 on download
    table, schema = frozen["windows"], frozen["schema"]

    sp = make_split(table, schema, pool=args.pool, k=args.k, q=args.q, seed=args.seed)
    bundle = {"windows": table, "folds": sp["folds"], "stats": sp["stats"], "name": frozen["name"],
              "schema": {**schema, "classes": sp["classes"], "log1p_features": sp["log1p_features"]}}
    seeds = tuple(int(s) for s in str(args.floor_seeds).split(",") if s.strip())
    fl = compute_floors(bundle, n_way=args.n_way, k_shot=args.k_shot, query=args.query,
                        seeds=seeds, episodes=args.floor_episodes)

    doc = {"version": SPLIT_VERSION, "dataset": frozen["name"], "data_sha": data_sha, **sp, **fl,
           "env": {"numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__,
                   "python": platform.python_version()}}
    doc["split_sha"] = split_sha(doc)

    # ---- acceptance check against what the freeze wrote --------------------- #
    cmp, legacy = None, None
    if frozen.get("folds") is not None and frozen.get("stats") is not None:
        cmp = compare_with_frozen(sp, frozen["folds"], frozen["stats"], schema.get("log1p_features", []))
        if args.legacy_floors_json.strip():
            legacy = json.loads(args.legacy_floors_json)
            diffs = [abs(float(legacy["floors"][k]) - v) for k, v in fl["floors"].items() if k in legacy.get("floors", {})]
            cmp["floors_max_abs_diff"] = max(diffs) if diffs else float("nan")
        print("comparison with the freeze:", json.dumps(cmp))
        bad = not (cmp["same_folds"] and cmp["same_log1p"] and cmp["same_stat_keys"]
                   and cmp["stats_max_abs_diff"] <= 1e-9)
        if bad and args.require_frozen_match:
            _write(args.report_html_path, _report(doc, cmp, legacy, "(not stored)"))
            raise SystemExit("the computed split differs from the frozen folds.json/stats.json — "
                             "stopping before anything trains (see the report)")

    uri = _store(doc, args.split_root or f"gs://{bucket}", doc["dataset"], args.project)
    fold_ids = [str(f["fold"]) for f in doc["folds"]]
    _write(args.split_uri_output_path, uri)
    _write(args.split_sha_output_path, doc["split_sha"])
    _write(args.fold_ids_output_path, json.dumps(fold_ids))
    _write(args.report_html_path, _report(doc, cmp, legacy, uri))
    line = (f"split {doc['split_sha'][:12]}: {len(fold_ids)} folds, floors "
            + ", ".join(f"f{k} {v:.4f}" for k, v in fl["floors"].items())
            + ("" if cmp is None else f" | freeze match: folds {cmp['same_folds']}, "
               f"stats |diff| {cmp['stats_max_abs_diff']:.1e}"))
    _write(args.summary_output_path, line)
    print(line)
    print("split_uri:", uri)


if __name__ == "__main__":
    main()
