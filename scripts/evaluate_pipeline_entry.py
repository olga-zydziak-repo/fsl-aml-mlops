#!/usr/bin/env python3
"""Heavy-evaluation entry point for the container component.

Loads the REGISTERED model artifact from GCS (state_dict + architecture.json),
rebuilds the exact architecture, recomputes test accuracy INDEPENDENTLY using
the same fsl core the training used, and gates on three checks:

  1. consistency_with_training - |acc_heavy - acc_train| <= ci95_train + ci95_heavy
     (statistical, not bitwise: episodes are re-sampled, so exact equality is
     not expected; a large gap means the saved artifact != the trained model)
  2. accuracy_lower_ci        - (mean - ci95) >= accuracy_threshold, on the
     RECOMPUTED numbers (the saved artifact must clear the bar itself)
  3. stability                - recomputed std <= max_std_threshold

It also renders a self-contained HTML report (pure-SVG histogram - no
matplotlib, no new image dependencies) and writes it to the KFP HTML artifact
path, so the console shows it inline.

Zero duplicated model logic: the encoder registry, evaluate and the data
loaders are imported from fsl (shipped in the image). model.eval() is applied
before scoring (BatchNorm determinism - caught this the hard way earlier);
fsl's evaluate() enforces it again internally.

EVERYTHING that defines the episode comes from architecture.json: the backbone,
the geometry, the seed (same class split), AND the episode shape (n_way,
k_shot, query) plus the class fold for tabular domains. Re-deriving any of
those from defaults would have this component score a DIFFERENT task than the
one trained, and then report the difference as an artifact inconsistency.
"""
import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace


def _write(path: str, value) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(value))


def svg_histogram(values, bins=30, width=640, height=300, color="#4C72B0",
                  vline=None, vline_label="", title=""):
    """Self-contained SVG histogram - numpy only, no plotting libraries."""
    import numpy as np
    counts, edges = np.histogram(values, bins=bins)
    max_c = counts.max() if counts.max() > 0 else 1
    pad_l, pad_b, pad_t = 50, 40, 30
    plot_w, plot_h = width - pad_l - 20, height - pad_b - pad_t
    lo, hi = float(edges[0]), float(edges[-1])
    span = (hi - lo) or 1.0

    def x_px(v): return pad_l + (v - lo) / span * plot_w

    bars = []
    for i, c in enumerate(counts):
        x0, x1 = x_px(edges[i]), x_px(edges[i + 1])
        y = pad_t + plot_h - (c / max_c) * plot_h
        bars.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(x1 - x0 - 1, 1):.1f}" '
                    f'height="{pad_t + plot_h - y:.1f}" fill="{color}"/>')
    ticks = []
    for v in [lo, (lo + hi) / 2, hi]:
        ticks.append(f'<text x="{x_px(v):.1f}" y="{height - 12}" font-size="11" '
                     f'text-anchor="middle" fill="#555">{v:.3f}</text>')
    vline_svg = ""
    if vline is not None:
        vx = x_px(vline)
        vline_svg = (f'<line x1="{vx:.1f}" y1="{pad_t}" x2="{vx:.1f}" y2="{pad_t + plot_h}" '
                     f'stroke="#C44E52" stroke-width="2" stroke-dasharray="5,3"/>'
                     f'<text x="{vx + 4:.1f}" y="{pad_t + 12}" font-size="11" '
                     f'fill="#C44E52">{vline_label}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{width / 2}" y="18" font-size="13" text-anchor="middle" '
            f'fill="#333">{title}</text>'
            f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - 20}" '
            f'y2="{pad_t + plot_h}" stroke="#999"/>'
            + "".join(bars) + "".join(ticks) + vline_svg + '</svg>')


def main() -> None:
    ap = argparse.ArgumentParser(description="Heavy evaluation of a saved ProtoNet artifact.")
    ap.add_argument("--project", required=True)
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--dataset", default="omniglot")
    ap.add_argument("--model-dir", required=True, help="gs://... dir with model.pt + architecture.json")
    ap.add_argument("--model-version", default="?", help="Registry version id (for the report header)")
    ap.add_argument("--train-accuracy-mean", type=float, required=True)
    ap.add_argument("--train-accuracy-ci95", type=float, required=True)
    ap.add_argument("--accuracy-threshold", type=float, required=True)
    ap.add_argument("--max-std-threshold", type=float, required=True)
    ap.add_argument("--test-episodes", type=int, default=1000)
    ap.add_argument("--passed-output-path", required=True)
    ap.add_argument("--reason-output-path", required=True)
    ap.add_argument("--accuracy-recomputed-output-path", required=True)
    ap.add_argument("--report-html-path", required=True)
    args = ap.parse_args()

    import numpy as np
    import torch
    from google.cloud import storage

    from fsl.config import TrainConfig
    from fsl.errors import explain_failure, require
    from fsl.data.registry import get_loaders
    from fsl.models.registry import get_encoder
    from fsl.training.loop import evaluate

    # --- 1. download the saved artifact (the thing we're actually judging) ---
    require(args.model_dir.startswith("gs://"),
            f"model_dir must be a gs:// path, got {args.model_dir!r}")
    bucket_name, _, prefix = args.model_dir[len("gs://"):].partition("/")
    client = storage.Client(project=args.project)
    gcs_bucket = client.bucket(bucket_name)
    local = Path(tempfile.mkdtemp())
    with explain_failure("downloading model artifact from GCS", bucket=bucket_name):
        for name in ("model.pt", "architecture.json"):
            gcs_bucket.blob(f"{prefix}/{name}").download_to_filename(str(local / name))
    arch = json.loads((local / "architecture.json").read_text())
    print(f"Loaded artifact from {args.model_dir}: {arch}")

    # --- 2. rebuild the EXACT architecture and load weights ---
    # Through the registry, from the name the artifact records. A hardcoded
    # constructor here would quietly build the wrong network for any run that
    # used a different backbone - a state_dict mismatch at best, a wrong-shaped
    # forward at worst. Artifacts frozen before the encoder was recorded are
    # Conv4 by construction, hence the default.
    encoder_name = arch.get("encoder", arch.get("architecture", "conv4")).strip().lower()
    model = get_encoder(SimpleNamespace(
        encoder=encoder_name,
        in_channels=arch["in_channels"],
        embedding_hid=arch["embedding_hid"],
        mlp_depth=arch.get("mlp_depth", 2),
        mlp_dropout=arch.get("mlp_dropout", 0.0),
    ))
    model.load_state_dict(torch.load(local / "model.pt", map_location="cpu"))
    model.eval()  # BatchNorm determinism; fsl's evaluate() also enforces this
    device = torch.device("cpu")
    model = model.to(device)

    # --- 3. same data, SAME split and SAME episode shape (from the artifact) ---
    cfg = TrainConfig(
        project=args.project, region=args.region, bucket=args.bucket,
        dataset=args.dataset, seed=arch["seed"],
        n_way=arch["n_way"], k_shot=arch["k_shot"],
        query=arch.get("query", 15),         # pre-fix artifacts trained at the default
        fold=arch.get("fold", 0),            # class split for tabular domains
        encoder=encoder_name,
        embedding_hid=arch["embedding_hid"],
        mlp_depth=arch.get("mlp_depth", 2),
        mlp_dropout=arch.get("mlp_dropout", 0.0),
        in_channels=arch["in_channels"],
        image_size=arch.get("image_size", 28),  # pre-RESISC45 artifacts lack it
        split_uri=arch.get("split_uri", ""),    # folds/stats the model was trained under

        test_episodes=args.test_episodes, log_to_vertex=False,
    )
    print(f"re-evaluating {encoder_name} on {args.dataset} fold {cfg.fold}: "
          f"{cfg.n_way}-way {cfg.k_shot}-shot, query={cfg.query}, seed={cfg.seed}")
    load_frozen, build_task_samplers = get_loaders(cfg)
    dataset, archive_sha = load_frozen(cfg)
    _, _, test_tasks = build_task_samplers(dataset, cfg)

    # --- 4. recompute independently ---
    accs = evaluate(model, test_tasks, cfg, cfg.test_episodes, device)
    mean = float(accs.mean())
    std = float(accs.std())
    ci95 = float(1.96 * std / np.sqrt(len(accs)))
    lower = mean - ci95
    print(f"Recomputed on saved artifact: {mean:.4f} +/- {ci95:.4f} (std {std:.4f})")

    # --- 5. gate checks ---
    checks = []
    diff = abs(mean - args.train_accuracy_mean)
    tol = args.train_accuracy_ci95 + ci95
    checks.append(("consistency_with_training", diff <= tol,
                   f"diff={diff:.4f} vs tol={tol:.4f} (train={args.train_accuracy_mean:.4f}, heavy={mean:.4f})"))
    checks.append(("accuracy_lower_ci", lower >= args.accuracy_threshold,
                   f"lowerCI={lower:.4f} vs threshold={args.accuracy_threshold:.4f}"))
    checks.append(("stability", std <= args.max_std_threshold,
                   f"std={std:.4f} vs max={args.max_std_threshold:.4f}"))
    passed = all(ok for _, ok, _ in checks)
    reason = "; ".join(f"{n}: {'PASS' if ok else 'FAIL'} ({d})" for n, ok, d in checks)
    print(f"Heavy gate: {'PASSED' if passed else 'FAILED'} - {reason}")

    # --- 6. self-contained HTML report ---
    hist = svg_histogram(accs, vline=mean, vline_label=f"mean {mean:.3f}",
                         title=f"Accuracy over {len(accs)} test episodes (recomputed)")
    rows = ""
    for n, ok, d in checks:
        col = "#2a7d4f" if ok else "#c0392b"
        rows += (f'<tr><td>{n}</td><td style="color:{col};font-weight:600">'
                 f'{"PASS" if ok else "FAIL"}</td><td>{d}</td></tr>')
    verdict_col = "#2a7d4f" if passed else "#c0392b"
    verdict = "ARTIFACT VERIFIED - eligible for promotion" if passed else "ARTIFACT REJECTED - not promoted"
    fold_note = f" | fold: {cfg.fold}" if "fold" in arch else ""
    html = f"""<!DOCTYPE html><html><head><style>
body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:24px;color:#1a1a1a;max-width:760px}}
table{{border-collapse:collapse;margin:16px 0}}td,th{{border:1px solid #ddd;padding:6px 12px;font-size:14px;text-align:left}}
.metric{{display:inline-block;margin:8px 28px 8px 0}}.metric .v{{font-size:26px;font-weight:600;color:#4C72B0}}
.metric .l{{font-size:12px;color:#666}}.verdict{{font-size:17px;font-weight:700;color:{verdict_col};margin:14px 0}}
.meta{{font-size:12px;color:#888}}</style></head><body>
<h2>Heavy Evaluation - model version {args.model_version}</h2>
<p class="meta">artifact: {args.model_dir}<br/>data sha: {archive_sha[:16]}... | encoder: {encoder_name} |
seed: {arch["seed"]}{fold_note} | {arch["n_way"]}-way {arch["k_shot"]}-shot, query {cfg.query} |
{len(accs)} episodes</p>
<div class="metric"><div class="v">{mean:.2%}</div><div class="l">recomputed mean</div></div>
<div class="metric"><div class="v">&plusmn;{ci95:.4f}</div><div class="l">95% CI</div></div>
<div class="metric"><div class="v">{std:.4f}</div><div class="l">std (spread)</div></div>
<div class="verdict">{verdict}</div>
<table><tr><th>check</th><th>result</th><th>detail</th></tr>{rows}</table>
{hist}
<p class="meta">Consistency is statistical, not bitwise: episodes are re-sampled, so the recomputed
mean is compared against training within combined confidence intervals.</p>
</body></html>"""
    _write(args.report_html_path, html)

    # --- 7. KFP outputs ---
    _write(args.passed_output_path, str(passed).lower())
    _write(args.reason_output_path, reason)
    _write(args.accuracy_recomputed_output_path, mean)
    print("Wrote all KFP outputs (passed, reason, accuracy, report).")


if __name__ == "__main__":
    main()