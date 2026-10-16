#!/usr/bin/env python3
"""Explainability entry point: why the model decided what it decided.

ProtoNet decides by distance: the support windows of each class are embedded
and averaged into a prototype, and a query goes to the nearest prototype. The
decision therefore has a built-in, auditable form ("it is class A because it
sits closest to THESE support examples"). This report shows that mechanism
directly and, for tabular domains, goes one level deeper: which INPUT FEATURES
moved the query toward the chosen class.

Two modes, picked by what the test sampler can do:

TABULAR (samld; the sampler exposes `sample_indexed`)
  * Contrastive Integrated Gradients on the decision margin
        m(x) = ||f(x) - p_contrast|| - ||f(x) - p_chosen||
    i.e. the exact logit gap fast_adapt uses (logits = -cdist). The contrast
    class is the runner-up for a correct decision and the TRUE class for a
    mistake ("why B and not A"). The baseline is the contrast class's mean
    support window, so the attribution answers: which differences between this
    window and a typical window of the contrast class made the model pick the
    chosen class. Attributions are additive: they sum to m(x) - m(baseline).
  * Features rolled up into families (volume, amounts, threshold signals,
    flow balance, counterparties, rhythm, instruments, cross-border).
  * Case studies with provenance (account/week for windows, case id and size for
    cases; nearest support examples) and
    a sentence in RAW units, not z-scores.
  * Global view: per-typology confusion, the families each typology is
    recognised by, and what drives the most common confusions.
  * Explanation-quality tests with thresholds fixed in advance (constants
    below): completeness, deletion faithfulness, stability under support
    resampling, plus baseline sensitivity (reported, no threshold).

IMAGE (omniglot, resisc45)
  Decision-level report only: distance bars, support images, PCA, margins.

Deliberately NOT produced: counterfactuals toward a normal class ("what would
make this window look normal"). For AML that is an evasion recipe, not an
explanation.

The model is rebuilt through the encoder registry from architecture.json, and
the episode shape (n_way, k_shot, query, fold) comes from the artifact too, so
this component explains the same task that was trained and evaluated. With the
seed from the artifact, the test sampler draws the same episodes the heavy
evaluation scored.

`--model-dir` accepts a gs:// path (pipeline) or a local directory holding
model.pt + architecture.json (retroactive runs from a notebook).
"""
from __future__ import annotations

import argparse
import html as _html
import base64
import io
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

# --------------------------------------------------------------------------- #
# Explanation-quality thresholds. Fixed BEFORE any run on real data; moving
# them after reading a report would turn the tests into decoration.
# --------------------------------------------------------------------------- #
IG_STEPS = 50                    # default midpoint Riemann steps (override: --ig-steps)
COMPLETENESS_MAX_P95 = 0.01      # p95 of |sum(attr) - delta_m| / |delta_m|
FAITHFULNESS_K = 5               # features removed in the deletion test
FAITHFULNESS_MIN_RATIO = 2.0     # margin drop, top-k vs random-k removal
STABILITY_QUERIES = 30           # queries re-explained under new support sets
STABILITY_RESAMPLES = 20         # support resamples per query
STABILITY_MIN_SPEARMAN = 0.7     # median over queries of mean pairwise Spearman
DELETION_CURVE_K = 10
RANDOM_DELETION_REPEATS = 10

GREEN, ORANGE, RED, BLUE, GREY = "#2a7d4f", "#d9822b", "#c0392b", "#4C72B0", "#b8c4d8"
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


# --------------------------------------------------------------------------- #
# Feature metadata: human label, family, display kind
# --------------------------------------------------------------------------- #
_FEATURES: dict[str, tuple[str, str, str]] = {
    "n": ("transactions", "volume", "count"),
    "n_out": ("outgoing transactions", "volume", "count"),
    "out_ratio": ("outgoing share of transactions", "volume", "pct"),
    "amt_mean": ("mean amount", "amounts", "money"),
    "amt_std": ("amount spread (std)", "amounts", "money"),
    "amt_max": ("largest amount", "amounts", "money"),
    "amt_min": ("smallest amount", "amounts", "money"),
    "r100": ("share of round amounts (x100)", "threshold signals", "pct"),
    "s10k": ("share of amounts 9k-10k", "threshold signals", "pct"),
    "amt_out_sum": ("total sent", "flow balance", "money"),
    "amt_in_sum": ("total received", "flow balance", "money"),
    "net": ("net outflow (sent - received)", "flow balance", "money"),
    "flow_ratio": ("log(sent) - log(received)", "flow balance", "ratio"),
    "cpty_u": ("distinct counterparties", "counterparties", "count"),
    "n_days": ("active days", "rhythm", "count"),
    "tx_per_day": ("transactions per active day", "rhythm", "ratio"),
    "night": ("night-time share (00-06)", "rhythm", "pct"),
    "cash": ("cash share", "instruments", "pct"),
    "xb": ("cross-border share", "cross-border", "pct"),
    "curm": ("currency-mismatch share", "cross-border", "pct"),
    # --- case level (samld-case): one example = one flagged multi-account case ---
    "n_tx": ("transactions in case", "size", "count"),
    "n_acc": ("accounts in case", "size", "count"),
    "duration_days": ("case duration", "size", "days"),
    "mean_gap": ("mean gap between transactions", "rhythm", "days"),
    "share_src": ("share of pure senders", "account roles", "pct"),
    "share_sink": ("share of pure receivers", "account roles", "pct"),
    "share_mid": ("share of pass-through accounts", "account roles", "pct"),
    "max_out_share": ("largest fan-out (share of accounts)", "structure", "pct"),
    "max_in_share": ("largest fan-in (share of accounts)", "structure", "pct"),
    "deg_cv": ("degree inequality (CV)", "structure", "ratio"),
    "edge_repeat": ("transactions per account pair", "structure", "ratio"),
    "density": ("graph density", "structure", "ratio"),
    "depth": ("layers (longest path)", "structure", "count"),
    "has_cycle": ("contains a cycle", "structure", "flag"),
    "reciprocity": ("reciprocal links", "structure", "pct"),
    "is_bipartite": ("bipartite graph", "structure", "flag"),
    "amt_total": ("total amount", "amounts", "money"),
    "amt_cv": ("amount variability (CV)", "amounts", "ratio"),
    "amt_max_med": ("largest / median amount", "amounts", "ratio"),
    "passthrough": ("pass-through balance (in vs out)", "flow", "pct"),
    "fwd_1d": ("forwarded within a day", "flow", "pct"),
    "sink_conc": ("share received by top receiver", "flow", "pct"),
    "src_conc": ("share sent by top sender", "flow", "pct"),
}
_PREFIXES: list[tuple[str, str, str]] = [   # prefix -> family, kind (label derived)
    ("pt_", "instruments", "pct"),
    ("fwd_", "network", "pct"), ("bal_", "network", "ratio"), ("n_bal", "network", "ratio"),
    ("in_nb_", "network", "ratio"), ("out_nb_", "network", "ratio"), ("cyc", "network", "ratio"),
    ("hist_", "history", "ratio"), ("acct_age", "history", "ratio"),
    ("new_cp", "history", "pct"), ("max_day", "history", "ratio"),
]


def feature_meta(name: str) -> tuple[str, str, str]:
    """(label, family, kind) for a feature name; unknown names fall into 'other'."""
    if name in _FEATURES:
        return _FEATURES[name]
    for pre, fam, kind in _PREFIXES:
        if name.startswith(pre):
            label = ("share paid by " + name[3:].replace("_", " ")) if pre == "pt_" else name
            return label, fam, kind
    return name, "other", "ratio"


def fmt_value(v: float, kind: str) -> str:
    if kind == "money":
        return f"{v:,.0f}"
    if kind == "pct":
        return f"{100 * v:.1f}%" if 0 < abs(v) < 0.01 else f"{100 * v:.0f}%"
    if kind == "flag":
        return ("yes" if v >= 0.5 else "no") if v in (0.0, 1.0) else f"{100 * v:.0f}% of cases"
    if kind == "days":
        return f"{v:.1f} d"
    if kind == "count":
        return f"{int(round(v)):,}" if abs(v - round(v)) < 1e-6 else f"{v:,.1f}"
    return f"{v:.2f}"


def fmt_margin(m: float) -> str:
    return f"{m:.3f}" if abs(m) < 0.1 else f"{m:.2f}"


def esc(s) -> str:
    return _html.escape(str(s))


def _write(path: str, value) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(value))


# --------------------------------------------------------------------------- #
# SVG / HTML building blocks (no plotting libraries in the image)
# --------------------------------------------------------------------------- #
def svg_distance_bars(distances, labels, chosen_idx, true_idx=None, width=440):
    n = len(distances)
    row_h, pad_t = 26, 6
    pad_l = int(min(190, 14 + 6.4 * max(len(str(l)) for l in labels)))
    height = pad_t * 2 + n * row_h
    max_d = max(distances) or 1.0
    bar_max_w = width - pad_l - 48
    rows = []
    for i, (d, lab) in enumerate(zip(distances, labels)):
        y = pad_t + i * row_h
        w = (d / max_d) * bar_max_w
        fill = GREEN if i == chosen_idx else GREY
        stroke = (f' stroke="{ORANGE}" stroke-width="2.5"'
                  if (true_idx is not None and i == true_idx and i != chosen_idx) else "")
        rows.append(
            f'<text x="{pad_l - 8}" y="{y + 14}" font-size="11" text-anchor="end" fill="#333">{esc(lab)}</text>'
            f'<rect x="{pad_l}" y="{y + 3}" width="{w:.1f}" height="15" fill="{fill}"{stroke} rx="3"/>'
            f'<text x="{pad_l + w + 6:.1f}" y="{y + 14}" font-size="10.5" fill="#555">{d:.2f}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" width="{width}" xmlns="http://www.w3.org/2000/svg">'
            + "".join(rows) + "</svg>")


def svg_diverging_bars(items, pos_label, neg_label, width=440):
    """items: [(label, value)]; positive -> right/green, negative -> left/orange."""
    row_h, pad_t, label_w = 22, 26, 130
    height = pad_t + len(items) * row_h + 6
    mid = label_w + (width - label_w) / 2
    half = (width - label_w) / 2 - 34
    vmax = max([abs(v) for _, v in items] + [1e-12])
    parts = [f'<text x="{mid - 6:.0f}" y="14" font-size="10.5" text-anchor="end" fill="{ORANGE}">'
             f'&#8592; toward {esc(neg_label)}</text>'
             f'<text x="{mid + 6:.0f}" y="14" font-size="10.5" fill="{GREEN}">toward {esc(pos_label)} &#8594;</text>'
             f'<line x1="{mid:.0f}" y1="{pad_t - 4}" x2="{mid:.0f}" y2="{height - 4}" stroke="#999"/>']
    for i, (lab, v) in enumerate(items):
        y = pad_t + i * row_h
        w = abs(v) / vmax * half
        x = mid if v >= 0 else mid - w
        col = GREEN if v >= 0 else ORANGE
        tx = mid + w + 4 if v >= 0 else mid - w - 4
        anchor = "start" if v >= 0 else "end"
        parts.append(
            f'<text x="{label_w - 8}" y="{y + 13}" font-size="11" text-anchor="end" fill="#333">{esc(lab)}</text>'
            f'<rect x="{x:.1f}" y="{y + 2}" width="{max(w, 0.5):.1f}" height="14" fill="{col}" rx="2"/>'
            f'<text x="{tx:.1f}" y="{y + 13}" font-size="10" text-anchor="{anchor}" fill="#555">{v:+.0%}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" width="{width}" xmlns="http://www.w3.org/2000/svg">'
            + "".join(parts) + "</svg>")


def svg_pca_scatter(proto_xy, query_xy, query_labels, names, width=560, height=400):
    import numpy as np
    allxy = np.vstack([proto_xy, query_xy])
    lo, hi = allxy.min(0), allxy.max(0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    lo, hi = lo - 0.06 * span, hi + 0.06 * span
    pad = 30

    def px(p):
        x = pad + (p[0] - lo[0]) / ((hi[0] - lo[0]) or 1) * (width - 2 * pad - 110)
        y = pad + (1 - (p[1] - lo[1]) / ((hi[1] - lo[1]) or 1)) * (height - 2 * pad)
        return x, y

    parts = []
    for i, q in enumerate(query_xy):
        x, y = px(q)
        c = PALETTE[int(query_labels[i]) % len(PALETTE)]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{c}" opacity="0.55"/>')
    for k in range(len(proto_xy)):
        x, y = px(proto_xy[k])
        c = PALETTE[k % len(PALETTE)]
        parts.append(f'<rect x="{x - 7:.1f}" y="{y - 7:.1f}" width="14" height="14" fill="{c}" '
                     f'stroke="#222" stroke-width="1.5" transform="rotate(45 {x:.1f} {y:.1f})"/>')
    for k, nm in enumerate(names):   # legend on the right, never overlapping points
        c = PALETTE[k % len(PALETTE)]
        ly = 30 + 20 * k
        parts.append(f'<rect x="{width - 128}" y="{ly - 9}" width="10" height="10" fill="{c}"/>'
                     f'<text x="{width - 113}" y="{ly}" font-size="10.5" fill="#333">{esc(nm)}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" width="{width}" xmlns="http://www.w3.org/2000/svg">'
            f'<rect width="{width}" height="{height}" fill="#fafafa" rx="8"/>'
            + "".join(parts) + "</svg>")


def svg_histogram(values, bins=25, width=640, height=260, color=BLUE, title=""):
    import numpy as np
    counts, edges = np.histogram(values, bins=bins)
    max_c = counts.max() if counts.max() > 0 else 1
    pad_l, pad_t = 40, 28
    plot_w, plot_h = width - pad_l - 20, height - 64
    lo, hi = float(edges[0]), float(edges[-1])
    span = (hi - lo) or 1.0
    bars, ticks = [], []
    for i, c in enumerate(counts):
        x0 = pad_l + (edges[i] - lo) / span * plot_w
        x1 = pad_l + (edges[i + 1] - lo) / span * plot_w
        y = pad_t + plot_h - (c / max_c) * plot_h
        bars.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(x1 - x0 - 1, 1):.1f}" '
                    f'height="{pad_t + plot_h - y:.1f}" fill="{color}"/>')
    for v in [lo, (lo + hi) / 2, hi]:
        x = pad_l + (v - lo) / span * plot_w
        ticks.append(f'<text x="{x:.1f}" y="{height - 14}" font-size="11" '
                     f'text-anchor="middle" fill="#555">{v:.2f}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" width="{width}" xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{width / 2}" y="16" font-size="12.5" text-anchor="middle" fill="#333">{esc(title)}</text>'
            f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - 20}" y2="{pad_t + plot_h}" stroke="#999"/>'
            + "".join(bars) + "".join(ticks) + "</svg>")


def svg_lines(xs, series, title, y_label, width=560, height=280, zero_line=True):
    """series: [(name, ys, colour, dashed)]"""
    import numpy as np
    pad_l, pad_r, pad_t, pad_b = 52, 150, 28, 40
    allv = np.concatenate([np.asarray(s[1], float) for s in series] + ([np.zeros(1)] if zero_line else []))
    lo, hi = float(allv.min()), float(allv.max())
    pad = 0.08 * ((hi - lo) or 1.0)
    lo, hi = lo - pad, hi + pad
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b

    def P(x, y):
        return (pad_l + (x - xs[0]) / ((xs[-1] - xs[0]) or 1) * pw,
                pad_t + (1 - (y - lo) / ((hi - lo) or 1)) * ph)

    parts = [f'<text x="{pad_l + pw / 2:.0f}" y="16" font-size="12.5" text-anchor="middle" fill="#333">{esc(title)}</text>',
             f'<line x1="{pad_l}" y1="{pad_t + ph}" x2="{pad_l + pw}" y2="{pad_t + ph}" stroke="#999"/>',
             f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + ph}" stroke="#999"/>',
             f'<text x="14" y="{pad_t + ph / 2:.0f}" font-size="11" fill="#555" '
             f'transform="rotate(-90 14 {pad_t + ph / 2:.0f})" text-anchor="middle">{esc(y_label)}</text>']
    for x in xs:
        X, _ = P(x, lo)
        parts.append(f'<text x="{X:.1f}" y="{pad_t + ph + 16}" font-size="10.5" text-anchor="middle" fill="#555">{x}</text>')
    parts.append(f'<text x="{pad_l + pw / 2:.0f}" y="{height - 6}" font-size="11" text-anchor="middle" '
                 f'fill="#555">features removed (replaced by the contrast class\'s typical value)</text>')
    for v in (lo + pad, hi - pad):
        _, Y = P(xs[0], v)
        parts.append(f'<text x="{pad_l - 6}" y="{Y + 4:.1f}" font-size="10.5" text-anchor="end" fill="#555">{v:.2f}</text>')
    if zero_line and lo < 0 < hi:
        _, Y = P(xs[0], 0.0)
        parts.append(f'<line x1="{pad_l}" y1="{Y:.1f}" x2="{pad_l + pw}" y2="{Y:.1f}" stroke="{RED}" '
                     f'stroke-dasharray="4,3"/><text x="{pad_l + pw + 6}" y="{Y + 4:.1f}" font-size="10" '
                     f'fill="{RED}">decision flips</text>')
    for i, (name, ys, col, dashed) in enumerate(series):
        pts = " ".join(f"{P(x, y)[0]:.1f},{P(x, y)[1]:.1f}" for x, y in zip(xs, ys))
        dash = ' stroke-dasharray="6,4"' if dashed else ""
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.2"{dash}/>')
        ly = pad_t + 14 + 18 * i
        parts.append(f'<line x1="{pad_l + pw + 8}" y1="{ly - 4}" x2="{pad_l + pw + 26}" y2="{ly - 4}" '
                     f'stroke="{col}" stroke-width="2.2"{dash}/><text x="{pad_l + pw + 30}" y="{ly}" '
                     f'font-size="10.5" fill="#333">{esc(name)}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" width="{width}" xmlns="http://www.w3.org/2000/svg">'
            + "".join(parts) + "</svg>")


def _cell(v, rgb, txt=None, bold=False):
    a = max(0.0, min(1.0, float(v)))
    fw = "font-weight:600;" if bold else ""
    return (f'<td style="background:rgba({rgb},{0.08 + 0.8 * a:.2f});{fw}text-align:center">'
            f'{txt if txt is not None else f"{v:.0%}"}</td>')


def html_confusion(classes, counts):
    import numpy as np
    C = np.asarray(counts, float)
    R = C / np.maximum(C.sum(1, keepdims=True), 1)
    head = "".join(f'<th class="rot"><div>{esc(c)}</div></th>' for c in classes)
    rows = ""
    for i, c in enumerate(classes):
        cells = "".join(_cell(R[i, j], "42,125,79" if i == j else "192,57,43", bold=(i == j))
                        if (i == j or R[i, j] > 0) else '<td style="text-align:center;color:#bbb">&middot;</td>'
                        for j in range(len(classes)))
        rows += f'<tr><th style="text-align:right">{esc(c)}</th>{cells}<td class="n">{int(C[i].sum())}</td></tr>'
    return (f'<table class="grid"><tr><th style="text-align:right">true &#8595; / predicted &#8594;</th>'
            f'{head}<th>n</th></tr>{rows}</table>')


def html_typology_map(classes, families, shares, competitor):
    head = "".join(f'<th class="rot"><div>{esc(f)}</div></th>' for f in families)
    rows = ""
    for c in classes:
        sh = shares.get(c)
        if sh is None:
            rows += (f'<tr><th style="text-align:right">{esc(c)}</th>'
                     f'<td colspan="{len(families) + 1}" style="color:#999">no correct decisions to explain</td></tr>')
            continue
        top = max(sh.values()) or 1.0
        cells = "".join(_cell(sh[f] / top, "76,114,176", f"{sh[f]:.0%}") for f in families)
        comp, frac = competitor[c]
        rows += (f'<tr><th style="text-align:right">{esc(c)}</th>{cells}'
                 f'<td style="font-size:12px;color:#555;padding-left:10px">{esc(comp)} ({frac:.0%})</td></tr>')
    return (f'<table class="grid"><tr><th></th>{head}<th style="text-align:left;padding-left:44px;vertical-align:bottom">'
            f'closest competitor</th></tr>{rows}</table>')


# --------------------------------------------------------------------------- #
# Episode mechanics
# --------------------------------------------------------------------------- #
def episode_breakdown(model, data, labels, n_way, k_shot, query, device, idx=None):
    """Raw view of one episode.

    MUST match fsl.models.protonet.fast_adapt: stable label sort, the first
    `k_shot` rows of each class block are support, logits = -cdist.
    """
    import numpy as np
    import torch
    order = torch.sort(labels, stable=True).indices
    data, labels = data[order].to(device), labels[order].to(device)
    with torch.no_grad():
        emb = model(data)
    sm = np.zeros(data.size(0), dtype=bool)
    sel = np.arange(n_way) * (k_shot + query)
    for off in range(k_shot):
        sm[sel + off] = True
    smt, qmt = torch.from_numpy(sm).to(device), torch.from_numpy(~sm).to(device)
    s_emb = emb[smt].reshape(n_way, k_shot, -1)
    protos = s_emb.mean(dim=1)
    q_emb = emb[qmt]
    dists = torch.cdist(q_emb, protos)
    sorted_d, sorted_i = dists.sort(dim=1)
    out = dict(
        protos=protos, support_x=data[smt].reshape(n_way, k_shot, *data.shape[1:]), support_emb=s_emb,
        q_x=data[qmt], q_emb=q_emb, q_labels=labels[qmt].long(), dists=dists,
        preds=sorted_i[:, 0], second=sorted_i[:, 1], margins=sorted_d[:, 1] - sorted_d[:, 0],
    )
    if idx is not None:
        idx = np.asarray(idx)[order.cpu().numpy()]
        out["support_idx"] = idx[sm].reshape(n_way, k_shot)
        out["q_idx"] = idx[~sm]
    return out


def margin_fn(model, x, p_c, p_r):
    """Contrastive margin = logit(chosen) - logit(contrast) under logits = -cdist."""
    e = model(x)
    return (e - p_r).norm(dim=-1) - (e - p_c).norm(dim=-1)


def contrastive_ig(model, x, x0, p_c, p_r, steps=IG_STEPS, chunk=256):
    """Integrated Gradients of the contrastive margin, x0 -> x, midpoint rule.

    Args: x, x0 (N, F); p_c, p_r (N, D). Returns (attr (N, F), m(x) (N,), m(x0) (N,)).
    """
    import torch
    alphas = (torch.arange(steps, dtype=x.dtype) + 0.5) / steps
    attrs, mx, m0 = [], [], []
    for s in range(0, x.shape[0], chunk):
        xb, x0b, pc, pr = x[s:s + chunk], x0[s:s + chunk], p_c[s:s + chunk], p_r[s:s + chunk]
        B, F = xb.shape
        path = (x0b[:, None, :] + alphas[None, :, None] * (xb - x0b)[:, None, :]).reshape(B * steps, F)
        path.requires_grad_(True)
        m = margin_fn(model, path, pc.repeat_interleave(steps, 0), pr.repeat_interleave(steps, 0))
        g = torch.autograd.grad(m.sum(), path)[0].reshape(B, steps, F).mean(1)
        attrs.append(((xb - x0b) * g).detach())
        with torch.no_grad():
            mx.append(margin_fn(model, xb, pc, pr))
            m0.append(margin_fn(model, x0b, pc, pr))
    return torch.cat(attrs), torch.cat(mx), torch.cat(m0)


def _ranks(a):
    import numpy as np
    return np.argsort(np.argsort(a, axis=-1), axis=-1).astype(float)


def spearman_rows(a, b):
    """Row-wise Spearman between two (N, F) arrays."""
    import numpy as np
    ra, rb = _ranks(a), _ranks(b)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    return (ra * rb).sum(1) / np.where(den > 0, den, 1.0)


def mean_pairwise_spearman(A):
    """Mean Spearman over all pairs of rows of A (R, F)."""
    import numpy as np
    r = _ranks(A)
    r -= r.mean(1, keepdims=True)
    r /= np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-12)
    C = r @ r.T
    iu = np.triu_indices(len(A), 1)
    return float(C[iu].mean())


# --------------------------------------------------------------------------- #
# Artifact
# --------------------------------------------------------------------------- #
def fetch_artifact(model_dir: str, project: str) -> Path:
    from fsl.errors import explain_failure, require
    if model_dir.startswith("gs://"):
        from google.cloud import storage
        bucket_name, _, prefix = model_dir[len("gs://"):].partition("/")
        local = Path(tempfile.mkdtemp())
        bucket = storage.Client(project=project).bucket(bucket_name)
        with explain_failure("downloading model artifact from GCS", bucket=bucket_name):
            for name in ("model.pt", "architecture.json"):
                bucket.blob(f"{prefix.rstrip('/')}/{name}").download_to_filename(str(local / name))
        return local
    local = Path(model_dir).expanduser()
    for name in ("model.pt", "architecture.json"):
        require((local / name).exists(), f"{local / name} not found (expected model.pt + architecture.json)")
    return local


# --------------------------------------------------------------------------- #
# Tabular explanation
# --------------------------------------------------------------------------- #
def explain_tabular(model, test_tasks, bundle, cfg, arch, args, device):
    import numpy as np
    import torch

    n_way, k_shot, q = cfg.n_way, cfg.k_shot, cfg.query
    schema = bundle["schema"]
    W = bundle["windows"]
    feats = list(schema["feature_names"])
    F = len(feats)
    meta = [feature_meta(f) for f in feats]
    families = list(dict.fromkeys(m[1] for m in meta))
    fam_of = np.array([families.index(m[1]) for m in meta])
    onehot = np.zeros((F, len(families)))
    onehot[np.arange(F), fam_of] = 1.0
    raw = W[feats].astype("float64").to_numpy()
    accs = W[schema["group_col"]].to_numpy()
    weeks = W["week"].to_numpy() if "week" in W.columns else np.full(len(W), -1)
    unit = str(schema.get("unit", "window"))
    is_case = unit == "case" and schema.get("id_col") in W.columns
    ids = W[schema["id_col"]].to_numpy() if is_case else accs
    n_acc_col = W["n_acc"].to_numpy() if "n_acc" in W.columns else np.zeros(len(W), int)
    n_tx_col = W["n_tx"].to_numpy() if "n_tx" in W.columns else np.zeros(len(W), int)

    def prov(row) -> dict:
        if is_case:
            return {"case": str(ids[row]), "accounts": int(n_acc_col[row]), "transactions": int(n_tx_col[row])}
        return {"account": str(accs[row]), "week": int(weeks[row])}

    def subject_html(row) -> str:
        if is_case:
            return (f"Case <b>#{esc(ids[row])}</b> ({int(n_acc_col[row])} accounts, "
                    f"{int(n_tx_col[row])} transactions)")
        return f"Window <b>account {esc(accs[row])} &middot; week {esc(weeks[row])}</b>"

    def prec_row(row, dist) -> str:
        a_, b_ = (ids[row], int(n_acc_col[row])) if is_case else (accs[row], weeks[row])
        return f"<tr><td>{esc(a_)}</td><td class='num'>{esc(b_)}</td><td class='num'>{dist:.2f}</td></tr>"

    prec_head = "<th>case</th><th>accounts</th>" if is_case else "<th>account</th><th>week</th>"
    scope_unit = (f"flagged cases ({esc(schema.get('framing', 'case level'))})" if is_case
                  else "windows (account &times; week)")
    lab_all = W[schema["label_col"]].to_numpy().astype(str)
    X_std = torch.from_numpy(np.asarray(test_tasks.X, dtype=np.float32))
    rng = np.random.default_rng(int(cfg.seed) + 777)
    torch.manual_seed(int(cfg.seed))

    # ---- 1. scan test episodes ------------------------------------------- #
    eps, rec = [], defaultdict(list)
    for e in range(args.n_episodes):
        data, labels, idx, names = test_tasks.sample_indexed()
        ep = episode_breakdown(model, data, labels, n_way, k_shot, q, device, idx=idx)
        ep["names"] = names
        eps.append(ep)
        true, pred, second = ep["q_labels"].cpu(), ep["preds"].cpu(), ep["second"].cpu()
        contrast = torch.where(pred == true, second, true)
        nq = len(true)
        rec["ep"] += [e] * nq
        rec["qi"] += list(range(nq))
        rec["true"] += true.tolist()
        rec["pred"] += pred.tolist()
        rec["contrast"] += contrast.tolist()
        rec["q_idx"] += ep["q_idx"].tolist()
        rec["dist_c"] += ep["dists"].cpu()[torch.arange(nq), pred].tolist()
        rec["dist_r"] += ep["dists"].cpu()[torch.arange(nq), contrast].tolist()
        rec["p_c"].append(ep["protos"].cpu()[pred])
        rec["p_r"].append(ep["protos"].cpu()[contrast])
        rec["x"].append(ep["q_x"].cpu())
        rec["x0"].append(ep["support_x"].cpu()[contrast].mean(1))
        rec["names_t"] += [names[i] for i in true.tolist()]
        rec["names_p"] += [names[i] for i in pred.tolist()]
        rec["names_r"] += [names[i] for i in contrast.tolist()]
    x = torch.cat(rec["x"])
    x0 = torch.cat(rec["x0"])
    p_c, p_r = torch.cat(rec["p_c"]), torch.cat(rec["p_r"])
    true = np.array(rec["true"]); pred = np.array(rec["pred"])
    q_idx = np.array(rec["q_idx"])
    nt, npred, nr = np.array(rec["names_t"]), np.array(rec["names_p"]), np.array(rec["names_r"])
    correct = true == pred
    N = len(true)
    print(f"scanned {args.n_episodes} episodes, {N} query decisions, accuracy {correct.mean():.3f}")

    # ---- 2. attributions (contrast-class baseline + average-window baseline) #
    steps = int(args.ig_steps)
    attr_t, m_x_t, m_0_t = contrastive_ig(model, x, x0, p_c, p_r, steps=steps)
    attr0_t, _, _ = contrastive_ig(model, x, torch.zeros_like(x), p_c, p_r, steps=steps)
    attr, attr0 = attr_t.numpy().astype(float), attr0_t.numpy().astype(float)
    m_x, m_0 = m_x_t.numpy().astype(float), m_0_t.numpy().astype(float)
    delta = m_x - m_0
    resid = np.abs(attr.sum(1) - delta) / np.maximum(np.abs(delta), 1e-8)
    fam_abs = np.abs(attr) @ onehot
    fam_abs_share = fam_abs / np.maximum(fam_abs.sum(1, keepdims=True), 1e-12)
    fam_pos = np.clip(attr, 0, None) @ onehot

    # ---- 3. global view: confusion, typology map, top confusions ---------- #
    classes = sorted(set(nt.tolist()))
    ci = {c: i for i, c in enumerate(classes)}
    conf = np.zeros((len(classes), len(classes)), int)
    for a, b in zip(nt, npred):
        conf[ci[a], ci[b]] += 1
    recall = {c: float(conf[i, i] / max(conf[i].sum(), 1)) for c, i in ci.items()}

    tmap, competitor = {}, {}
    for c in classes:
        m = (nt == c) & correct
        if m.sum() == 0:
            continue
        tmap[c] = {f: float(v) for f, v in zip(families, fam_abs_share[m].mean(0))}
        comp, cnt = Counter(nr[m].tolist()).most_common(1)[0]
        competitor[c] = (comp, cnt / m.sum())
    fam_order = sorted(families, key=lambda f: -np.mean([tmap[c][f] for c in tmap]) if tmap else 0)

    rows = conf / np.maximum(conf.sum(1, keepdims=True), 1)
    off = [(rows[i, j], classes[i], classes[j]) for i in range(len(classes))
           for j in range(len(classes)) if i != j and conf[i, j] > 0]
    top_conf = []
    for rate, a, b in sorted(off, reverse=True)[:3]:
        m = (nt == a) & (npred == b)
        fp = fam_pos[m].mean(0)
        fp = fp / max(fp.sum(), 1e-12)
        fpos = np.clip(attr[m], 0, None).mean(0)
        top_conf.append({
            "true": a, "pred": b, "rate": float(rate), "n": int(m.sum()),
            "families": {families[k]: float(fp[k]) for k in np.argsort(-fp)[:3]},
            "features": [meta[k][0] for k in np.argsort(-fpos)[:3]],
        })

    # ---- 4. explanation-quality tests ------------------------------------- #
    comp_p95, comp_med = float(np.quantile(resid, 0.95)), float(np.median(resid))
    completeness = {"p95_rel_residual": comp_p95, "median_rel_residual": comp_med,
                    "threshold_p95": COMPLETENESS_MAX_P95, "steps": steps,
                    "passed": comp_p95 <= COMPLETENESS_MAX_P95}

    order_top = np.argsort(-attr, axis=1)
    ks = list(range(0, min(DELETION_CURVE_K, F) + 1))

    def margins_after(order_mat, k):
        if k == 0:
            return m_x
        mask = np.zeros((N, F), bool)
        np.put_along_axis(mask, order_mat[:, :k], True, axis=1)
        xk = torch.where(torch.from_numpy(mask), x0, x)
        with torch.no_grad():
            return margin_fn(model, xk, p_c, p_r).numpy().astype(float)

    curve_top = [margins_after(order_top, k) for k in ks]
    rand_orders = [np.argsort(rng.random((N, F)), axis=1) for _ in range(RANDOM_DELETION_REPEATS)]
    curve_rand = [np.mean([margins_after(o, k) for o in rand_orders], axis=0) for k in ks]
    kf = min(FAITHFULNESS_K, F)
    m_top_k, m_rand_k = margins_after(order_top, kf), np.mean([margins_after(o, kf) for o in rand_orders], 0)
    drop_top, drop_rand = float(np.mean(m_x - m_top_k)), float(np.mean(m_x - m_rand_k))
    ratio = drop_top / drop_rand if drop_rand > 1e-9 else (float("inf") if drop_top > 0 else 0.0)
    flip_top = float(np.mean(m_top_k < 0))
    flip_rand = float(np.mean([(margins_after(o, kf) < 0).mean() for o in rand_orders]))
    faithfulness = {"k": kf, "margin_drop_top": drop_top, "margin_drop_random": drop_rand,
                    "ratio": ratio, "threshold_ratio": FAITHFULNESS_MIN_RATIO,
                    "flip_rate_top": flip_top, "flip_rate_random": flip_rand,
                    "passed": ratio >= FAITHFULNESS_MIN_RATIO}

    # stability: same query, same (chosen, contrast) pair, new support windows
    cand = np.flatnonzero(correct)
    rng.shuffle(cand)
    by_cls = defaultdict(list)
    for n in cand:
        by_cls[nt[n]].append(n)
    picked, r = [], 0
    while len(picked) < STABILITY_QUERIES and any(len(v) > r for v in by_cls.values()):
        picked += [v[r] for v in by_cls.values() if len(v) > r]
        r += 1
    picked = picked[:STABILITY_QUERIES]
    rhos, kept, pears, topfam = [], [], [], []
    for n in picked:
        acc_q = accs[q_idx[n]]
        pools = [np.flatnonzero((lab_all == nm) & (accs != acc_q)) for nm in (npred[n], nr[n])]
        if min(len(p) for p in pools) < k_shot:
            continue
        sc = np.stack([rng.choice(pools[0], k_shot, replace=False) for _ in range(STABILITY_RESAMPLES)])
        sr = np.stack([rng.choice(pools[1], k_shot, replace=False) for _ in range(STABILITY_RESAMPLES)])
        with torch.no_grad():
            pc = model(X_std[sc.ravel()]).reshape(STABILITY_RESAMPLES, k_shot, -1).mean(1)
            pr = model(X_std[sr.ravel()]).reshape(STABILITY_RESAMPLES, k_shot, -1).mean(1)
        xb = x[n:n + 1].repeat(STABILITY_RESAMPLES, 1)
        x0b = X_std[sr.ravel()].reshape(STABILITY_RESAMPLES, k_shot, -1).mean(1)
        a, mx, _ = contrastive_ig(model, xb, x0b, pc, pr, steps=steps)
        a = a.numpy().astype(float)
        rhos.append(mean_pairwise_spearman(a))
        kept.append(float((mx > 0).float().mean()))
        C = np.corrcoef(a)                                   # magnitude-weighted agreement
        pears.append(float(C[np.triu_indices(len(a), 1)].mean()))
        tf = np.argmax(a @ onehot, axis=1)                   # family pushing hardest toward chosen
        topfam.append(float(np.bincount(tf).max() / len(tf)))
    stab_med = float(np.median(rhos)) if rhos else float("nan")
    stability = {"queries": len(rhos), "resamples": STABILITY_RESAMPLES,
                 "median_spearman": stab_med, "threshold": STABILITY_MIN_SPEARMAN,
                 "decision_retained": float(np.mean(kept)) if kept else float("nan"),
                 # diagnostics, no threshold: WHERE any instability lives
                 "median_pearson": float(np.median(pears)) if pears else float("nan"),
                 "top_family_agreement": float(np.mean(topfam)) if topfam else float("nan"),
                 "passed": bool(rhos) and stab_med >= STABILITY_MIN_SPEARMAN}
    base_agree = float(np.median(spearman_rows(attr, attr0)))
    tests = {"completeness": completeness, "faithfulness": faithfulness, "stability": stability,
             "baseline_agreement_spearman": base_agree}

    # ---- 5. case studies -------------------------------------------------- #
    def raw_mean(rows_):
        return raw[np.asarray(rows_)].mean(0)

    def case(n, kind, title, subtitle):
        ep = eps[rec["ep"][n]]
        qi, c, rr, t = rec["qi"][n], pred[n], rec["contrast"][n], true[n]
        names = ep["names"]
        row = q_idx[n]
        sup_c, sup_r = ep["support_idx"][c], ep["support_idx"][rr]
        rv, rc, rrv = raw[row], raw_mean(sup_c), raw_mean(sup_r)
        a = attr[n]
        tot = a.sum() if abs(a.sum()) > 1e-12 else 1.0
        share = a / tot
        top = np.argsort(-np.abs(a))[:8]
        pos = [k for k in np.argsort(-a) if a[k] > 0][:3]
        neg = [k for k in np.argsort(a) if a[k] < 0 and abs(a[k]) > 0.10 * abs(tot)][:1]

        def phrase(k):
            lab_, _, kind_ = meta[k]
            return (f"<b>{esc(lab_)}</b> = {fmt_value(rv[k], kind_)} "
                    f"({esc(names[c])} support avg {fmt_value(rc[k], kind_)}, "
                    f"{esc(names[rr])} support avg {fmt_value(rrv[k], kind_)})")

        why = "; ".join(phrase(k) for k in pos) or "no single feature dominates"
        sent = (f"{subject_html(row)} was classified as "
                f"<b>{esc(names[c])}</b> rather than <b>{esc(names[rr])}</b> "
                f"(margin {fmt_margin(rec['dist_r'][n] - rec['dist_c'][n])}). Main reasons: {why}.")
        if neg:
            sent += f" Pulling the other way: {phrase(neg[0])}."
        if t != c:
            sent += f" <span style='color:{RED}'>True typology: <b>{esc(names[t])}</b>.</span>"
        plain = _html.unescape(sent.replace("<b>", "").replace("</b>", "")
                               .replace(f"<span style='color:{RED}'>", "").replace("</span>", ""))

        fam_items = sorted(((families[j], float((a @ onehot)[j] / tot)) for j in range(len(families))),
                           key=lambda z: -abs(z[1]))
        fam_svg = svg_diverging_bars(fam_items, names[c], names[rr])
        dist_svg = svg_distance_bars([float(d) for d in ep["dists"][qi]], names, chosen_idx=c,
                                     true_idx=(t if t != c else None))
        trs = ""
        for k in top:
            lab_, fam_, kind_ = meta[k]
            col = GREEN if a[k] >= 0 else ORANGE
            w = min(100, abs(share[k]) / max(np.abs(share[top]).max(), 1e-12) * 100)
            trs += (f"<tr><td>{esc(lab_)}</td><td class='fam'>{esc(fam_)}</td>"
                    f"<td class='num'><b>{fmt_value(rv[k], kind_)}</b></td>"
                    f"<td class='num'>{fmt_value(rc[k], kind_)}</td><td class='num'>{fmt_value(rrv[k], kind_)}</td>"
                    f"<td class='num' style='color:{col}'>{share[k]:+.0%}</td>"
                    f"<td style='width:90px'><div style='height:9px;width:{w:.0f}%;background:{col};"
                    f"border-radius:2px'></div></td></tr>")
        with torch.no_grad():
            d_sup = (ep["support_emb"][c] - ep["q_emb"][qi]).norm(dim=-1).cpu().numpy()
        prec = "".join(prec_row(sup_c[s], d_sup[s]) for s in np.argsort(d_sup))
        verdict = (f"<span style='color:{GREEN}'>correct</span>" if t == c else
                   f"<span style='color:{RED}'>wrong &mdash; true: {esc(names[t])}</span>")
        html = f"""<div class="case">
<h3>{esc(title)}</h3><p class="sub">{esc(subtitle)}</p>
<div class="narr">{sent}</div>
<div class="row">
 <div><div class="cap">distance to each prototype (shortest wins) &mdash; predicted <b>{esc(names[c])}</b>, {verdict}</div>{dist_svg}</div>
 <div><div class="cap">what moved the decision, by feature family (share of the swing)</div>{fam_svg}</div>
</div>
<div class="row">
 <div style="flex:2"><div class="cap">top features by contribution (raw units; support avg = the episode's {k_shot} support {unit}s)</div>
 <table class="t"><tr><th>feature</th><th>family</th><th>this {unit}</th><th>{esc(names[c])}<br/>support avg</th>
 <th>{esc(names[rr])}<br/>support avg</th><th colspan="2">push</th></tr>{trs}</table></div>
 <div style="flex:1"><div class="cap">precedents: {esc(names[c])} support {unit}s, nearest first</div>
 <table class="t"><tr>{prec_head}<th>distance</th></tr>{prec}</table></div>
</div></div>"""
        js = {"kind": kind, "provenance": prov(row), "true": names[t],
              "predicted": names[c], "contrast": names[rr], "margin": float(rec["dist_r"][n] - rec["dist_c"][n]),
              "narrative": plain,
              "top_features": [{"feature": feats[k], "label": meta[k][0], "family": meta[k][1],
                                "value": float(rv[k]), "chosen_support_avg": float(rc[k]),
                                "contrast_support_avg": float(rrv[k]), "share_of_margin": float(share[k])}
                               for k in top],
              "precedents": [{**prov(sup_c[s]), "distance": float(d_sup[s])} for s in np.argsort(d_sup)]}
        return html, js

    dm = np.array(rec["dist_r"]) - np.array(rec["dist_c"])
    case_html, case_js = "", []
    ok_idx = np.flatnonzero(correct)
    if len(ok_idx):
        n = ok_idx[np.argmax(dm[ok_idx])]
        h, j = case(n, "confident", "Most confident correct decision", "largest gap to the runner-up")
        case_html += h; case_js.append(j)
        n = ok_idx[np.argmin(dm[ok_idx])]
        h, j = case(n, "borderline", "Most borderline correct decision",
                    "smallest gap: the runner-up almost won")
        case_html += h; case_js.append(j)
    if top_conf:
        tc = top_conf[0]
        err = np.flatnonzero((nt == tc["true"]) & (npred == tc["pred"]))
        n = err[np.argmax(dm[err])]
        h, j = case(n, "mistake", f"Most common mistake: {tc['true']} taken for {tc['pred']}",
                    "the most confident instance of the most frequent confusion; contrast = the true class")
        case_html += h; case_js.append(j)
    else:
        case_html += ('<div class="case" style="border-style:dashed;color:#666">No misclassification in '
                      f'the sampled {args.n_episodes} episodes.</div>')

    # ---- 6. embedding map + margin histogram ------------------------------ #
    ep0 = eps[0]
    both = torch.cat([ep0["protos"], ep0["q_emb"]]).cpu()
    Xc = both - both.mean(0, keepdim=True)
    _, _, V = torch.linalg.svd(Xc, full_matrices=False)
    xy = (Xc @ V[:2].T).numpy()
    pca_svg = svg_pca_scatter(xy[:n_way], xy[n_way:], ep0["q_labels"].cpu().numpy(), ep0["names"])
    margins_all = np.concatenate([e["margins"].cpu().numpy() for e in eps])
    hist = svg_histogram(margins_all, title=f"Decision margins, {len(margins_all)} decisions "
                                             f"({args.n_episodes} episodes)")

    # ---- 7. assemble ------------------------------------------------------- #
    def badge(name, t, detail):
        col = GREEN if t["passed"] else RED
        return (f'<div class="badge" style="border-color:{col}"><div class="bn">{name}</div>'
                f'<div class="bv" style="color:{col}">{"PASS" if t["passed"] else "FAIL"}</div>'
                f'<div class="bd">{detail}</div></div>')

    fr = "&infin;" if faithfulness["ratio"] == float("inf") else f"{faithfulness['ratio']:.1f}&times;"
    badges = (badge("completeness", completeness, f"p95 residual {comp_p95:.2%}")
              + badge("faithfulness", faithfulness, f"top-{kf} vs random-{kf}: {fr}")
              + badge("stability", stability, f"median &rho; {stab_med:.2f}"))
    conf_lines = "".join(
        f"<li><b>{esc(t['true'])}</b> taken for <b>{esc(t['pred'])}</b> in {t['rate']:.0%} of its queries "
        f"(n={t['n']}). Push toward {esc(t['pred'])} came from "
        + ", ".join(f"{esc(f)} ({v:.0%})" for f, v in t["families"].items())
        + f"; strongest features: {', '.join(esc(z) for z in t['features'])}.</li>" for t in top_conf)
    curve_svg = svg_lines(ks, [
        ("top-attributed first", [float(np.mean(c)) for c in curve_top], GREEN, False),
        ("random order", [float(np.mean(c)) for c in curve_rand], "#888", True)],
        "Deletion test: mean margin as features are removed", "mean margin")
    acc_mean = float(correct.mean())
    failed = [k for k, t in (("completeness", completeness), ("faithfulness", faithfulness),
                             ("stability", stability)) if not t["passed"]]
    trust_line = ("All three tests pass: the attributions are exact, they name what the model actually uses, "
                  f"and they do not hinge on which support {unit}s were drawn." if not failed else
                  f"Failed: {', '.join(failed)}. Read the case studies with that in mind; details at the end "
                  "of the report.")
    if not stability["passed"] and stability["top_family_agreement"] >= 0.7:
        stab_line = ("Feature-level rankings move with the support set while the leading family mostly "
                     "holds: read single cases at the family level, not as an exact feature order.")
    elif not stability["passed"]:
        stab_line = (f"Explanations depend strongly on which support {unit}s were drawn: treat single-case "
                     "feature rankings as illustrative and rely on the aggregated views above.")
    else:
        stab_line = "Single-case feature rankings can be read as stated."
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;color:#1a1a1a;max-width:1060px;background:#fff}}
h2{{margin-bottom:2px}} h3{{margin:0 0 4px 0}} .meta{{font-size:12px;color:#888;margin-bottom:14px}}
.primer{{background:#f2f6fb;border-radius:10px;padding:14px 18px;font-size:14px;line-height:1.55}}
.badges{{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}}
.badge{{border:2px solid;border-radius:10px;padding:8px 14px;min-width:170px}}
.bn{{font-size:12px;color:#666}} .bv{{font-size:20px;font-weight:700}} .bd{{font-size:12px;color:#555}}
.case{{border:1px solid #e0e0e0;border-radius:10px;padding:16px 18px;margin:18px 0}}
.sub{{margin:0 0 10px 0;font-size:13px;color:#666}} .cap{{font-size:12px;color:#666;margin-bottom:6px}}
.narr{{background:#fbfaf4;border-left:3px solid #CCB974;padding:9px 12px;font-size:13.5px;line-height:1.55;margin-bottom:12px}}
.row{{display:flex;gap:26px;flex-wrap:wrap;margin-top:10px}} .row>div{{flex:1;min-width:300px}}
table.t{{border-collapse:collapse;font-size:12.5px;width:100%}} .t th,.t td{{border-bottom:1px solid #eee;padding:4px 6px;text-align:left}}
.t th{{font-weight:600;color:#555;font-size:11.5px}} .num{{text-align:right!important;font-variant-numeric:tabular-nums}}
.fam{{color:#777;font-size:11.5px}}
table.grid{{border-collapse:separate;border-spacing:2px;font-size:12.5px;margin:8px 0}}
.grid td{{padding:6px 8px;border-radius:3px;min-width:44px}} .grid th{{font-weight:500;color:#444;padding:2px 6px}}
.grid th.rot{{height:120px;vertical-align:bottom;white-space:nowrap}} .grid th.rot>div{{transform:rotate(-45deg);width:30px;transform-origin:left bottom;margin-left:18px}}
.grid td.n{{color:#888;font-size:11.5px}}
.note{{font-size:12.5px;color:#666;line-height:1.5}} ul{{font-size:13.5px;line-height:1.6}}
</style></head><body>
<h2>Explainability report &mdash; model version {esc(args.model_version)}</h2>
<p class="meta">artifact: {esc(args.model_dir)} | {esc(cfg.dataset)} fold {cfg.fold} | encoder {esc(arch.get('encoder', '?'))} |
seed {arch['seed']} | {n_way}-way {k_shot}-shot, query {q} | {args.n_episodes} test episodes, {N} decisions |
accuracy on these episodes {acc_mean:.1%}</p>
<div class="primer"><b>How to read this.</b> Each typology's few support {unit}s are averaged into a
<i>prototype</i>; a {unit} is assigned to the nearest prototype. Every decision is therefore a comparison:
<i>"closer to typology A than to B"</i>; the gap between the two distances is the <b>margin</b>. To see
<i>which features</i> decided it, walk from a typical {unit} of the competing typology (where the competitor
wins) to this {unit} (where the chosen typology wins) and credit every feature with its part of that
<b>swing</b> (Integrated Gradients). The parts add up exactly to the swing. Green pushes toward the chosen
typology, orange toward the competitor.</div>
<h3 style="margin-top:20px">Can these explanations be trusted?</h3>
<div class="badges">{badges}</div>
<p class="note">{trust_line}</p>
<h3 style="margin-top:22px">Where the model is right and where it confuses typologies</h3>
<p class="note">Row = true typology, column = prediction, share of that row's decisions. These are the
fold's held-out test typologies: the model never saw them in training and recognises them from {k_shot} examples.</p>
{html_confusion(classes, conf)}
<ul>{conf_lines or "<li>No confusions in the sampled episodes.</li>"}</ul>
<h3 style="margin-top:22px">What each typology is recognised by</h3>
<p class="note">Share of the decision swing carried by each feature family, averaged over correct decisions
of that typology, measured against its closest competitor (named on the right). This is the model's working
definition of each typology, as learned from {k_shot} examples.</p>
{html_typology_map(classes, fam_order, tmap, competitor)}
<h3 style="margin-top:22px">Case studies</h3>
{case_html}
<h3 style="margin-top:22px">Explanation quality, in detail</h3>
<div class="row"><div>{curve_svg}</div><div class="note">
<p><b>Faithfulness.</b> Removing the {kf} features the explanation ranks highest (replacing them with the
competitor's typical values) cuts the margin by {drop_top:.3f} on average; removing {kf} random features cuts it
by {drop_rand:.3f} (ratio {fr}, required &ge; {FAITHFULNESS_MIN_RATIO:.0f}&times;). The competitor overtakes the
chosen typology in {flip_top:.0%} of decisions after top-{kf} removal vs {flip_rand:.0%} after random removal.</p>
<p><b>Completeness.</b> Attributions sum to the swing up to a {comp_med:.2%} median / {comp_p95:.2%} p95
residual at {steps} steps (required p95 &le; {COMPLETENESS_MAX_P95:.0%}).</p>
<p><b>Stability.</b> {stability['queries']} decisions re-explained with {STABILITY_RESAMPLES} freshly drawn
support sets each: median feature-ranking Spearman {stab_med:.2f} (required &ge; {STABILITY_MIN_SPEARMAN}).
Diagnostics, no threshold: magnitude-weighted agreement (Pearson) {stability['median_pearson']:.2f}; the same
family leads the explanation in {stability['top_family_agreement']:.0%} of resamples; the decision itself
survives {stability['decision_retained']:.0%} of them. {stab_line}</p>
<p><b>Baseline sensitivity</b> (reported, no threshold): rankings against the competitor's typical {unit} vs
against an average training {unit} agree at median Spearman {base_agree:.2f}. The two answer different
questions ("why A rather than B" vs "why A rather than an ordinary {unit}"); this report uses the first.</p>
</div></div>
<h3 style="margin-top:22px">The embedding space of one episode</h3>
<p class="note">Diamonds = prototypes, dots = query {unit}s coloured by true typology. 2D PCA projection,
illustrative only.</p>
{pca_svg}
<h3>How decisive are the decisions?</h3>
{hist}
<p class="note">{float((margins_all < 0.5).mean()):.0%} of decisions had a margin below 0.5.</p>
<p class="meta" style="margin-top:22px">Scope. Explains the decisions of the encoder as trained, on
{esc(cfg.dataset)} {scope_unit}. Attributions describe the model, not ground truth about
laundering. No counterfactuals toward normal behaviour are generated, by design: in AML they would describe how
to evade detection. Thresholds for the quality tests were fixed before this run.</p>
</body></html>"""

    summary = {
        "mode": "tabular", "model_dir": args.model_dir, "model_version": args.model_version,
        "dataset": cfg.dataset, "fold": int(cfg.fold), "encoder": arch.get("encoder"),
        "episodes": args.n_episodes, "decisions": int(N), "accuracy": acc_mean,
        "per_class_recall": recall,
        "confusion": {"classes": classes, "counts": conf.tolist()},
        "top_confusions": top_conf, "typology_map": tmap,
        "closest_competitor": {c: {"class": v[0], "share": float(v[1])} for c, v in competitor.items()},
        "tests": tests, "cases": case_js,
        "deletion_curve": {"k": ks, "top": [float(np.mean(c)) for c in curve_top],
                           "random": [float(np.mean(c)) for c in curve_rand]},
    }
    tc = top_conf[0] if top_conf else None
    line = (f"acc {acc_mean:.3f} | completeness {'PASS' if completeness['passed'] else 'FAIL'} "
            f"(p95 {comp_p95:.2%}) | faithfulness {'PASS' if faithfulness['passed'] else 'FAIL'} "
            f"({faithfulness['ratio']:.1f}x, flips {flip_top:.0%} vs {flip_rand:.0%}) | stability "
            f"{'PASS' if stability['passed'] else 'FAIL'} (rho {stab_med:.2f})"
            + (f" | top confusion {tc['true']}->{tc['pred']} {tc['rate']:.0%}" if tc else ""))
    return html, summary, line


# --------------------------------------------------------------------------- #
# Image explanation (decision level)
# --------------------------------------------------------------------------- #
def tensor_to_png_b64(t, scale=3):
    import numpy as np
    from PIL import Image
    a = t.cpu().numpy()
    if a.ndim == 3 and a.shape[0] == 3:
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        a = (a * std + mean).clip(0, 1)
        arr = (a.transpose(1, 2, 0) * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")
    else:
        arr = (a.squeeze() * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
    side = arr.shape[0]
    scale = max(1, min(scale, 252 // max(side, 1)))
    img = img.resize((side * scale, side * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def img_tag(b64, size=84):
    return (f'<img src="data:image/png;base64,{b64}" style="width:{size}px;height:{size}px;'
            f'image-rendering:pixelated;border-radius:6px;margin:2px"/>')


def explain_image(model, test_tasks, cfg, arch, args, device):
    import numpy as np
    import torch
    n_way, k_shot, q = cfg.n_way, cfg.k_shot, cfg.query
    names = [f"class {chr(65 + i)}" for i in range(n_way)]
    all_m, best, worst, miss, first = [], None, None, None, None
    n_dec = n_ok = 0
    for _ in range(args.n_episodes):
        data, labels = test_tasks.sample()
        ep = episode_breakdown(model, data, labels, n_way, k_shot, q, device)
        all_m.extend(ep["margins"].tolist())
        ok = (ep["preds"] == ep["q_labels"]).cpu()
        n_dec += len(ok); n_ok += int(ok.sum())
        for i in range(len(ok)):
            m = float(ep["margins"][i])
            if ok[i] and (best is None or m > best[0]):
                best = (m, ep, i)
            if ok[i] and (worst is None or m < worst[0]):
                worst = (m, ep, i)
            if not ok[i] and (miss is None or m > miss[0]):
                miss = (m, ep, i)
        first = first or ep

    def render(title, sub, tup):
        if tup is None:
            return ""
        m, ep, i = tup
        c, t = int(ep["preds"][i]), int(ep["q_labels"][i])
        bars = svg_distance_bars([float(d) for d in ep["dists"][i]], names, c, t if t != c else None)
        sup = "".join(img_tag(tensor_to_png_b64(ep["support_x"][c][s]), 60) for s in range(k_shot))
        verdict = (f"<span style='color:{GREEN}'>correct</span>" if t == c
                   else f"<span style='color:{RED}'>true: {names[t]}</span>")
        return f"""<div class="case"><h3>{title}</h3><p class="sub">{sub} (margin {m:.2f})</p>
<div class="row"><div style="flex:0 0 120px"><div class="cap">query</div>{img_tag(tensor_to_png_b64(ep['q_x'][i]))}
<div style="font-size:13px">predicted <b>{names[c]}</b>, {verdict}</div></div>
<div><div class="cap">distance to each prototype (shortest wins)</div>{bars}</div>
<div><div class="cap">support examples of {names[c]} (its prototype = their average)</div>{sup}</div></div></div>"""

    cases = (render("Most confident decision", "largest gap to the runner-up", best)
             + render("Most borderline correct decision", "the runner-up almost won", worst)
             + (render("Most confident mistake", "orange outline = the true class", miss) if miss else
                '<div class="case" style="border-style:dashed;color:#666">No misclassification sampled.</div>'))
    both = torch.cat([first["protos"], first["q_emb"]]).cpu()
    Xc = both - both.mean(0, keepdim=True)
    _, _, V = torch.linalg.svd(Xc, full_matrices=False)
    xy = (Xc @ V[:2].T).numpy()
    pca = svg_pca_scatter(xy[:n_way], xy[n_way:], first["q_labels"].cpu().numpy(), names)
    margins = np.array(all_m)
    hist = svg_histogram(margins, title=f"Decision margins, {len(margins)} decisions")
    acc = n_ok / max(n_dec, 1)
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:24px;color:#1a1a1a;max-width:1000px;background:#fff}}
.meta{{font-size:12px;color:#888}} .primer{{background:#f2f6fb;border-radius:10px;padding:14px 18px;font-size:14px;line-height:1.5}}
.case{{border:1px solid #e0e0e0;border-radius:10px;padding:16px;margin:18px 0}} .sub{{font-size:13px;color:#666;margin:0 0 8px 0}}
.cap{{font-size:12px;color:#666;margin-bottom:6px}} .row{{display:flex;gap:24px;flex-wrap:wrap}} .note{{font-size:13px;color:#666}}
</style></head><body>
<h2>Explainability report &mdash; model version {esc(args.model_version)}</h2>
<p class="meta">artifact: {esc(args.model_dir)} | {esc(cfg.dataset)} | encoder {esc(arch.get('encoder', 'conv4'))} |
seed {arch['seed']} | {n_way}-way {k_shot}-shot, query {q} | {args.n_episodes} episodes | accuracy {acc:.1%}</p>
<div class="primer"><b>How to read this.</b> Each class's support examples are averaged into a prototype; a
query goes to the nearest prototype. The bars are those distances, the margin is the gap between the best and
second-best.</div>{cases}
<h3>The embedding space of one episode</h3><p class="note">Diamonds = prototypes, dots = queries. 2D PCA, illustrative.</p>{pca}
<h3>How decisive are the decisions?</h3>{hist}
<p class="note">{float((margins < 0.5).mean()):.0%} of decisions had a margin below 0.5. Decision-level explanation:
why the encoder finds two images similar stays inside the network.</p></body></html>"""
    summary = {"mode": "image", "model_dir": args.model_dir, "dataset": cfg.dataset, "accuracy": acc,
               "episodes": args.n_episodes, "tight_margin_share": float((margins < 0.5).mean())}
    line = (f"acc {acc:.3f} | confident m={best[0]:.2f} | borderline m={worst[0]:.2f} | "
            f"misclass={'yes' if miss else 'no'} | tight-margin {(margins < 0.5).mean():.1%}")
    return html, summary, line


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Explainability report for a saved ProtoNet artifact.")
    ap.add_argument("--project", required=True)
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--dataset", default="omniglot",
                    help="fallback only: the artifact's own dataset (architecture.json) wins")
    ap.add_argument("--model-dir", required=True, help="gs://... or local dir with model.pt + architecture.json")
    ap.add_argument("--model-version", default="?")
    ap.add_argument("--n-episodes", type=int, default=60)
    ap.add_argument("--report-html-path", required=True)
    ap.add_argument("--summary-output-path", required=True)
    ap.add_argument("--json-output-path", default=None, help="optional: full machine-readable results")
    ap.add_argument("--ig-steps", type=int, default=IG_STEPS,
                    help="Integrated Gradients steps (numerical budget only; thresholds do not move)")
    args = ap.parse_args()

    import torch

    from fsl.config import TrainConfig
    from fsl.data.registry import get_loaders
    from fsl.models.registry import get_encoder

    local = fetch_artifact(args.model_dir, args.project)
    arch = json.loads((local / "architecture.json").read_text())
    encoder = str(arch.get("encoder", arch.get("architecture", "conv4"))).strip().lower()
    dataset = arch.get("dataset", args.dataset)
    if dataset != args.dataset:
        print(f"note: --dataset {args.dataset!r} ignored; the artifact was trained on {dataset!r}")

    model = get_encoder(SimpleNamespace(encoder=encoder, in_channels=arch["in_channels"],
                                        embedding_hid=arch["embedding_hid"],
                                        mlp_depth=arch.get("mlp_depth", 2),
                                        mlp_dropout=arch.get("mlp_dropout", 0.0)))
    model.load_state_dict(torch.load(local / "model.pt", map_location="cpu"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = torch.device("cpu")

    cfg = TrainConfig(project=args.project, region=args.region, bucket=args.bucket, dataset=dataset,
                      seed=arch["seed"], n_way=arch["n_way"], k_shot=arch["k_shot"],
                      query=arch.get("query", 15), fold=arch.get("fold", 0), encoder=encoder,
                      embedding_hid=arch["embedding_hid"], mlp_depth=arch.get("mlp_depth", 2),
                      mlp_dropout=arch.get("mlp_dropout", 0.0), in_channels=arch["in_channels"],
                      image_size=arch.get("image_size", 28), log_to_vertex=False)
    print(f"explaining {encoder} on {dataset} fold {cfg.fold}: {cfg.n_way}-way {cfg.k_shot}-shot, "
          f"query={cfg.query}, seed={cfg.seed}")
    load_frozen, build_task_samplers = get_loaders(cfg)
    bundle, _ = load_frozen(cfg)
    _, _, test_tasks = build_task_samplers(bundle, cfg)

    if hasattr(test_tasks, "sample_indexed") and isinstance(bundle, dict) and "schema" in bundle:
        html, summary, line = explain_tabular(model, test_tasks, bundle, cfg, arch, args, device)
    else:
        html, summary, line = explain_image(model, test_tasks, cfg, arch, args, device)

    _write(args.report_html_path, html)
    _write(args.summary_output_path, line)
    if args.json_output_path:
        _write(args.json_output_path, json.dumps(summary, indent=2, default=float))
    print("Explainability report written.", line)


if __name__ == "__main__":
    main()
