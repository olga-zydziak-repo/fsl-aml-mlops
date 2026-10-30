#!/usr/bin/env python3
"""Active few-shot learning, evaluated inside the pipeline run.

Question this component answers for the model trained in THIS run, on its
fold's test typologies (never seen in training): if an analyst labels cases
one at a time, does letting the model choose WHICH case to label next reach
a given accuracy with fewer labels than labelling cases at random?

Protocol (same as notebooks/samld_case_active_learning.ipynb, where the
criterion was fixed before the first run and passed on all 5 folds):
  * cases of the test typologies are split by account group into a POOL
    (unlabelled, the model may ask for labels) and an EVALUATION set (never
    labelled); groups are disjoint, so one launderer is never on both sides;
  * start with 1 labelled case per typology; each round a strategy picks one
    pool case, the oracle reveals its typology, that typology's prototype is
    recomputed. No retraining: a new label just joins its prototype;
  * strategies: random (reference), margin (smallest gap between the two
    nearest prototypes: the validated one), hybrid (most typical cases for
    the first rounds, then margin: reported);
  * balanced accuracy on the evaluation set after every label.

Representations: `raw` (standardised case features, the strongest in the
campaign) and `protonet` (this run's encoder). The label QUEUE shown to the
analyst uses `raw`: for nearest-prototype classification in raw feature space
the explanation is EXACT, no attribution method needed:
    d(x, B)^2 - d(x, A)^2 = sum_j [(x_j - B_j)^2 - (x_j - A_j)^2]
so every feature's share of "why A rather than B" is a closed-form term.

Image datasets: not applicable (no case table); the component writes a short
report saying so and exits cleanly. numpy + torch only, no new dependencies.

Experiments (v1.2): with --experiment-name and --experiment-run (the run the
training container opened) the headline numbers of the validated strategy are
added to that run, so the Experiments table carries training, gate and active
learning of one model in one row.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explain_pipeline_entry import (  # noqa: E402  (same image, same helpers)
    BLUE, GREEN, ORANGE, RED, _write, esc, feature_meta, fetch_artifact, fmt_value,
)

STRATEGY_COLOURS = {"random": "#888888", "margin": RED, "hybrid": GREEN}


# --------------------------------------------------------------------------- #
# Simulation (numpy only)
# --------------------------------------------------------------------------- #
def _kmeans(Z, k, rng, iters=25):
    import numpy as np
    idx = [int(rng.integers(len(Z)))]
    d2 = ((Z - Z[idx[0]]) ** 2).sum(1)
    for _ in range(1, k):                                  # k-means++ init
        p = d2 / d2.sum() if d2.sum() > 0 else None
        idx.append(int(rng.choice(len(Z), p=p)))
        d2 = np.minimum(d2, ((Z - Z[idx[-1]]) ** 2).sum(1))
    C = Z[idx].copy()
    for _ in range(iters):
        lab = ((Z[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if np.any(lab == j):
                C[j] = Z[lab == j].mean(0)
    return lab


def typical_pick(Z, pool, L, cand, rng, knn=20):
    """TypiClust (Hacohen et al. 2022): |L|+1 clusters over pool+labelled; in the
    largest cluster without a label, the candidate with the highest typicality."""
    import numpy as np
    allp = np.union1d(pool, L)
    k = min(len(L) + 1, len(allp))
    lab = dict(zip(allp.tolist(), _kmeans(Z[allp], k, rng).tolist()))
    covered = {lab[i] for i in L.tolist()}
    sizes = np.bincount([lab[i] for i in allp.tolist()], minlength=k)
    members = cand
    for c in [c for c in np.argsort(-sizes) if c not in covered]:
        m = np.array([i for i in cand if lab.get(int(i)) == c])
        if len(m):
            members = m
            break
    if len(members) == 1:
        return members[0]
    D = np.sqrt(((Z[members][:, None] - Z[members][None]) ** 2).sum(-1))
    kk = min(knn, len(members) - 1)
    return members[np.argmax(1.0 / (np.sort(D, axis=1)[:, 1:kk + 1].mean(1) + 1e-9))]


def split_pool(grp, pool_share, rng):
    import numpy as np
    ug = np.unique(grp)
    pool_g = set(rng.choice(ug, int(round(pool_share * len(ug))), replace=False).tolist())
    in_pool = np.array([g in pool_g for g in grp])
    return np.flatnonzero(in_pool), np.flatnonzero(~in_pool)


def prototypes(Z, y, L, n_cls):
    import numpy as np
    return np.stack([Z[L[y[L] == c]].mean(0) for c in range(n_cls)])


def margin_rank(Z, P, cand):
    import numpy as np
    d = np.sort(np.sqrt(((Z[cand][:, None] - P[None]) ** 2).sum(-1)), axis=1)
    return d[:, 1] - d[:, 0]


def simulate(Z, y, grp, n_cls, strategy, rng, rounds, warmup, pool_share):
    import numpy as np
    pool, ev = split_pool(grp, pool_share, rng)
    if any(not np.any(y[pool] == c) or not np.any(y[ev] == c) for c in range(n_cls)):
        return None
    labeled = [int(rng.choice(pool[y[pool] == c])) for c in range(n_cls)]
    curve = []
    for r in range(rounds + 1):
        L = np.array(labeled)
        P = prototypes(Z, y, L, n_cls)
        pred = ((Z[ev][:, None] - P[None]) ** 2).sum(-1).argmin(1)
        curve.append(float(np.mean([np.mean(pred[y[ev] == c] == c) for c in range(n_cls)])))
        if r == rounds:
            break
        cand = np.setdiff1d(pool, L)
        if strategy == "random":
            pick = rng.choice(cand)
        elif strategy == "hybrid" and r < warmup:
            pick = typical_pick(Z, pool, L, cand, rng)
        else:                                              # margin, and hybrid after warm-up
            pick = cand[np.argmin(margin_rank(Z, P, cand))]
        labeled.append(int(pick))
    return curve


# --------------------------------------------------------------------------- #
# Report pieces
# --------------------------------------------------------------------------- #
def svg_curves(x, series, title, width=520, height=290):
    import numpy as np
    pad_l, pad_r, pad_t, pad_b = 48, 96, 28, 40
    allv = np.concatenate([np.asarray(v) for _, v, _ in series])
    lo, hi = float(allv.min()) - 0.01, float(allv.max()) + 0.01
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    P = lambda a, b: (pad_l + (a - x[0]) / ((x[-1] - x[0]) or 1) * pw,
                      pad_t + (1 - (b - lo) / ((hi - lo) or 1)) * ph)
    out = [f'<text x="{pad_l + pw / 2:.0f}" y="16" font-size="12.5" text-anchor="middle" fill="#333">{esc(title)}</text>',
           f'<line x1="{pad_l}" y1="{pad_t + ph}" x2="{pad_l + pw}" y2="{pad_t + ph}" stroke="#999"/>',
           f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + ph}" stroke="#999"/>',
           f'<text x="{pad_l + pw / 2:.0f}" y="{height - 6}" font-size="11" text-anchor="middle" fill="#555">'
           f'labelled cases</text>']
    for v in (lo + 0.01, (lo + hi) / 2, hi - 0.01):
        _, Y = P(x[0], v)
        out.append(f'<text x="{pad_l - 6}" y="{Y + 4:.1f}" font-size="10.5" text-anchor="end" fill="#555">{v:.2f}</text>')
    for t in x[::4]:
        X, _ = P(t, lo)
        out.append(f'<text x="{X:.1f}" y="{pad_t + ph + 15}" font-size="10.5" text-anchor="middle" fill="#555">{t}</text>')
    for i, (name, ys, col) in enumerate(series):
        pts = " ".join(f"{P(a, b)[0]:.1f},{P(a, b)[1]:.1f}" for a, b in zip(x, ys))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.3"/>')
        ly = pad_t + 12 + 18 * i
        out.append(f'<line x1="{pad_l + pw + 8}" y1="{ly - 4}" x2="{pad_l + pw + 24}" y2="{ly - 4}" '
                   f'stroke="{col}" stroke-width="2.3"/><text x="{pad_l + pw + 28}" y="{ly}" '
                   f'font-size="10.5" fill="#333">{esc(name)}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" width="{width}" xmlns="http://www.w3.org/2000/svg">'
            + "".join(out) + "</svg>")


def greedy_batch(Z, L_pos, L_lab, cand, size, n_cls):
    """The batch the analyst labels in one go, chosen one case at a time: the case with the
    smallest margin, then the model's own guess stands in for its label (its prototype moves)
    and the next smallest margin is taken, and so on. Plain top-`size` by margin piles
    near-duplicates onto one boundary and drags that prototype; this keeps the batch close to
    the validated one-label-at-a-time protocol without asking the analyst in between."""
    import numpy as np
    L, lab, cand, out = [int(i) for i in L_pos], [int(c) for c in L_lab], np.asarray(cand), []
    for _ in range(min(size, len(cand))):
        La, la = np.asarray(L), np.asarray(lab)
        P = np.stack([Z[La[la == c]].mean(0) for c in range(n_cls)])
        d = np.sqrt(((Z[cand][:, None] - P[None]) ** 2).sum(-1))
        s = np.sort(d, axis=1)
        j = int(np.argmin(s[:, 1] - s[:, 0]))
        out.append(int(cand[j]))
        L.append(int(cand[j]))
        lab.append(int(d[j].argmin()))
        cand = np.delete(cand, j)
    return out


def explain_queue(Zraw, raw, names, feats, ids, n_acc, n_tx, L_pos, L_lab, cand, queue_size, y=None):
    """Pick a batch from `cand` (greedy_batch) against prototypes of the labelled cases (L_pos with
    labels L_lab) and explain each queued case exactly against those prototypes. `y` (true typologies)
    is optional: only the simulation knows it and uses it for the 'oracle' fields."""
    import numpy as np
    n_cls = len(names)
    L_pos, L_lab = np.asarray(L_pos), np.asarray(L_lab)
    P = np.stack([Zraw[L_pos[L_lab == c]].mean(0) for c in range(n_cls)])
    raw_proto = np.stack([raw[L_pos[L_lab == c]].mean(0) for c in range(n_cls)])
    meta = [feature_meta(f) for f in feats]
    items = []
    for i in greedy_batch(Zraw, L_pos, L_lab, cand, queue_size, n_cls):
        d2 = ((Zraw[i] - P) ** 2).sum(1)
        a, b = np.argsort(d2)[:2]
        contrib = (Zraw[i] - P[b]) ** 2 - (Zraw[i] - P[a]) ** 2      # sums exactly to d_b^2 - d_a^2
        tot = float(contrib.sum())
        assert abs(tot - float(d2[b] - d2[a])) < 1e-3 * max(1.0, abs(tot))
        top_a = [k for k in np.argsort(-contrib) if contrib[k] > 0][:3]
        top_b = [k for k in np.argsort(contrib) if contrib[k] < 0][:3]

        def part(k):
            lab, _, kind = meta[k]
            return (f"{lab} = {fmt_value(raw[i, k], kind)} ({names[a]} avg {fmt_value(raw_proto[a, k], kind)}, "
                    f"{names[b]} avg {fmt_value(raw_proto[b, k], kind)})")
        item = {
            "case": str(ids[i]), "pos": int(i), "accounts": int(n_acc[i]), "transactions": int(n_tx[i]),
            "nearest": names[a], "runner_up": names[b],
            "margin": float(np.sqrt(d2[b]) - np.sqrt(d2[a])),
            "for_nearest": [part(k) for k in top_a], "for_runner_up": [part(k) for k in top_b],
            "exact_sum": tot,
            # structured, for the demo section: (feature, push, case value, avg nearest, avg runner-up)
            "pulls": [(feats[k], float(contrib[k]), float(raw[i, k]), float(raw_proto[a, k]),
                       float(raw_proto[b, k])) for k in top_a + top_b],
            "share_nearest": float(contrib[contrib > 0].sum() / max(np.abs(contrib).sum(), 1e-12)),
        }
        if y is not None:
            item.update(oracle_label=names[y[i]], model_was_right=bool(y[i] == a))
        items.append(item)
    return items


def balanced_accuracy(Zraw, y, ev, L_pos, L_lab, n_cls):
    import numpy as np
    L_pos, L_lab = np.asarray(L_pos), np.asarray(L_lab)
    P = np.stack([Zraw[L_pos[L_lab == c]].mean(0) for c in range(n_cls)])
    pred = ((Zraw[ev][:, None] - P[None]) ** 2).sum(-1).argmin(1)
    return float(np.mean([np.mean(pred[y[ev] == c] == c) for c in range(n_cls) if np.any(y[ev] == c)]))


def queue_with_explanations(Zraw, raw, y, grp, names, feats, ids, n_acc, n_tx, rng,
                            queue_after, queue_size, pool_share):
    """One realistic round: seed + `queue_after` margin labels, then the next
    `queue_size` cases the model asks about, each with an exact explanation.
    Also returns the batch effect and the state (labelled / pool / evaluation rows)
    that the analyst's round starts from."""
    import numpy as np
    n_cls = len(names)
    pool, ev = split_pool(grp, pool_share, rng)
    labeled = [int(rng.choice(pool[y[pool] == c])) for c in range(n_cls)]
    for _ in range(queue_after):
        L = np.array(labeled)
        cand = np.setdiff1d(pool, L)
        labeled.append(int(cand[np.argmin(margin_rank(Zraw, prototypes(Zraw, y, L, n_cls), cand))]))
    L = np.array(labeled)
    cand = np.setdiff1d(pool, L)
    items = explain_queue(Zraw, raw, names, feats, ids, n_acc, n_tx, L, y[L], cand, queue_size, y=y)
    # batch effect: labelling THIS queue vs the same number of random pool cases (oracle = true typology)
    queued = [it["pos"] for it in items]
    bacc = lambda Lx: balanced_accuracy(Zraw, y, ev, Lx, y[np.asarray(Lx)], n_cls)
    rand = [bacc(list(L) + rng.choice(cand, len(queued), replace=False).tolist()) for _ in range(50)]
    effect = {"before": bacc(L), "after_queue": bacc(list(L) + queued), "after_random": float(np.mean(rand)),
              "batch": len(queued)}
    state = {"labeled": [int(i) for i in L], "labels": [int(c) for c in y[L]],
             "pool": [int(i) for i in pool], "eval": [int(i) for i in ev], "queue": queued}
    return items, len(L), effect, state


# --------------------------------------------------------------------------- #
# Demo section (Polish): what active learning does, readable in ten seconds
# --------------------------------------------------------------------------- #
_PL = {
    "n_tx": ("liczba transakcji", "count"), "n_acc": ("liczba kont", "count"),
    "duration_days": ("czas trwania", "days"), "n_days": ("aktywne dni", "count"),
    "tx_per_day": ("transakcje na aktywny dzień", "ratio"),
    "mean_gap": ("średni odstęp między transakcjami", "days"),
    "share_src": ("udział kont tylko wysyłających", "pct"),
    "share_sink": ("udział kont tylko odbierających", "pct"),
    "share_mid": ("udział kont pośredniczących", "pct"),
    "max_out_share": ("największe rozgałęzienie wypływów", "pct"),
    "max_in_share": ("największe zbieganie wpływów", "pct"),
    "deg_cv": ("nierówność połączeń między kontami", "ratio"),
    "edge_repeat": ("transakcje na parę kont", "ratio"), "density": ("gęstość sieci kont", "ratio"),
    "depth": ("liczba warstw pośredników", "count"),
    "has_cycle": ("pieniądze wracają w kółko", "flag"),
    "reciprocity": ("przepływy w obie strony", "pct"),
    "is_bipartite": ("dwie rozłączne grupy kont", "flag"),
    "amt_total": ("suma kwot", "money"), "amt_mean": ("średnia kwota", "money"),
    "amt_cv": ("zmienność kwot", "ratio"), "amt_max_med": ("największa kwota / mediana", "ratio"),
    "r100": ("udział okrągłych kwot", "pct"), "s10k": ("udział kwot tuż pod 10 000", "pct"),
    "passthrough": ("przepływ przez pośredników", "pct"),
    "fwd_1d": ("przekazane dalej w ciągu doby", "pct"),
    "sink_conc": ("udział głównego odbiorcy", "pct"), "src_conc": ("udział głównego nadawcy", "pct"),
    "cash": ("udział gotówki", "pct"), "xb": ("udział przelewów zagranicznych", "pct"),
    "curm": ("udział zmiany waluty", "pct"),
}
DEMO_BLUE, DEMO_GRAY = "#2a78d6", "#8f8e88"


def pl_n(n: int, one: str, few: str, many: str) -> str:
    """Polish plural: 1 konto, 2-4 konta (not 12-14), 5+ kont."""
    n = int(n)
    if n == 1:
        return f"{n} {one}"
    return f"{n} {few if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) else many}"


def fmt_pl(v: float, kind: str) -> str:
    if kind == "money":
        return f"{v:,.0f}".replace(",", " ")
    if kind == "pct":
        return f"{100 * v:.0f}%"
    if kind == "flag":
        return ("tak" if v >= 0.5 else "nie") if v in (0.0, 1.0) else f"{100 * v:.0f}% przypadków"
    if kind == "days":
        return f"{v:.1f} dnia".replace(".", ",")
    if kind == "count":
        return f"{int(round(v))}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}".replace(".", ",")
    return f"{v:.2f}".replace(".", ",")


def svg_hero_curve(x, model, rand, need, budget, width=640, height=330):
    """Two lines: labels the model asked for vs random labels; target = random's accuracy at the full budget."""
    import numpy as np
    pl, pr, pt, pb = 56, 150, 18, 46
    ys = np.r_[model, rand]
    lo, hi = float(ys.min()) - 0.02, float(ys.max()) + 0.02
    pw, ph = width - pl - pr, height - pt - pb
    X = lambda a: pl + (a - x[0]) / ((x[-1] - x[0]) or 1) * pw
    Y = lambda b: pt + (1 - (b - lo) / ((hi - lo) or 1)) * ph
    o = []
    for t in np.linspace(lo + 0.02, hi - 0.02, 4):                        # recessive grid
        o.append(f'<line x1="{pl}" y1="{Y(t):.1f}" x2="{pl + pw}" y2="{Y(t):.1f}" stroke="#ececea"/>'
                 f'<text x="{pl - 8}" y="{Y(t) + 4:.1f}" font-size="11" text-anchor="end" fill="#6b6a66">{100 * t:.0f}%</text>')
    for t in x[::4]:
        o.append(f'<text x="{X(t):.1f}" y="{pt + ph + 18}" font-size="11" text-anchor="middle" fill="#6b6a66">{t}</text>')
    o.append(f'<text x="{pl + pw / 2:.0f}" y="{height - 6}" font-size="12" text-anchor="middle" fill="#52514e">'
             f'liczba przypadków oznaczonych przez analityka</text>')
    target = float(rand[-1])
    o.append(f'<line x1="{pl}" y1="{Y(target):.1f}" x2="{pl + pw}" y2="{Y(target):.1f}" stroke="#52514e" '
             f'stroke-width="1" stroke-dasharray="3 4"/>')
    pts = lambda v: " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(x, v))
    o.append(f'<polyline points="{pts(rand)}" fill="none" stroke="{DEMO_GRAY}" stroke-width="2" stroke-dasharray="6 4"/>')
    o.append(f'<polyline points="{pts(model)}" fill="none" stroke="{DEMO_BLUE}" stroke-width="2.5"/>')
    for a, b, c, n in [(a, b, DEMO_BLUE, "model pyta") for a, b in zip(x, model)] + \
                      [(a, b, DEMO_GRAY, "losowo") for a, b in zip(x, rand)]:
        o.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="7" fill="transparent">'
                 f'<title>{n}: {a} oznaczeń, trafność {100 * b:.0f}%</title></circle>')
    o.append(f'<text x="{pl + pw + 8}" y="{Y(model[-1]) + 4:.1f}" font-size="12.5" font-weight="600" fill="#0b0b0b">'
             f'model pyta</text>')
    o.append(f'<text x="{pl + pw + 8}" y="{Y(rand[-1]) + 18:.1f}" font-size="12.5" fill="#52514e">losowo</text>')
    if need:
        o.append(f'<line x1="{X(need):.1f}" y1="{Y(target):.1f}" x2="{X(need):.1f}" y2="{pt + ph}" stroke="{DEMO_BLUE}" '
                 f'stroke-width="1.2" stroke-dasharray="2 3"/>')
        o.append(f'<circle cx="{X(need):.1f}" cy="{Y(target):.1f}" r="5.5" fill="{DEMO_BLUE}" stroke="#fff" stroke-width="2"/>')
        o.append(f'<circle cx="{X(budget):.1f}" cy="{Y(target):.1f}" r="5.5" fill="{DEMO_GRAY}" stroke="#fff" stroke-width="2"/>')
        o.append(f'<text x="{X(need) + 8:.1f}" y="{Y(target) + 18:.1f}" font-size="12.5" font-weight="600" fill="#0b0b0b">'
                 f'{pl_n(need, "oznaczenie", "oznaczenia", "oznaczeń")}</text>')
        o.append(f'<text x="{X(budget) - 6:.1f}" y="{Y(target) - 10:.1f}" font-size="12.5" text-anchor="end" '
                 f'fill="#52514e">{pl_n(budget, "oznaczenie", "oznaczenia", "oznaczeń")}</text>')
    o.append(f'<line x1="{pl}" y1="{pt + ph}" x2="{pl + pw}" y2="{pt + ph}" stroke="#b5b4ae"/>')
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px" role="img" '
            f'aria-label="trafność a liczba oznaczeń: model pyta vs losowo" xmlns="http://www.w3.org/2000/svg">'
            + "".join(o) + "</svg>")


def svg_loop(width=380, height=300):
    """The loop in four steps."""
    boxes = [(20, 18, ["1. Model ocenia", "nowe przypadki"]),
             (210, 18, ["2. Wybiera te, przy", "których się waha"]),
             (210, 206, ["3. Analityk oznacza", "je (kolejka.csv)"]),
             (20, 206, ["4. Wzorce typologii", "się poprawiają"])]
    bw, bh = 150, 76
    o = ['<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#52514e"/></marker></defs>']
    for x0, y0, lines in boxes:
        o.append(f'<rect x="{x0}" y="{y0}" width="{bw}" height="{bh}" rx="10" fill="#f2f6fb" stroke="#c9d8ee"/>')
        for i, t in enumerate(lines):
            o.append(f'<text x="{x0 + bw / 2}" y="{y0 + 32 + 20 * i}" font-size="13" text-anchor="middle" '
                     f'fill="#0b0b0b">{esc(t)}</text>')
    arrows = [(172, 56, 208, 56), (285, 96, 285, 204), (208, 244, 172, 244), (95, 204, 95, 96)]
    for x1, y1, x2, y2 in arrows:
        o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#52514e" stroke-width="1.6" marker-end="url(#ah)"/>')
    o.append(f'<text x="{width / 2 - 0}" y="146" font-size="12" text-anchor="middle" fill="#52514e">bez trenowania:</text>'
             f'<text x="{width / 2 - 0}" y="163" font-size="12" text-anchor="middle" fill="#52514e">każda etykieta działa od razu</text>')
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px" role="img" '
            f'aria-label="pętla active learning w czterech krokach" xmlns="http://www.w3.org/2000/svg">'
            + "".join(o) + "</svg>")


def cards_html(queue) -> str:
    """Case cards (Polish): size, the two typologies, the tug-of-war bar, the exact reasons."""
    cards = ""
    for it in queue:
        a, b = it["nearest"], it["runner_up"]
        share = 100 * it["share_nearest"]
        def reasons(sign):
            out = []
            for f, push, val, avg_a, avg_b in it["pulls"]:
                if (push > 0) != sign:
                    continue
                lab, kind = _PL.get(f, (f, "ratio"))
                out.append(f"<li>{esc(lab)}: <b>{esc(fmt_pl(val, kind))}</b> "
                           f"<span class='mut'>(typowo {esc(a)}: {esc(fmt_pl(avg_a, kind))}; "
                           f"{esc(b)}: {esc(fmt_pl(avg_b, kind))})</span></li>")
            return "".join(out) or "<li class='mut'>brak wyraźnych cech</li>"
        cards += f"""<div class="card"><div class="ch"><b>Przypadek {esc(it['case'])}</b> ·
{pl_n(it['accounts'], 'konto', 'konta', 'kont')} · {pl_n(it['transactions'], 'transakcja', 'transakcje', 'transakcji')}</div>
<div class="cq">Model się waha: <b>{esc(a)}</b> czy <b>{esc(b)}</b>?</div>
<div class="tug"><div class="ta" style="width:{share:.0f}%">{esc(a) + ' ' if share >= 35 else ''}{share:.0f}%</div>
<div class="tb" style="width:{100 - share:.0f}%">{100 - share:.0f}%{' ' + esc(b) if share <= 65 else ''}</div></div>
<div class="tc">siła cech przemawiających za każdą typologią (rozkład dokładny, nie przybliżony)</div>
<div class="cols2"><div><div class="cap2">Za {esc(a)}</div><ul>{reasons(True)}</ul></div>
<div><div class="cap2">Za {esc(b)}</div><ul>{reasons(False)}</ul></div></div></div>"""
    return cards


def demo_section_pl(args, fold, names, x, model, rand, need, budget, queue, effect, files_link) -> str:
    v = str(args.model_version)
    if v == "not-registered":
        model_txt = "model odrzucony przez bramkę (kolejka liczona na surowych cechach przypadków, więc i tak obowiązuje)"
    elif v.isdigit():
        model_txt = f"model w rejestrze, wersja {v}"
    else:
        model_txt = f"model: {v}"
    if need:
        t1 = (f"{need} zamiast {budget}", "oznaczonych przypadków wystarcza, żeby dojść do trafności, którą losowy wybór "
                                          f"osiąga dopiero przy {budget}")
        t2 = (f"{round(100 * (1 - need / budget))}% mniej", "pracy analityka przy tej samej trafności")
    else:
        t1 = ("—", f"model nie doszedł do trafności losowego wyboru przy {budget} oznaczeniach")
        t2 = ("—", "brak oszczędności w tym budżecie")
    t3 = (f"{100 * model[-1]:.0f}% vs {100 * rand[-1]:.0f}%",
          f"trafność przy {pl_n(budget, 'oznaczeniu', 'oznaczeniach', 'oznaczeniach')}: model pyta vs losowo")
    tiles = "".join(f'<div class="tile"><div class="tv">{esc(a)}</div><div class="tl">{esc(b)}</div></div>'
                    for a, b in (t1, t2, t3))
    cards = cards_html(queue[:3])
    link = (f'<a href="{esc(files_link)}">kolejka.csv do wypełnienia i transakcje.csv (folder w GCS)</a>'
            if files_link and files_link != "-" else "kolejka.csv i transakcje.csv w folderze analyst/do_oznaczenia/")
    return f"""<section class="demo">
<h1>Active learning: model pyta analityka tylko o przypadki, których nie jest pewny</h1>
<p class="lead">Analityk ma setki wykrytych przypadków i czas na kilkanaście. Zamiast losowych model podsuwa te, przy
których waha się między dwiema typologiami. Każda odpowiedź od razu poprawia wzorzec typologii, bez trenowania.</p>
<p class="ctx">fold {fold} · typologie nowe dla modelu: {esc(', '.join(names))} · {esc(model_txt)}</p>
<div class="tiles">{tiles}</div>
<div class="viz"><div class="vc"><div class="vt">Trafność a liczba oznaczeń</div>
{svg_hero_curve(x, model, rand, need, budget)}</div><div class="vl"><div class="vt">Jak działa pętla</div>{svg_loop()}</div></div>
<h2>Następne do sprawdzenia</h2>
<p class="ctx">Trzy pierwsze z {len(queue)} przypadków w kolejce. Pełna lista: {link}.</p>
<div class="cards">{cards}</div>
<p class="foot">Odpowiedzi analityka w tej symulacji to prawdziwe typologie z danych (SAML-D, dane syntetyczne).
Przewagę wyboru przez model sprawdzono na 5 foldach kryterium ustalonym przed uruchomieniem; ta strona pokazuje jeden
fold. Zadanie: przypisanie typologii do już wykrytego przypadku, nie wykrywanie.</p>
</section>"""

# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Active few-shot learning simulation for a saved artifact.")
    ap.add_argument("--project", required=True)
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--dataset", default="samld-case", help="fallback; the artifact's dataset wins")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-version", default="?")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--pool-share", type=float, default=0.6)
    ap.add_argument("--queue-after", type=int, default=4)
    ap.add_argument("--queue-size", type=int, default=8)
    ap.add_argument("--report-html-path", required=True)
    ap.add_argument("--summary-output-path", required=True)
    ap.add_argument("--json-output-path", default=None)
    ap.add_argument("--run-name", default="",
                    help="pipeline job name (dsl.PIPELINE_JOB_NAME_PLACEHOLDER); builds the console link in the digest")
    ap.add_argument("--digest-queue-size", type=int, default=5,
                    help="how many queued cases go into the one-line analyst digest")
    ap.add_argument("--emit-digest", default="true", type=lambda s: str(s).lower() not in ("false", "0", "no"),
                    help="print the FSL_ANALYST_DIGEST line (the per-run analyst e-mail); "
                         "the fold campaign sets false and sends one combined e-mail instead")
    ap.add_argument("--digest-output-path", default=None,
                    help="also write the digest line to this file (the campaign summary reads it)")
    ap.add_argument("--analyst-root", default="",
                    help="where analyst batches go; default gs://<bucket>/analyst (a local dir works for tests)")
    ap.add_argument("--case-transactions", default="",
                    help="reference table case -> transactions; default <analyst-root>/reference/<dataset>/"
                         "case_transactions.parquet (built once by samld_case_analyst_data.ipynb)")
    ap.add_argument("--write-analyst-files", default="true",
                    type=lambda s: str(s).lower() not in ("false", "0", "no"))
    ap.add_argument("--experiment-name", default="",
                    help="Vertex Experiments experiment holding the training's run; empty = no logging")
    ap.add_argument("--experiment-run", default="",
                    help="the run the training container opened (its experiment_run output); empty = skip")
    args = ap.parse_args()

    import numpy as np
    import torch

    from fsl.config import TrainConfig
    from fsl.data.registry import get_loaders
    from fsl.models.registry import get_encoder

    local = fetch_artifact(args.model_dir, args.project)
    arch = json.loads((local / "architecture.json").read_text())
    dataset = arch.get("dataset", args.dataset)
    cfg = TrainConfig(project=args.project, region=args.region, bucket=args.bucket, dataset=dataset,
                      seed=arch["seed"], n_way=arch["n_way"], k_shot=arch["k_shot"],
                      query=arch.get("query", 15), fold=arch.get("fold", 0),
                      encoder=str(arch.get("encoder", "conv4")).lower(), embedding_hid=arch["embedding_hid"],
                      in_channels=arch["in_channels"], image_size=arch.get("image_size", 28),
                      split_uri=arch.get("split_uri", ""), log_to_vertex=False)
    load_frozen, build = get_loaders(cfg)
    bundle, sha = load_frozen(cfg)
    if not (isinstance(bundle, dict) and "schema" in bundle):
        msg = f"active learning simulation not applicable to image dataset {dataset!r}"
        _write(args.report_html_path, f"<html><body><p>{esc(msg)}</p></body></html>")
        _write(args.summary_output_path, msg)
        if args.digest_output_path:
            _write(args.digest_output_path, msg)
        print(msg)
        return

    sch, W = bundle["schema"], bundle["windows"]
    feats = list(sch["feature_names"])
    names = list(bundle["folds"][cfg.fold]["test"])
    lab_all = W[sch["label_col"]].to_numpy().astype(str)
    rows = np.flatnonzero(np.isin(lab_all, names))
    y = np.array([names.index(c) for c in lab_all[rows]])
    grp = W[sch["group_col"]].to_numpy()[rows]
    _, _, te = build(bundle, cfg)
    Zraw = np.asarray(te.X, dtype=np.float32)[rows]
    raw = W[feats].astype("float64").to_numpy()[rows]
    id_col = sch.get("id_col")
    ids = W[id_col].to_numpy()[rows] if id_col in W.columns else rows
    n_acc = W["n_acc"].to_numpy()[rows] if "n_acc" in W.columns else np.zeros(len(rows), int)
    n_tx = W["n_tx"].to_numpy()[rows] if "n_tx" in W.columns else np.zeros(len(rows), int)

    encoder = str(arch.get("encoder", "mlp")).lower()
    model = get_encoder(SimpleNamespace(encoder=encoder, in_channels=arch["in_channels"],
                                        embedding_hid=arch["embedding_hid"],
                                        mlp_depth=arch.get("mlp_depth", 2),
                                        mlp_dropout=arch.get("mlp_dropout", 0.0)))
    model.load_state_dict(torch.load(local / "model.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        Zp = model(torch.from_numpy(Zraw)).numpy()

    strategies = ("random", "margin", "hybrid")
    n_cls, seed0 = len(names), 10_000 * int(cfg.fold)
    curves = {}
    for rn, Z in (("raw", Zraw), ("protonet", Zp)):
        curves[rn] = {s: [] for s in strategies}
        for rep in range(args.reps):
            for s in strategies:                           # identical split and start for every strategy
                c = simulate(Z, y, grp, n_cls, s, np.random.default_rng(seed0 + rep),
                             args.rounds, args.warmup, args.pool_share)
                if c is not None:
                    curves[rn][s].append(c)
        curves[rn] = {s: np.array(v) for s, v in curves[rn].items()}
    budgets = np.arange(n_cls, n_cls + args.rounds + 1)
    b_idx = [int(np.flatnonzero(budgets == b)[0]) for b in (8, 12, 16) if b in budgets]

    res = {}
    for rn in curves:
        mean = {s: curves[rn][s].mean(0) for s in strategies}
        for s in ("margin", "hybrid"):
            d = curves[rn][s][:, b_idx].mean(1) - curves[rn]["random"][:, b_idx].mean(1)
            h = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
            target = mean["random"][-1]
            need = next((int(budgets[r]) for r, v in enumerate(mean[s]) if v >= target), None)
            res[f"{rn}-{s}"] = {"delta_8_16": float(d.mean()), "ci95": float(h), "labels_to_match_random_max": need}
        res[f"{rn}-curves"] = {s: mean[s].tolist() for s in strategies}

    queue, n_lab, effect, state = queue_with_explanations(Zraw, raw, y, grp, names, feats, ids, n_acc, n_tx,
                                                          np.random.default_rng(seed0 + 999), args.queue_after,
                                                          args.queue_size, args.pool_share)
    # state of this batch in case ids (rows of the fold's test typologies -> ids): the analyst's round starts here
    to_id = lambda rows_: [int(ids[i]) for i in rows_]
    stan = {"dataset": dataset, "fold": int(cfg.fold), "round": 0, "names": names, "sha": sha,
            # the experiment design this batch belongs to; rounds refuse to mix splits
            "split_uri": bundle.get("split_uri", ""), "split_sha": bundle.get("split_sha", ""),
            "labeled": [[int(ids[i]), names[c]] for i, c in zip(state["labeled"], state["labels"])],
            "pool": to_id(state["pool"]), "eval": to_id(state["eval"]), "queue": to_id(state["queue"]),
            "history": [{"round": 0, "labels": len(state["labeled"]),
                         "accuracy": effect["before"]}]}
    files_link, batch_dir = "-", None
    if args.write_analyst_files:
        try:
            files_link, batch_dir = write_analyst_batch(args, dataset, int(cfg.fold), names, queue, stan=stan)
            print("Analyst batch written:", files_link)
        except Exception as e:  # the report is the component's job; the batch is an add-on
            print(f"WARNING: analyst batch not written: {e!r}")

    # ---- HTML ------------------------------------------------------------- #
    rm, rr = res["raw-margin"], res["raw-curves"]
    log_to_experiment(args, rm, rr, effect, int(budgets[-1]))
    eff = (f"{rm['labels_to_match_random_max']} labels instead of {int(budgets[-1])}"
           if rm["labels_to_match_random_max"] else "not reached within the budget")
    charts = "".join(
        f"<div>{svg_curves(budgets.tolist(), [(s, res[f'{rn}-curves'][s], STRATEGY_COLOURS[s]) for s in strategies], f'{rn}: balanced accuracy vs labels')}</div>"
        for rn in ("raw", "protonet"))
    rows_html = ""
    for s in ("margin", "hybrid"):
        for rn in ("raw", "protonet"):
            r_ = res[f"{rn}-{s}"]
            rows_html += (f"<tr><td>{s}</td><td>{rn}</td><td class='num'>{r_['delta_8_16']:+.3f} "
                          f"&plusmn; {r_['ci95']:.3f}</td><td class='num'>{r_['labels_to_match_random_max'] or '&mdash;'}</td></tr>")
    q_html = ""
    for it in queue:
        right = (f"<span style='color:{GREEN}'>current guess right</span>" if it["model_was_right"]
                 else f"<span style='color:{RED}'>current guess wrong</span>")
        q_html += f"""<div class="q"><div class="qh"><b>Case #{esc(it['case'])}</b> ({it['accounts']} accounts,
{it['transactions']} transaction{'s' if it['transactions'] != 1 else ''}) &mdash; torn between <b>{esc(it['nearest'])}</b> and <b>{esc(it['runner_up'])}</b>
(margin {it['margin']:.3f}). Analyst's answer (oracle): <b>{esc(it['oracle_label'])}</b> &middot; {right}</div>
<div class="cols"><div><div class="cap" style="color:{GREEN}">pulls toward {esc(it['nearest'])}</div><ul>{''.join(f'<li>{esc(p)}</li>' for p in it['for_nearest']) or '<li>&mdash;</li>'}</ul></div>
<div><div class="cap" style="color:{ORANGE}">pulls toward {esc(it['runner_up'])}</div><ul>{''.join(f'<li>{esc(p)}</li>' for p in it['for_runner_up']) or '<li>&mdash;</li>'}</ul></div></div></div>"""
    demo = demo_section_pl(args, int(cfg.fold), names, budgets.tolist(), np.asarray(rr["margin"]),
                           np.asarray(rr["random"]), rm["labels_to_match_random_max"], int(budgets[-1]),
                           queue, effect, files_link)
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;color:#1a1a1a;max-width:1060px;background:#fff}}
.meta{{font-size:12px;color:#888;margin-bottom:14px}} h3{{margin:22px 0 6px}}
.primer{{background:#f2f6fb;border-radius:10px;padding:14px 18px;font-size:14px;line-height:1.55}}
.badges{{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}} .badge{{border:2px solid {BLUE};border-radius:10px;padding:8px 14px;min-width:200px}}
.bn{{font-size:12px;color:#666}} .bv{{font-size:20px;font-weight:700;color:{BLUE}}}
.row{{display:flex;gap:20px;flex-wrap:wrap}} table.t{{border-collapse:collapse;font-size:13px}}
.t th,.t td{{border-bottom:1px solid #eee;padding:5px 10px;text-align:left}} .num{{text-align:right!important;font-variant-numeric:tabular-nums}}
.q{{border:1px solid #e0e0e0;border-radius:10px;padding:12px 16px;margin:10px 0}} .qh{{font-size:13.5px;line-height:1.5}}
.cols{{display:flex;gap:24px;flex-wrap:wrap}} .cols>div{{flex:1;min-width:300px}} .cap{{font-size:12px;margin-top:6px}}
ul{{margin:4px 0 0 18px;padding:0;font-size:12.5px;line-height:1.5}} .note{{font-size:12.5px;color:#666;line-height:1.5}}
.demo h1{{font-size:24px;line-height:1.25;margin:4px 0 8px}} .demo h2{{font-size:18px;margin:26px 0 4px}}
.lead{{font-size:16px;line-height:1.5;max-width:860px;margin:0 0 6px}} .ctx{{font-size:13px;color:#52514e;margin:4px 0 12px}}
.tiles{{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0 18px}}
.tile{{flex:1;min-width:230px;background:#f2f6fb;border-radius:12px;padding:14px 18px}}
.tv{{font-size:30px;font-weight:700;color:#1f4e8c;line-height:1.1}} .tl{{font-size:13px;color:#33322f;margin-top:6px;line-height:1.4}}
.viz{{display:flex;gap:28px;flex-wrap:wrap;align-items:flex-start}} .vc{{flex:1.6;min-width:320px}} .vl{{flex:1;min-width:260px}}
.vt{{font-size:13px;font-weight:600;color:#33322f;margin-bottom:4px}}
.cards{{display:flex;gap:14px;flex-wrap:wrap}} .card{{flex:1;min-width:300px;border:1px solid #e3e2dd;border-radius:12px;padding:14px 16px}}
.ch{{font-size:15px}} .cq{{font-size:14px;margin:6px 0 10px}}
.tug{{display:flex;height:26px;border-radius:6px;overflow:hidden;font-size:12px;font-weight:600;gap:2px}}
.ta{{background:#2a78d6;color:#fff;display:flex;align-items:center;padding:0 8px;white-space:nowrap;overflow:hidden}}
.tb{{background:#d9d8d2;color:#33322f;display:flex;align-items:center;justify-content:flex-end;padding:0 8px;white-space:nowrap;overflow:hidden}}
.tc{{font-size:11.5px;color:#6b6a66;margin:4px 0 8px}} .cols2{{display:flex;gap:16px;flex-wrap:wrap}} .cols2>div{{flex:1;min-width:200px}}
.cap2{{font-size:12.5px;font-weight:600;color:#33322f}} .mut{{color:#6b6a66}}
.foot{{font-size:12px;color:#6b6a66;line-height:1.5;margin-top:18px;max-width:900px}}
details.tech{{margin-top:30px;border-top:1px solid #e3e2dd;padding-top:12px}} details.tech>summary{{cursor:pointer;font-size:14px;font-weight:600;color:#33322f}}
</style></head><body>
{demo}
<details class="tech"><summary>Technical details (EN): all strategies, both representations, full queue with explanations</summary>
<h2>Active few-shot learning &mdash; model version {esc(args.model_version)}</h2>
<p class="meta">artifact: {esc(args.model_dir)} | {esc(dataset)} fold {cfg.fold} | data sha {esc(sha[:12])} |
test typologies (unseen in training): {esc(', '.join(names))} | {args.reps} repetitions x {args.rounds} rounds</p>
<div class="primer"><b>What this shows.</b> An analyst labels flagged cases one at a time. Starting from one labelled
case per typology, the model either receives labels at random or <i>asks</i> for the case it is least sure about
(smallest gap between its two nearest typology prototypes). No retraining: each new label joins its typology's
prototype. The analyst is simulated by revealing the true typology. Task framing: typology assignment for an
already detected case, not detection.</div>
<div class="badges"><div class="badge"><div class="bn">model asks (margin) vs random, 8&ndash;16 labels</div>
<div class="bv">{rm['delta_8_16']:+.1%}</div></div><div class="badge"><div class="bn">labels to reach random's accuracy at {int(budgets[-1])}</div>
<div class="bv">{esc(eff)}</div></div><div class="badge"><div class="bn">accuracy at {int(budgets[-1])} labels: margin / random</div>
<div class="bv">{rr['margin'][-1]:.3f} / {rr['random'][-1]:.3f}</div></div></div>
<h3>Learning curves</h3><div class="row">{charts}</div>
<table class="t"><tr><th>strategy</th><th>representation</th><th>gain over random, 8&ndash;16 labels (&plusmn; 95% CI over repetitions)</th><th>labels to match random @ {int(budgets[-1])}</th></tr>{rows_html}</table>
<p class="note">Strategy choice was validated across all 5 folds in samld_case_active_learning.ipynb with a criterion
fixed before the run (margin vs random). This page reports one fold; repetitions within a fold share the same cases,
so the interval above is descriptive. hybrid (typical cases first, then margin) is reported, not validated.</p>
<p class="note"><b>Batch effect (this fold, one split):</b> labelling the whole queue below at once moves balanced accuracy
on the evaluation set from {effect['before']:.3f} to {effect['after_queue']:.3f}; the same number of random pool cases
gives {effect['after_random']:.3f} (mean of 50 draws). The queue is picked greedily (see below), not as the top
{effect['batch']} by margin. Each analyst round re-checks this on real answers: its gate compares the prototypes before
and after the batch on the same evaluation set.</p>
<h3>Next cases the model asks the analyst to label</h3>
<p class="note">After {n_lab} labels ({n_cls} seed + {args.queue_after} chosen by the model). The batch is chosen one case at
a time: the smallest margin first, then the model's own guess stands in for that label and the next smallest margin is
taken, so the batch does not pile near-duplicates onto one boundary. For prototypes in raw feature space each feature's
push is exact: the contributions sum to d(case, runner-up)&sup2; &minus; d(case, nearest)&sup2; with no approximation.
Raw values; "avg" = mean of the cases labelled so far for that typology.</p>
{q_html}
<p class="meta" style="margin-top:20px">Scope: synthetic SAML-D, flagged cases (boundaries from Is_laundering). The queue
ranks typology ambiguity among already detected cases; it does not find new laundering.</p>
</details>
</body></html>"""

    summary = {"dataset": dataset, "fold": int(cfg.fold), "test_typologies": names, "sha": sha,
               "reps": args.reps, "rounds": args.rounds, "budgets": budgets.tolist(), "results": res, "queue": queue}
    line = (f"fold {cfg.fold} | margin vs random {rm['delta_8_16']:+.3f} (8-16 labels) | "
            f"margin reaches random@{int(budgets[-1])} at {rm['labels_to_match_random_max']} labels | "
            f"acc@{int(budgets[-1])} margin {rr['margin'][-1]:.3f} vs random {rr['random'][-1]:.3f}")
    _write(args.report_html_path, html)
    report_link = "-"
    if batch_dir:
        try:   # the same page next to the CSVs: one click from the e-mail, no Vertex console needed
            _put(f"{batch_dir}/raport.html", html.encode("utf-8"), args.project, "text/html; charset=utf-8")
            report_link = (f"https://storage.cloud.google.com/{batch_dir[len('gs://'):]}/raport.html"
                           if batch_dir.startswith("gs://") else f"{batch_dir}/raport.html")
        except Exception as e:
            print(f"WARNING: report copy not written: {e!r}")
    _write(args.summary_output_path, line)
    if args.json_output_path:
        _write(args.json_output_path, json.dumps(summary, indent=2, default=float))
    print("Active learning report written.", line)
    digest = analyst_digest(args, cfg.fold, names, rm, int(budgets[-1]), queue, files_link, report_link)
    if args.digest_output_path:
        _write(args.digest_output_path, digest)
    if args.emit_digest:
        print(digest)


DIGEST_MARKER = "FSL_ANALYST_DIGEST"


def al_experiment_record(args, rm: dict, rr: dict, effect: dict, budget: int) -> tuple[dict, dict]:
    """(params, metrics) this component adds to the training's Experiments run.

    The validated strategy only (margin, raw features): its gain over random at
    8-16 labels, the labels it needs to reach random's accuracy at the full
    budget (absent when never reached: a missing metric, not a fake number),
    both accuracies at the budget, and the batch effect of the queue shown to
    the analyst versus the same number of random cases.
    """
    params = {"al_strategy": "margin", "al_representation": "raw", "al_reps": int(args.reps),
              "al_rounds": int(args.rounds), "al_budget": int(budget), "al_queue_size": int(args.queue_size)}
    metrics = {"al_margin_vs_random_8_16": float(rm["delta_8_16"]),
               "al_acc_margin_at_budget": float(rr["margin"][-1]),
               "al_acc_random_at_budget": float(rr["random"][-1]),
               "al_queue_effect_vs_random": float(effect["after_queue"] - effect["after_random"])}
    if rm.get("labels_to_match_random_max"):
        metrics["al_labels_to_match_random"] = int(rm["labels_to_match_random_max"])
    return params, metrics


def log_to_experiment(args, rm: dict, rr: dict, effect: dict, budget: int) -> None:
    """Resume the training's run and add the active-learning numbers; never fails the report."""
    name, run = (getattr(args, "experiment_name", "") or "").strip(), (getattr(args, "experiment_run", "") or "").strip()
    if not (name and run):
        return
    params, metrics = al_experiment_record(args, rm, rr, effect, budget)
    try:
        from google.cloud import aiplatform

        aiplatform.init(project=args.project, location=args.region, experiment=name)
        aiplatform.start_run(run, resume=True)
        aiplatform.log_params(params)
        aiplatform.log_metrics(metrics)
        aiplatform.end_run()
        print(f"Active learning logged to experiment run {run}: {metrics}")
    except Exception as e:  # noqa: BLE001 - the report is the component's job; the run is an add-on
        print(f"WARNING: active learning not logged to Experiments run {run!r}: {e!r}")


def _run_id(args) -> str:
    """Pipeline job name, or a local timestamp outside Vertex (placeholder not substituted)."""
    import time
    if args.run_name and "{{" not in args.run_name:
        return args.run_name
    return time.strftime("local-%Y%m%d-%H%M%S")


def _put(dest: str, data: bytes, project: str, content_type: str) -> None:
    if dest.startswith("gs://"):
        from google.cloud import storage
        bucket, _, name = dest[len("gs://"):].partition("/")
        storage.Client(project=project).bucket(bucket).blob(name).upload_from_string(data, content_type=content_type)
    else:
        p = Path(dest)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def _read_parquet(src: str, project: str):
    """Reference table, or None when it does not exist yet."""
    import pandas as pd
    if src.startswith("gs://"):
        import tempfile
        from google.cloud import storage
        bucket, _, name = src[len("gs://"):].partition("/")
        blob = storage.Client(project=project).bucket(bucket).blob(name)
        if not blob.exists():
            return None
        local = Path(tempfile.mkdtemp()) / "t.parquet"
        blob.download_to_filename(str(local))
        return pd.read_parquet(local)
    return pd.read_parquet(src) if Path(src).exists() else None


def write_analyst_batch(args, dataset, fold, names, queue, stan=None, batch=None) -> str:
    """One batch for the analyst: <root>/do_oznaczenia/<run>/fold<k>/
         kolejka.csv     one row per queued case, empty column typologia_analityka to fill in
         transakcje.csv  the raw transactions of those cases (no labels)
         README.txt      what to fill in and where to put the filled file
    The filled kolejka.csv goes back to <root>/oznaczone/<run>/fold<k>/kolejka.csv.
    CSV: ';' separator, UTF-8 with BOM (opens directly in Polish Excel). Returns a link."""
    import pandas as pd

    root = (args.analyst_root or f"gs://{args.bucket}/analyst").rstrip("/")
    batch = batch or f"{_run_id(args)}/fold{fold}"
    out = f"{root}/do_oznaczenia/{batch}"
    if stan is not None:      # where the analyst's round starts from (labelled cases, pool, evaluation set)
        _put(f"{out}/stan.json", json.dumps(stan, indent=1, ensure_ascii=False).encode("utf-8"), args.project,
             "application/json")
    T = _read_parquet(args.case_transactions or f"{root}/reference/{dataset}/case_transactions.parquet",
                      args.project)
    ids = [int(it["case"]) for it in queue]
    rows = []
    for rank, it in enumerate(queue, 1):
        span = ("", "")
        if T is not None:
            d = T.loc[T["przypadek"] == int(it["case"]), "data"]
            if len(d):
                span = (str(d.min()), str(d.max()))
        rows.append({
            "kolejnosc": rank, "przypadek": int(it["case"]),
            "liczba_kont": it["accounts"], "liczba_transakcji": it["transactions"],
            "okres_od": span[0], "okres_do": span[1],
            "podpowiedz_modelu": it["nearest"], "druga_typologia": it["runner_up"],
            "margines": round(float(it["margin"]), 3),
            "za_podpowiedzia": " | ".join(it["for_nearest"]),
            "za_druga_typologia": " | ".join(it["for_runner_up"]),
            "typologia_analityka": "", "komentarz": "",
        })
    csv_kw = dict(index=False, sep=";")
    _put(f"{out}/kolejka.csv", pd.DataFrame(rows).to_csv(**csv_kw).encode("utf-8-sig"), args.project,
         "text/csv; charset=utf-8")
    if T is not None:
        tx = T[T["przypadek"].isin(ids)].sort_values(["przypadek", "data", "godzina"])
        _put(f"{out}/transakcje.csv", tx.to_csv(**csv_kw).encode("utf-8-sig"), args.project,
             "text/csv; charset=utf-8")
    back = f"{root}/oznaczone/{batch}/kolejka.csv"
    readme = (
        f"Partia do oznaczenia: {batch}\n"
        f"Zbior: {dataset}, fold {fold}. Przypadkow: {len(rows)}.\n\n"
        "kolejka.csv: jeden wiersz na przypadek (grupe podejrzanych transakcji).\n"
        "  kolejnosc 1 = przypadek, przy ktorym model waha sie najbardziej (najmniejszy margines).\n"
        "  Kolejne model wybiera po kolei, zakladajac na chwile, ze jego podpowiedzi do poprzednich\n"
        "  sa trafne; dzieki temu w partii nie ma kilku prawie identycznych przypadkow.\n"
        "  podpowiedz_modelu / druga_typologia: dwie typologie, miedzy ktorymi model sie waha.\n"
        "  za_podpowiedzia / za_druga_typologia: cechy, ktore ciagna przypadek w kazda strone\n"
        "  (wartosc przypadku i srednie obu typologii; wyjasnienie jest dokladne, nie przyblizone).\n"
        "transakcje.csv: surowe transakcje tych przypadkow (kolumna przypadek laczy oba pliki).\n\n"
        "Co zrobic:\n"
        f"  1. W kolumnie typologia_analityka wpisz jedna z: {', '.join(names)}.\n"
        "     Inna typologia: wpisz jej nazwe. Pusta komorka = przypadek pominiety.\n"
        "  2. Pozostalych kolumn nie zmieniaj; komentarz jest opcjonalny.\n"
        f"  3. Zapisz plik jako CSV z separatorem ';' i odloz go tutaj:\n     {back}\n"
    )
    if T is None:
        readme += ("\nUWAGA: brak tabeli transakcji (samld_case_analyst_data.ipynb nie byl uruchomiony),\n"
                   "dlatego nie ma transakcje.csv ani okresow w kolejce.\n")
    _put(f"{out}/README.txt", readme.encode("utf-8"), args.project, "text/plain; charset=utf-8")
    if out.startswith("gs://"):
        bucket, _, prefix = out[len("gs://"):].partition("/")
        return f"https://console.cloud.google.com/storage/browser/{bucket}/{prefix}?project={args.project}", out
    return out, out


def analyst_digest(args, fold, names, rm, budget, queue, files_link="-", report_link="-") -> str:
    """One log line that a Cloud Monitoring log-based alert turns into the analyst's e-mail.

    Format: `FSL_ANALYST_DIGEST | key=value | key=value | ...`. The alert policy pulls
    each value out with REGEXP_EXTRACT(jsonPayload.message, "key=([^|]+)"), so values
    must never contain "|".
    """
    def clean(v) -> str:
        return str(v).replace("|", "/").replace("\n", " ").strip()

    need = rm["labels_to_match_random_max"]
    saving = (f"{need} labels instead of {budget} (margin {rm['delta_8_16']:+.3f} vs random, 8-16 labels)"
              if need else f"random@{budget} not reached (margin {rm['delta_8_16']:+.3f} vs random, 8-16 labels)")
    items = []
    for it in queue[:args.digest_queue_size]:
        items.append("#{} ({} vs {})".format(it["case"], it["nearest"], it["runner_up"]))
    run = args.run_name if args.run_name and "{{" not in args.run_name else ""
    link = (f"https://console.cloud.google.com/vertex-ai/locations/{args.region}/pipelines/runs/{run}"
            f"?project={args.project}" if run else "-")
    fields = [
        ("fold", fold),
        ("model", args.model_version),
        ("typologies", ", ".join(names)),
        ("saving", saving),
        ("queue", "; ".join(items) or "-"),
        ("report", report_link),
        ("files", files_link),
        ("run", link),
    ]
    return " | ".join([DIGEST_MARKER] + [f"{k}={clean(v)}" for k, v in fields])


if __name__ == "__main__":
    main()
