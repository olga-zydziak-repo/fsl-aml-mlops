#!/usr/bin/env python3
"""One analyst round of active learning: the filled kolejka.csv goes back into the model.

Input (written by al_simulation_entry.py or by the previous round):
    <root>/do_oznaczenia/<batch>/stan.json    labelled cases so far, pool, evaluation set, queue
    <root>/oznaczone/<batch>/kolejka.csv      the same queue with typologia_analityka filled in

What a round does:
  1. reads the analyst's answers (empty cell = skipped, a name outside the fold's typologies
     = new typology, kept aside: one case cannot form a prototype the evaluation set could test);
  2. builds candidate prototypes from the labels in use plus these answers (one mean per typology,
     raw standardised case features; no training);
  3. gate: balanced accuracy of the candidate on the evaluation set must not fall more than
     --max-drop below the prototypes in use, AND no single typology may lose more than
     --max-typology-drop of its evaluation cases (counting at least --min-typology-cases cases).
     The second rule exists because the average hides trades: on SAML-D fold 3, round 2 moved
     Gather-Scatter 67% -> 92% and Layered_Fan_In 80% -> 49% for a balanced change of -1.7 pp.
     Pass -> the answers join the labels in use and the
     candidate goes into use (<root>/wzorce/<base>/produkcja.json). Fail -> the prototypes in use
     stay and the answers are held for a second review (stan.json "held"), so one wrong batch does
     not poison every later round. The gate cannot tell a wrong answer from a correct one that
     drags a mean prototype; it judges the prototypes, and a person decides about held answers;
  4. picks the next batch from the prototypes in use (greedy_batch: one case at a time, the
     model's guess standing in for the label) with exact explanations, and writes it as
     <batch base>-r<N+1> for the analyst;
  5. writes a Polish round report and prints FSL_ROUND_DIGEST, which a log-based alert turns
     into the analyst's e-mail (analyst_alert_setup.ipynb, rule "FSL - runda active learning").

The evaluation set is the one fixed when the batch chain started, disjoint by account group from
the pool. Its typologies come from the data (SAML-D); in production it is a set of cases analysts
confirmed earlier. Same image as the pipeline, numpy + pandas only.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import al_simulation_entry as A  # noqa: E402  (same image, same helpers)
from explain_pipeline_entry import _write, esc  # noqa: E402

ROUND_MARKER = "FSL_ROUND_DIGEST"


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def _get(src: str, project: str) -> bytes | None:
    """File from GCS or a local path; None when it does not exist."""
    if src.startswith("gs://"):
        from google.cloud import storage
        bucket, _, name = src[len("gs://"):].partition("/")
        blob = storage.Client(project=project).bucket(bucket).blob(name)
        return blob.download_as_bytes() if blob.exists() else None
    p = Path(src)
    return p.read_bytes() if p.exists() else None


def read_filled(data: bytes):
    """kolejka.csv as the analyst saved it: UTF-8 (with or without BOM) or Windows-1250 from Polish
    Excel; separator ';', ',' or tab, whichever the header uses."""
    import io

    import pandas as pd
    for enc in ("utf-8-sig", "cp1250"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    header = text.splitlines()[0] if text else ""
    sep = max((";", ",", "\t"), key=header.count)
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"przypadek", "typologia_analityka"} - set(df.columns)
    if missing:
        raise SystemExit(f"kolejka.csv: brak kolumn {sorted(missing)} (naglowek: {header[:200]!r})")
    return df


def match_typology(text: str, names: list[str]) -> str | None:
    """'cash withdrawal', 'Cash-Withdrawal ' -> 'Cash_Withdrawal'; None when it is not one of `names`."""
    key = lambda s: re.sub(r"[\s_\-]+", "", str(s)).lower()
    return {key(n): n for n in names}.get(key(text))


# --------------------------------------------------------------------------- #
# Data of the fold (same rows and features as al_simulation_entry.py)
# --------------------------------------------------------------------------- #
def load_fold(stan: dict, args):
    import numpy as np

    from fsl.config import TrainConfig
    from fsl.data.registry import get_loaders

    # n_way/k_shot/query only shape the episode samplers, which are not used here;
    # the standardised feature table depends on the fold alone
    cfg = TrainConfig(project=args.project, region=args.region, bucket=args.bucket, dataset=stan["dataset"],
                      fold=int(stan["fold"]), seed=0, n_way=2, k_shot=1, query=1,
                      split_uri=stan.get("split_uri", ""), log_to_vertex=False)
    load_frozen, build = get_loaders(cfg)
    bundle, sha = load_frozen(cfg)
    if bundle.get("split_sha", "") != stan.get("split_sha", ""):
        raise SystemExit(f"split {bundle.get('split_sha', '')[:12] or 'z freeze'} != split partii "
                         f"{str(stan.get('split_sha', ''))[:12] or 'z freeze'}: runda nie miesza podzialow")
    sch, W = bundle["schema"], bundle["windows"]
    names = list(bundle["folds"][cfg.fold]["test"])
    if names != list(stan["names"]):
        raise SystemExit(f"typologie foldu {names} != typologie partii {stan['names']}: inny zbior danych?")
    if sha != stan.get("sha"):
        print(f"WARNING: data sha {sha[:12]} differs from the batch's {str(stan.get('sha'))[:12]}")
    feats = list(sch["feature_names"])
    lab_all = W[sch["label_col"]].to_numpy().astype(str)
    rows = np.flatnonzero(np.isin(lab_all, names))
    _, _, te = build(bundle, cfg)
    id_col = sch.get("id_col")
    ids = W[id_col].to_numpy()[rows] if id_col in W.columns else rows
    col = lambda c: W[c].to_numpy()[rows] if c in W.columns else np.zeros(len(rows), int)
    return {
        "names": names, "feats": feats, "sha": sha, "split_sha": bundle.get("split_sha", ""),
        "Zraw": np.asarray(te.X, dtype=np.float32)[rows],
        "raw": W[feats].astype("float64").to_numpy()[rows],
        "y": np.array([names.index(c) for c in lab_all[rows]]),
        "ids": ids, "n_acc": col("n_acc"), "n_tx": col("n_tx"),
        "pos": {int(c): i for i, c in enumerate(ids)},
    }


# --------------------------------------------------------------------------- #
# Report (Polish)
# --------------------------------------------------------------------------- #
CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;color:#1a1a1a;max-width:1060px;background:#fff}
.demo h1{font-size:24px;line-height:1.25;margin:4px 0 8px} .demo h2{font-size:18px;margin:26px 0 4px}
.lead{font-size:16px;line-height:1.5;max-width:860px;margin:0 0 6px} .ctx{font-size:13px;color:#52514e;margin:4px 0 12px}
.tiles{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0 18px}
.tile{flex:1;min-width:230px;background:#f2f6fb;border-radius:12px;padding:14px 18px}
.tile.ok{background:#eef7ee} .tile.no{background:#fbf1e8}
.tv{font-size:30px;font-weight:700;color:#1f4e8c;line-height:1.1} .ok .tv{color:#2d6a2f} .no .tv{color:#a3501a}
.tl{font-size:13px;color:#33322f;margin-top:6px;line-height:1.4}
.viz{display:flex;gap:28px;flex-wrap:wrap;align-items:flex-start} .vc{flex:1.2;min-width:320px} .vl{flex:1;min-width:300px}
.vt{font-size:13px;font-weight:600;color:#33322f;margin-bottom:6px}
table.ans{border-collapse:collapse;font-size:13px;width:100%} .ans td:last-child{white-space:nowrap} .ans th,.ans td{border-bottom:1px solid #ecebe6;padding:5px 8px;text-align:left}
.ans th{font-weight:600;color:#52514e;font-size:12px} .fix{color:#a3501a;font-weight:600} .same{color:#2d6a2f} .mut{color:#6b6a66}
.cards{display:flex;gap:14px;flex-wrap:wrap} .card{flex:1;min-width:300px;border:1px solid #e3e2dd;border-radius:12px;padding:14px 16px}
.ch{font-size:15px} .cq{font-size:14px;margin:6px 0 10px}
.tug{display:flex;height:26px;border-radius:6px;overflow:hidden;font-size:12px;font-weight:600;gap:2px}
.ta{background:#2a78d6;color:#fff;display:flex;align-items:center;padding:0 8px;white-space:nowrap;overflow:hidden}
.tb{background:#d9d8d2;color:#33322f;display:flex;align-items:center;justify-content:flex-end;padding:0 8px;white-space:nowrap;overflow:hidden}
.tc{font-size:11.5px;color:#6b6a66;margin:4px 0 8px} .cols2{display:flex;gap:16px;flex-wrap:wrap} .cols2>div{flex:1;min-width:200px}
.cap2{font-size:12.5px;font-weight:600;color:#33322f} ul{margin:4px 0 0 18px;padding:0;font-size:12.5px;line-height:1.5}
.foot{font-size:12px;color:#6b6a66;line-height:1.5;margin-top:18px;max-width:900px}
details.tech{margin-top:30px;border-top:1px solid #e3e2dd;padding-top:12px}
details.tech>summary{cursor:pointer;font-size:14px;font-weight:600;color:#33322f}
.note{font-size:12.5px;color:#52514e;line-height:1.55;max-width:900px}
table.typ{max-width:760px;margin:6px 0 4px} .typ tr.bad td{background:#fbf1e8}
.warn{font-size:13px;color:#7a3d12;background:#fbf1e8;border-radius:8px;padding:8px 12px;margin:0 0 16px;max-width:900px}
"""


def pct(v: float) -> str:
    return f"{100 * v:.1f}%".replace(".", ",")


def which(r: int) -> str:
    """'wzorce z rundy 2' / 'wzorce startowe' (round 0 = the labels the batch chain started with)."""
    return "wzorce startowe" if r == 0 else f"wzorce z rundy {r}"


def svg_history(history, width=560, height=290):
    """Accuracy of the prototypes in use after each round (line) and every round's candidate
    (filled dot = accepted, hollow = rejected by the gate)."""
    pl, pr, pt, pb = 56, 24, 18, 58
    xs = [h["round"] for h in history]
    prod = [h.get("production_accuracy", h["accuracy"]) for h in history]
    cand = [h["accuracy"] for h in history]
    # axis on round 5 pp steps, at least 10 pp tall, so small changes do not crowd the labels
    lo = math.floor((min(prod + cand) - 0.02) * 20) / 20
    hi = math.ceil((max(prod + cand) + 0.02) * 20) / 20
    while hi - lo < 0.10 - 1e-9 and (lo > 0 or hi < 1):
        lo = max(0.0, round(lo - 0.05, 2)) if lo > 0 else lo
        hi = min(1.0, round(hi + 0.05, 2)) if hi - lo < 0.10 - 1e-9 and hi < 1 else hi
    lo, hi = max(0.0, lo), min(1.0, hi)
    pw, ph = width - pl - pr, height - pt - pb
    X = lambda r: pl + (r - xs[0]) / ((xs[-1] - xs[0]) or 1) * pw if len(xs) > 1 else pl + pw / 2
    Y = lambda v: pt + (1 - (v - lo) / ((hi - lo) or 1)) * ph
    o = []
    for t in [lo + 0.05 * i for i in range(int(round((hi - lo) / 0.05)) + 1)]:
        o.append(f'<line x1="{pl}" y1="{Y(t):.1f}" x2="{pl + pw}" y2="{Y(t):.1f}" stroke="#ececea"/>'
                 f'<text x="{pl - 8}" y="{Y(t) + 4:.1f}" font-size="11" text-anchor="end" fill="#6b6a66">{100 * t:.0f}%</text>')
    for h in history:
        lab = "start" if h["round"] == 0 else f"runda {h['round']}"
        n_use = h.get("labels_in_use", h["labels"])
        o.append(f'<text x="{X(h["round"]):.1f}" y="{pt + ph + 18}" font-size="11.5" text-anchor="middle" '
                 f'fill="#33322f">{lab}</text><text x="{X(h["round"]):.1f}" y="{pt + ph + 33}" font-size="10.5" '
                 f'text-anchor="middle" fill="#6b6a66">{A.pl_n(n_use, "etykieta", "etykiety", "etykiet")}</text>')
    o.append(f'<line x1="{pl}" y1="{pt + ph}" x2="{pl + pw}" y2="{pt + ph}" stroke="#b5b4ae"/>')
    if len(xs) > 1:
        pts = " ".join(f"{X(r):.1f},{Y(v):.1f}" for r, v in zip(xs, prod))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{A.DEMO_BLUE}" stroke-width="2.5"/>')
    for h, p, c in zip(history, prod, cand):
        r = h["round"]
        o.append(f'<circle cx="{X(r):.1f}" cy="{Y(p):.1f}" r="5" fill="{A.DEMO_BLUE}" stroke="#fff" stroke-width="2">'
                 f'<title>w użyciu po rundzie {r}: {pct(p)}</title></circle>')
        if h.get("accepted") is False:
            right_edge = r == xs[-1] and len(xs) > 1                      # label to the left at the last round
            o.append(f'<circle cx="{X(r):.1f}" cy="{Y(c):.1f}" r="5.5" fill="#fff" stroke="#a3501a" stroke-width="2">'
                     f'<title>runda {r}: nowe wzorce {pct(c)}, odrzucone przez bramkę</title></circle>'
                     f'<text x="{X(r) + (-9 if right_edge else 9):.1f}" y="{Y(c) + 4:.1f}" font-size="11" '
                     f'text-anchor="{"end" if right_edge else "start"}" fill="#a3501a">odrzucone {pct(c)}</text>')
    last = history[-1]
    o.append(f'<text x="{X(last["round"]) - (8 if len(xs) > 1 else -10):.1f}" y="{Y(prod[-1]) - 10:.1f}" font-size="12.5" '
             f'font-weight="600" text-anchor="{"end" if len(xs) > 1 else "start"}" fill="#0b0b0b">{pct(prod[-1])}</text>')
    o.append(f'<text x="{pl + pw / 2:.0f}" y="{height - 4}" font-size="11.5" text-anchor="middle" fill="#52514e">'
             f'trafność wzorców w użyciu na zbiorze kontrolnym</text>')
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px" role="img" '
            f'aria-label="trafność wzorców po kolejnych rundach" xmlns="http://www.w3.org/2000/svg">'
            + "".join(o) + "</svg>")


def answers_table(answers) -> str:
    rows = ""
    for a in answers:
        if a["status"] == "oznaczony":
            verdict = ("<span class='same'>zgodna</span>" if a["label"] == a["guess"]
                       else "<span class='fix'>poprawka</span>")
            ans = esc(a["label"])
        elif a["status"] == "inna":
            verdict, ans = "<span class='mut'>spoza foldu</span>", esc(a["text"])
        else:
            verdict, ans = "<span class='mut'>pominięty</span>", "&mdash;"
        rows += (f"<tr><td>{esc(a['case'])}</td><td>{esc(a['guess'])}</td><td><b>{ans}</b></td>"
                 f"<td>{verdict}</td></tr>")
    return ("<table class='ans'><tr><th>przypadek</th><th>podpowiedź modelu</th><th>analityk</th><th></th></tr>"
            f"{rows}</table>")


def gate_verdict(before, after, per_typology, max_drop, max_typology_drop, min_typology_cases):
    """(accepted, collapses). collapses: typologies whose evaluation accuracy fell by more than
    max_typology_drop AND by at least min_typology_cases cases (one flip in a small class is noise)."""
    collapses = []
    for name, n, b, a in per_typology:
        if b != b or a != a:                                   # NaN: typology absent from evaluation set
            continue
        lost = int(round((b - a) * n))
        if a < b - max_typology_drop and lost >= min_typology_cases:
            collapses.append((name, n, b, a, lost))
    return (after >= before - max_drop) and not collapses, collapses


def collapse_text(collapses, ascii_only=False) -> str:
    parts = []
    for name, n, b, a, lost in collapses:
        t = f"{name} {pct(b)} -> {pct(a)} (-{lost} z {n})" if ascii_only else \
            f"{name} {pct(b)} → {pct(a)} (−{lost} z {n} przypadków)"
        parts.append(t)
    return "; ".join(parts)


def round_html(ctx) -> str:
    c = ctx
    n_new, n_fix = c["n_new"], c["n_fix"]
    head = (f"Runda {c['round']}: analityk oznaczył {A.pl_n(n_new, 'przypadek', 'przypadki', 'przypadków')}, "
            f"trafność {pct(c['before'])} &rarr; {pct(c['after'])}")
    if c["accepted"]:
        lead = ("Odpowiedzi analityka weszły do wzorców typologii od razu, bez trenowania. Bramka porównała nowe "
                "wzorce z tymi w użyciu na zbiorze kontrolnym i je przyjęła: od teraz to one oceniają przypadki.")
        t2 = ("przyjęte", f"bramka: średnio nie gorzej niż o {100 * c['max_drop']:.0f} pp i żadna typologia nie "
                          f"traci więcej niż {100 * c['max_typology_drop']:.0f} pp; w użyciu od tej rundy")
    elif c["collapses"]:
        avg = ("Średnio nowe wzorce wypadły prawie tak samo, ale" if c["after"] >= c["before"] - c["max_drop"]
               else "Nowe wzorce wypadły gorzej, a do tego")
        lead = (f"Odpowiedzi analityka weszły do wzorców typologii od razu, bez trenowania. {avg} jedna typologia "
                "się załamała, więc bramka ich nie przyjęła: zysk na jednej typologii nie może być kupiony utratą "
                "innej. W użyciu zostają dotychczasowe wzorce, a odpowiedzi z tej rundy czekają na drugą "
                "weryfikację.")
        t2 = ("odrzucone", f"załamanie typologii: {collapse_text(c['collapses'])}; w użyciu zostają "
                           f"{which(c['prod_round'])}")
    else:
        lead = ("Odpowiedzi analityka weszły do wzorców typologii od razu, bez trenowania, ale nowe wzorce wypadły "
                "na zbiorze kontrolnym gorzej niż te w użyciu, więc bramka ich nie przyjęła. W użyciu zostają "
                "dotychczasowe wzorce, a odpowiedzi z tej rundy czekają na drugą weryfikację; następna partia "
                "liczy się od wzorców w użyciu.")
        t2 = ("odrzucone", f"bramka: spadek o {100 * (c['before'] - c['after']):.1f} pp, dopuszczalny "
                           f"{100 * c['max_drop']:.0f} pp; w użyciu zostają {which(c['prod_round'])}".replace(".", ","))
    tiles = (f'<div class="tile"><div class="tv">{pct(c["before"])} &rarr; {pct(c["after"])}</div>'
             f'<div class="tl">trafność wzorców na zbiorze kontrolnym '
             f'({A.pl_n(c["n_eval"], "przypadek", "przypadki", "przypadków")} o znanej typologii, '
             f'nieużywanych do nauki)</div></div>'
             f'<div class="tile {"ok" if c["accepted"] else "no"}"><div class="tv">{t2[0]}</div>'
             f'<div class="tl">{esc(t2[1])}</div></div>'
             f'<div class="tile"><div class="tv">{n_fix} z {n_new}</div>'
             f'<div class="tl">odpowiedzi, w których analityk poprawił podpowiedź modelu; pozostałe ją '
             f'potwierdziły</div></div>')
    if c["next_items"]:
        link = (f'<a href="{esc(c["files_link"])}">kolejka.csv i transakcje.csv (folder w GCS)</a>'
                if c["files_link"].startswith("http") else esc(c["files_link"]))
        nxt = (f"<h2>Następne do sprawdzenia</h2><p class='ctx'>Trzy pierwsze z {len(c['next_items'])} przypadków "
               f"w nowej partii {esc(c['next_batch'])}. Pełna lista: {link}.</p>"
               f"<div class='cards'>{A.cards_html(c['next_items'][:3])}</div>")
    else:
        nxt = "<h2>Pula wyczerpana</h2><p class='ctx'>Wszystkie przypadki z puli są już oznaczone albo odłożone.</p>"
    small = [(t, n) for t, n, _, _ in c["per_typology"] if n <= 3]
    warn = ""
    if small:
        t, n = min(small, key=lambda x: x[1])
        step = f"{100 / len(c['per_typology']) / n:.1f}".replace(".", ",")      # one case flip, in pp
        warn = (f"<p class='warn'>Uwaga: typologia {esc(t)} ma w zbiorze kontrolnym tylko "
                f"{A.pl_n(n, 'przypadek', 'przypadki', 'przypadków')}, więc jeden przypadek zmienia trafność o "
                f"{step} pp. Werdykt bramki jest tu mało pewny; rozbicie na typologie jest niżej.</p>")
    bad = {name for name, *_ in c["collapses"]}

    def delta_cell(b, a):
        if b != b or a != a:
            return "&mdash;"
        dd = 100 * (a - b)
        if abs(dd) < 0.05:
            return "<span class='mut'>bez zmian</span>"
        cls = "fix" if dd < 0 else "same"
        return f"<span class='{cls}'>{dd:+.1f} pp</span>".replace(".", ",").replace("-", "−")
    row_open = lambda t: '<tr class="bad">' if t in bad else "<tr>"
    pt_rows = "".join(f"{row_open(t)}<td>{esc(t)}</td><td>{n}</td>"
                      f"<td>{pct(b) if b == b else '&mdash;'}</td><td>{pct(a) if a == a else '&mdash;'}</td>"
                      f"<td>{delta_cell(b, a)}</td></tr>" for t, n, b, a in c["per_typology"])
    typ_table = (f"<table class='ans typ'><tr><th>typologia</th><th>przypadków w zbiorze kontrolnym</th>"
                 f"<th>wzorce w użyciu</th><th>nowe wzorce</th><th>zmiana</th></tr>{pt_rows}</table>")
    extra = ""
    if c["other"]:
        extra += ("<p class='note'><b>Typologie spoza foldu</b> (odłożone, nie wchodzą do wzorców): "
                  + esc("; ".join(f"{cid}: {t}" for cid, t in c["other"])) + "</p>")
    if c["held"]:
        extra += ("<p class='note'><b>Odpowiedzi czekające na drugą weryfikację</b> (z rund odrzuconych przez bramkę; "
                  "nie wchodzą do wzorców, dopóki ktoś ich nie potwierdzi): "
                  + esc("; ".join(f"{cid}: {t} (runda {r})" for cid, t, r in c["held"])) + "</p>")
    if c["ignored"]:
        extra += ("<p class='note'><b>Pominięte wiersze pliku</b> (przypadek spoza tej partii albo już oznaczony): "
                  + esc(", ".join(c["ignored"])) + "</p>")
    return f"""<!DOCTYPE html><html lang="pl"><head><meta charset="utf-8"><title>Runda {c['round']} active learning</title>
<style>{CSS}</style></head><body><section class="demo">
<h1>{head}</h1>
<p class="lead">{lead}</p>
<p class="ctx">fold {c['fold']} · typologie: {esc(', '.join(c['names']))} · partia {esc(c['batch'])}</p>
<div class="tiles">{tiles}</div>
{warn}
<div class="viz"><div class="vc"><div class="vt">Trafność po kolejnych rundach</div>{svg_history(c['history'])}</div>
<div class="vl"><div class="vt">Odpowiedzi analityka w tej rundzie</div>{answers_table(c['answers'])}</div></div>
<h2>Trafność per typologia</h2>
<p class="ctx">Bramka patrzy na średnią i na każdą typologię osobno: nowe wzorce nie wchodzą do użycia, jeśli
któraś typologia traci więcej niż {100 * c['max_typology_drop']:.0f} pp i co najmniej {c['min_typology_cases']} przypadki.</p>
{typ_table}
{nxt}
<p class="foot">Wzorzec typologii to średnia oznaczonych przypadków tej typologii w przestrzeni cech przypadku; nowy
przypadek trafia do najbliższego wzorca. Zbiór kontrolny jest stały od startu partii i rozłączny z pulą (inne grupy
kont). Tu typologie zbioru kontrolnego pochodzą z danych (SAML-D, dane syntetyczne); w produkcji to przypadki
wcześniej potwierdzone przez analityków. Zadanie: przypisanie typologii do wykrytego przypadku, nie wykrywanie.</p>
</section>
<details class="tech"><summary>Szczegóły techniczne</summary>
<p class="note">Wzorce w użyciu przed rundą: {which(c['prod_round_before'])}, {c['n_prod_before']} etykiet, trafność
zrównoważona {c['before']:.3f}. Nowe wzorce (te etykiety + {n_new} z tej rundy): {c['after']:.3f}. Reguła bramki: nowe &ge;
w użyciu &minus; {c['max_drop']:.2f} oraz żadna typologia nie traci więcej niż {c['max_typology_drop']:.2f} przy co najmniej
{c['min_typology_cases']} przypadkach. Etykiet w użyciu po rundzie: {c['n_labels']}.
Następna partia: {len(c['next_items'])} przypadków, wybieranych po jednym; po każdym wyborze podpowiedź modelu zastępuje
etykietę do czasu odpowiedzi analityka, więc partia nie skupia się na jednej granicy. Wyjaśnienia są dokładne:
wkłady cech sumują się do d(przypadek, druga)&sup2; &minus; d(przypadek, najbliższa)&sup2;.
Wzorce: {esc(c['wzorce_link'])}. Dane: sha {esc(c['sha'][:12])}.</p>
{extra}
</details></body></html>"""


def round_digest(args, ctx, files_link, report_link) -> str:
    """One log line for the alert rule 'FSL - runda active learning'. Values never contain '|'."""
    clean = lambda v: str(v).replace("|", "/").replace("\n", " ").strip()
    c = ctx
    if c["accepted"]:
        verdict = "przyjeta, nowe wzorce w uzyciu"
    elif c["collapses"]:
        verdict = (f"odrzucona, zalamanie typologii: {collapse_text(c['collapses'], ascii_only=True)}; "
                   f"w uzyciu zostaja {which(c['prod_round'])}")
    else:
        verdict = f"odrzucona, w uzyciu zostaja {which(c['prod_round'])}"
    delta = 100 * (c["after"] - c["before"])
    accuracy = f"{pct(c['before'])} -> {pct(c['after'])} ({delta:+.1f} pp)".replace(".", ",")
    queue = "; ".join("#{} ({} vs {})".format(it["case"], it["nearest"], it["runner_up"])
                      for it in c["next_items"][:args.digest_queue_size]) or "pula wyczerpana"
    run = args.run_name if args.run_name and "{{" not in args.run_name else ""
    link = (f"https://console.cloud.google.com/vertex-ai/locations/{args.region}/pipelines/runs/{run}"
            f"?project={args.project}" if run else "-")
    fields = [("fold", c["fold"]), ("round", c["round"]), ("verdict", verdict), ("accuracy", accuracy),
              ("labels", f"{c['n_new']} nowych, w uzyciu razem {c['n_labels']}"),
              ("agreement", f"analityk poprawil model w {c['n_fix']} z {c['n_new']}"),
              ("queue", queue), ("report", report_link), ("files", files_link), ("run", link)]
    return " | ".join([ROUND_MARKER] + [f"{k}={clean(v)}" for k, v in fields])


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="One analyst round of active learning.")
    ap.add_argument("--project", required=True)
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--batch", required=True, help="<run>/fold<k> or <run>/fold<k>-r<N>, as under do_oznaczenia/")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--analyst-root", default="", help="default gs://<bucket>/analyst (a local dir works for tests)")
    ap.add_argument("--case-transactions", default="")
    ap.add_argument("--queue-size", type=int, default=8)
    ap.add_argument("--max-drop", type=float, default=0.02,
                    help="gate: new prototypes may be at most this much below the ones in use (balanced accuracy)")
    ap.add_argument("--max-typology-drop", type=float, default=0.10,
                    help="gate: no typology may lose more than this share of its evaluation cases ...")
    ap.add_argument("--min-typology-cases", type=int, default=3,
                    help="... counting at least this many cases (one flip in a small class is noise)")
    ap.add_argument("--digest-queue-size", type=int, default=5)
    ap.add_argument("--report-html-path", required=True)
    ap.add_argument("--summary-output-path", required=True)
    ap.add_argument("--json-output-path", default=None)
    args = ap.parse_args()

    import numpy as np

    root = (args.analyst_root or f"gs://{args.bucket}/analyst").rstrip("/")
    batch = args.batch.strip().strip("/")
    raw_stan = _get(f"{root}/do_oznaczenia/{batch}/stan.json", args.project)
    if raw_stan is None:
        raise SystemExit(f"brak {root}/do_oznaczenia/{batch}/stan.json: partia sprzed aktualizacji obrazu "
                         "albo zla nazwa partii (lista partii: al_runda.ipynb, komorka 2)")
    filled = _get(f"{root}/oznaczone/{batch}/kolejka.csv", args.project)
    if filled is None:
        raise SystemExit(f"brak wypelnionego pliku {root}/oznaczone/{batch}/kolejka.csv")
    stan = json.loads(raw_stan)
    names = list(stan["names"])
    n_cls, rnd = len(names), int(stan["round"]) + 1
    d = load_fold(stan, args)
    Zraw, y, pos = d["Zraw"], d["y"], d["pos"]

    # ---- the analyst's answers ------------------------------------------- #
    df = read_filled(filled)
    queue_ids = [int(c) for c in stan["queue"]]
    labelled_before = {int(c) for c, _ in stan["labeled"]}
    by_case, ignored = {}, []
    for _, row in df.iterrows():
        try:
            cid = int(float(str(row["przypadek"]).strip()))
        except ValueError:
            continue
        if cid not in queue_ids or cid in labelled_before:
            ignored.append(str(cid))
            continue
        by_case[cid] = str(row["typologia_analityka"]).strip()   # last row wins for duplicates

    L_prev = [pos[int(c)] for c, _ in stan["labeled"]]
    lab_prev = [names.index(t) for _, t in stan["labeled"]]
    P_prev = np.stack([Zraw[np.asarray(L_prev)[np.asarray(lab_prev) == k]].mean(0) for k in range(n_cls)])
    answers, new, other, skipped = [], [], [], []
    for cid in queue_ids:
        guess = names[int(((Zraw[pos[cid]] - P_prev) ** 2).sum(1).argmin())]  # the suggestion the analyst saw
        text = by_case.get(cid, "")
        name = match_typology(text, names) if text else None
        if name:
            answers.append({"case": cid, "guess": guess, "label": name, "status": "oznaczony"})
            new.append([cid, name])
        elif text:
            answers.append({"case": cid, "guess": guess, "text": text, "status": "inna"})
            other.append([cid, text])
        else:
            answers.append({"case": cid, "guess": guess, "status": "pominiety"})
            skipped.append(cid)
    if not new:
        raise SystemExit(f"{root}/oznaczone/{batch}/kolejka.csv: zadnej typologii z {names} w kolumnie "
                         "typologia_analityka, nie ma czego dodac")

    # ---- prototypes before / after, gate ---------------------------------- #
    ev = np.array([pos[int(c)] for c in stan["eval"]])
    in_use = [[int(c), t] for c, t in stan["labeled"]]                 # labels behind the prototypes in use
    candidate = in_use + new
    acc = lambda labs: A.balanced_accuracy(Zraw, y, ev, [pos[c] for c, _ in labs],
                                           [names.index(t) for _, t in labs], n_cls)
    prod_round_before = int(stan.get("production_round", 0))
    before, after = acc(in_use), acc(candidate)

    def per_class(labs):
        L = np.asarray([pos[c] for c, _ in labs])
        lab = np.asarray([names.index(t) for _, t in labs])
        P = np.stack([Zraw[L[lab == k]].mean(0) for k in range(n_cls)])
        pred = ((Zraw[ev][:, None] - P[None]) ** 2).sum(-1).argmin(1)
        return [float(np.mean(pred[y[ev] == k] == k)) if np.any(y[ev] == k) else float("nan") for k in range(n_cls)]
    pc_before, pc_after = per_class(in_use), per_class(candidate)
    per_typology = [(names[k], int(np.sum(y[ev] == k)), pc_before[k], pc_after[k]) for k in range(n_cls)]
    accepted, collapses = gate_verdict(before, after, per_typology, args.max_drop,
                                       args.max_typology_drop, args.min_typology_cases)
    reason = ("ok" if accepted else
              f"zalamanie typologii: {collapse_text(collapses, ascii_only=True)}" if collapses else
              f"srednia spadla o {100 * (before - after):.1f} pp".replace(".", ","))
    labeled = candidate if accepted else in_use
    held = [list(h) for h in stan.get("held", [])] + ([] if accepted else [[c, t, rnd] for c, t in new])
    prod_round = rnd if accepted else prod_round_before
    print(f"round {rnd}: {len(new)} new labels | prototypes in use {before:.3f} -> new {after:.3f} | "
          f"gate {'PASS' if accepted else 'FAIL'} ({reason})")

    # ---- prototypes as an artifact: what a scoring job would read ---------- #
    base = re.sub(r"-r\d+$", "", batch)

    def prototypes_doc(labs, r, accuracy, in_use):
        L = np.asarray([pos[c] for c, _ in labs])
        lab = np.asarray([names.index(t) for _, t in labs])
        return {"dataset": stan["dataset"], "fold": int(stan["fold"]), "round": r, "in_use": in_use,
                "balanced_accuracy_eval": accuracy, "labels": len(labs), "names": names,
                "features": d["feats"], "sha": d["sha"], "split_sha": d["split_sha"],
                "space": ("case features after the fold's log1p + standardisation "
                          + ("(split.json " + d["split_sha"][:12] + ")" if d["split_sha"]
                             else "(stats.json of the frozen dataset)")),
                "prototypes": [Zraw[L[lab == k]].mean(0).round(6).tolist() for k in range(n_cls)],
                "prototypes_raw_units": [d["raw"][L[lab == k]].mean(0).round(6).tolist() for k in range(n_cls)],
                "cases": labs}
    wz = f"{root}/wzorce/{base}"
    A._put(f"{wz}/runda{rnd}.json", json.dumps(prototypes_doc(candidate, rnd, after, accepted), indent=1,
                                                   ensure_ascii=False).encode("utf-8"), args.project, "application/json")
    prod_doc = prototypes_doc(labeled, prod_round, after if accepted else before, True)
    A._put(f"{wz}/produkcja.json", json.dumps(prod_doc, indent=1, ensure_ascii=False).encode("utf-8"),
           args.project, "application/json")
    wzorce_link = (f"https://console.cloud.google.com/storage/browser/{wz[len('gs://'):]}?project={args.project}"
                   if wz.startswith("gs://") else wz)

    # ---- next batch -------------------------------------------------------- #
    skipped_all = [int(c) for c in stan.get("skipped", [])] + skipped
    other_all = [[int(c), t] for c, t in stan.get("other", [])] + other
    taken = {c for c, _ in candidate} | {int(h[0]) for h in held} | set(skipped_all) | {c for c, _ in other_all}
    cand = np.array([pos[int(c)] for c in stan["pool"] if int(c) not in taken], dtype=int)
    items = []
    if len(cand):
        items = A.explain_queue(Zraw, d["raw"], names, d["feats"], d["ids"], d["n_acc"], d["n_tx"],
                                [pos[c] for c, _ in labeled], [names.index(t) for _, t in labeled],
                                cand, args.queue_size)
    history = list(stan["history"]) + [{
        "round": rnd, "labels": len(candidate), "accuracy": after, "production_accuracy": after if accepted else before,
        "labels_in_use": len(labeled),
        "accepted": accepted, "reason": reason, "new_labels": len(new), "analyst_fixed": sum(a.get("label") not in (None, a["guess"])
                                                                         for a in answers if a["status"] == "oznaczony")}]
    next_batch = f"{base}-r{rnd}"
    new_stan = {**stan, "round": rnd, "labeled": labeled, "held": held, "production_round": prod_round,
                "queue": [int(it["case"]) for it in items], "skipped": skipped_all, "other": other_all,
                "history": history, "previous_batch": batch}
    files_link, next_dir = "-", None
    if items:
        files_link, next_dir = A.write_analyst_batch(args, stan["dataset"], int(stan["fold"]), names, items,
                                                     stan=new_stan, batch=next_batch)
        print("Analyst batch written:", files_link)

    # ---- report, result, digest -------------------------------------------- #
    ctx = {"round": rnd, "fold": int(stan["fold"]), "names": names, "batch": batch, "next_batch": next_batch,
           "before": before, "after": after, "accepted": accepted, "max_drop": args.max_drop,
           "max_typology_drop": args.max_typology_drop, "min_typology_cases": args.min_typology_cases,
           "collapses": collapses,
           "prod_round": prod_round, "prod_round_before": prod_round_before, "n_prod_before": len(in_use),
           "n_new": len(new), "n_fix": history[-1]["analyst_fixed"], "n_labels": len(labeled), "held": held,
           "n_eval": len(ev), "per_typology": per_typology, "answers": answers, "other": other, "ignored": ignored, "history": history,
           "next_items": items, "files_link": files_link, "wzorce_link": wzorce_link, "sha": d["sha"]}
    html = round_html(ctx)
    _write(args.report_html_path, html)
    report_dest = f"{next_dir}/raport.html" if next_dir else f"{root}/oznaczone/{batch}/raport_rundy.html"
    report_link = "-"
    try:
        A._put(report_dest, html.encode("utf-8"), args.project, "text/html; charset=utf-8")
        report_link = (f"https://storage.cloud.google.com/{report_dest[len('gs://'):]}"
                       if report_dest.startswith("gs://") else report_dest)
    except Exception as e:
        print(f"WARNING: report copy not written: {e!r}")
    result = {"batch": batch, "round": rnd, "accepted": accepted, "reason": reason,
              "per_typology": [{"typology": t, "eval_cases": n, "in_use": b, "new": a} for t, n, b, a in per_typology],
              "accuracy_in_use_before": before,
              "accuracy_new": after, "new_labels": len(new), "labels_in_use": len(labeled),
              "held_for_review": [] if accepted else new,
              "analyst_fixed": ctx["n_fix"], "next_batch": next_batch if items else None, "report": report_link}
    A._put(f"{root}/oznaczone/{batch}/wynik.json", json.dumps(result, indent=1, ensure_ascii=False).encode("utf-8"),
           args.project, "application/json")
    line = (f"round {rnd} | fold {stan['fold']} | {len(new)} new labels (analyst fixed {ctx['n_fix']}) | "
            f"in use {before:.3f} -> new {after:.3f} | gate {'PASS' if accepted else 'FAIL'} ({reason}) | "
            f"next batch {next_batch if items else '-'}")
    _write(args.summary_output_path, line)
    if args.json_output_path:
        _write(args.json_output_path, json.dumps({**result, "history": history, "next_queue": items}, indent=2,
                                                 default=float, ensure_ascii=False))
    print("Active learning round written.", line)
    print(round_digest(args, ctx, files_link, report_link))


if __name__ == "__main__":
    main()