"""Synthetic SAML-D-shaped transactions for tests (no real data, no network).

Shape mirrors what the freeze needs: flagged transactions grouped into cases by
shared accounts within a few days, a typology per case, some accounts reused by
later cases of the same typology (so account groups span several cases), one
typology too small to form a grouped episode (dropped by feasibility), one mixed
case (dropped by purity) and some normal traffic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TYPOLOGIES = [
    "Behavioural_Change_1", "Behavioural_Change_2", "Bipartite", "Cash_Withdrawal", "Cycle",
    "Deposit-Send", "Fan_In", "Fan_Out", "Gather-Scatter", "Layered_Fan_In", "Layered_Fan_Out",
    "Over-Invoicing", "Scatter-Gather", "Single_large", "Smurfing", "Stacked Bipartite", "Structuring",
]
PAYMENT = ["Cash Deposit", "Cross-border", "Credit card", "Debit card", "ACH", "Cheque", "Cash Withdrawal"]
PLACES = ["UK", "Spain", "UAE", "Morocco", "Turkey"]
CURR = ["UK pounds", "Euro", "Dirham", "Moroccan dirham", "Turkish lira"]


def make_transactions(seed: int = 0, n_cases=(24, 40)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    next_acc = [10_000]
    day0 = pd.Timestamp("2022-10-07")

    def new_acc():
        next_acc[0] += 1
        return next_acc[0]

    def add_case(label, start, reuse=None):
        n_acc = int(rng.integers(2, 8))
        accs = [new_acc() for _ in range(n_acc)]
        if reuse is not None:
            accs[0] = reuse
        n_tx = int(rng.integers(n_acc - 1, n_acc + 6))
        scale = 300 + 900 * (TYPOLOGIES.index(label) if label in TYPOLOGIES else 3)
        for i in range(n_tx):
            if i < n_acc - 1:              # a chain first, so the case is connected
                s, r = accs[i], accs[i + 1]
            else:
                s, r = rng.choice(accs, 2, replace=False)
            d = start + int(rng.integers(0, 4))
            rows.append({
                "Time": "10:35:19", "Date": (day0 + pd.Timedelta(days=d)).strftime("%Y-%m-%d"),
                "Sender_account": int(s), "Receiver_account": int(r),
                "Amount": round(float(rng.lognormal(np.log(scale), 0.6)), 2),
                "Payment_currency": CURR[int(rng.integers(0, 2))],
                "Received_currency": CURR[int(rng.integers(0, 3))],
                "Sender_bank_location": PLACES[int(rng.integers(0, 2))],
                "Receiver_bank_location": PLACES[int(rng.integers(0, 3))],
                "Payment_type": PAYMENT[int(rng.integers(0, len(PAYMENT)))],
                "Is_laundering": 1, "Laundering_type": label,
            })
        return accs

    for label in TYPOLOGIES:
        reusable = []
        for c in range(int(rng.integers(*n_cases))):
            start = 20 * c + int(rng.integers(0, 5))          # cases of a typology >= ~2 weeks apart
            reuse = reusable[int(rng.integers(0, len(reusable)))] if reusable and rng.random() < 0.3 else None
            reusable += add_case(label, start, reuse)[:1]
    # a typology with too few account groups for a grouped 5+5 episode: dropped by feasibility
    acc = add_case("Tiny_typology", 5)[0]
    for c in range(1, 3):
        add_case("Tiny_typology", 5 + 30 * c, reuse=acc)
    # one mixed case: two typologies half/half, purity 0.5 -> dropped by the purity rule
    a, b = new_acc(), new_acc()
    for lab in ("Smurfing", "Fan_In"):
        rows.append({"Time": "11:00:00", "Date": "2023-07-01", "Sender_account": a, "Receiver_account": b,
                     "Amount": 999.0, "Payment_currency": "UK pounds", "Received_currency": "UK pounds",
                     "Sender_bank_location": "UK", "Receiver_bank_location": "UK", "Payment_type": "ACH",
                     "Is_laundering": 1, "Laundering_type": lab})
    # normal traffic
    for _ in range(400):
        rows.append({"Time": "12:00:00",
                     "Date": (day0 + pd.Timedelta(days=int(rng.integers(0, 300)))).strftime("%Y-%m-%d"),
                     "Sender_account": int(rng.integers(1, 5000)), "Receiver_account": int(rng.integers(1, 5000)),
                     "Amount": round(float(rng.lognormal(6, 1)), 2), "Payment_currency": "UK pounds",
                     "Received_currency": "UK pounds", "Sender_bank_location": "UK",
                     "Receiver_bank_location": "UK", "Payment_type": "Debit card",
                     "Is_laundering": 0, "Laundering_type": "Normal_Small_Fan_Out"})
    return pd.DataFrame(rows)


def write_csv(path: Path, seed: int = 0) -> Path:
    df = make_transactions(seed)
    df.to_csv(path, index=False)
    return path
