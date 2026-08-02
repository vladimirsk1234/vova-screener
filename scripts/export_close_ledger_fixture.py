"""Export the Streamlit close-scan replay, trade by trade, as the TS parity oracle.

`export_parity_fixture.py` records what a close scan answers on its last bar, which is all the
Streamlit table shows. The tracked positions in the app need the rest of the replay too — which
trade a symbol is in right now and where it was entered — so this dumps every trade the scan
takes over several series, including the one still running.

    python scripts/export_close_ledger_fixture.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sequence_vova import run_sequence_vova_close_ledger, run_sequence_vova_close_scan

OUT = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "engine"
    / "fixtures"
    / "close_ledger_parity.json"
)

# Trends, chop and volatility all produce different trade counts, and `no_rr_req` decides which
# bars are allowed to open one at all — the app scans with it on, Streamlit defaults to off.
CASES = [
    {"label": "drift-up", "seed": 7, "n": 400, "drift": 0.08, "vol": 1.2, "no_rr_req": False},
    {"label": "drift-down", "seed": 11, "n": 400, "drift": -0.08, "vol": 1.2, "no_rr_req": False},
    {"label": "choppy", "seed": 23, "n": 400, "drift": 0.0, "vol": 0.6, "no_rr_req": False},
    {"label": "volatile", "seed": 31, "n": 400, "drift": 0.02, "vol": 3.0, "no_rr_req": False},
    {"label": "no-rr-gate", "seed": 7, "n": 400, "drift": 0.08, "vol": 1.2, "no_rr_req": True},
    {"label": "short-series", "seed": 5, "n": 60, "drift": 0.0, "vol": 1.0, "no_rr_req": True},
    # Cut mid-trade, which is the state most of the universe is in on any given bar and the one
    # the app has to answer "where was this position entered" for.
    {
        "label": "position-open",
        "seed": 31,
        "n": 400,
        "drift": 0.02,
        "vol": 3.0,
        "no_rr_req": False,
        "trim": 1,
    },
    {
        "label": "position-open-no-rr",
        "seed": 7,
        "n": 400,
        "drift": 0.08,
        "vol": 1.2,
        "no_rr_req": True,
        "trim": 37,
    },
]

OPTS = {"atr_len": 14, "min_rr": 1.5, "use_last_hl_sl": True, "risk_dollars": 100}


def _synthetic(n: int, seed: int, drift: float, vol: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(drift, vol, n))
    close = np.maximum(close, 1.0)
    high = close + rng.uniform(0.2, 1.5, n)
    low = np.maximum(close - rng.uniform(0.2, 1.5, n), 0.5)
    open_ = close + rng.normal(0, 0.3, n)
    vol_col = rng.integers(1_000_000, 5_000_000, n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol_col},
        index=idx,
    )


def _bars(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "date": ts.strftime("%Y-%m-%d"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
        }
        for ts, row in df.iterrows()
    ]


def _num(v) -> float | None:
    if v is None:
        return None
    f = float(v)
    return None if np.isnan(f) else f


def _trade(trade: dict, dates: list[str]) -> dict:
    exit_index = trade["exit_index"]
    return {
        "entry_index": int(trade["entry_index"]),
        "entry_date": dates[int(trade["entry_index"])],
        "entry_price": _num(trade["entry_price"]),
        "entry_sl": _num(trade["entry_sl"]),
        "entry_tp": _num(trade["entry_tp"]),
        "entry_rr": _num(trade["entry_rr"]),
        "position_size": _num(trade["position_size"]),
        "exit_index": None if exit_index is None else int(exit_index),
        "exit_date": None if exit_index is None else dates[int(exit_index)],
        "exit_price": _num(trade["exit_price"]),
        "close_rr": _num(trade["close_rr"]),
        "pnl_dollars": _num(trade["pnl_dollars"]),
        "pnl_pct": _num(trade["pnl_pct"]),
    }


def main() -> None:
    cases = []
    for case in CASES:
        df = _synthetic(case["n"], case["seed"], case["drift"], case["vol"])
        trim = case.get("trim", 0)
        if trim:
            df = df.iloc[:-trim]
        bars = _bars(df)
        dates = [b["date"] for b in bars]
        opts = {**OPTS, "no_rr_req": case["no_rr_req"]}
        ledger = run_sequence_vova_close_ledger(df, **opts) or []
        scan = run_sequence_vova_close_scan(df, **opts)
        cases.append(
            {
                "label": case["label"],
                "opts": opts,
                "bars": bars,
                "trades": [_trade(t, dates) for t in ledger],
                # The last-bar answer the Streamlit table renders, so the two stay tied together.
                "scan": {
                    "Valid": bool(scan["Valid"]) if scan else False,
                    "entry_price": _num(scan["entry_price"]) if scan else None,
                    "exit_price": _num(scan["exit_price"]) if scan else None,
                    "entry_rr": _num(scan["entry_rr"]) if scan else None,
                    "close_rr": _num(scan["close_rr"]) if scan else None,
                    "pnl_dollars": _num(scan["pnl_dollars"]) if scan else None,
                },
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Compact: this is generated data nobody reads by hand, and the bars dominate the file.
    OUT.write_text(json.dumps({"cases": cases}, separators=(",", ":")), encoding="utf-8")
    total = sum(len(c["trades"]) for c in cases)
    print(f"Wrote {OUT} — {len(cases)} series, {total} trades")


if __name__ == "__main__":
    main()
