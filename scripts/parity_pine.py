#!/usr/bin/env python3
"""Parity check: Python vs Numba run_sequence_vova_pine core + close-scan smoke."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

from data_utils import fill_last_bar_ohlc, resample_to_timeframe
from sequence_vova import (
    _NUMBA_AVAILABLE,
    _PINE_USE_NUMBA,
    _calc_atr_numpy,
    _pine_result_dict,
    _run_sequence_vova_pine_numba,
    _run_sequence_vova_pine_python,
    run_sequence_vova_close_scan,
    run_sequence_vova_pine,
)

TICKERS = ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN"]
TF = "Daily"
ATOL = 1e-6


def _load_df(ticker: str):
    df = yf.download(
        ticker,
        period="2y",
        interval="1d",
        progress=False,
        auto_adjust=False,
        multi_level_index=False,
    )
    if df is None or df.empty:
        return None
    df = fill_last_bar_ohlc(df)
    return resample_to_timeframe(df, TF)


def _compare_buy(ticker: str) -> list[str]:
    df = _load_df(ticker)
    if df is None or len(df) < 50:
        return [f"{ticker}/buy: skip (no data)"]

    c = np.ascontiguousarray(df["Close"].values, dtype=np.float64)
    h = np.ascontiguousarray(df["High"].values, dtype=np.float64)
    l = np.ascontiguousarray(df["Low"].values, dtype=np.float64)
    atr = _calc_atr_numpy(h, l, c, 14)

    py = _pine_result_dict(
        _run_sequence_vova_pine_python(c, h, l, atr, 1.5, True, 100.0, False)
    )
    if not _NUMBA_AVAILABLE or _run_sequence_vova_pine_numba is None:
        return [f"{ticker}/buy: numba not available, skip"]

    nb = _pine_result_dict(
        _run_sequence_vova_pine_numba(c, h, l, atr, 1.5, True, 100.0, False)
    )
    pub = run_sequence_vova_pine(df, min_rr=1.5, direction="buy")

    errs = []
    tag = f"{ticker}/buy"
    for key in ("Valid", "New", "Strong"):
        if py[key] != nb[key]:
            errs.append(f"{tag}: {key} py={py[key]} nb={nb[key]}")
    for key in ("TP", "SL", "RR", "position_size", "position_value", "Close", "ATR"):
        if not np.isclose(py[key], nb[key], rtol=0, atol=ATOL, equal_nan=True):
            errs.append(f"{tag}: {key} py={py[key]} nb={nb[key]}")
    if pub is not None:
        for key in ("Valid", "New", "Strong"):
            if pub[key] != py[key]:
                errs.append(f"{tag}: public {key} != python")
        for key in ("TP", "SL", "RR"):
            if not np.isclose(pub[key], py[key], rtol=0, atol=ATOL, equal_nan=True):
                errs.append(f"{tag}: public {key} != python")
    return errs


def _smoke_close_scan() -> list[str]:
    """Synthetic OHLC: close-scan returns dict with expected keys; P&L math when Valid."""
    errs: list[str] = []
    n = 80
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0.05, 0.8, n))
    high = close + rng.uniform(0.2, 1.0, n)
    low = close - rng.uniform(0.2, 1.0, n)
    open_ = close + rng.uniform(-0.3, 0.3, n)
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1e6})

    out = run_sequence_vova_close_scan(df, risk_dollars=100.0)
    if out is None:
        return ["close_scan: returned None"]
    required = (
        "Valid", "New", "entry_price", "exit_price", "entry_sl",
        "position_size", "pnl_dollars", "pnl_pct", "entry_rr", "close_rr",
        "Close", "ATR",
    )
    for key in required:
        if key not in out:
            errs.append(f"close_scan: missing key {key}")
    if out.get("Valid"):
        entry = out["entry_price"]
        exit_p = out["exit_price"]
        size = out["position_size"]
        if not np.isnan(size) and size > 0:
            expected_pnl = (exit_p - entry) * size
            if not np.isclose(out["pnl_dollars"], expected_pnl, rtol=0, atol=ATOL, equal_nan=True):
                errs.append(
                    f"close_scan: pnl mismatch got={out['pnl_dollars']} expected={expected_pnl}"
                )
    return errs


def main() -> int:
    print(f"numba_available={_NUMBA_AVAILABLE} pine_use_numba={_PINE_USE_NUMBA}")
    all_errs: list[str] = []
    for t in TICKERS:
        all_errs.extend(_compare_buy(t))
    all_errs.extend(_smoke_close_scan())
    if all_errs:
        for e in all_errs:
            print(f"FAIL: {e}")
        return 1
    print(f"OK: parity passed for {len(TICKERS)} tickers (buy) + close-scan smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
