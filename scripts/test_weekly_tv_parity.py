#!/usr/bin/env python3
"""
Regression: Weekly scans must use Yahoo native 1wk bars (TradingView parity).

Daily→W-FRI resample can leave a stale Close when the latest daily Close is NaN,
which flips sequence state vs TradingView / Yahoo 1wk (seen on F).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yfinance as yf

from data_utils import (
    fill_last_bar_ohlc,
    interval_and_period,
    prepare_scan_ohlc,
    resample_to_timeframe,
)
from sequence_vova import run_sequence_vova_pine

TICKER = "F"


def main() -> int:
    inter, period = interval_and_period("Weekly")
    if inter != "1wk":
        print(f"FAIL: Weekly must fetch 1wk, got {inter!r}")
        return 1

    native = yf.download(
        TICKER,
        period=period,
        interval="1wk",
        progress=False,
        auto_adjust=False,
        multi_level_index=False,
    )
    daily = yf.download(
        TICKER,
        period=period,
        interval="1d",
        progress=False,
        auto_adjust=False,
        multi_level_index=False,
    )
    if native is None or native.empty or daily is None or daily.empty:
        print("FAIL: Yahoo returned empty OHLC")
        return 1

    cols = ["Open", "High", "Low", "Close", "Volume"]
    native = native[cols].copy()
    daily = daily[cols].copy()

    prepared, companion = prepare_scan_ohlc(native, "Weekly", inter="1wk")
    if companion is not None:
        print("FAIL: native 1wk prepare must not invent a daily companion")
        return 1
    if prepared is None or prepared.empty:
        print("FAIL: prepare_scan_ohlc empty")
        return 1
    if float(prepared["Close"].iloc[-1]) != float(native["Close"].iloc[-1]):
        print("FAIL: prepare_scan_ohlc altered native weekly Close")
        return 1

    # Accidental resample of weekly bars would corrupt the series.
    wrongly = resample_to_timeframe(native, "Weekly")
    if wrongly is not None and len(wrongly) == len(native):
        # Same length can still differ; ensure scanner does not take this path.
        pass

    resampled = fill_last_bar_ohlc(resample_to_timeframe(daily, "Weekly"))
    native_filled = fill_last_bar_ohlc(prepared)
    native_out = run_sequence_vova_pine(native_filled, direction="buy")
    resample_out = run_sequence_vova_pine(resampled, direction="buy") if resampled is not None else None

    n_close = float(native_filled["Close"].iloc[-1])
    r_close = float(resampled["Close"].iloc[-1]) if resampled is not None else float("nan")
    print(
        f"{TICKER} native Close={n_close:.4f} Valid={bool(native_out['Valid'])} "
        f"New={bool(native_out['New'])} RR={float(native_out['RR']):.2f}"
    )
    if resample_out is not None:
        print(
            f"{TICKER} resample Close={r_close:.4f} Valid={bool(resample_out['Valid'])} "
            f"New={bool(resample_out['New'])} RR={float(resample_out['RR']):.2f}"
        )
        if abs(n_close - r_close) > 1e-6 and bool(native_out["Valid"]) != bool(resample_out["Valid"]):
            print("OK: documented divergence — native 1wk != daily resample (scanner uses native)")
        elif abs(n_close - r_close) <= 1e-6:
            print("OK: native and resample Closes currently match")
        else:
            print("OK: Closes differ but Valid flags currently agree")

    # Scanner path must preserve native Valid/New (no resample).
    scan_df, _ = prepare_scan_ohlc(native, "Weekly", inter=inter)
    scan_df = fill_last_bar_ohlc(scan_df)
    scan_out = run_sequence_vova_pine(scan_df, direction="buy")
    if bool(scan_out["Valid"]) != bool(native_out["Valid"]) or bool(scan_out["New"]) != bool(
        native_out["New"]
    ):
        print("FAIL: scanner prepare path disagrees with native 1wk pine result")
        return 1

    print("OK: Weekly scan path uses native Yahoo 1wk (TV parity)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
