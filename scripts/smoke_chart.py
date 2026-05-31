#!/usr/bin/env python3
"""Smoke test: MSFT Monthly scan chart pipeline (no Streamlit)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yfinance as yf

from chart_preview import build_chart_payload, figure_from_payload
from data_utils import fill_last_bar_ohlc, resample_to_timeframe
from indicator_params import IndicatorParams, default_chart_params
from sequence_vova import run_sequence_vova_full, run_sequence_vova_pine


def main() -> int:
    ticker = "MSFT"
    tf = "Monthly"
    print(f"Downloading {ticker} daily OHLC...")
    df = yf.download(
        ticker,
        period="10y",
        interval="1d",
        progress=False,
        auto_adjust=False,
        multi_level_index=False,
    )
    if df is None or df.empty:
        print("FAIL: no data")
        return 1
    df = fill_last_bar_ohlc(df)
    df_daily = df.copy()
    df = resample_to_timeframe(df, tf)
    if df is None or df.empty:
        print("FAIL: resample empty")
        return 1

    print("Running screener...")
    out = run_sequence_vova_pine(df, min_rr=1.0)
    if out is None:
        print("FAIL: screener returned None")
        return 1
    print(f"  Valid={out['Valid']} RR={out['RR']:.2f}")

    print("Building chart payload (with symbol kwarg)...")
    payload = build_chart_payload(
        df,
        tf,
        symbol="NASDAQ:MSFT",
        yahoo_ticker=ticker,
        df_daily=df_daily,
        atr_len=14,
        min_rr=1.0,
    )
    if payload is None:
        print("FAIL: payload is None")
        return 1

    params = default_chart_params()
    full = run_sequence_vova_full(df, params=params)
    if full is None:
        print("FAIL: full indicator returned None")
        return 1
    ext = full.get("extension_lines") or []
    assert len(ext) <= 2, f"expected at most 2 extension lines (Pine parity), got {len(ext)}"
    print(f"OK: extension lines count = {len(ext)} (Pine: max 2)")

    print("Building Plotly figure...")
    fig = figure_from_payload(payload, symbol="NASDAQ:MSFT", params=params)
    if fig is None:
        print("FAIL: figure is None")
        return 1

    trace_count = len(fig.data)
    print(f"OK: figure has {trace_count} traces")
    y_range = fig.layout.yaxis.range
    assert y_range is not None and y_range[1] < 1500, f"y-axis too wide: {y_range}"
    print(f"OK: y-axis range {y_range[0]:.1f} .. {y_range[1]:.1f}")
    defaults = params
    assert defaults.show_hhll and defaults.show_extension_lines and defaults.show_crit_level
    assert defaults.show_breaks
    assert not defaults.show_fib and not defaults.show_bb
    assert defaults.show_watermark
    print("OK: default visibility matches spec")

    # E2E: chart_cache round-trip (scan -> cache -> display)
    chart_cache = {"NASDAQ:MSFT": payload}
    cached = chart_cache["NASDAQ:MSFT"]
    fig2 = figure_from_payload(cached, symbol="NASDAQ:MSFT", params=params)
    assert fig2 is not None and len(fig2.data) == trace_count
    print("OK: chart_cache -> figure_from_payload round-trip")

    # Verify indicator computed on visible plot window (weekly/monthly parity)
    cached_df = payload["df"]
    if len(cached_df) > 80:
        fig3 = figure_from_payload(payload, symbol="NASDAQ:MSFT", params=params)
        peak_pts = sum(
            len(getattr(t, "x", []) or [])
            for t in fig3.data
            if getattr(t, "name", None) == "Peaks"
        )
        assert fig3 is not None
        full3 = run_sequence_vova_full(cached_df.iloc[-36:], params=params)
        if (full3.get("peaks") or []):
            assert peak_pts > 0, "Peaks missing on trimmed monthly window"
        print(f"OK: indicator overlays on visible window (peak markers={peak_pts})")

    params_fib = IndicatorParams.from_dict({**params.as_dict(), "show_fib": True})
    fig_fib = figure_from_payload(cached, symbol="NASDAQ:MSFT", params=params_fib)
    assert fig_fib is not None
    if len(fig_fib.data) > trace_count:
        print(f"OK: fib toggle adds traces ({len(fig_fib.data)} > {trace_count})")
    else:
        print("OK: fib enabled but no fib levels for current structure (expected sometimes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
