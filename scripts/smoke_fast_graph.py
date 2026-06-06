#!/usr/bin/env python3
"""Smoke test FAST Graphs scanner metrics and charts on AAPL, ADBE, TD."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yfinance as yf

from data_utils import resample_to_timeframe
from fast_graph_chart import build_fast_graph_figure, build_fg_radar_figure
from fast_graph_metrics import (
    FastGraphFilterConfig,
    est_annual_ror_pct,
    eps_cagr_over_years,
    resolve_fair_pe,
)
from fast_graph_scanner import run_fast_graph_scan


def _test_pure_metrics() -> bool:
    """Unit checks for math helpers (no network)."""
    ok = True
    if resolve_fair_pe(15.0) != 15.0:
        print("  FAIL: resolve_fair_pe high growth")
        ok = False
    if resolve_fair_pe(8.0, sidebar_fair_pe=15.0) != 15.0:
        print("  FAIL: resolve_fair_pe low growth")
        ok = False
    ror = est_annual_ror_pct(100.0, 200.0, years=3)
    if ror is None or ror < 24.0 or ror > 27.0:
        print(f"  FAIL: est_annual_ror_pct expected ~26, got {ror}")
        ok = False
    cagr = eps_cagr_over_years({2020: 1.0, 2021: 1.1, 2022: 1.21, 2023: 1.33}, 3)
    if cagr is None or cagr < 9.0 or cagr > 11.0:
        print(f"  FAIL: eps_cagr_over_years expected ~10, got {cagr}")
        ok = False
    if ok:
        print("  Pure metrics: OK")
    return ok


def _fetch_weekly(ticker: str):
    df = yf.download(ticker, period="10y", interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None, None
    if hasattr(df.columns, "levels"):
        df = df.droplevel(1, axis=1) if df.columns.nlevels > 1 else df
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    weekly = resample_to_timeframe(df, "Weekly")
    return weekly, df


def main() -> int:
    print("=== Pure metrics ===")
    failures = 0 if _test_pure_metrics() else 1

    tickers = ["AAPL", "ADBE", "TD"]
    cfg = FastGraphFilterConfig(min_est_eps_growth=0.0, min_est_annual_ror=0.0)

    for t in tickers:
        print(f"\n=== {t} ===")
        weekly, daily = _fetch_weekly(t)
        if weekly is None or weekly.empty:
            print("  FAIL: no price data")
            failures += 1
            continue

        metrics = run_fast_graph_scan(weekly, ticker=t, df_daily=daily, filter_cfg=cfg)
        if not metrics:
            print("  FAIL: no metrics")
            failures += 1
            continue

        print(f"  Valid: {metrics.get('Valid')}")
        print(f"  Close: {metrics.get('close')}")
        print(f"  Growth Rate: {metrics.get('growth_rate')}%")
        print(f"  Fair P/E: {metrics.get('fair_pe')}")
        print(f"  Normal P/E: {metrics.get('normal_pe')}")
        print(f"  Est ROR: {metrics.get('est_annual_ror')}%")
        print(f"  FG Score: {metrics.get('fg_score')}")

        hist = build_fast_graph_figure(
            df_weekly=weekly,
            df_daily=daily,
            metrics=metrics,
            mode="historical",
        )
        fcst = build_fast_graph_figure(
            df_weekly=weekly,
            df_daily=daily,
            metrics=metrics,
            mode="forecast",
        )
        radar = build_fg_radar_figure(metrics.get("fg_axes") or {})

        if hist is None or fcst is None:
            print("  FAIL: chart build")
            failures += 1
        else:
            print("  Charts: OK (historical + forecast)")
        if radar is None:
            print("  WARN: no radar chart")
        else:
            print("  Radar: OK")

    if failures:
        print(f"\n{failures} ticker(s) failed")
        return 1
    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
