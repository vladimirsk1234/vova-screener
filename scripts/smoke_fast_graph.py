#!/usr/bin/env python3
"""Smoke test FAST Graphs scanner metrics and charts on AAPL, ADBE, TD, AGI."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yfinance as yf

from data_utils import resample_to_timeframe
from fast_graph_chart import build_fast_graph_figure
from fast_graph_metrics import (
    FastGraphFilterConfig,
    compute_forecast_growth_pct,
    compute_historical_growth_rate_pct,
    est_annual_ror_pct,
    eps_cagr_over_years,
    resolve_chart_growth_rate,
    resolve_fair_pe,
    resolve_target_year_eps,
)
from fast_graph_scanner import run_fast_graph_scan
from ticker_data import filter_eps_outliers


def _test_pure_metrics() -> bool:
    """Unit checks for math helpers (no network)."""
    ok = True
    if resolve_fair_pe(15.0) != 15.0:
        print("  FAIL: resolve_fair_pe high growth")
        ok = False
    if resolve_fair_pe(8.0, sidebar_fair_pe=15.0) != 15.0:
        print("  FAIL: resolve_fair_pe low growth")
        ok = False
    if resolve_fair_pe(185.0, growth_cap_pct=100.0) != 100.0:
        print("  FAIL: resolve_fair_pe growth cap")
        ok = False
    ror = est_annual_ror_pct(100.0, 200.0, years=3)
    if ror is None or ror < 24.0 or ror > 27.0:
        print(f"  FAIL: est_annual_ror_pct expected ~26, got {ror}")
        ok = False
    cagr = eps_cagr_over_years({2020: 1.0, 2021: 1.1, 2022: 1.21, 2023: 1.33}, 3)
    if cagr is None or cagr < 9.0 or cagr > 11.0:
        print(f"  FAIL: eps_cagr_over_years expected ~10, got {cagr}")
        ok = False

    agi_eps = {2020: 0.12, 2021: 0.15, 2022: 0.09, 2023: 0.53, 2024: 0.69}
    filtered = filter_eps_outliers({2022: 0.09, 2023: 0.53, 2024: 0.7, 2025: 2.11}, min_frac_of_median=0.25)
    if 2022 in filtered:
        print(f"  FAIL: filter_eps_outliers(25%) should drop 2022 turnaround year, got {filtered}")
        ok = False
    hist_growth = compute_historical_growth_rate_pct(agi_eps)
    if hist_growth is None or hist_growth >= 100.0:
        print(f"  FAIL: AGI-like historical growth expected <100%, got {hist_growth}")
        ok = False
    fair_agi = resolve_fair_pe(hist_growth, growth_cap_pct=100.0)
    if fair_agi >= 100.0:
        print(f"  FAIL: AGI-like fair P/E expected <100x, got {fair_agi}")
        ok = False

    estimates = {
        "0y": {"avg": 1.96, "growth": 0.77},
        "+1y": {"avg": 3.70, "growth": 0.31},
    }
    fc_growth = compute_forecast_growth_pct(estimates, agi_eps, hist_growth)
    if fc_growth is None or fc_growth <= 0:
        print(f"  FAIL: forecast growth expected positive, got {fc_growth}")
        ok = False

    chart_hist = resolve_chart_growth_rate(hist_growth, fc_growth, mode="historical")
    chart_fair = resolve_fair_pe(chart_hist, growth_cap_pct=100.0)
    if chart_hist is None or chart_fair < 10.0:
        print(f"  FAIL: AGI chart growth/fair P/E expected >=10, got growth={chart_hist} fair={chart_fair}")
        ok = False

    future_eps = resolve_target_year_eps(
        agi_eps,
        estimates,
        horizon_years=3,
        growth_rate=fc_growth,
    )
    if future_eps is None or future_eps <= 0:
        print(f"  FAIL: resolve_target_year_eps expected positive, got {future_eps}")
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


def _print_metrics(ticker: str, metrics: dict) -> None:
    print(f"  Valid: {metrics.get('Valid')}")
    print(f"  Close: {metrics.get('close')}")
    print(f"  Hist Growth: {metrics.get('historical_growth_rate')}%")
    print(f"  Hist Fair P/E: {metrics.get('historical_fair_pe')}")
    print(f"  Hist Normal P/E: {metrics.get('historical_normal_pe')}")
    print(f"  Fcst Growth: {metrics.get('forecast_growth_rate')}%")
    print(f"  Fcst Fair P/E: {metrics.get('forecast_fair_pe')}")
    print(f"  Est ROR: {metrics.get('est_annual_ror')}%")
    print(f"  Future Price: {metrics.get('future_price')}")


def main() -> int:
    print("=== Pure metrics ===")
    failures = 0 if _test_pure_metrics() else 1

    tickers = ["AAPL", "ADBE", "TD", "AGI"]
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

        _print_metrics(t, metrics)

        if t == "AGI":
            hist_fair = metrics.get("historical_fair_pe")
            chart_growth = metrics.get("chart_historical_growth_rate")
            if hist_fair is None or hist_fair > 100.0:
                print(f"  FAIL: AGI historical fair P/E should be <=100x, got {hist_fair}")
                failures += 1
            if chart_growth is None or chart_growth < 10.0:
                print(f"  FAIL: AGI chart historical growth should be >=10%, got {chart_growth}")
                failures += 1
            ror = metrics.get("est_annual_ror")
            if ror is not None and ror > 150.0:
                print(f"  FAIL: AGI est ROR should be <150%, got {ror}")
                failures += 1

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
        if hist is None or fcst is None:
            print("  FAIL: chart build")
            failures += 1
        else:
            y_range = hist.layout.yaxis.range
            if y_range and y_range[1] > float(metrics.get("close", 0)) * 10:
                print(f"  WARN: historical y-axis may still be wide: {y_range}")
            print("  Charts: OK (historical + forecast)")

    if failures:
        print(f"\n{failures} check(s) failed")
        return 1
    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
