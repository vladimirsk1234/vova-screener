#!/usr/bin/env python3
"""Print FAST Graph-style metrics for manual comparison against FAST Graphs Premium."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from data_utils import resample_to_timeframe
from fast_graph_metrics import FastGraphFilterConfig
from fast_graph_scanner import run_fast_graph_scan
from fundamentals_fast import get_fast_graph_panel_data
from ticker_data import resolve_annual_eps_map


def _fetch_weekly(ticker: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    import yfinance as yf

    t = yf.Ticker(ticker)
    daily = t.history(period="max", auto_adjust=True)
    if daily is None or daily.empty:
        return None, None
    weekly = resample_to_timeframe(daily, "Weekly")
    return weekly, daily


def _print_ticker(ticker: str) -> None:
    eps_map, eps_source = resolve_annual_eps_map(ticker, min_years=6)
    years = sorted(eps_map.keys())
    print(f"\n{'=' * 60}")
    print(f"{ticker}  EPS source: {eps_source}  years: {years[0]}-{years[-1]} ({len(years)} pts)")
    if years:
        tail = {y: round(eps_map[y], 2) for y in years[-5:]}
        print(f"  Latest EPS: {tail}")

    weekly, daily = _fetch_weekly(ticker)
    if weekly is None or weekly.empty:
        print("  No price data")
        return

    cfg = FastGraphFilterConfig(
        min_est_eps_growth=0.0,
        require_analyst_forward_growth=False,
        require_cagr_1y=False,
        require_cagr_3y=False,
        require_cagr_10y=False,
        price_below_fair=False,
    )
    metrics = run_fast_graph_scan(weekly, ticker=ticker, df_daily=daily, filter_cfg=cfg)
    if not metrics:
        print("  No metrics")
        return

    fields = (
        "historical_growth_rate",
        "historical_fair_pe",
        "historical_normal_pe",
        "blended_pe",
        "valuation_eps",
        "valuation_eps_basis",
        "market_cap",
        "country",
        "industry",
        "lt_debt_capital",
        "cagr_1y",
        "cagr_3y",
        "cagr_10y",
        "vs_fair_pct",
    )
    for key in fields:
        print(f"  {key}: {metrics.get(key)}")

    panel = get_fast_graph_panel_data(
        ticker,
        close=metrics.get("close"),
        df_daily=daily,
        fair_pe=float(metrics.get("fair_pe") or 15.0),
        scanner_metrics=metrics,
    )
    details = {label: val for label, val in panel.get("details", [])}
    print(f"  panel_market_cap: {details.get('Market Cap')}")
    print(f"  panel_tev: {details.get('TEV')}")
    print(f"  panel_country: {details.get('Country')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="FAST Graphs parity snapshot")
    parser.add_argument(
        "tickers",
        nargs="*",
        default=["BIIB", "CDE", "OC"],
        help="Tickers to compare (default: BIIB CDE OC)",
    )
    args = parser.parse_args()
    for ticker in args.tickers:
        _print_ticker(ticker.upper())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
