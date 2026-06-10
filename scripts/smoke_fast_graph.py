#!/usr/bin/env python3
"""Smoke test FAST Graphs scanner metrics and charts on AAPL, ADBE, TD, AGI, FITB."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yfinance as yf
import pandas as pd

from data_utils import resample_to_timeframe
from fast_graph_chart import _annual_eps_table_rows, build_fast_graph_figure
from fast_graph_metrics import (
    FastGraphFilterConfig,
    _estimate_eps_chain,
    chart_annual_eps,
    compute_forecast_growth_pct,
    compute_forward_eps_growth_pct,
    compute_historical_growth_rate_pct,
    est_annual_ror_pct,
    eps_cagr_over_years,
    passes_fast_graph_filters,
    resolve_chart_growth_rate,
    resolve_fair_pe,
    resolve_target_year_eps,
    resolve_valuation_eps,
)
from eps_yield import avg_historical_pe_5y, fair_and_normal_price, MAX_YEARLY_PE
from fast_graph_scanner import run_fast_graph_scan, fast_graph_table_row
from ticker_data import filter_eps_outliers, resolve_annual_eps_map, strip_incomplete_eps_years
from sec_eps import _parse_operating_eps_from_facts


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
    if hist_growth is None:
        print("  FAIL: AGI-like historical growth expected a value")
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
    if chart_hist != hist_growth:
        print(f"  FAIL: historical chart growth must not use forecast, got {chart_hist} vs {hist_growth}")
        ok = False

    # FITB-like: low historical CAGR must not be replaced by analyst +1y spike.
    fitb_hist = 1.74
    fitb_fc = 59.88
    fitb_chart = resolve_chart_growth_rate(fitb_hist, fitb_fc, mode="historical")
    fitb_fair = resolve_fair_pe(fitb_chart, sidebar_fair_pe=15.0)
    if fitb_chart != fitb_hist:
        print(f"  FAIL: FITB-like chart growth should be historical {fitb_hist}, got {fitb_chart}")
        ok = False
    if fitb_fair != 15.0:
        print(f"  FAIL: FITB-like fair P/E should be sidebar 15x, got {fitb_fair}")
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


def _test_partial_year_eps() -> bool:
    """Partial in-progress FY must use analyst 0y on chart, not summed quarterly EPS."""
    ok = True
    annual_eps = {2024: 14.0, 2025: 16.0, 2026: 4.6}
    earnings_history = [
        {"date": "2026-02-28", "eps_actual": 6.06, "eps_estimate": 5.87},
    ]
    estimates = {"0y": {"avg": 23.5}, "+1y": {"avg": 26.0}}

    completed, last_completed = strip_incomplete_eps_years(
        annual_eps,
        earnings_history=earnings_history,
        current_year=2026,
    )
    if 2026 in completed:
        print(f"  FAIL: partial 2026 should be stripped, got {completed}")
        ok = False
    if last_completed != 2025:
        print(f"  FAIL: last_completed_year should be 2025, got {last_completed}")
        ok = False

    chain = _estimate_eps_chain(
        annual_eps,
        estimates,
        last_completed_year=last_completed,
        years_ahead=4,
        growth_rate=10.0,
    )
    reported_years = [y for y, _, is_est in chain if not is_est]
    if reported_years != [2024, 2025]:
        print(f"  FAIL: reported years expected [2024, 2025], got {reported_years}")
        ok = False

    est_2026 = next((e for y, e, _ in chain if y == 2026), None)
    if est_2026 != 23.5:
        print(f"  FAIL: 2026 estimate expected 23.5, got {est_2026}")
        ok = False

    table = _annual_eps_table_rows(chain, estimates, include_estimates=True)
    row_2026 = table[table["fy"].str.contains("26", na=False)]
    if row_2026.empty:
        print("  FAIL: no 2026 row in EPS table")
        ok = False
    else:
        chg = row_2026.iloc[-1]["chg_yr"]
        if chg is None or chg <= 0:
            print(f"  FAIL: 2026 Chg/Yr should be positive, got {chg}")
            ok = False

    cache_path = ROOT / ".cache" / "fg_bundle" / "ADBE.json"
    if cache_path.is_file():
        with open(cache_path, encoding="utf-8") as f:
            bundle = json.load(f)
        adbe_eps = {int(k): float(v) for k, v in bundle["annual_eps"].items()}
        adbe_hist = bundle.get("earnings_history") or []
        adbe_est = bundle.get("earnings_estimates") or {}
        completed_adbe, last_adbe = chart_annual_eps(adbe_eps, adbe_hist)
        if 2026 in completed_adbe and completed_adbe.get(2026, 0) < 10:
            print(f"  FAIL: ADBE completed_eps should strip partial 2026, got {completed_adbe}")
            ok = False
        chain_adbe = _estimate_eps_chain(
            adbe_eps,
            adbe_est,
            last_completed_year=last_adbe,
            years_ahead=4,
            growth_rate=13.0,
        )
        est_0y = adbe_est.get("0y", {}).get("avg")
        fy2026 = next((e for y, e, _ in chain_adbe if y == 2026), None)
        if est_0y and fy2026 != est_0y:
            print(f"  FAIL: ADBE 2026 chart EPS should be analyst 0y {est_0y}, got {fy2026}")
            ok = False
        fy2025 = next((e for y, e, is_est in chain_adbe if y == 2025 and not is_est), None)
        if fy2025 and fy2026 and fy2026 <= fy2025:
            print(f"  FAIL: ADBE 2026 EPS {fy2026} should exceed 2025 {fy2025}")
            ok = False
    else:
        print("  WARN: ADBE cache bundle missing — skipping cache regression")

    if ok:
        print("  Partial-year EPS: OK")
    return ok


def _test_cpfs_undervaluation() -> bool:
    """Cheap gate uses vs Fair % on valuation_eps (not blended P/E)."""
    ok = True
    cfg = FastGraphFilterConfig(
        price_below_fair=True,
        require_cagr_1y=False,
        require_cagr_3y=False,
        require_cagr_10y=False,
        require_analyst_forward_growth=False,
        min_est_eps_growth=0.0,
        max_lt_debt_capital=0.0,
    )

    cheap = {
        "valuation_eps": 10.0,
        "fair_pe": 15.0,
        "fair_price": 150.0,
        "vs_fair_pct": -20.0,
        "country": "United States",
    }
    passed, reason = passes_fast_graph_filters(cheap, cfg)
    if not passed or reason:
        print(f"  FAIL: below-fair name should pass cheap gate, got passed={passed} reason={reason!r}")
        ok = False

    expensive = {
        "valuation_eps": 10.0,
        "fair_pe": 15.0,
        "fair_price": 150.0,
        "vs_fair_pct": 33.33,
        "country": "United States",
    }
    passed, reason = passes_fast_graph_filters(expensive, cfg)
    if passed or reason != "NOT_BELOW_FAIR":
        print(f"  FAIL: above-fair name should fail cheap gate, got passed={passed} reason={reason!r}")
        ok = False

    slow_grower_cheap = {
        "valuation_eps": 10.0,
        "fair_pe": 15.0,
        "fair_price": 150.0,
        "vs_fair_pct": -33.33,
        "country": "United States",
    }
    passed, reason = passes_fast_graph_filters(slow_grower_cheap, cfg)
    if not passed or reason:
        print(f"  FAIL: slow grower below fair should pass, got passed={passed} reason={reason!r}")
        ok = False

    if ok:
        print("  CPFS undervaluation: OK")
    return ok


def _test_fg_preset_filters() -> bool:
    """FG Undervalued Quality preset gates."""
    ok = True
    cfg = FastGraphFilterConfig.fg_undervalued_quality()

    cheap_enough = {
        "valuation_eps": 10.0,
        "fair_price": 100.0,
        "vs_fair_pct": 5.0,
        "fair_pe": 15.0,
        "cagr_5y": 3.0,
        "historical_growth_rate": 9.0,
        "growth_rate": 9.0,
        "est_eps_growth": 10.0,
        "forward_eps_growth": 10.0,
        "est_annual_ror": 12.0,
        "lt_debt_capital": 40.0,
        "blended_pe": 12.0,
        "historical_normal_pe": 15.0,
        "normal_pe": 15.0,
        "eps_persistence_pct": 80.0,
        "country": "United States",
    }
    passed, reason = passes_fast_graph_filters(cheap_enough, cfg)
    if not passed:
        print(f"  FAIL: FG preset should pass slightly above fair (+5%), got reason={reason!r}")
        ok = False

    too_rich = {**cheap_enough, "vs_fair_pct": 8.0}
    passed, reason = passes_fast_graph_filters(too_rich, cfg)
    if passed or reason != "VS_FAIR":
        print(f"  FAIL: vs Fair +8% should fail VS_FAIR, got passed={passed} reason={reason!r}")
        ok = False

    high_pe = {**cheap_enough, "vs_fair_pct": -10.0, "blended_pe": 20.0, "historical_normal_pe": 15.0}
    passed, reason = passes_fast_graph_filters(high_pe, cfg)
    if passed or reason != "PE_GT_NORMAL":
        print(f"  FAIL: blended P/E > normal should fail PE_GT_NORMAL, got {reason!r}")
        ok = False

    if ok:
        print("  FG preset filters: OK")
    return ok


def _test_fg_preset_more_permissive_than_cpfs() -> bool:
    """FG Undervalued Quality should pass names CPFS-G Strict rejects."""
    ok = True
    fg = FastGraphFilterConfig.fg_undervalued_quality()
    cpfs = FastGraphFilterConfig.cpfs_strict(countries=("United States",))

    fg_friendly = {
        "valuation_eps": 10.0,
        "fair_price": 100.0,
        "vs_fair_pct": 4.0,
        "fair_pe": 15.0,
        "cagr_1y": -5.0,
        "cagr_3y": -3.0,
        "cagr_5y": 2.0,
        "cagr_10y": 8.0,
        "historical_growth_rate": 6.0,
        "growth_rate": 6.0,
        "est_eps_growth": 9.0,
        "forward_eps_growth": 9.0,
        "est_annual_ror": 8.0,
        "lt_debt_capital": 45.0,
        "blended_pe": 14.0,
        "historical_normal_pe": 16.0,
        "normal_pe": 16.0,
        "eps_persistence_pct": 75.0,
        "country": "United States",
    }
    fg_ok, fg_reason = passes_fast_graph_filters(fg_friendly, fg)
    cpfs_ok, cpfs_reason = passes_fast_graph_filters(fg_friendly, cpfs)
    if not fg_ok:
        print(f"  FAIL: FG preset should pass permissive profile, reason={fg_reason!r}")
        ok = False
    if cpfs_ok:
        print("  FAIL: CPFS strict should reject +4% vs fair with negative 1Y/3Y CAGR")
        ok = False
    elif cpfs_reason not in ("NOT_BELOW_FAIR", "CAGR_1Y", "CAGR_3Y", "EST_GROWTH"):
        print(f"  WARN: CPFS reject reason={cpfs_reason!r} (expected strict gate)")

    if ok:
        print("  FG vs CPFS permissiveness: OK")
    return ok


def _test_finviz_undervalued_filters() -> bool:
    """Finviz S: UNDERVALUED filter gates (synthetic + known passers)."""
    from finviz_metrics import FinvizUndervaluedConfig, passes_finviz_filters

    ok = True
    cfg = FinvizUndervaluedConfig.s_undervalued()

    good = {
        "trailing_pe": 15.0,
        "eps_growth_yoy_eff": 5.0,
        "eps_growth_5y_eff": 8.0,
        "eps_growth_next_y_eff": 10.0,
        "eps_growth_next_5y": 9.0,
        "lt_debt_equity": 0.3,
        "eps_growth_ttm_eff": 4.0,
        "sales_growth_ttm_eff": 6.0,
        "eps_growth_3y_eff": 7.0,
        "sales_cagr_3y": 5.0,
        "gross_margin_pct": 35.0,
    }
    passed, reason = passes_finviz_filters(good, cfg)
    if not passed:
        print(f"  FAIL: Finviz synthetic profile should pass, reason={reason!r}")
        ok = False

    high_pe = {**good, "trailing_pe": 25.0}
    passed, reason = passes_finviz_filters(high_pe, cfg)
    if passed or reason != "PE":
        print(f"  FAIL: P/E 25 should fail PE, got passed={passed} reason={reason!r}")
        ok = False

    if ok:
        print("  Finviz S: UNDERVALUED filters: OK")
    return ok


def _test_tv_to_yahoo_suffix() -> bool:
    """Canadian TV symbols map to Yahoo .TO / .V suffixes."""
    from ticker_data import _parse_list_entry, tv_part_to_yahoo

    ok = True
    cases = [
        ("TSX:SHOP", "SHOP.TO"),
        ("TSXV:ABC", "ABC.V"),
        ("NASDAQ:AAPL", "AAPL"),
        ("NYSE:BRK.B", "BRK-B"),
    ]
    for tv_part, expected in cases:
        yahoo = tv_part_to_yahoo(tv_part)
        parsed = _parse_list_entry(f"{tv_part}|Test")
        parsed_yahoo = parsed[0] if parsed else None
        if yahoo != expected or parsed_yahoo != expected:
            print(
                f"  FAIL: {tv_part} expected Yahoo {expected!r}, "
                f"got tv_part_to_yahoo={yahoo!r} parse={parsed_yahoo!r}"
            )
            ok = False
    if ok:
        print("  TV -> Yahoo suffix mapping: OK")
    return ok


def _test_cpfs_g_growth() -> bool:
    """CPFS-G: reject falling EPS (BLDR-like); pass growing + cheap names."""
    ok = True
    cfg = FastGraphFilterConfig.cpfs_strict()

    if compute_forward_eps_growth_pct(None, {2020: 1.0, 2021: 2.0}) is not None:
        print("  FAIL: forward growth must not use historical fallback")
        ok = False

    estimates = {"0y": {"avg": 1.0}, "+1y": {"avg": 1.12, "growth": 0.12}}
    fwd = compute_forward_eps_growth_pct(estimates)
    if fwd is None or fwd < 11.0 or fwd > 13.0:
        print(f"  FAIL: forward growth from analyst expected ~12, got {fwd}")
        ok = False

    fc_with_fallback = compute_forecast_growth_pct(estimates, {2020: 1.0}, 14.0)
    if fc_with_fallback is None or fc_with_fallback < 11.0:
        print(f"  FAIL: forecast growth with fallback expected ~12, got {fc_with_fallback}")
        ok = False

    bldr_like = {
        "valuation_eps": 10.0,
        "fair_price": 150.0,
        "vs_fair_pct": -10.0,
        "fair_pe": 15.0,
        "cagr_1y": -15.0,
        "cagr_3y": -10.0,
        "cagr_10y": 14.0,
        "forward_eps_growth": 12.0,
        "lt_debt_capital": 30.0,
        "country": "United States",
    }
    passed, reason = passes_fast_graph_filters(bldr_like, cfg)
    if passed or reason != "CAGR_1Y":
        print(f"  FAIL: BLDR-like should fail CAGR_1Y, got passed={passed} reason={reason!r}")
        ok = False

    no_forward = {
        "valuation_eps": 10.0,
        "fair_price": 150.0,
        "vs_fair_pct": -10.0,
        "fair_pe": 15.0,
        "cagr_1y": 5.0,
        "cagr_3y": 8.0,
        "cagr_10y": 12.0,
        "forward_eps_growth": None,
        "est_eps_growth": 14.0,
        "lt_debt_capital": 30.0,
        "country": "United States",
    }
    passed, reason = passes_fast_graph_filters(no_forward, cfg)
    if passed or reason != "NO_FORWARD_GROWTH":
        print(f"  FAIL: missing forward growth should fail, got passed={passed} reason={reason!r}")
        ok = False

    growing_cheap = {
        "valuation_eps": 10.0,
        "fair_price": 150.0,
        "vs_fair_pct": -10.0,
        "fair_pe": 15.0,
        "cagr_1y": 5.0,
        "cagr_3y": 8.0,
        "cagr_10y": 12.0,
        "forward_eps_growth": 12.0,
        "lt_debt_capital": 30.0,
        "country": "United States",
    }
    passed, reason = passes_fast_graph_filters(growing_cheap, cfg)
    if not passed or reason:
        print(f"  FAIL: growing cheap name should pass CPFS-G, got passed={passed} reason={reason!r}")
        ok = False

    if ok:
        print("  CPFS-G growth quality: OK")
    return ok


def _test_discrepancy_fixes() -> bool:
    """Regression: FSLR gate/table, OC negative EPS, FISV normal P/E, filtered CAGR alignment."""
    ok = True
    cfg_cheap = FastGraphFilterConfig(
        price_below_fair=True,
        require_cagr_1y=False,
        require_cagr_3y=False,
        require_cagr_10y=False,
        require_analyst_forward_growth=False,
        min_est_eps_growth=0.0,
        max_lt_debt_capital=0.0,
    )

    # FSLR-like: blended P/E would pass but vs Fair % is above fair → must fail.
    fslr_like = {
        "valuation_eps": 15.49,
        "fair_pe": 15.0,
        "fair_price": 232.35,
        "vs_fair_pct": 20.08,
        "blended_pe": 14.96,
        "country": "United States",
    }
    passed, reason = passes_fast_graph_filters(fslr_like, cfg_cheap)
    if passed or reason != "NOT_BELOW_FAIR":
        print(f"  FAIL: FSLR-like above fair should fail, got passed={passed} reason={reason!r}")
        ok = False

    # OC-like: negative valuation EPS must reject before cheap gate.
    oc_like = {
        "valuation_eps": -5.0,
        "vs_fair_pct": -211.0,
        "blended_pe": 10.0,
        "fair_pe": 22.62,
        "country": "United States",
    }
    passed, reason = passes_fast_graph_filters(oc_like, cfg_cheap)
    if passed or reason != "NEGATIVE_EPS":
        print(f"  FAIL: OC-like negative EPS should fail NEGATIVE_EPS, got passed={passed} reason={reason!r}")
        ok = False

    fair, normal = fair_and_normal_price(-5.0, 15.0, 12.0)
    if fair is not None or normal is not None:
        print(f"  FAIL: negative EPS should yield None fair/normal, got {fair}, {normal}")
        ok = False

    val_eps, basis = resolve_valuation_eps({2023: -1.0, 2024: 3.5}, trailing_eps=-5.0)
    if val_eps != 3.5 or basis != "annual":
        print(f"  FAIL: resolve_valuation_eps expected latest positive annual 3.5, got {val_eps} ({basis})")
        ok = False

    # CAGR 1Y/3Y/10Y use same filtered series as Growth Rate.
    raw_eps = {2018: 1.0, 2019: 1.1, 2020: 0.05, 2021: 1.2, 2022: 1.3, 2023: 1.4, 2024: 1.5}
    filtered = filter_eps_outliers(raw_eps, min_frac_of_median=0.25)
    hist = compute_historical_growth_rate_pct(raw_eps)
    cagr_1y = eps_cagr_over_years(filtered, 1)
    cagr_3y = eps_cagr_over_years(filtered, 3)
    cagr_10y = eps_cagr_over_years(filtered, 10)
    if cagr_1y is None or cagr_3y is None:
        print(f"  FAIL: filtered CAGR expected values, got 1y={cagr_1y} 3y={cagr_3y}")
        ok = False
    if hist is not None and cagr_10y is not None:
        if hist == cagr_10y:
            print(f"  NOTE: growth rate equals cagr_10y ({hist}) — geometric mean matched CAGR on this series")

    # FISV-like: cap extreme yearly P/E from split mismatch.
    dates = pd.date_range("2020-01-01", periods=5, freq="YE")
    prices = pd.DataFrame({"Close": [200.0, 180.0, 150.0, 60.0, 54.0]}, index=dates)
    annual_eps = {2020: 4.0, 2021: 4.2, 2022: 4.5, 2023: 4.8, 2024: 5.0}
    norm_pe = avg_historical_pe_5y(prices, annual_eps)
    if norm_pe is None or norm_pe > MAX_YEARLY_PE:
        print(f"  FAIL: FISV-like normal P/E should be capped <= {MAX_YEARLY_PE}, got {norm_pe}")
        ok = False

    stale_row = fast_graph_table_row(
        {"fair_price": -107.22, "normal_price": -57.16, "vs_fair_pct": -211.0, "Valid": False},
        tv_url="OC",
        tv_sym="OC",
        company_name="OC",
    )
    if stale_row.get("Fair $") != "N/A" or stale_row.get("Normal $") != "N/A":
        print(f"  FAIL: table must not show negative Fair/Normal, got {stale_row.get('Fair $')}")
        ok = False

    if ok:
        print("  Discrepancy fixes: OK")
    return ok


def _test_sec_operating_eps() -> bool:
    """SEC operating EPS parser + resolve_annual_eps_map priority."""
    ok = True
    mock_facts = {
        "facts": {
            "us-gaap": {
                "OperatingIncomeLoss": {
                    "units": {
                        "USD": [
                            {"fy": 2022, "fp": "FY", "form": "10-K", "filed": "2023-02-01", "val": 100_000_000.0},
                            {"fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-01", "val": 120_000_000.0},
                        ]
                    }
                },
                "WeightedAverageNumberOfDilutedSharesOutstanding": {
                    "units": {
                        "shares": [
                            {"fy": 2022, "fp": "FY", "form": "10-K", "filed": "2023-02-01", "val": 10_000_000.0},
                            {"fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-01", "val": 10_000_000.0},
                        ]
                    }
                },
            }
        }
    }
    parsed = _parse_operating_eps_from_facts(mock_facts)
    if parsed.get(2022) != 10.0 or parsed.get(2023) != 12.0:
        print(f"  FAIL: operating EPS parser expected 10/12, got {parsed}")
        ok = False

    try:
        eps_map, source = resolve_annual_eps_map("AAPL", min_years=6)
        if source not in (
            "sec_operating",
            "sec_operating+yahoo",
            "sec",
            "sec+yahoo",
            "yahoo_annual",
            "yahoo_quarterly",
        ):
            print(f"  FAIL: unknown EPS source {source!r}")
            ok = False
        elif source == "sec_operating" and len(eps_map) < 6:
            print(f"  FAIL: sec_operating expected >=6 years, got {len(eps_map)}")
            ok = False
        elif source != "sec_operating":
            print(f"  WARN: AAPL EPS source is {source!r} (SEC operating unavailable or <6y)")
    except Exception as exc:
        print(f"  WARN: live AAPL EPS resolve skipped: {exc}")

    if ok:
        print("  SEC operating EPS: OK")
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
    print(f"  EPS source: {metrics.get('eps_source')}")
    print(f"  Close: {metrics.get('close')}")
    print(f"  Hist Growth: {metrics.get('historical_growth_rate')}%")
    print(f"  Hist Fair P/E: {metrics.get('historical_fair_pe')}")
    print(f"  Hist Normal P/E: {metrics.get('historical_normal_pe')}")
    print(f"  Fcst Growth: {metrics.get('forecast_growth_rate')}%")
    print(f"  Fcst Fair P/E: {metrics.get('forecast_fair_pe')}")
    print(f"  Est ROR: {metrics.get('est_annual_ror')}%")
    print(f"  Future Price: {metrics.get('future_price')}")


def _test_fitb_live(metrics: dict) -> bool:
    """Regression: FITB historical box must not show analyst +1y growth."""
    hist = metrics.get("historical_growth_rate")
    fcst = metrics.get("forecast_growth_rate")
    chart_hist = metrics.get("chart_historical_growth_rate")
    hist_fair = metrics.get("historical_fair_pe")
    ok = True
    if hist is not None and fcst is not None and hist < 15.0 and fcst > 30.0:
        if chart_hist == fcst:
            print(f"  FAIL: chart historical growth leaked forecast {fcst}%")
            ok = False
        if hist_fair is not None and hist_fair == fcst:
            print(f"  FAIL: historical fair P/E leaked forecast growth {fcst}x")
            ok = False
        if hist_fair is not None and hist_fair > 25.0:
            print(f"  FAIL: FITB historical fair P/E should be ~15x, got {hist_fair}")
            ok = False
    if ok:
        print("  FITB regression: OK")
    return ok


def _test_parity_tickers() -> bool:
    """Regression cases from BIIB / CDE / OC FAST Graphs parity screenshots."""
    ok = True
    cfg = FastGraphFilterConfig(
        min_est_eps_growth=0.0,
        min_est_annual_ror=0.0,
        require_analyst_forward_growth=False,
        require_cagr_1y=False,
        require_cagr_3y=False,
        require_cagr_10y=False,
        price_below_fair=False,
    )

    biib_eps, biib_src = resolve_annual_eps_map("BIIB", min_years=6)
    latest_biib = max(biib_eps.keys()) if biib_eps else 0
    if latest_biib < 2022:
        print(f"  FAIL: BIIB EPS should include 2022+, latest keys={sorted(biib_eps.keys())[-3:]}")
        ok = False
    weekly, daily = _fetch_weekly("BIIB")
    if weekly is not None and not weekly.empty:
        biib = run_fast_graph_scan(weekly, ticker="BIIB", df_daily=daily, filter_cfg=cfg)
        if biib:
            hist = biib.get("historical_growth_rate")
            if hist is None or hist >= 5.0:
                print(f"  FAIL: BIIB historical growth expected <5%, got {hist} (src={biib_src})")
                ok = False
            else:
                print(f"  BIIB growth={hist}% src={biib.get('eps_source')}: OK")
        else:
            print("  WARN: BIIB scan returned no metrics")
    else:
        print("  WARN: BIIB price data unavailable")

    cde_eps, cde_src = resolve_annual_eps_map("CDE", min_years=6)
    window_end = max(cde_eps.keys()) if cde_eps else 0
    window_start = window_end - 9
    points_in_window = sum(1 for y in range(window_start, window_end + 1) if y in cde_eps)
    if points_in_window < 5:
        print(
            f"  FAIL: CDE expected >=5 EPS points in 10Y window, got {points_in_window} "
            f"keys={sorted(cde_eps.keys())}"
        )
        ok = False
    weekly, daily = _fetch_weekly("CDE")
    if weekly is not None and not weekly.empty:
        cde = run_fast_graph_scan(weekly, ticker="CDE", df_daily=daily, filter_cfg=cfg)
        if cde:
            hist = cde.get("historical_growth_rate")
            fair = cde.get("historical_fair_pe")
            if hist is not None and (hist > 150.0 or fair is None or fair >= 100.0):
                print(f"  FAIL: CDE sanity check failed growth={hist} fair={fair}")
                ok = False
            else:
                print(f"  CDE growth={hist}% fair={fair}x src={cde.get('eps_source')}: OK")
        else:
            print("  WARN: CDE scan returned no metrics")
    else:
        print("  WARN: CDE price data unavailable")

    weekly, daily = _fetch_weekly("OC")
    if weekly is not None and not weekly.empty:
        oc = run_fast_graph_scan(weekly, ticker="OC", df_daily=daily, filter_cfg=cfg)
        if oc:
            norm = oc.get("historical_normal_pe")
            mcap = oc.get("market_cap")
            if mcap is None or mcap <= 0:
                print(f"  FAIL: OC market_cap should be populated, got {mcap}")
                ok = False
            if norm is None or norm <= 0:
                print(f"  FAIL: OC normal P/E should be positive, got {norm}")
                ok = False
            else:
                print(f"  OC normal_pe={norm}x market_cap={mcap}: OK")
        else:
            print("  WARN: OC scan returned no metrics")
    else:
        print("  WARN: OC price data unavailable")

    if ok:
        print("  Parity tickers (BIIB/CDE/OC): OK")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", action="append", dest="extra_tickers", default=[])
    args = parser.parse_args()

    print("=== Pure metrics ===")
    failures = 0 if _test_pure_metrics() else 1

    print("\n=== CPFS undervaluation ===")
    if not _test_cpfs_undervaluation():
        failures += 1

    print("\n=== CPFS-G growth quality ===")
    if not _test_cpfs_g_growth():
        failures += 1

    print("\n=== FG preset filters ===")
    if not _test_fg_preset_filters():
        failures += 1

    print("\n=== FG vs CPFS permissiveness ===")
    if not _test_fg_preset_more_permissive_than_cpfs():
        failures += 1

    print("\n=== TV Yahoo suffix mapping ===")
    if not _test_tv_to_yahoo_suffix():
        failures += 1

    print("\n=== Finviz S: UNDERVALUED filters ===")
    if not _test_finviz_undervalued_filters():
        failures += 1

    print("\n=== Discrepancy fixes ===")
    if not _test_discrepancy_fixes():
        failures += 1

    print("\n=== SEC operating EPS ===")
    if not _test_sec_operating_eps():
        failures += 1

    print("\n=== Partial-year EPS ===")
    if not _test_partial_year_eps():
        failures += 1

    print("\n=== Parity tickers (BIIB/CDE/OC) ===")
    if not _test_parity_tickers():
        failures += 1

    tickers = ["AAPL", "ADBE", "TD", "AGI"]
    for extra in args.extra_tickers:
        if extra.upper() not in {t.upper() for t in tickers}:
            tickers.append(extra.upper())
    if "FITB" not in tickers:
        tickers.append("FITB")

    cfg = FastGraphFilterConfig(
        min_est_eps_growth=0.0,
        min_est_annual_ror=0.0,
        require_analyst_forward_growth=False,
        require_cagr_1y=False,
        require_cagr_3y=False,
        price_below_fair=False,
    )

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

        if t == "AAPL":
            src = metrics.get("eps_source")
            if src == "sec_operating":
                print("  AAPL sec_operating: OK")
            elif src in ("sec", "yahoo_annual"):
                print(f"  WARN: AAPL EPS source {src!r} (expected sec_operating when SEC data available)")

        if t == "AGI":
            hist_fair = metrics.get("historical_fair_pe")
            chart_growth = metrics.get("chart_historical_growth_rate")
            hist_growth = metrics.get("historical_growth_rate")
            if hist_fair is None or hist_fair > 100.0:
                print(f"  FAIL: AGI historical fair P/E should be <=100x, got {hist_fair}")
                failures += 1
            if chart_growth != hist_growth:
                print(f"  FAIL: AGI chart growth should equal historical {hist_growth}, got {chart_growth}")
                failures += 1
            ror = metrics.get("est_annual_ror")
            if ror is not None and ror > 150.0:
                print(f"  FAIL: AGI est ROR should be <150%, got {ror}")
                failures += 1

        if t == "FITB":
            if not _test_fitb_live(metrics):
                failures += 1

        if t == "ADBE":
            annual = metrics.get("annual_eps") or {}
            if 2026 in annual and float(annual[2026]) < 10:
                print(f"  FAIL: ADBE metrics should not include partial 2026 EPS, got {annual.get(2026)}")
                failures += 1
            last_completed = metrics.get("last_completed_year")
            est_0y = (metrics.get("earnings_estimates") or {}).get("0y", {}).get("avg")
            if last_completed is None:
                print("  FAIL: ADBE last_completed_year missing")
                failures += 1
            elif est_0y is not None:
                chain = _estimate_eps_chain(
                    metrics.get("annual_eps"),
                    metrics.get("earnings_estimates"),
                    last_completed_year=last_completed,
                )
                eps_current = next((e for y, e, _ in chain if y == last_completed + 1), None)
                if eps_current != est_0y:
                    print(f"  FAIL: ADBE chart 0y EPS {eps_current} != estimate {est_0y}")
                    failures += 1
                else:
                    print(f"  ADBE partial-year EPS uses 0y estimate ({est_0y}): OK")

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
