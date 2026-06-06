"""
FAST Graphs–style fundamental scanner. No Streamlit.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from fast_graph_data import annual_eps_from_bundle, fetch_fast_graph_bundle
from fast_graph_metrics import (
    FastGraphFilterConfig,
    build_fast_graph_metrics,
    passes_fast_graph_filters,
)


def run_fast_graph_scan(
    df: pd.DataFrame,
    *,
    ticker: str,
    df_daily: pd.DataFrame | None = None,
    bundle: dict[str, Any] | None = None,
    filter_cfg: FastGraphFilterConfig | None = None,
    fetch_bundle: bool = True,
) -> dict[str, Any] | None:
    """
    Run FAST Graphs scan on one symbol.
    Returns metrics dict with Valid=True when filters pass, else Valid=False with reject_reason.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return None

    close_s = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if close_s.empty:
        return None
    close = float(close_s.iloc[-1])

    cfg = filter_cfg or FastGraphFilterConfig()
    if bundle is None and fetch_bundle:
        try:
            bundle = fetch_fast_graph_bundle(ticker)
        except Exception:
            return {"Valid": False, "reject_reason": "FUNDAMENTALS_ERROR"}

    if not bundle:
        return {"Valid": False, "reject_reason": "NO_FUNDAMENTALS"}

    annual_eps = annual_eps_from_bundle(bundle)
    info = bundle.get("info") or {}
    metrics = build_fast_graph_metrics(
        close=close,
        annual_eps=annual_eps or None,
        df_daily=df_daily,
        info=info,
        earnings_estimates=bundle.get("earnings_estimates"),
        earnings_history=bundle.get("earnings_history"),
        lt_debt_capital=bundle.get("lt_debt_capital"),
        cfg=cfg,
    )
    metrics["company_name"] = info.get("company_name") or ticker
    metrics["country"] = info.get("country")
    metrics["exchange"] = info.get("exchange")

    passed, reason = passes_fast_graph_filters(metrics, cfg)
    metrics["Valid"] = passed
    metrics["reject_reason"] = reason if not passed else ""
    metrics["bundle"] = bundle
    return metrics


def fast_graph_table_row(
    metrics: dict[str, Any],
    *,
    tv_url: str,
    tv_sym: str,
    company_name: str,
) -> dict:
    """Map metrics to results table row."""
    def _fmt(val, suffix=""):
        if val is None:
            return "N/A"
        return f"{val}{suffix}"

    return {
        "Symbol": tv_url,
        "tv_symbol": tv_sym,
        "Company Name": company_name,
        "Close": metrics.get("close"),
        "Growth Rate": metrics.get("growth_rate"),
        "Fair P/E": metrics.get("fair_pe"),
        "Normal P/E": metrics.get("normal_pe"),
        "Blended P/E": metrics.get("blended_pe"),
        "Fair $": metrics.get("fair_price"),
        "Normal $": metrics.get("normal_price"),
        "vs Fair %": metrics.get("vs_fair_pct"),
        "Est EPS Growth": metrics.get("est_eps_growth"),
        "Est ROR": metrics.get("est_annual_ror"),
        "LT Debt/Cap": metrics.get("lt_debt_capital"),
        "Pass": 1 if metrics.get("Valid") else 0,
    }
