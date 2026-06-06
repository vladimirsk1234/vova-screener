"""
Yahoo Finance data bundle for FAST Graphs scanner. No Streamlit.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yfinance as yf

from fundamentals_fast import _info_field_str, _lt_debt_to_capital_pct, _yield_pct_from_yahoo
from ticker_data import (
    _eps_from_yf_info,
    _float_field,
    _resolve_fundamentals_info,
    get_annual_eps_history_10y,
    get_earnings_estimates_yf,
    get_earnings_history_yf,
)

_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "fg_bundle"
_CACHE_TTL_SEC = 86400.0


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("/", "_").upper()
    return _CACHE_DIR / f"{safe}.json"


def _load_cache(ticker: str) -> dict | None:
    path = _cache_path(ticker)
    if not path.is_file():
        return None
    try:
        mtime = path.stat().st_mtime
        if time.time() - mtime > _CACHE_TTL_SEC:
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(ticker: str, data: dict) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_cache_path(ticker), "w", encoding="utf-8") as f:
            json.dump(data, f, default=str)
    except Exception:
        pass


def _parse_earnings_estimates(ticker_obj: yf.Ticker) -> dict[str, Any]:
    raw = get_earnings_estimates_yf(ticker_obj)
    out: dict[str, Any] = {}
    if not raw:
        return out
    for period in ("0y", "+1y"):
        row = raw.get(period, {})
        if not row:
            continue
        entry: dict[str, Any] = {}
        for key in ("avg", "low", "high", "growth", "numberOfAnalysts", "yearAgoEps"):
            val = row.get(key)
            if val is not None:
                try:
                    entry[key] = float(val) if key != "numberOfAnalysts" else int(val)
                except (TypeError, ValueError):
                    entry[key] = val
        if entry:
            out[period] = entry
    return out


def _parse_earnings_history(ticker_obj: yf.Ticker) -> list[dict]:
    raw = get_earnings_history_yf(ticker_obj)
    out: list[dict] = []
    for item in raw or []:
        est = item.get("epsEstimate")
        act = item.get("epsActual")
        beat = False
        if est is not None and act is not None:
            try:
                beat = float(act) >= float(est)
            except (TypeError, ValueError):
                beat = False
        out.append({
            "date": item.get("date"),
            "eps_estimate": est,
            "eps_actual": act,
            "surprise_pct": item.get("surprisePercent"),
            "beat": beat,
        })
    return out


def _info_bundle(info: dict, merged: dict) -> dict[str, Any]:
    trailing_eps, forward_eps = _eps_from_yf_info(merged)
    if trailing_eps is None:
        trailing_eps, forward_eps = _eps_from_yf_info(info)

    div_yld = _yield_pct_from_yahoo(
        _float_field(info, "dividendYield"),
        close=_float_field(info, "regularMarketPrice") or _float_field(merged, "regularMarketPrice"),
        dividend_rate=_float_field(info, "dividendRate"),
    )

    return {
        "company_name": str(merged.get("company_name") or info.get("longName") or ""),
        "country": _info_field_str(info, "country"),
        "exchange": _info_field_str(info, "exchange", "fullExchangeName"),
        "industry": _info_field_str(info, "industryDisp", "industry"),
        "currency": _info_field_str(info, "currency") or "USD",
        "trailing_eps": trailing_eps,
        "forward_eps": forward_eps,
        "trailing_pe": _float_field(info, "trailingPE") or _float_field(merged, "trailingPE"),
        "forward_pe": _float_field(info, "forwardPE") or _float_field(merged, "forwardPE"),
        "market_cap": _float_field(info, "marketCap") or _float_field(merged, "marketCap"),
        "profit_margin": _float_field(info, "profitMargins"),
        "roe": _float_field(info, "returnOnEquity"),
        "roa": _float_field(info, "returnOnAssets"),
        "debt_to_equity": _float_field(info, "debtToEquity"),
        "operating_cashflow": _float_field(info, "operatingCashflow"),
        "dividend_yield_pct": div_yld,
        "description": str(info.get("longBusinessSummary") or ""),
    }


def fetch_fast_graph_bundle(ticker: str, *, use_cache: bool = True) -> dict[str, Any]:
    """
    Fetch all fundamental data needed for FAST Graphs scan/chart.
    Returns dict with keys: info, annual_eps, earnings_estimates, earnings_history, lt_debt_capital.
    """
    if use_cache:
        hit = _load_cache(ticker)
        if hit is not None:
            return hit

    merged, ticker_obj = _resolve_fundamentals_info(ticker)
    info: dict = {}
    if ticker_obj is not None:
        try:
            info = ticker_obj.info or {}
        except Exception:
            info = {}

    for key, val in info.items():
        if key not in merged or merged.get(key) is None:
            if isinstance(val, (int, float, str, bool)) or val is None:
                merged[key] = val

    annual_eps = get_annual_eps_history_10y(ticker) or {}
    earnings_estimates = _parse_earnings_estimates(ticker_obj) if ticker_obj else {}
    earnings_history = _parse_earnings_history(ticker_obj) if ticker_obj else []
    lt_debt_capital = _lt_debt_to_capital_pct(ticker_obj) if ticker_obj else None

    bundle = {
        "ticker": ticker,
        "info": _info_bundle(info, merged),
        "annual_eps": {str(k): v for k, v in annual_eps.items()},
        "earnings_estimates": earnings_estimates,
        "earnings_history": earnings_history,
        "lt_debt_capital": lt_debt_capital,
    }
    _save_cache(ticker, bundle)
    return bundle


def annual_eps_from_bundle(bundle: dict) -> dict[int, float]:
    raw = bundle.get("annual_eps") or {}
    out: dict[int, float] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out
