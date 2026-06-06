"""
Yahoo Finance data bundle for FAST Graphs scanner. No Streamlit.

Self-contained module: avoids importing new ticker_data symbols at load time
(prevents ImportError on partial deploys / circular imports).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "fg_bundle"
_CACHE_TTL_SEC = 86400.0


def _float_field(i: dict, key: str) -> float | None:
    val = i.get(key)
    if val is None:
        return None
    try:
        out = float(val)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _eps_from_yf_info(i: dict) -> tuple[float | None, float | None]:
    te = i.get("trailingEps")
    fe = i.get("forwardEps")
    trailing = _float_field({"trailingEps": te}, "trailingEps") if te is not None else None
    forward = _float_field({"forwardEps": fe}, "forwardEps") if fe is not None else None
    return trailing, forward


def _info_field_str(info: dict, *keys: str) -> str | None:
    for key in keys:
        val = info.get(key)
        if val is None or val == "":
            continue
        return str(val).strip()
    return None


def _yield_pct_from_yahoo(
    raw: float | None,
    *,
    close: float | None = None,
    dividend_rate: float | None = None,
) -> float | None:
    pct: float | None = None
    if raw is not None:
        try:
            v = float(raw)
        except (TypeError, ValueError):
            v = None
        if v is not None and math.isfinite(v):
            if 0 < v < 0.5:
                pct = v * 100.0
            elif 0 < v <= 25.0:
                pct = v
    if pct is not None and pct > 20.0:
        pct = None
    if pct is None and dividend_rate is not None and close is not None:
        try:
            rate = float(dividend_rate)
            c = float(close)
        except (TypeError, ValueError):
            rate, c = None, None
        if rate is not None and c and c > 0 and math.isfinite(rate):
            alt = rate / c * 100.0
            if 0 < alt <= 20.0:
                pct = alt
    return round(pct, 2) if pct is not None and math.isfinite(pct) else None


def _find_balance_row(bs: pd.DataFrame, candidates: tuple[str, ...]) -> float | None:
    if not isinstance(bs, pd.DataFrame) or bs.empty:
        return None
    for name in candidates:
        if name in bs.index:
            col = bs.columns[0]
            try:
                val = float(bs.loc[name, col])
                if math.isfinite(val):
                    return val
            except (TypeError, ValueError):
                continue
    return None


def _lt_debt_to_capital_pct(ticker_obj: yf.Ticker | None) -> float | None:
    if ticker_obj is None:
        return None
    try:
        bs = ticker_obj.balance_sheet
    except Exception:
        return None
    if not isinstance(bs, pd.DataFrame) or bs.empty:
        return None
    lt_debt = _find_balance_row(
        bs,
        (
            "Long Term Debt",
            "Long Term Debt And Capital Lease Obligation",
            "Long Term Debt Noncurrent",
        ),
    )
    equity = _find_balance_row(
        bs,
        (
            "Stockholders Equity",
            "Total Stockholder Equity",
            "Common Stock Equity",
            "Total Equity Gross Minority Interest",
        ),
    )
    if lt_debt is None or equity is None:
        return None
    denom = lt_debt + equity
    if denom <= 0:
        return None
    return round(lt_debt / denom * 100.0, 2)


def _extract_annual_eps_map(financials: pd.DataFrame | None) -> dict[int, float]:
    if not isinstance(financials, pd.DataFrame) or financials.empty:
        return {}
    eps_row = None
    for candidate in ("Diluted EPS", "Basic EPS", "DilutedEPS", "BasicEPS"):
        if candidate in financials.index:
            eps_row = financials.loc[candidate]
            break
    if eps_row is None:
        return {}
    out: dict[int, float] = {}
    for col, raw_val in eps_row.items():
        try:
            year = pd.Timestamp(col).year
            eps_val = float(raw_val)
        except Exception:
            continue
        if not math.isfinite(eps_val):
            continue
        out[year] = eps_val
    return out


def _annual_eps_history_10y(ticker: str, ticker_obj: yf.Ticker | None = None) -> dict[int, float]:
    try:
        t = ticker_obj or yf.Ticker(ticker)
        eps_map = _extract_annual_eps_map(getattr(t, "financials", None))
        if not eps_map:
            eps_map = _extract_annual_eps_map(getattr(t, "income_stmt", None))
        if not eps_map:
            return {}
        years_desc = sorted(eps_map.keys(), reverse=True)[:10]
        return {y: eps_map[y] for y in sorted(years_desc)}
    except Exception:
        return {}


def _earnings_estimates_yf(ticker_obj: yf.Ticker) -> dict[str, dict]:
    out: dict[str, dict] = {}
    try:
        df = ticker_obj.get_earnings_estimate()
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return out
        for period in df.index:
            row = df.loc[period]
            entry: dict = {}
            for col in df.columns:
                val = row.get(col) if hasattr(row, "get") else row[col]
                if val is not None and (isinstance(val, (int, float)) or not pd.isna(val)):
                    try:
                        entry[str(col)] = float(val) if col != "numberOfAnalysts" else int(val)
                    except (TypeError, ValueError):
                        entry[str(col)] = val
            if entry:
                out[str(period)] = entry
    except Exception:
        pass
    return out


def _earnings_history_yf(ticker_obj: yf.Ticker) -> list[dict]:
    out: list[dict] = []
    try:
        df = ticker_obj.get_earnings_history()
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return out
        for idx, row in df.iterrows():
            entry: dict = {"date": str(idx)}
            for col in ("epsEstimate", "epsActual", "epsDifference", "surprisePercent"):
                if col in df.columns:
                    val = row[col]
                    if val is not None and not (isinstance(val, float) and math.isnan(val)):
                        try:
                            entry[col] = float(val)
                        except (TypeError, ValueError):
                            entry[col] = val
            out.append(entry)
    except Exception:
        pass
    return out


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
    raw = _earnings_estimates_yf(ticker_obj)
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
    raw = _earnings_history_yf(ticker_obj)
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


def _resolve_fundamentals_info(ticker: str) -> tuple[dict, yf.Ticker | None]:
    """Lazy import from ticker_data to avoid circular imports at module load."""
    from ticker_data import _resolve_fundamentals_info as _resolve

    return _resolve(ticker)


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

    annual_eps = _annual_eps_history_10y(ticker, ticker_obj) if ticker_obj else {}
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
