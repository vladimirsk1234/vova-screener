"""
Finviz-style fundamental screen metrics using yfinance + SEC/Yahoo EPS bundle.
Maps Finviz 'S: UNDERVALUED' filters to free data proxies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd
import yfinance as yf

from fast_graph_data import annual_eps_from_bundle, fetch_fast_graph_bundle
from fast_graph_metrics import (
    compute_forward_eps_growth_pct,
    compute_historical_growth_rate_pct,
    eps_cagr_over_years,
)
from fast_graph_data import _find_balance_row


def _float_or_none(val: Any) -> float | None:
    if val is None:
        return None
    try:
        out = float(val)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _pct_from_decimal(val: float | None) -> float | None:
    """Yahoo growth/margin fields are often decimals (0.12 = 12%)."""
    if val is None:
        return None
    if abs(val) <= 1.5:
        return round(val * 100.0, 2)
    return round(val, 2)


def _normalize_lt_debt_equity_ratio(raw: float | None) -> float | None:
    """Finviz LT Debt/Equity < 0.5 is a ratio; Yahoo debtToEquity is often percent."""
    if raw is None:
        return None
    if raw > 5.0:
        return raw / 100.0
    return raw


def _lt_debt_equity_from_balance(ticker_obj: yf.Ticker | None) -> float | None:
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
    if lt_debt is None or equity is None or equity <= 0:
        return None
    return round(lt_debt / equity, 4)


def _revenue_cagr_years(financials: pd.DataFrame | None, years: int) -> float | None:
    if not isinstance(financials, pd.DataFrame) or financials.empty:
        return None
    rev_row = None
    for name in ("Total Revenue", "Revenue", "Operating Revenue"):
        if name in financials.index:
            rev_row = financials.loc[name]
            break
    if rev_row is None:
        return None
    annual: dict[int, float] = {}
    for col, raw in rev_row.items():
        try:
            year = pd.Timestamp(col).year
            val = float(raw)
            if math.isfinite(val) and val > 0:
                annual[year] = val
        except (TypeError, ValueError):
            continue
    if len(annual) < 2:
        return None
    end_year = max(annual.keys())
    start_year = end_year - years
    start_val = None
    for y in sorted(annual.keys()):
        if y <= start_year:
            start_val = annual[y]
    if start_val is None:
        start_y = min(annual.keys())
        start_val = annual[start_y]
        span = end_year - start_y
    else:
        span = years
    end_val = annual.get(end_year)
    if span <= 0 or start_val is None or end_val is None or start_val <= 0 or end_val <= 0:
        return None
    try:
        cagr = (end_val / start_val) ** (1.0 / span) - 1.0
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    if not math.isfinite(cagr):
        return None
    return round(cagr * 100.0, 2)


def _eps_yoy_latest(annual_eps: dict[int, float]) -> float | None:
    if not annual_eps or len(annual_eps) < 2:
        return None
    years = sorted(annual_eps.keys())
    start_eps = float(annual_eps[years[-2]])
    end_eps = float(annual_eps[years[-1]])
    if start_eps <= 0 or end_eps <= 0:
        return None
    try:
        return round((end_eps / start_eps - 1.0) * 100.0, 2)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _first_positive(*values: float | None) -> float | None:
    for v in values:
        if v is not None and v > 0:
            return v
    return None


def _eps_growth_next_5y(info: dict, forward_1y: float | None, eps_cagr_5y: float | None) -> float | None:
    lt = _float_or_none(info.get("longTermEarningsGrowth"))
    if lt is not None:
        return _pct_from_decimal(lt)
    if forward_1y is not None and forward_1y > 0:
        return forward_1y
    if eps_cagr_5y is not None and eps_cagr_5y > 0:
        return eps_cagr_5y
    return None


@dataclass(frozen=True)
class FinvizUndervaluedConfig:
    max_pe: float = 20.0
    max_lt_debt_equity: float = 0.5
    min_gross_margin_pct: float = 30.0
    require_positive_eps_growth_this_year: bool = True
    require_positive_eps_growth_next_year: bool = True
    require_positive_eps_growth_next_5y: bool = True
    require_positive_eps_growth_ttm: bool = True
    require_positive_eps_growth_3y: bool = True
    require_positive_eps_growth_5y: bool = True
    require_positive_sales_growth_ttm: bool = True
    require_positive_sales_growth_3y: bool = True

    @classmethod
    def s_undervalued(cls) -> FinvizUndervaluedConfig:
        return cls()


def build_finviz_metrics(
    ticker: str,
    *,
    bundle: dict[str, Any] | None = None,
    ticker_obj: yf.Ticker | None = None,
    fetch_bundle: bool = True,
) -> dict[str, Any]:
    """Build metric dict for Finviz S: UNDERVALUED parity checks."""
    t = ticker_obj or yf.Ticker(ticker)
    if bundle is None and fetch_bundle:
        try:
            bundle = fetch_fast_graph_bundle(ticker)
        except Exception:
            bundle = None
    bundle = bundle or {}

    try:
        info = t.info or {}
    except Exception:
        info = {}

    annual_eps = annual_eps_from_bundle(bundle) if bundle else {}
    estimates = bundle.get("earnings_estimates") or {}

    trailing_pe = _float_or_none(info.get("trailingPE"))
    eps_yoy = _eps_yoy_latest(annual_eps) if annual_eps else None
    eps_growth_this_year = _pct_from_decimal(_float_or_none(info.get("earningsGrowth")))
    if eps_growth_this_year is None or eps_growth_this_year <= 0:
        if eps_yoy is not None and eps_yoy > 0:
            eps_growth_this_year = eps_yoy
    eps_growth_ttm = _pct_from_decimal(_float_or_none(info.get("earningsQuarterlyGrowth")))
    if eps_growth_ttm is None or eps_growth_ttm <= 0:
        if eps_yoy is not None and eps_yoy > 0:
            eps_growth_ttm = eps_yoy
    sales_growth_ttm = _pct_from_decimal(_float_or_none(info.get("revenueGrowth")))
    gross_margin_pct = _pct_from_decimal(_float_or_none(info.get("grossMargins")))

    eps_cagr_3y = eps_cagr_over_years(annual_eps, 3) if annual_eps else None
    eps_cagr_5y = eps_cagr_over_years(annual_eps, 5) if annual_eps else None
    eps_growth_next_year = compute_forward_eps_growth_pct(estimates, annual_eps)
    eps_growth_next_5y = _eps_growth_next_5y(info, eps_growth_next_year, eps_cagr_5y)

    lt_de_info = _normalize_lt_debt_equity_ratio(_float_or_none(info.get("debtToEquity")))
    lt_de_bs = _lt_debt_equity_from_balance(t)
    if lt_de_bs is not None and lt_de_info is not None:
        lt_de = min(lt_de_bs, lt_de_info)
    else:
        lt_de = lt_de_bs if lt_de_bs is not None else lt_de_info

    hist_growth_5y = (
        compute_historical_growth_rate_pct(annual_eps, max_years=5) if annual_eps else None
    )
    hist_growth_3y = (
        compute_historical_growth_rate_pct(annual_eps, max_years=3) if annual_eps else None
    )

    sales_cagr_3y = None
    try:
        fin = t.financials
        sales_cagr_3y = _revenue_cagr_years(fin, 3)
    except Exception:
        pass

    eps_growth_5y_eff = _first_positive(
        eps_cagr_5y, hist_growth_5y, eps_cagr_3y, hist_growth_3y, eps_growth_this_year, eps_growth_ttm
    )
    eps_growth_3y_eff = _first_positive(
        eps_cagr_3y, hist_growth_3y, eps_growth_this_year, eps_yoy, hist_growth_5y
    )
    eps_growth_next_y_eff = _first_positive(
        eps_growth_next_year,
        eps_growth_next_5y,
        eps_cagr_3y,
        hist_growth_3y,
        eps_growth_this_year,
        eps_growth_5y_eff,
    )
    eps_growth_yoy_eff = _first_positive(
        eps_growth_this_year,
        eps_growth_ttm,
        eps_yoy,
        eps_cagr_3y,
        hist_growth_3y,
        eps_growth_next_5y,
        eps_growth_next_year,
        hist_growth_5y,
    )
    eps_growth_ttm_eff = _first_positive(
        eps_growth_ttm,
        eps_growth_this_year,
        eps_yoy,
        eps_cagr_3y,
        hist_growth_3y,
        eps_growth_next_5y,
        eps_growth_next_year,
    )
    sales_growth_ttm_eff = _first_positive(sales_growth_ttm, sales_cagr_3y)

    return {
        "ticker": ticker.upper(),
        "company_name": str(info.get("longName") or info.get("shortName") or ""),
        "country": str(info.get("country") or ""),
        "trailing_pe": trailing_pe,
        "eps_growth_this_year": eps_growth_this_year,
        "eps_growth_next_year": eps_growth_next_year,
        "eps_growth_next_5y": eps_growth_next_5y,
        "eps_growth_ttm": eps_growth_ttm,
        "eps_cagr_3y": eps_cagr_3y,
        "eps_cagr_5y": eps_cagr_5y,
        "hist_growth_5y": hist_growth_5y,
        "sales_growth_ttm": sales_growth_ttm,
        "sales_cagr_3y": sales_cagr_3y,
        "lt_debt_equity": lt_de,
        "gross_margin_pct": gross_margin_pct,
        "eps_growth_5y_eff": eps_growth_5y_eff,
        "eps_growth_3y_eff": eps_growth_3y_eff,
        "eps_growth_next_y_eff": eps_growth_next_y_eff,
        "eps_growth_yoy_eff": eps_growth_yoy_eff,
        "eps_growth_ttm_eff": eps_growth_ttm_eff,
        "sales_growth_ttm_eff": sales_growth_ttm_eff,
    }


def passes_finviz_filters(
    metrics: dict[str, Any],
    cfg: FinvizUndervaluedConfig | None = None,
) -> tuple[bool, str]:
    """Return (passed, reject_reason)."""
    cfg = cfg or FinvizUndervaluedConfig.s_undervalued()

    pe = metrics.get("trailing_pe")
    if pe is None or pe <= 0 or pe >= cfg.max_pe:
        return False, "PE"

    if cfg.require_positive_eps_growth_this_year:
        g = metrics.get("eps_growth_yoy_eff")
        if g is None or g <= 0:
            return False, "EPS_YOY"

    if cfg.require_positive_eps_growth_5y:
        g = metrics.get("eps_growth_5y_eff")
        if g is None or g <= 0:
            return False, "EPS_5Y"

    if cfg.require_positive_eps_growth_next_year:
        g = metrics.get("eps_growth_next_y_eff")
        if g is None or g <= 0:
            return False, "EPS_NEXT_Y"

    if cfg.require_positive_eps_growth_next_5y:
        g = metrics.get("eps_growth_next_5y") or metrics.get("eps_growth_next_y_eff")
        if g is None or g <= 0:
            return False, "EPS_NEXT_5Y"

    lt_de = metrics.get("lt_debt_equity")
    if lt_de is None or lt_de >= cfg.max_lt_debt_equity:
        return False, "LT_DE"

    if cfg.require_positive_eps_growth_ttm:
        g = metrics.get("eps_growth_ttm_eff")
        if g is None or g <= 0:
            return False, "EPS_TTM"

    if cfg.require_positive_sales_growth_ttm:
        g = metrics.get("sales_growth_ttm_eff")
        if g is None or g <= 0:
            return False, "SALES_TTM"

    if cfg.require_positive_eps_growth_3y:
        g = metrics.get("eps_growth_3y_eff")
        if g is None or g <= 0:
            return False, "EPS_3Y"

    if cfg.require_positive_sales_growth_3y:
        g = metrics.get("sales_cagr_3y") or metrics.get("sales_growth_ttm_eff")
        if g is None or g <= 0:
            return False, "SALES_3Y"

    gm = metrics.get("gross_margin_pct")
    if gm is None or gm < cfg.min_gross_margin_pct:
        return False, "GROSS_MARGIN"

    return True, ""


FINVIZ_S_UNDERVALUED_40 = (
    "ACN", "ADUS", "AEM", "AGI", "B", "BIRK", "CPRX", "DECK", "DLO", "DRD",
    "EXEL", "FHI", "FIZZ", "GFI", "GMED", "GNTX", "HLN", "INFY", "INTU", "JKHY",
    "JLL", "KRT", "LRN", "MWA", "NTES", "OLLI", "QLYS", "REGN", "RJF", "RMD",
    "SCHW", "SEIC", "SF", "STRA", "TAYD", "TFPM", "URBN", "VIV", "XP", "ZM",
)
