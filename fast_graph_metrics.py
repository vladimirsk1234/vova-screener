"""
FAST Graphs–style valuation metrics. Pure math; no I/O.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from eps_yield import (
    avg_historical_pe_5y,
    eps_row_metrics,
    eps_yield_pct,
    fair_and_normal_price,
    pe_ttm,
    vs_fair_pct,
)


def eps_cagr_over_years(annual_eps: dict[int, float] | None, years: int) -> float | None:
    """CAGR % over the last `years` span in annual_eps."""
    if not annual_eps or len(annual_eps) < 2:
        return None
    sorted_years = sorted(annual_eps.keys())
    end_year = sorted_years[-1]
    start_year = end_year - years
    candidates = [y for y in sorted_years if y <= end_year]
    if len(candidates) < 2:
        return None
    start_y = None
    for y in reversed(candidates):
        if y <= start_year:
            start_y = y
            break
    if start_y is None:
        start_y = candidates[0]
    if start_y >= end_year:
        return None
    start_eps = float(annual_eps[start_y])
    end_eps = float(annual_eps[end_year])
    span = end_year - start_y
    if span <= 0 or start_eps <= 0 or end_eps <= 0:
        return None
    try:
        cagr = (end_eps / start_eps) ** (1.0 / span) - 1.0
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    if not math.isfinite(cagr):
        return None
    return round(cagr * 100.0, 2)


def eps_cagr_pct(annual_eps: dict[int, float] | None) -> float | None:
    """CAGR % over full available annual EPS history."""
    if not annual_eps or len(annual_eps) < 2:
        return None
    years = sorted(annual_eps.keys())
    return eps_cagr_over_years(annual_eps, years[-1] - years[0])


def resolve_fair_pe(
    growth_rate_pct: float | None,
    *,
    sidebar_fair_pe: float = 15.0,
    growth_threshold: float = 10.0,
) -> float:
    """Auto rule: P/E = growth when growth >= threshold, else fixed fair P/E."""
    if growth_rate_pct is not None and growth_rate_pct >= growth_threshold:
        return round(float(growth_rate_pct), 2)
    return float(sidebar_fair_pe)


def _blended_pe(trailing_pe: float | None, forward_pe: float | None) -> float | None:
    vals = [v for v in (trailing_pe, forward_pe) if v is not None and math.isfinite(v) and v > 0]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)


def _pct_to_decimal(val: float | None) -> float | None:
    if val is None:
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    if abs(v) > 1.5:
        return v / 100.0
    return v


def project_future_eps(
    base_eps: float,
    *,
    years: int,
    growth_rate: float | None,
) -> float | None:
    """Compound EPS forward `years` at growth_rate (decimal or percent)."""
    if years <= 0 or base_eps <= 0:
        return None
    g = _pct_to_decimal(growth_rate)
    if g is None:
        return None
    try:
        return base_eps * ((1.0 + g) ** years)
    except (OverflowError, ValueError):
        return None


def est_annual_ror_pct(
    current_price: float,
    future_price: float,
    *,
    years: int,
    dividend_yield_pct: float | None = None,
) -> float | None:
    """CAGR from current to future price over N years (+ optional dividend yield)."""
    if current_price <= 0 or future_price <= 0 or years <= 0:
        return None
    try:
        total_return = future_price / current_price
        if dividend_yield_pct is not None and dividend_yield_pct > 0:
            div_dec = dividend_yield_pct / 100.0
            total_return *= (1.0 + div_dec * years)
        ror = total_return ** (1.0 / years) - 1.0
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    if not math.isfinite(ror):
        return None
    return round(ror * 100.0, 2)


def compute_est_ror(
    close: float,
    future_eps: float | None,
    valuation_pe: float,
    *,
    horizon_years: int = 3,
    dividend_yield_pct: float | None = None,
) -> tuple[float | None, float | None]:
    """Return (future_price, est_annual_ror_pct)."""
    if future_eps is None or future_eps <= 0 or valuation_pe <= 0:
        return None, None
    future_price = future_eps * valuation_pe
    ror = est_annual_ror_pct(
        close,
        future_price,
        years=horizon_years,
        dividend_yield_pct=dividend_yield_pct,
    )
    return round(future_price, 2), ror


def _clamp_score(val: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, val))


def _score_margin(margin: float | None) -> float:
    if margin is None:
        return 50.0
    pct = margin * 100.0 if abs(margin) <= 1.0 else margin
    return _clamp_score(pct * 3.0)


def _score_roe(roe: float | None) -> float:
    if roe is None:
        return 50.0
    pct = roe * 100.0 if abs(roe) <= 1.0 else roe
    return _clamp_score(pct * 2.5)


def _score_growth(cagr: float | None, est_growth: float | None) -> float:
    vals = [v for v in (cagr, est_growth) if v is not None and math.isfinite(v)]
    if not vals:
        return 50.0
    avg = sum(vals) / len(vals)
    return _clamp_score(avg * 4.0)


def _score_financial_strength(lt_debt_cap: float | None, debt_to_equity: float | None) -> float:
    score = 70.0
    if lt_debt_cap is not None:
        if lt_debt_cap <= 30:
            score += 20
        elif lt_debt_cap <= 55:
            score += 10
        elif lt_debt_cap > 70:
            score -= 25
        elif lt_debt_cap > 55:
            score -= 10
    if debt_to_equity is not None:
        if debt_to_equity > 200:
            score -= 15
        elif debt_to_equity < 50:
            score += 5
    return _clamp_score(score)


def _score_cash_flow(ocf_to_mcap: float | None) -> float:
    if ocf_to_mcap is None:
        return 50.0
    pct = ocf_to_mcap * 100.0 if abs(ocf_to_mcap) <= 1.0 else ocf_to_mcap
    return _clamp_score(pct * 5.0)


def _score_predictability(beat_pct: float | None) -> float:
    if beat_pct is None:
        return 50.0
    return _clamp_score(beat_pct)


def compute_fg_scores(raw: dict[str, Any]) -> dict[str, float]:
    """Approximate FAST Graphs pentagon scores from Yahoo proxies."""
    profitability = _clamp_score(
        (_score_margin(raw.get("profit_margin"))
         + _score_roe(raw.get("roe"))
         + _score_roe(raw.get("roa"))) / 3.0
    )
    growth = _score_growth(raw.get("eps_cagr"), raw.get("est_eps_growth"))
    financial = _score_financial_strength(
        raw.get("lt_debt_capital"),
        raw.get("debt_to_equity"),
    )
    cash_flow = _score_cash_flow(raw.get("ocf_to_mcap"))
    predictability = _score_predictability(raw.get("analyst_beat_pct"))

    axes = {
        "Profitability": round(profitability, 1),
        "Growth": round(growth, 1),
        "Financial Strength": round(financial, 1),
        "Cash Flow Generation": round(cash_flow, 1),
        "Predictability": round(predictability, 1),
    }
    axes["FG Score"] = round(sum(axes.values()) / len(axes), 1)
    return axes


@dataclass(frozen=True)
class FastGraphFilterConfig:
    countries: tuple[str, ...] = ("United States", "Canada")
    exclude_otc: bool = True
    min_est_eps_growth: float = 10.0
    require_cagr_1y: bool = True
    require_cagr_3y: bool = True
    require_cagr_5y: bool = True
    require_cagr_10y: bool = True
    min_cagr_1y: float = 0.0
    min_cagr_3y: float = 0.0
    min_cagr_5y: float = 0.0
    min_cagr_10y: float = 0.0
    ror_gte_growth: bool = True
    max_lt_debt_capital: float = 55.0
    min_est_annual_ror: float = 0.0
    price_below_fair: bool = False
    min_fg_score: float = 0.0
    horizon_years: int = 3
    sidebar_fair_pe: float = 15.0
    growth_threshold: float = 10.0
    valuation_pe_mode: str = "fair"  # fair | normal


def _is_otc_exchange(exchange: str | None) -> bool:
    if not exchange:
        return False
    ex = exchange.upper()
    return "OTC" in ex or ex in ("PNK", "OQB", "OQX", "GREY")


def passes_fast_graph_filters(metrics: dict[str, Any], cfg: FastGraphFilterConfig) -> tuple[bool, str]:
    """Return (passed, reject_reason)."""
    country = str(metrics.get("country") or "")
    if cfg.countries and country and country not in cfg.countries:
        return False, "COUNTRY"

    if cfg.exclude_otc and _is_otc_exchange(metrics.get("exchange")):
        return False, "OTC"

    est_growth = metrics.get("est_eps_growth")
    if cfg.min_est_eps_growth > 0:
        if est_growth is None or est_growth < cfg.min_est_eps_growth:
            return False, "EST_GROWTH"

    cagr_checks = [
        (cfg.require_cagr_1y, "cagr_1y", cfg.min_cagr_1y),
        (cfg.require_cagr_3y, "cagr_3y", cfg.min_cagr_3y),
        (cfg.require_cagr_5y, "cagr_5y", cfg.min_cagr_5y),
        (cfg.require_cagr_10y, "cagr_10y", cfg.min_cagr_10y),
    ]
    for required, key, minimum in cagr_checks:
        if not required:
            continue
        val = metrics.get(key)
        if val is None or val < minimum:
            return False, key.upper()

    lt_debt = metrics.get("lt_debt_capital")
    if cfg.max_lt_debt_capital > 0 and lt_debt is not None and lt_debt > cfg.max_lt_debt_capital:
        return False, "DEBT_CAP"

    ror = metrics.get("est_annual_ror")
    if cfg.min_est_annual_ror > 0:
        if ror is None or ror < cfg.min_est_annual_ror:
            return False, "MIN_ROR"

    if cfg.ror_gte_growth and est_growth is not None and ror is not None:
        if ror < est_growth:
            return False, "ROR_LT_GROWTH"

    if cfg.price_below_fair:
        vs_fair = metrics.get("vs_fair_pct")
        if vs_fair is None or vs_fair >= 0:
            return False, "NOT_BELOW_FAIR"

    fg = metrics.get("fg_score")
    if cfg.min_fg_score > 0:
        if fg is None or fg < cfg.min_fg_score:
            return False, "FG_SCORE"

    return True, ""


def build_fast_graph_metrics(
    *,
    close: float,
    annual_eps: dict[int, float] | None,
    df_daily: pd.DataFrame | None,
    info: dict,
    earnings_estimates: dict[str, Any] | None,
    earnings_history: list[dict] | None,
    lt_debt_capital: float | None,
    cfg: FastGraphFilterConfig,
) -> dict[str, Any]:
    """Assemble all FAST Graph metrics for one symbol."""
    growth_rate = eps_cagr_pct(annual_eps)
    fair_pe = resolve_fair_pe(
        growth_rate,
        sidebar_fair_pe=cfg.sidebar_fair_pe,
        growth_threshold=cfg.growth_threshold,
    )
    norm_pe = avg_historical_pe_5y(df_daily, annual_eps)

    trailing_eps = info.get("trailing_eps")
    forward_eps = info.get("forward_eps")
    trailing_pe = info.get("trailing_pe")
    forward_pe = info.get("forward_pe")

    blended = _blended_pe(trailing_pe, forward_pe)
    if blended is None and trailing_eps:
        blended = pe_ttm(close, trailing_eps)

    norm_pe_val = norm_pe if norm_pe is not None else 0.0
    row_m = eps_row_metrics(close, trailing_eps, fair_pe=fair_pe, norm_pe=norm_pe_val)
    fair_price = row_m.get("Fair $")
    vs_fair = row_m.get("vs Fair %")

    est_0y = (earnings_estimates or {}).get("0y", {})
    est_1y = (earnings_estimates or {}).get("+1y", {})
    est_growth_raw = est_1y.get("growth") or est_0y.get("growth")
    est_eps_growth = None
    if est_growth_raw is not None:
        try:
            g = float(est_growth_raw)
            est_eps_growth = round(g * 100.0 if abs(g) <= 1.5 else g, 2)
        except (TypeError, ValueError):
            pass

    base_future_eps = est_1y.get("avg") or forward_eps or trailing_eps
    proj_growth = est_eps_growth or growth_rate
    future_eps = project_future_eps(
        float(base_future_eps) if base_future_eps else 0.0,
        years=cfg.horizon_years,
        growth_rate=proj_growth,
    ) if base_future_eps else None

    val_pe = fair_pe if cfg.valuation_pe_mode == "fair" else (norm_pe or fair_pe)
    future_price, est_ror = compute_est_ror(
        close,
        future_eps,
        val_pe,
        horizon_years=cfg.horizon_years,
        dividend_yield_pct=info.get("dividend_yield_pct"),
    )

    beat_pct = None
    if earnings_history:
        beats = sum(1 for h in earnings_history if h.get("beat"))
        total = len(earnings_history)
        if total > 0:
            beat_pct = round(beats / total * 100.0, 1)

    mcap = info.get("market_cap")
    ocf = info.get("operating_cashflow")
    ocf_to_mcap = None
    if mcap and ocf and mcap > 0:
        ocf_to_mcap = ocf / mcap

    fg_raw = {
        "profit_margin": info.get("profit_margin"),
        "roe": info.get("roe"),
        "roa": info.get("roa"),
        "eps_cagr": growth_rate,
        "est_eps_growth": est_eps_growth,
        "lt_debt_capital": lt_debt_capital,
        "debt_to_equity": info.get("debt_to_equity"),
        "ocf_to_mcap": ocf_to_mcap,
        "analyst_beat_pct": beat_pct,
    }
    fg_scores = compute_fg_scores(fg_raw)

    metrics: dict[str, Any] = {
        "close": round(close, 2),
        "growth_rate": growth_rate,
        "fair_pe": fair_pe,
        "normal_pe": norm_pe,
        "blended_pe": blended,
        "eps_yield": eps_yield_pct(trailing_eps, close),
        "fair_price": fair_price,
        "normal_price": row_m.get("Normal $"),
        "vs_fair_pct": vs_fair,
        "est_eps_growth": est_eps_growth,
        "est_annual_ror": est_ror,
        "future_price": future_price,
        "future_eps": round(future_eps, 4) if future_eps else None,
        "lt_debt_capital": lt_debt_capital,
        "fg_score": fg_scores.get("FG Score"),
        "fg_axes": fg_scores,
        "cagr_1y": eps_cagr_over_years(annual_eps, 1),
        "cagr_3y": eps_cagr_over_years(annual_eps, 3),
        "cagr_5y": eps_cagr_over_years(annual_eps, 5),
        "cagr_10y": eps_cagr_over_years(annual_eps, 10),
        "country": info.get("country"),
        "exchange": info.get("exchange"),
        "analyst_beat_pct": beat_pct,
        "annual_eps": annual_eps or {},
        "earnings_estimates": earnings_estimates or {},
        "trailing_eps": trailing_eps,
        "forward_eps": forward_eps,
        "dividend_yield_pct": info.get("dividend_yield_pct"),
    }
    return metrics
