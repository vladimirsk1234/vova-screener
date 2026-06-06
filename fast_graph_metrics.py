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
from ticker_data import filter_eps_outliers


DEFAULT_GROWTH_CAP_PCT = 100.0
YOY_CAP_PCT = 150.0


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
    """CAGR % over full available annual EPS history (legacy; prefer compute_historical_growth_rate_pct)."""
    if not annual_eps or len(annual_eps) < 2:
        return None
    years = sorted(annual_eps.keys())
    return eps_cagr_over_years(annual_eps, years[-1] - years[0])


def _yoy_changes_pct(
    annual_eps: dict[int, float],
    *,
    min_base_frac: float = 0.25,
) -> list[float]:
    """Year-over-year EPS % changes for consecutive positive years."""
    positive = [float(v) for v in annual_eps.values() if v is not None and float(v) > 0]
    if not positive:
        return []
    positive.sort()
    mid = len(positive) // 2
    median_eps = positive[mid] if len(positive) % 2 else (positive[mid - 1] + positive[mid]) / 2.0
    min_base = median_eps * min_base_frac

    changes: list[float] = []
    years = sorted(annual_eps.keys())
    prev_eps: float | None = None
    for y in years:
        eps = float(annual_eps[y])
        if eps <= 0:
            prev_eps = None
            continue
        if prev_eps is not None and prev_eps >= min_base and prev_eps > 0:
            chg = (eps - prev_eps) / abs(prev_eps) * 100.0
            if math.isfinite(chg):
                changes.append(chg)
        prev_eps = eps
    return changes


def _geometric_mean_yoy(changes: list[float], *, yoy_cap: float = YOY_CAP_PCT) -> float | None:
    """Geometric mean of YoY % changes, excluding spikes above yoy_cap."""
    valid = [c for c in changes if math.isfinite(c) and abs(c) <= yoy_cap]
    if len(valid) < 2:
        return None
    product = 1.0
    for c in valid:
        product *= 1.0 + c / 100.0
    if product <= 0:
        return None
    gm = product ** (1.0 / len(valid)) - 1.0
    if not math.isfinite(gm):
        return None
    return round(gm * 100.0, 2)


def compute_historical_growth_rate_pct(
    annual_eps: dict[int, float] | None,
    *,
    yoy_cap: float = YOY_CAP_PCT,
    max_years: int = 5,
) -> float | None:
    """
    FAST Graphs–style historical growth: geometric mean of YoY changes
    on outlier-filtered positive EPS, with 5y CAGR fallback.
    """
    if not annual_eps or len(annual_eps) < 2:
        return None
    filtered = filter_eps_outliers(annual_eps, min_frac_of_median=0.25)
    if len(filtered) < 2:
        filtered = {y: float(e) for y, e in annual_eps.items() if float(e) > 0}
    years = sorted(filtered.keys())
    if len(years) > max_years:
        years = years[-max_years:]
        filtered = {y: filtered[y] for y in years}
    if len(filtered) < 2:
        return None

    changes = _yoy_changes_pct(filtered)
    valid = [c for c in changes if math.isfinite(c) and abs(c) <= yoy_cap]

    gm = _geometric_mean_yoy(valid, yoy_cap=yoy_cap) if len(valid) >= 2 else None
    if gm is not None:
        return gm
    if len(valid) == 1:
        return round(valid[0], 2)

    # Fallback: 1-year CAGR between last two filtered years (avoids tiny base years).
    last_two = years[-2:]
    start_eps = float(filtered[last_two[0]])
    end_eps = float(filtered[last_two[1]])
    if start_eps > 0 and end_eps > 0 and last_two[1] > last_two[0]:
        try:
            cagr = (end_eps / start_eps) - 1.0
            if math.isfinite(cagr):
                return round(cagr * 100.0, 2)
        except (ValueError, ZeroDivisionError, OverflowError):
            pass

    span = min(max_years, years[-1] - years[0])
    if span >= 1:
        return eps_cagr_over_years(filtered, span)
    return None


def compute_forecast_growth_pct(
    estimates: dict[str, Any] | None,
    annual_eps: dict[int, float] | None,
    historical_growth: float | None,
) -> float | None:
    """
    Forecast growth: analyst +1y growth, else implied 0y→+1y, else historical.
    """
    est = estimates or {}
    est_0y = est.get("0y", {})
    est_1y = est.get("+1y", {})

    raw_growth = est_1y.get("growth") or est_0y.get("growth")
    if raw_growth is not None:
        try:
            g = float(raw_growth)
            return round(g * 100.0 if abs(g) <= 1.5 else g, 2)
        except (TypeError, ValueError):
            pass

    avg_0y = est_0y.get("avg")
    avg_1y = est_1y.get("avg")
    if avg_0y is not None and avg_1y is not None:
        try:
            e0, e1 = float(avg_0y), float(avg_1y)
            if e0 > 0 and e1 > 0:
                implied = (e1 / e0 - 1.0) * 100.0
                if math.isfinite(implied):
                    return round(implied, 2)
        except (TypeError, ValueError):
            pass

    return historical_growth


def resolve_fair_pe(
    growth_rate_pct: float | None,
    *,
    sidebar_fair_pe: float = 15.0,
    growth_threshold: float = 10.0,
    growth_cap_pct: float = DEFAULT_GROWTH_CAP_PCT,
) -> float:
    """Auto rule: P/E = growth when growth >= threshold, else fixed fair P/E."""
    if growth_rate_pct is not None and growth_rate_pct >= growth_threshold:
        capped = min(float(growth_rate_pct), growth_cap_pct)
        return round(capped, 2)
    return float(sidebar_fair_pe)


def resolve_chart_growth_rate(
    historical_growth: float | None,
    forecast_growth: float | None,
    *,
    mode: str,
    growth_threshold: float = 10.0,
) -> float | None:
    """
    Growth rate shown on FAST Graph chart boxes.
    Historical view: prefer historical when >= threshold; else use strong forecast
    (volatile EPS histories like AGI otherwise fall back to sidebar fair P/E).
    """
    if mode == "forecast":
        return forecast_growth if forecast_growth is not None else historical_growth
    if historical_growth is not None and historical_growth >= growth_threshold:
        return historical_growth
    if forecast_growth is not None and forecast_growth >= growth_threshold:
        return forecast_growth
    if historical_growth is not None:
        return historical_growth
    return forecast_growth


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


def _estimate_eps_chain(
    annual_eps: dict[int, float] | None,
    estimates: dict[str, Any] | None,
    *,
    years_ahead: int = 4,
    growth_rate: float | None = None,
) -> list[tuple[int, float, bool]]:
    """
    Build (year, eps, is_estimate) chain: historical + analyst 0y/+1y + projected.
    """
    points: list[tuple[int, float, bool]] = []
    for y, e in sorted((annual_eps or {}).items()):
        points.append((int(y), float(e), False))

    est = estimates or {}
    est_0y = est.get("0y", {})
    est_1y = est.get("+1y", {})
    last_year = max(annual_eps.keys()) if annual_eps else pd.Timestamp.now().year

    if est_0y.get("avg"):
        points.append((last_year + 1, float(est_0y["avg"]), True))
    if est_1y.get("avg"):
        points.append((last_year + 2, float(est_1y["avg"]), True))

    base = est_1y.get("avg") or est_0y.get("avg")
    if base and years_ahead > 2 and growth_rate is not None:
        for i in range(3, years_ahead + 1):
            proj = project_future_eps(float(base), years=i - 2, growth_rate=growth_rate)
            if proj:
                points.append((last_year + i, proj, True))
    return points


def resolve_target_year_eps(
    annual_eps: dict[int, float] | None,
    estimates: dict[str, Any] | None,
    *,
    horizon_years: int,
    growth_rate: float | None,
) -> float | None:
    """
    EPS at target fiscal year = last reported year + horizon_years.
    Uses analyst chain when available; compounds only beyond last estimate.
    """
    if not annual_eps:
        return None
    last_year = max(annual_eps.keys())
    target_year = last_year + horizon_years
    chain = _estimate_eps_chain(
        annual_eps,
        estimates,
        years_ahead=horizon_years + 2,
        growth_rate=growth_rate,
    )
    by_year = {y: e for y, e, _ in chain}
    if target_year in by_year:
        return by_year[target_year]

    est_years = sorted(y for y, _, is_est in chain if is_est)
    if not est_years:
        base = float(annual_eps[last_year])
        if base <= 0:
            return None
        return project_future_eps(base, years=horizon_years, growth_rate=growth_rate)

    last_est_year = est_years[-1]
    last_est_eps = by_year[last_est_year]
    years_remaining = target_year - last_est_year
    if years_remaining <= 0:
        return last_est_eps
    return project_future_eps(float(last_est_eps), years=years_remaining, growth_rate=growth_rate)


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
    growth_cap_pct: float = DEFAULT_GROWTH_CAP_PCT
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
    historical_growth = compute_historical_growth_rate_pct(annual_eps)
    forecast_growth = compute_forecast_growth_pct(
        earnings_estimates,
        annual_eps,
        historical_growth,
    )

    chart_historical_growth = resolve_chart_growth_rate(
        historical_growth,
        forecast_growth,
        mode="historical",
        growth_threshold=cfg.growth_threshold,
    )
    chart_forecast_growth = resolve_chart_growth_rate(
        historical_growth,
        forecast_growth,
        mode="forecast",
        growth_threshold=cfg.growth_threshold,
    )

    historical_fair_pe = resolve_fair_pe(
        chart_historical_growth,
        sidebar_fair_pe=cfg.sidebar_fair_pe,
        growth_threshold=cfg.growth_threshold,
        growth_cap_pct=cfg.growth_cap_pct,
    )
    forecast_fair_pe = resolve_fair_pe(
        chart_forecast_growth,
        sidebar_fair_pe=cfg.sidebar_fair_pe,
        growth_threshold=cfg.growth_threshold,
        growth_cap_pct=cfg.growth_cap_pct,
    )

    historical_normal_pe = avg_historical_pe_5y(df_daily, annual_eps)
    forecast_normal_pe = historical_normal_pe

    # Backward-compatible primary fields (historical mode)
    growth_rate = historical_growth
    fair_pe = historical_fair_pe
    norm_pe = historical_normal_pe

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
    est_eps_growth = forecast_growth
    if est_eps_growth is None and est_growth_raw is not None:
        try:
            g = float(est_growth_raw)
            est_eps_growth = round(g * 100.0 if abs(g) <= 1.5 else g, 2)
        except (TypeError, ValueError):
            pass

    proj_growth = forecast_growth or historical_growth
    future_eps = resolve_target_year_eps(
        annual_eps,
        earnings_estimates,
        horizon_years=cfg.horizon_years,
        growth_rate=proj_growth,
    )

    val_pe = (
        forecast_fair_pe if cfg.valuation_pe_mode == "fair"
        else (forecast_normal_pe or forecast_fair_pe)
    )
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
        "historical_growth_rate": historical_growth,
        "chart_historical_growth_rate": chart_historical_growth,
        "historical_fair_pe": historical_fair_pe,
        "historical_normal_pe": historical_normal_pe,
        "forecast_growth_rate": forecast_growth,
        "chart_forecast_growth_rate": chart_forecast_growth,
        "forecast_fair_pe": forecast_fair_pe,
        "sidebar_fair_pe": cfg.sidebar_fair_pe,
        "growth_threshold": cfg.growth_threshold,
        "growth_cap_pct": cfg.growth_cap_pct,
        "forecast_normal_pe": forecast_normal_pe,
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
