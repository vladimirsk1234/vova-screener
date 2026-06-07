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
    sanitize_display_price,
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


def compute_historical_cagr_pct(
    annual_eps: dict[int, float] | None,
    *,
    years: int = 10,
) -> float | None:
    """
    FAST Graphs–style historical growth: CAGR over `years` on outlier-filtered EPS.
    """
    if not annual_eps or len(annual_eps) < 2:
        return None
    filtered = filter_eps_outliers(annual_eps, min_frac_of_median=0.25)
    if len(filtered) < 2:
        filtered = {y: float(e) for y, e in annual_eps.items() if float(e) > 0}
    if len(filtered) < 2:
        return None
    return eps_cagr_over_years(filtered, years)


def compute_historical_growth_rate_pct(
    annual_eps: dict[int, float] | None,
    *,
    years: int = 10,
    yoy_cap: float = YOY_CAP_PCT,
    max_years: int | None = None,
) -> float | None:
    """
    Historical growth for FAST Graph panels: geometric mean of YoY changes
    on outlier-filtered EPS within the chart window; CAGR fallback.
    """
    span = max_years if max_years is not None else years
    if not annual_eps or len(annual_eps) < 2:
        return None
    filtered = filter_eps_outliers(annual_eps, min_frac_of_median=0.25)
    if len(filtered) < 2:
        filtered = {y: float(e) for y, e in annual_eps.items() if float(e) > 0}
    if len(filtered) < 2:
        return None

    end_year = max(filtered.keys())
    start_window = end_year - span
    windowed = {y: float(e) for y, e in filtered.items() if y >= start_window}
    if len(windowed) < 2:
        windowed = filtered

    changes = _yoy_changes_pct(windowed)
    if len(changes) >= 5:
        gm = _geometric_mean_yoy(changes, yoy_cap=yoy_cap)
        if gm is not None:
            return gm

    return compute_historical_cagr_pct(annual_eps, years=span)


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


def compute_forward_eps_growth_pct(
    estimates: dict[str, Any] | None,
    annual_eps: dict[int, float] | None = None,
) -> float | None:
    """
    Analyst-only forward EPS growth (+1y growth or implied 0y→+1y).
    Returns None when no analyst data — never falls back to historical CAGR.
    """
    del annual_eps  # API symmetry with compute_forecast_growth_pct
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

    return None


def recent_reported_yoy_pct(
    annual_eps: dict[int, float] | None,
    *,
    n: int = 2,
) -> list[float]:
    """Last `n` reported fiscal YoY EPS % changes (most recent last)."""
    if not annual_eps or n <= 0:
        return []
    changes = _yoy_changes_pct(annual_eps)
    if not changes:
        return []
    return changes[-n:]


def _filtered_annual_eps(annual_eps: dict[int, float] | None) -> dict[int, float]:
    """Outlier-filtered annual EPS (same basis as Growth Rate)."""
    if not annual_eps:
        return {}
    filtered = filter_eps_outliers(annual_eps, min_frac_of_median=0.25)
    if len(filtered) < 2:
        filtered = {y: float(e) for y, e in annual_eps.items() if float(e) > 0}
    return filtered


def resolve_valuation_eps(
    annual_eps: dict[int, float] | None,
    trailing_eps: float | None,
    earnings_estimates: dict[str, Any] | None = None,
) -> tuple[float | None, str]:
    """
    EPS for Fair $, vs Fair %, cheap gate, and chart anchor.
    Prefers current-year analyst estimate, then latest positive fiscal year, then TTM.
    """
    if earnings_estimates:
        est0 = earnings_estimates.get("0y") or {}
        avg = est0.get("avg") if isinstance(est0, dict) else None
        if avg is not None:
            try:
                est_eps = float(avg)
            except (TypeError, ValueError):
                est_eps = None
            if est_eps is not None and est_eps > 0:
                return est_eps, "estimate_0y"

    if annual_eps:
        latest_year = max(annual_eps.keys())
        try:
            latest_eps = float(annual_eps[latest_year])
        except (TypeError, ValueError):
            latest_eps = None
        if latest_eps is not None and latest_eps > 0:
            return latest_eps, "annual"
        for year in sorted(annual_eps.keys(), reverse=True):
            try:
                eps = float(annual_eps[year])
            except (TypeError, ValueError):
                continue
            if eps > 0:
                return eps, "annual"
    if trailing_eps is not None:
        try:
            te = float(trailing_eps)
        except (TypeError, ValueError):
            te = None
        if te is not None and te > 0:
            return te, "ttm"
    return None, "none"


def resolve_fair_pe(
    growth_rate_pct: float | None,
    *,
    sidebar_fair_pe: float = 15.0,
    growth_threshold: float = 15.0,
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
    growth_threshold: float = 15.0,
) -> float | None:
    """
    Growth rate shown on FAST Graph chart boxes.
    Historical view always uses historical CAGR (never analyst forecast).
    """
    del growth_threshold  # kept for call-site compatibility
    if mode == "forecast":
        return forecast_growth if forecast_growth is not None else historical_growth
    return historical_growth if historical_growth is not None else forecast_growth


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


def compute_eps_persistence_pct(
    annual_eps: dict[int, float] | None,
    *,
    years: int = 10,
) -> float | None:
    """Share of last `years` fiscal years with positive EPS (FG persistence proxy)."""
    if not annual_eps:
        return None
    end_year = max(annual_eps.keys())
    start_year = end_year - years + 1
    window = [y for y in range(start_year, end_year + 1) if y in annual_eps]
    if not window:
        return None
    positive = sum(1 for y in window if float(annual_eps[y]) > 0)
    return round(positive / len(window) * 100.0, 1)


@dataclass(frozen=True)
class FastGraphFilterConfig:
    countries: tuple[str, ...] = ("United States", "Canada")
    exclude_otc: bool = True
    min_est_eps_growth: float = 10.0
    require_cagr_1y: bool = True
    require_cagr_3y: bool = True
    require_cagr_5y: bool = False
    require_cagr_10y: bool = True
    min_cagr_1y: float = 0.0
    min_cagr_3y: float = 0.0
    min_cagr_5y: float = 0.0
    min_cagr_10y: float = 0.0
    cagr_5y_uses_historical_growth: bool = False
    require_analyst_forward_growth: bool = True
    require_recent_yoy_positive: bool = False
    min_recent_yoy_years: int = 2
    ror_gte_growth: bool = False
    max_lt_debt_capital: float = 55.0
    min_est_annual_ror: float = 0.0
    price_below_fair: bool = True
    max_vs_fair_pct: float | None = None
    require_pe_lte_normal: bool = False
    min_eps_persistence_pct: float = 0.0
    horizon_years: int = 3
    sidebar_fair_pe: float = 15.0
    growth_threshold: float = 15.0
    growth_cap_pct: float = DEFAULT_GROWTH_CAP_PCT
    valuation_pe_mode: str = "fair"  # fair | normal
    growth_years: int = 10

    @classmethod
    def fg_undervalued_quality(
        cls,
        *,
        sidebar_fair_pe: float = 15.0,
        horizon_years: int = 3,
        valuation_pe_mode: str = "fair",
        countries: tuple[str, ...] = ("United States",),
    ) -> FastGraphFilterConfig:
        """FAST Graphs 'Undervalued high quality stocks' preset (free-data approximation)."""
        return cls(
            countries=countries,
            exclude_otc=True,
            min_est_eps_growth=8.0,
            require_analyst_forward_growth=False,
            require_cagr_1y=False,
            require_cagr_3y=False,
            require_cagr_5y=True,
            min_cagr_5y=5.0,
            cagr_5y_uses_historical_growth=True,
            require_cagr_10y=False,
            max_lt_debt_capital=50.0,
            min_est_annual_ror=5.0,
            price_below_fair=True,
            max_vs_fair_pct=6.0,
            require_pe_lte_normal=True,
            min_eps_persistence_pct=70.0,
            horizon_years=horizon_years,
            sidebar_fair_pe=sidebar_fair_pe,
            valuation_pe_mode=valuation_pe_mode,
        )

    @classmethod
    def cpfs_strict(
        cls,
        *,
        countries: tuple[str, ...] = ("United States", "Canada"),
        exclude_otc: bool = True,
        min_est_eps_growth: float = 10.0,
        require_cagr_1y: bool = True,
        require_cagr_3y: bool = True,
        require_cagr_10y: bool = True,
        require_analyst_forward_growth: bool = True,
        require_recent_yoy_positive: bool = False,
        ror_gte_growth: bool = False,
        max_lt_debt_capital: float = 55.0,
        price_below_fair: bool = True,
        horizon_years: int = 3,
        sidebar_fair_pe: float = 15.0,
        valuation_pe_mode: str = "fair",
    ) -> FastGraphFilterConfig:
        """Original CPFS-G strict screen (1Y+3Y+10Y CAGR, vs Fair < 0, forward ≥ min)."""
        return cls(
            countries=countries,
            exclude_otc=exclude_otc,
            min_est_eps_growth=min_est_eps_growth,
            require_cagr_1y=require_cagr_1y,
            require_cagr_3y=require_cagr_3y,
            require_cagr_5y=False,
            require_cagr_10y=require_cagr_10y,
            require_analyst_forward_growth=require_analyst_forward_growth,
            require_recent_yoy_positive=require_recent_yoy_positive,
            ror_gte_growth=ror_gte_growth,
            max_lt_debt_capital=max_lt_debt_capital,
            min_est_annual_ror=0.0,
            price_below_fair=price_below_fair,
            max_vs_fair_pct=None,
            require_pe_lte_normal=False,
            min_eps_persistence_pct=0.0,
            horizon_years=horizon_years,
            sidebar_fair_pe=sidebar_fair_pe,
            valuation_pe_mode=valuation_pe_mode,
        )


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

    val_eps = metrics.get("valuation_eps")
    if val_eps is None:
        return False, "NO_EPS"
    try:
        if float(val_eps) <= 0:
            return False, "NEGATIVE_EPS"
    except (TypeError, ValueError):
        return False, "NO_EPS"

    fair_price = metrics.get("fair_price")
    if sanitize_display_price(fair_price) is None:
        return False, "NO_EPS"

    vs_fair = metrics.get("vs_fair_pct")
    if cfg.price_below_fair:
        if vs_fair is None:
            return False, "VS_FAIR"
        max_allowed = cfg.max_vs_fair_pct if cfg.max_vs_fair_pct is not None else 0.0
        try:
            if float(vs_fair) > max_allowed:
                return False, "NOT_BELOW_FAIR" if max_allowed <= 0 else "VS_FAIR"
        except (TypeError, ValueError):
            return False, "VS_FAIR"

    if cfg.require_pe_lte_normal:
        blended = metrics.get("blended_pe")
        normal = metrics.get("historical_normal_pe") or metrics.get("normal_pe")
        if normal is not None:
            try:
                npe = float(normal)
                if npe > 0:
                    if blended is None:
                        close = metrics.get("close")
                        veps = metrics.get("valuation_eps")
                        if close is not None and veps is not None:
                            try:
                                blended = float(close) / float(veps)
                            except (TypeError, ValueError, ZeroDivisionError):
                                blended = None
                    if blended is not None and float(blended) > npe:
                        return False, "PE_GT_NORMAL"
            except (TypeError, ValueError):
                pass

    if cfg.min_eps_persistence_pct > 0:
        persist = metrics.get("eps_persistence_pct")
        if persist is None or float(persist) < cfg.min_eps_persistence_pct:
            return False, "EPS_PERSISTENCE"

    est_growth = metrics.get("est_eps_growth")
    fwd_growth = metrics.get("forward_eps_growth")
    if cfg.require_analyst_forward_growth:
        growth_for_filter = fwd_growth
    else:
        growth_for_filter = est_growth if est_growth is not None else fwd_growth
    if cfg.min_est_eps_growth > 0:
        if cfg.require_analyst_forward_growth and fwd_growth is None:
            return False, "NO_FORWARD_GROWTH"
        if growth_for_filter is None or growth_for_filter < cfg.min_est_eps_growth:
            return False, "EST_GROWTH"

    cagr_checks = [
        (cfg.require_cagr_1y, "cagr_1y", cfg.min_cagr_1y, False),
        (cfg.require_cagr_3y, "cagr_3y", cfg.min_cagr_3y, False),
        (cfg.require_cagr_5y, "cagr_5y", cfg.min_cagr_5y, cfg.cagr_5y_uses_historical_growth),
        (cfg.require_cagr_10y, "cagr_10y", cfg.min_cagr_10y, False),
    ]
    for required, key, minimum, use_hist in cagr_checks:
        if not required:
            continue
        if use_hist:
            val = metrics.get("historical_growth_rate") or metrics.get("growth_rate")
        else:
            val = metrics.get(key)
        if val is None or val < minimum:
            return False, key.upper()

    if cfg.require_recent_yoy_positive:
        recent = metrics.get("recent_yoy_pct") or []
        needed = cfg.min_recent_yoy_years
        if len(recent) < needed or any(float(y) < 0 for y in recent[-needed:]):
            return False, "RECENT_YOY_DECLINE"

    lt_debt = metrics.get("lt_debt_capital")
    if cfg.max_lt_debt_capital > 0 and lt_debt is not None and lt_debt > cfg.max_lt_debt_capital:
        return False, "DEBT_CAP"

    ror = metrics.get("est_annual_ror")
    if cfg.min_est_annual_ror > 0:
        if ror is None or ror < cfg.min_est_annual_ror:
            return False, "MIN_ROR"

    if cfg.ror_gte_growth and growth_for_filter is not None and ror is not None:
        if ror < growth_for_filter:
            return False, "ROR_LT_GROWTH"

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
    historical_growth = compute_historical_growth_rate_pct(
        annual_eps,
        years=cfg.growth_years,
    )
    forecast_growth = compute_forecast_growth_pct(
        earnings_estimates,
        annual_eps,
        historical_growth,
    )
    forward_eps_growth = compute_forward_eps_growth_pct(
        earnings_estimates,
        annual_eps,
    )
    recent_yoy = recent_reported_yoy_pct(annual_eps, n=cfg.min_recent_yoy_years)

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
        historical_growth,
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

    valuation_eps, valuation_eps_basis = resolve_valuation_eps(
        annual_eps,
        trailing_eps,
        earnings_estimates,
    )
    filtered_eps = _filtered_annual_eps(annual_eps)

    blended = _blended_pe(trailing_pe, forward_pe)
    if blended is None and valuation_eps:
        blended = pe_ttm(close, valuation_eps)

    norm_pe_val = norm_pe if norm_pe is not None else 0.0
    row_m = eps_row_metrics(close, valuation_eps, fair_pe=fair_pe, norm_pe=norm_pe_val)
    fair_price = sanitize_display_price(row_m.get("Fair $"))
    normal_price = sanitize_display_price(row_m.get("Normal $"))
    vs_fair = row_m.get("vs Fair %") if fair_price is not None else None
    if vs_fair is not None:
        try:
            vs_fair = round(float(vs_fair), 2)
        except (TypeError, ValueError):
            vs_fair = None

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

    eps_persistence = compute_eps_persistence_pct(annual_eps)
    pe_vs_normal = None
    if blended is not None and norm_pe is not None:
        try:
            npe = float(norm_pe)
            if npe > 0:
                pe_vs_normal = round(float(blended) / npe, 2)
        except (TypeError, ValueError):
            pe_vs_normal = None

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
        "valuation_eps": valuation_eps,
        "valuation_eps_basis": valuation_eps_basis,
        "eps_yield": eps_yield_pct(valuation_eps, close),
        "fair_price": fair_price,
        "normal_price": normal_price,
        "vs_fair_pct": vs_fair,
        "est_eps_growth": est_eps_growth,
        "forward_eps_growth": forward_eps_growth,
        "recent_yoy_pct": recent_yoy,
        "est_annual_ror": est_ror,
        "future_price": future_price,
        "future_eps": round(future_eps, 4) if future_eps else None,
        "lt_debt_capital": lt_debt_capital,
        "cagr_1y": eps_cagr_over_years(filtered_eps, 1),
        "cagr_3y": eps_cagr_over_years(filtered_eps, 3),
        "cagr_5y": eps_cagr_over_years(filtered_eps, 5),
        "cagr_10y": eps_cagr_over_years(filtered_eps, 10),
        "eps_persistence_pct": eps_persistence,
        "pe_vs_normal": pe_vs_normal,
        "country": info.get("country"),
        "industry": info.get("industry"),
        "market_cap": info.get("market_cap"),
        "exchange": info.get("exchange"),
        "analyst_beat_pct": beat_pct,
        "annual_eps": annual_eps or {},
        "eps_source": info.get("eps_source"),
        "earnings_estimates": earnings_estimates or {},
        "trailing_eps": trailing_eps,
        "forward_eps": forward_eps,
        "dividend_yield_pct": info.get("dividend_yield_pct"),
    }
    return metrics
