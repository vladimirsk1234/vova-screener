"""
EPS yield and FAST Graphs–style fair value (eps * P/E). Pure math; no I/O.
Aligns with Pine: eps_yield_pct = (eps / close) * 100.
"""
from __future__ import annotations
import math
import pandas as pd


def eps_yield_pct(eps: float | None, close: float | None) -> float | None:
    """(eps / close) * 100 when eps and close are valid and close > 0."""
    if eps is None or close is None:
        return None
    try:
        e, c = float(eps), float(close)
    except (TypeError, ValueError):
        return None
    if c <= 0:
        return None
    return (e / c) * 100.0


def pe_ttm(close: float | None, eps: float | None) -> float | None:
    """Price / EPS when eps != 0."""
    if eps is None or close is None:
        return None
    try:
        e, c = float(eps), float(close)
    except (TypeError, ValueError):
        return None
    if e == 0:
        return None
    return c / e


def fair_and_normal_price(eps: float | None, fair_pe: float, norm_pe: float) -> tuple[float | None, float | None]:
    """Fair value = eps * fair_pe; normal P/E line = eps * norm_pe (Pine overlay)."""
    if eps is None:
        return None, None
    try:
        e = float(eps)
    except (TypeError, ValueError):
        return None, None
    if e <= 0:
        return None, None
    return e * float(fair_pe), e * float(norm_pe)


def vs_fair_pct(close: float | None, fair_value: float | None) -> float | None:
    """(close - fair) / fair * 100 when fair != 0."""
    if close is None or fair_value is None:
        return None
    try:
        c, f = float(close), float(fair_value)
    except (TypeError, ValueError):
        return None
    if f <= 0:
        return None
    return ((c - f) / f) * 100.0


def sanitize_display_price(val: float | None) -> float | None:
    """Table/chart dollar targets must be positive finite values."""
    if val is None:
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0:
        return None
    return round(v, 2)


def eps_row_metrics(
    close: float,
    eps: float | None,
    fair_pe: float,
    norm_pe: float,
) -> dict:
    """
    Build optional display columns for one symbol. Omits keys with no value where appropriate.
    `eps` should be valuation_eps (latest positive annual EPS, or positive TTM fallback).
    """
    y = eps_yield_pct(eps, close)
    pe = pe_ttm(close, eps)
    fair, normal = fair_and_normal_price(eps, fair_pe, norm_pe)
    vs_fair = vs_fair_pct(close, fair)

    out: dict = {}
    if y is not None:
        out["EPS Yield %"] = round(y, 2)
    if pe is not None:
        out["P/E (TTM)"] = round(pe, 2)
    if fair is not None:
        out["Fair $"] = round(fair, 2)
    if normal is not None:
        out["Normal $"] = round(normal, 2)
    if vs_fair is not None:
        out["vs Fair %"] = round(vs_fair, 2)
    out["Close"] = round(float(close), 2)
    return out


def passes_eps_filters(
    eps_yield: float | None,
    eps: float | None,
    *,
    min_eps_yield: float | None,
    max_eps_yield: float | None,
    require_eps: bool,
    include_negative_eps: bool,
) -> bool:
    """
    Screening rules for EPS yield modes.
    - If require_eps: must have finite eps; if not include_negative_eps, eps must be > 0.
    - Yield min/max: applied when yield is computed; missing yield fails if require_eps.
    """
    if eps is None or eps != eps:  # nan
        if require_eps:
            return False
        # no EPS data: cannot satisfy yield bounds meaningfully
        if min_eps_yield is not None or max_eps_yield is not None:
            return False
        return True

    if not include_negative_eps and eps <= 0:
        return False

    if eps_yield is None:
        return not require_eps and min_eps_yield is None and max_eps_yield is None

    if min_eps_yield is not None and eps_yield < min_eps_yield:
        return False
    if max_eps_yield is not None and eps_yield > max_eps_yield:
        return False
    return True


MAX_YEARLY_PE = 150.0


def avg_historical_pe_5y(price_df: pd.DataFrame | None, annual_eps_by_year: dict[int, float] | None) -> float | None:
    """
    Average of yearly P/E values over the last 5 fiscal years with positive EPS.
    Uses last available close in each matching calendar year.
    Caps extreme yearly P/E (split/EPS mismatch) before averaging.
    """
    if price_df is None or not isinstance(price_df, pd.DataFrame) or price_df.empty:
        return None
    if "Close" not in price_df.columns or not annual_eps_by_year:
        return None

    closes = pd.to_numeric(price_df["Close"], errors="coerce").dropna()
    if closes.empty:
        return None
    year_end_close = closes.groupby(closes.index.year).last()

    positive_years = sorted(
        y for y, e in annual_eps_by_year.items()
        if e is not None and float(e) > 0
    )
    if not positive_years:
        return None
    window_years = positive_years[-5:]

    pe_values: list[float] = []
    for year in window_years:
        try:
            e = float(annual_eps_by_year[year])
        except (TypeError, ValueError):
            continue
        if e <= 0:
            continue
        if year not in year_end_close.index:
            continue
        price = float(year_end_close.loc[year])
        pe = price / e
        if math.isfinite(pe) and pe > 0:
            pe_values.append(min(pe, MAX_YEARLY_PE))

    if not pe_values:
        return None
    if len(pe_values) >= 5:
        pe_values = sorted(pe_values)[1:-1]
    return round(sum(pe_values) / len(pe_values), 2)
