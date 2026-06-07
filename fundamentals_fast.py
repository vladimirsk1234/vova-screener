"""
FAST Graphs–style fundamental panel data from Yahoo Finance (yfinance). No Streamlit.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd
import yfinance as yf

from eps_yield import (
    avg_historical_pe_5y,
    eps_row_metrics,
    eps_yield_pct,
    pe_ttm,
)
from ticker_data import (
    _days_to_earnings,
    _eps_from_yf_info,
    _float_field,
    _resolve_fundamentals_info,
    get_annual_eps_history_5y,
)


def _highlights_from_scanner_metrics(
    scanner_metrics: dict[str, Any],
    *,
    chart_mode: str = "historical",
) -> list[dict[str, str]] | None:
    """Build highlight boxes from scanner metrics (mode-aware)."""
    if not scanner_metrics:
        return None
    if chart_mode == "forecast":
        growth = (
            scanner_metrics.get("chart_forecast_growth_rate")
            or scanner_metrics.get("forecast_growth_rate")
        )
        fair = scanner_metrics.get("forecast_fair_pe")
        normal = scanner_metrics.get("forecast_normal_pe")
    else:
        growth = (
            scanner_metrics.get("historical_growth_rate")
            or scanner_metrics.get("growth_rate")
        )
        fair = scanner_metrics.get("historical_fair_pe") or scanner_metrics.get("fair_pe")
        normal = scanner_metrics.get("historical_normal_pe") or scanner_metrics.get("normal_pe")
    if growth is None and fair is None and normal is None:
        return None
    return [
        {
            "label": "Growth Rate",
            "value": _fmt_pct(growth) if growth is not None else "N/A",
            "css": "growth",
        },
        {
            "label": "Fair Value Ratio",
            "value": _fmt_ratio(float(fair)) if fair is not None else "N/A",
            "css": "fair",
        },
        {
            "label": "Normal P/E Ratio",
            "value": _fmt_ratio(normal) if normal is not None else "N/A",
            "css": "normal",
        },
    ]


def format_mcap_tev(value: float | None, currency: str | None = None) -> str:
    """Format market cap / TEV with currency prefix, e.g. CAD 307.94M."""
    if value is None:
        return "N/A"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(v) or v <= 0:
        return "N/A"
    cur = str(currency or "USD").strip().upper() or "USD"
    if v >= 1e12:
        return f"{cur} {v / 1e12:.2f}T"
    if v >= 1e9:
        return f"{cur} {v / 1e9:.2f}B"
    if v >= 1e6:
        return f"{cur} {v / 1e6:.2f}M"
    if v >= 1e3:
        return f"{cur} {v / 1e3:.2f}K"
    return f"{cur} {v:.2f}"


def _fmt_ratio(val: float | None, *, decimals: int = 2) -> str:
    if val is None:
        return "N/A"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(v):
        return "N/A"
    return f"{v:.{decimals}f}x"


def _fmt_pct(val: float | None, *, decimals: int = 2) -> str:
    if val is None:
        return "N/A"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(v):
        return "N/A"
    return f"{v:.{decimals}f}%"


def _yield_pct_from_yahoo(raw: float | None, *, close: float | None = None, dividend_rate: float | None = None) -> float | None:
    """
    Normalize Yahoo yield fields (decimal or percent) to a percent value.
    Falls back to dividendRate / close when the primary field looks invalid.
    """
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


def _fmt_pct_from_decimal(val: float | None, *, decimals: int = 2) -> str:
    """Format a value already normalized to percent."""
    if val is None:
        return "N/A"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(v):
        return "N/A"
    return f"{v:.{decimals}f}%"


def _blended_pe(trailing_pe: float | None, forward_pe: float | None) -> float | None:
    vals = [v for v in (trailing_pe, forward_pe) if v is not None and math.isfinite(v) and v > 0]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)


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


def _info_field_str(info: dict, *keys: str) -> str | None:
    for key in keys:
        val = info.get(key)
        if val is None or val == "":
            continue
        return str(val).strip()
    return None


def _extended_metrics(
    info: dict,
    merged: dict,
    *,
    ticker_obj: yf.Ticker | None,
    close: float | None,
    trailing_eps: float | None,
    earn_days: str,
    fair_pe: float,
    norm_pe: float | None,
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []

    def add(label: str, raw_key: str, fmt: str = "raw") -> None:
        val = _float_field(info, raw_key) if raw_key in info else merged.get(raw_key)
        if val is None and raw_key in merged:
            val = merged.get(raw_key)
        if val is None:
            v_str = _info_field_str(info, raw_key)
            if v_str:
                rows.append((label, v_str))
            return
        if fmt == "pct":
            pct = _yield_pct_from_yahoo(val)
            rows.append((label, _fmt_pct(pct)))
        elif fmt == "ratio":
            rows.append((label, _fmt_ratio(val)))
        elif fmt == "pct_growth":
            rows.append((label, _fmt_pct_from_decimal(val)))
        else:
            try:
                rows.append((label, f"{float(val):.4g}"))
            except (TypeError, ValueError):
                rows.append((label, str(val)))

    add("Trailing P/E", "trailingPE", "ratio")
    add("Forward P/E", "forwardPE", "ratio")
    add("PEG Ratio", "pegRatio", "ratio")
    add("Price/Book", "priceToBook", "ratio")
    add("Price/Sales (TTM)", "priceToSalesTrailing12Months", "ratio")
    add("Profit Margin", "profitMargins", "pct")
    add("Operating Margin", "operatingMargins", "pct")
    add("Gross Margin", "grossMargins", "pct")
    add("ROE", "returnOnEquity", "pct")
    add("ROA", "returnOnAssets", "pct")
    add("Revenue Growth", "revenueGrowth", "pct_growth")
    add("Earnings Growth", "earningsGrowth", "pct_growth")
    add("Beta", "beta", "raw")
    add("52-Week Change", "52WeekChange", "pct_growth")
    add("Book Value", "bookValue", "raw")
    add("Total Cash", "totalCash", "raw")
    add("Total Debt", "totalDebt", "raw")
    add("Payout Ratio", "payoutRatio", "pct")
    add("Full-Time Employees", "fullTimeEmployees", "raw")

    sector = _info_field_str(info, "sector", "sectorDisp")
    if sector:
        rows.append(("Sector", sector))
    exchange = _info_field_str(info, "exchange", "fullExchangeName")
    if exchange:
        rows.append(("Exchange", exchange))
    if close is not None:
        rows.append(("Close (scan bar)", f"{close:.2f}"))
    if trailing_eps is not None:
        rows.append(("Valuation EPS", f"{trailing_eps:.4f}"))
    rows.append(("Next Earnings", earn_days))

    metrics = (
        eps_row_metrics(
            float(close),
            trailing_eps,
            fair_pe=fair_pe,
            norm_pe=norm_pe if norm_pe is not None else 0.0,
        )
        if close is not None
        else {}
    )
    for key in ("Fair $", "Normal $", "vs Fair %"):
        if key in metrics:
            rows.append((key, str(metrics[key])))

    return rows


def get_fast_graph_panel_data(
    ticker: str,
    *,
    close: float | None = None,
    df_daily: pd.DataFrame | None = None,
    fair_pe: float = 15.0,
    scanner_metrics: dict[str, Any] | None = None,
    chart_mode: str = "historical",
) -> dict[str, Any]:
    """
    Build panel payload: highlights (3 boxes), details table, extended Yahoo fields.
    When scanner_metrics is provided, highlight boxes use scanner values (mode-aware).
    """
    eps_source = (scanner_metrics or {}).get("eps_source") or "yahoo"
    eps_label = {
        "sec_operating": "SEC operating income / shares",
        "sec": "SEC GAAP diluted/basic EPS",
        "yahoo_annual": "Yahoo annual EPS",
        "yahoo_quarterly": "Yahoo quarterly EPS (aggregated)",
    }.get(eps_source, eps_source)
    warnings = [
        "Данные SEC Operating EPS (free proxy) + Yahoo — не FAST Graphs Premium Adjusted EPS.",
        f"Growth Rate — Historical CAGR ({eps_label}, без выбросов).",
        "Normal P/E — split-adjusted year-end prices (auto_adjust).",
        "Est EPS Growth — отдельно, из Yahoo analyst estimates.",
        "GICS Sub-industry — поле industry Yahoo.",
        "S&P Credit Rating недоступен в Yahoo.",
    ]

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

    currency = _info_field_str(info, "currency") or "USD"
    company_name = str(merged.get("company_name") or info.get("longName") or ticker)

    px = close
    if px is None:
        px = merged.get("regularMarketPrice") or merged.get("currentPrice")
    if px is not None:
        try:
            px = float(px)
        except (TypeError, ValueError):
            px = None

    trailing_eps, forward_eps = _eps_from_yf_info(merged)
    if trailing_eps is None:
        trailing_eps, forward_eps = _eps_from_yf_info(info)

    valuation_eps = scanner_metrics.get("valuation_eps") if scanner_metrics else None
    display_eps = valuation_eps if valuation_eps is not None else trailing_eps

    annual_eps = get_annual_eps_history_5y(ticker)
    norm_pe = avg_historical_pe_5y(df_daily, annual_eps)

    if scanner_metrics:
        display_fair_pe = float(
            scanner_metrics.get("forecast_fair_pe" if chart_mode == "forecast" else "historical_fair_pe")
            or scanner_metrics.get("fair_pe")
            or fair_pe
        )
        display_norm_pe = (
            scanner_metrics.get("forecast_normal_pe" if chart_mode == "forecast" else "historical_normal_pe")
            or scanner_metrics.get("normal_pe")
            or norm_pe
        )
        if scanner_metrics.get("lt_debt_capital") is not None:
            lt_debt_cap = scanner_metrics.get("lt_debt_capital")
    else:
        display_fair_pe = float(fair_pe)
        display_norm_pe = norm_pe

    trailing_pe = _float_field(info, "trailingPE") or _float_field(merged, "trailingPE")
    forward_pe = _float_field(info, "forwardPE") or _float_field(merged, "forwardPE")
    blended = _blended_pe(trailing_pe, forward_pe)
    if blended is None and px is not None and display_eps:
        blended = pe_ttm(px, display_eps)

    eps_yld = eps_yield_pct(display_eps, px) if px is not None else None
    div_yld = _yield_pct_from_yahoo(
        _float_field(info, "dividendYield"),
        close=px,
        dividend_rate=_float_field(info, "dividendRate"),
    )

    mcap = _float_field(info, "marketCap") or _float_field(merged, "marketCap")
    tev = _float_field(info, "enterpriseValue")
    if scanner_metrics is None or scanner_metrics.get("lt_debt_capital") is None:
        lt_debt_cap = _lt_debt_to_capital_pct(ticker_obj)

    country = _info_field_str(info, "country") or "—"
    industry = (
        _info_field_str(info, "industryDisp", "industry")
        or "—"
    )
    quote_type = _info_field_str(info, "quoteType") or "SHARE"
    if quote_type.upper() == "EQUITY":
        quote_type = "SHARE"

    earn_days = _days_to_earnings(ticker_obj) if ticker_obj is not None else "N/A"

    scanner_highlights = _highlights_from_scanner_metrics(scanner_metrics, chart_mode=chart_mode)
    if scanner_highlights:
        highlights = scanner_highlights
    else:
        from fast_graph_metrics import compute_historical_growth_rate_pct

        growth_pct = compute_historical_growth_rate_pct(annual_eps)
        highlights = [
            {
                "label": "Growth Rate",
                "value": _fmt_pct(growth_pct) if growth_pct is not None else "N/A",
                "css": "growth",
            },
            {
                "label": "Fair Value Ratio",
                "value": _fmt_ratio(display_fair_pe),
                "css": "fair",
            },
            {
                "label": "Normal P/E Ratio",
                "value": _fmt_ratio(display_norm_pe) if display_norm_pe is not None else "N/A",
                "css": "normal",
            },
        ]

    details: list[tuple[str, str]] = [
        ("Blended P/E", _fmt_ratio(blended)),
        ("EPS Yld", _fmt_pct(eps_yld)),
        ("Div Yld", _fmt_pct(div_yld) if div_yld is not None else "0.00%"),
        ("S&P Credit Rating", "N/A"),
        ("Market Cap", format_mcap_tev(mcap, currency)),
        ("TEV", format_mcap_tev(tev, currency)),
        (
            "LT Debt/Capital",
            _fmt_pct(lt_debt_cap) if lt_debt_cap is not None else "N/A",
        ),
        ("Country", country),
        ("GICS Sub-industry", industry),
        ("Type", quote_type.upper()),
    ]

    extended = _extended_metrics(
        info,
        merged,
        ticker_obj=ticker_obj,
        close=px,
        trailing_eps=display_eps,
        earn_days=earn_days,
        fair_pe=display_fair_pe,
        norm_pe=display_norm_pe if display_norm_pe is not None else norm_pe,
    )

    return {
        "ticker": ticker,
        "company_name": company_name,
        "currency": currency,
        "highlights": highlights,
        "details": details,
        "extended": extended,
        "warnings": warnings,
        "fair_pe": display_fair_pe,
        "chart_mode": chart_mode,
    }
