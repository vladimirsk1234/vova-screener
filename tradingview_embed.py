"""
TradingView link helpers: interval mapping and symbol/URL building.
Used by the scanner to put a clickable link in each result row.
"""
from __future__ import annotations

import re
from urllib.parse import urlencode, urlparse, parse_qs

import yfinance as yf

# Yahoo / yfinance exchange codes -> TradingView exchange prefix
_YAHOO_EXCHANGE_TO_TV: dict[str, str] = {
    "NMS": "NASDAQ",
    "NGM": "NASDAQ",
    "NCM": "NASDAQ",
    "NAS": "NASDAQ",
    "NASDAQ": "NASDAQ",
    "NASDAQGS": "NASDAQ",
    "NASDAQCM": "NASDAQ",
    "NASDAQGM": "NASDAQ",
    "NYQ": "NYSE",
    "NYS": "NYSE",
    "NYSE": "NYSE",
    "ASE": "AMEX",
    "AMEX": "AMEX",
    "BTS": "BATS",
    "BAT": "BATS",
    "BATS": "BATS",
    "PCX": "ARCA",
    "ARCA": "ARCA",
}


def tf_to_tv_interval(tf: str) -> str:
    """Map screener timeframe to TradingView interval param."""
    return {"Daily": "D", "Weekly": "W", "Monthly": "M"}.get(tf, "D")


def normalize_tv_symbol(symbol: str) -> str:
    """Ensure EXCHANGE:TICKER form when possible; strip whitespace."""
    s = str(symbol or "").strip()
    if not s:
        return "NASDAQ:AAPL"
    if s.startswith("http"):
        try:
            qs = parse_qs(urlparse(s).query)
            if qs.get("symbol"):
                s = qs["symbol"][0]
        except Exception:
            m = re.search(r"symbol=([^&]+)", s)
            if m:
                s = m.group(1)
    return s.upper() if ":" in s else s.upper()


def infer_tv_symbol(yahoo_ticker: str) -> str:
    """Build EXCHANGE:SYMBOL for manual tickers via Yahoo metadata."""
    t = str(yahoo_ticker or "").strip().upper()
    if not t:
        return t
    if ":" in t:
        return t
    try:
        ex = (yf.Ticker(t).info.get("exchange") or "").upper()
        tv_ex = _YAHOO_EXCHANGE_TO_TV.get(ex)
        if tv_ex:
            return f"{tv_ex}:{t}"
    except Exception:
        pass
    return t


def build_chart_url(tv_symbol: str, interval: str, layout_id: str | None = None) -> str:
    """Deep link to TradingView chart with symbol and scan timeframe."""
    sym = normalize_tv_symbol(tv_symbol)
    if layout_id:
        base = f"https://www.tradingview.com/chart/{layout_id.strip()}/"
    else:
        base = "https://www.tradingview.com/chart/"
    return f"{base}?{urlencode({'symbol': sym, 'interval': interval})}"
