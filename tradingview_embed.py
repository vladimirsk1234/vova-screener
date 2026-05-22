"""
Preview helpers: TradingView URL builders and a native Plotly chart that mirrors
the Sequence Vova Pine indicator. The chart is rendered in-page from Yahoo data
running the same Python port used by the scanner.
"""
from __future__ import annotations

import re
from urllib.parse import urlencode, urlparse, parse_qs

import pandas as pd
import streamlit as st
import yfinance as yf

from data_utils import (
    fill_last_bar_ohlc,
    interval_and_period,
    resample_to_timeframe,
)
from sequence_vova import run_sequence_vova_full
from vova_chart import build_sequence_vova_figure, compute_chart_layers

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


def build_ideas_url(tv_symbol: str) -> str:
    """Symbol ideas / community page on TradingView."""
    sym = normalize_tv_symbol(tv_symbol)
    slug = sym.replace(":", "-")
    return f"https://www.tradingview.com/symbols/{slug}/ideas/"


def yahoo_ticker_from_tv_symbol(tv_symbol: str) -> str:
    """Strip EXCHANGE: prefix and switch dots to dashes for Yahoo."""
    s = normalize_tv_symbol(tv_symbol)
    if ":" in s:
        s = s.split(":", 1)[1]
    return s.replace(".", "-")


_REQ_COLS = ["Open", "High", "Low", "Close", "Volume"]


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_ohlcv(yahoo_ticker: str, tf: str) -> pd.DataFrame | None:
    """Single-ticker OHLCV fetch + resample to chart timeframe."""
    if not yahoo_ticker:
        return None
    inter, period = interval_and_period(tf)
    try:
        df = yf.download(
            yahoo_ticker,
            period=period,
            interval=inter,
            progress=False,
            auto_adjust=False,
            multi_level_index=False,
        )
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if not all(c in df.columns for c in _REQ_COLS):
        return None
    df = df[_REQ_COLS].copy()
    df = resample_to_timeframe(df, tf)
    if df is None or df.empty:
        return None
    df = fill_last_bar_ohlc(df)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


def render_symbol_preview(
    tv_symbol: str,
    tf: str,
    *,
    yahoo_ticker: str = "",
    company_name: str = "",
    tp: float | None = None,
    sl: float | None = None,
    risk_per_trade: int = 100,
    min_rr: float = 1.5,
    use_last_hl_sl: bool = True,
) -> None:
    """Native Plotly preview drawing the real Sequence Vova outputs."""
    sym = normalize_tv_symbol(tv_symbol)
    interval = tf_to_tv_interval(tf)
    chart_url = build_chart_url(sym, interval)
    ideas_url = build_ideas_url(sym)
    yahoo = yahoo_ticker or yahoo_ticker_from_tv_symbol(sym)

    title = company_name.strip() or sym
    st.subheader(f"Preview: {title}")
    st.caption(
        f"Native chart drawn from Python with your real Sequence Vova logic at **{tf}** "
        "timeframe. Use **Open full chart** for your published Pine on TradingView."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.link_button("Open full chart on TradingView", chart_url, use_container_width=True)
    with c2:
        st.link_button("Open ideas & community", ideas_url, use_container_width=True)

    if tp is not None and sl is not None:
        st.caption(
            f"Screener levels — TP: **{tp}** · SL: **{sl}** (drawn as horizontal lines below)."
        )

    with st.spinner(f"Loading {yahoo} {tf} data..."):
        df = _fetch_ohlcv(yahoo, tf)
    if df is None or df.empty or len(df) < 50:
        st.warning(f"Could not load enough OHLCV bars for {yahoo} ({tf}).")
        return

    try:
        seq_full = run_sequence_vova_full(
            df,
            min_rr=float(min_rr),
            use_last_hl_sl=bool(use_last_hl_sl),
            risk_dollars=int(risk_per_trade),
        )
        layers = compute_chart_layers(df)
        fig = build_sequence_vova_figure(
            df, seq_full, layers,
            tp=tp, sl=sl,
            title=f"{sym} — {tf}",
            height=800,
        )
    except Exception as e:
        st.error(f"Chart render failed: {type(e).__name__}: {e}")
        return

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
    )
