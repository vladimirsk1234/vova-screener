"""
TradingView embed helpers: interval mapping, URLs, and Streamlit preview panel.
No scan logic; used by headless_scanner results UI.
"""
from __future__ import annotations

import json
import re
from urllib.parse import urlencode, urlparse, parse_qs

import streamlit as st
import streamlit.components.v1 as components
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


def build_ideas_url(tv_symbol: str) -> str:
    """Symbol ideas / community page on TradingView."""
    sym = normalize_tv_symbol(tv_symbol)
    slug = sym.replace(":", "-")
    return f"https://www.tradingview.com/symbols/{slug}/ideas/"


_FULL_DOC_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><style>
  html, body {{ height: 100%; width: 100%; margin: 0; padding: 0; background: #0e1116; overflow: hidden; }}
  .tradingview-widget-container {{ height: {h}px !important; width: 100% !important; }}
  .tradingview-widget-container__widget {{ height: {h}px !important; width: 100% !important; }}
  .tradingview-widget-container iframe {{ height: {h}px !important; width: 100% !important; }}
</style></head><body>
<div class="tradingview-widget-container" style="height:{h}px;width:100%;">
  <div class="tradingview-widget-container__widget" style="height:{h}px;width:100%;"></div>
  <script type="text/javascript" src="{src}" async>{config}</script>
</div>
</body></html>"""


def _widget_doc(src: str, config: dict, height: int) -> str:
    return _FULL_DOC_TEMPLATE.format(
        src=src,
        config=json.dumps(config, separators=(",", ":")),
        h=int(height),
    )


def render_advanced_chart_html(tv_symbol: str, interval: str, height: int = 720) -> str:
    sym = normalize_tv_symbol(tv_symbol)
    cfg = {
        "width": "100%",
        "height": int(height),
        "symbol": sym,
        "interval": interval,
        "timezone": "exchange",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "allow_symbol_change": False,
        "calendar": False,
        "hide_side_toolbar": False,
        "support_host": "https://www.tradingview.com",
    }
    return _widget_doc(
        "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js",
        cfg,
        height,
    )


def render_timeline_html(tv_symbol: str, height: int = 520) -> str:
    """Timeline widget: symbol-specific ideas and news (not in-chart chat)."""
    sym = normalize_tv_symbol(tv_symbol)
    cfg = {
        "feedMode": "symbol",
        "symbol": sym,
        "colorTheme": "dark",
        "isTransparent": False,
        "displayMode": "regular",
        "width": "100%",
        "height": int(height),
        "locale": "en",
    }
    return _widget_doc(
        "https://s3.tradingview.com/external-embedding/embed-widget-timeline.js",
        cfg,
        height,
    )


def render_symbol_preview(
    tv_symbol: str,
    tf: str,
    *,
    company_name: str = "",
    tp: float | None = None,
    sl: float | None = None,
) -> None:
    """Embedded chart + ideas feed and links for one selected scan row."""
    sym = normalize_tv_symbol(tv_symbol)
    interval = tf_to_tv_interval(tf)
    chart_url = build_chart_url(sym, interval)
    ideas_url = build_ideas_url(sym)

    title = company_name.strip() or sym
    st.subheader(f"Preview: {title}")
    st.caption(
        f"Candles at **{tf}** ({interval}) — same timeframe as the scan. "
        "Sequence Vova / custom Pine is not shown in the embed; use **Open full chart** "
        "(log in on TradingView for indicators and in-chart chat)."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.link_button("Open full chart on TradingView", chart_url, use_container_width=True)
    with c2:
        st.link_button("Open ideas & community", ideas_url, use_container_width=True)

    if tp is not None and sl is not None:
        st.caption(f"Screener levels — TP: **{tp}** · SL: **{sl}** (from Yahoo data + Sequence Vova, not drawn on embed).")

    tab_chart, tab_social = st.tabs(["Chart", "Ideas & news"])

    with tab_chart:
        chart_h = st.slider(
            "Chart height (px)",
            min_value=400,
            max_value=1400,
            value=760,
            step=20,
            key=f"tv_chart_h_{sym}",
        )
        components.html(
            render_advanced_chart_html(sym, interval, height=chart_h),
            height=chart_h + 8,
            scrolling=False,
        )

    with tab_social:
        st.caption(
            "Headlines and symbol feed from TradingView. Live **chat** is only on the full chart page when you are logged in."
        )
        components.html(render_timeline_html(sym, height=520), height=540, scrolling=True)
