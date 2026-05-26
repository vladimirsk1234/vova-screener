"""
Terminal UI styles for the screener. Injects dark-theme CSS via Streamlit.
"""
import html
import json
import re
from urllib.parse import parse_qs, urlparse

import streamlit as st
import streamlit.components.v1 as components

from chart_preview import chart_json_for_mobile

_STYLES = """
<style>
    /* GLOBAL DARK THEME */
    .stApp { background-color: #050505; }
    /* Main pane: room for tall results table */
    section[data-testid="stMain"] {
        min-height: calc(100vh - 6rem);
    }

    /* FIX: Top padding to prevent header overlap; iPhone safe area */
    .block-container {
        padding-top: max(4rem, env(safe-area-inset-top, 0px)) !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    /* Dual view: mobile cards vs desktop dataframe */
    .mobile-results { display: none; }
    @media (max-width: 768px) {
        .mobile-results { display: block; }
        div[data-testid="stDataFrame"] { display: none !important; }
        .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
    }
    @media (min-width: 769px) {
        .mobile-results { display: none !important; }
        .mobile-only-chart-component { display: none !important; }
        div[data-testid="stIFrame"]:has(iframe[srcdoc*="chart-hover-popup"]) { display: none !important; }
    }

    /* Desktop/tablet: horizontal scroll for wide table */
    div[data-testid="stDataFrame"] {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }

    /* Sidebar: touch-friendly tap targets (iPhone) */
    @media (max-width: 768px) {
        section[data-testid="stSidebar"] button,
        section[data-testid="stSidebar"] [role="radiogroup"] label {
            min-height: 44px;
        }
    }

    /* LUXURY COMPACT CARD - Responsive */
    .ticker-card {
        background: linear-gradient(135deg, #0a0a0a 0%, #151515 100%);
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        box-shadow: 0 4px 12px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.05);
        transition: all 0.3s ease;
        position: relative;
        overflow: visible;
        width: 100%;
        box-sizing: border-box;
    }
    .ticker-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00e676, transparent);
        opacity: 0;
        transition: opacity 0.3s;
    }
    .ticker-card:hover,
    .ticker-card:active {
        border-color: #00e676;
        box-shadow: 0 6px 20px rgba(0,230,118,0.3), inset 0 1px 0 rgba(255,255,255,0.1);
        transform: translateY(-1px);
    }
    .ticker-card:hover::before,
    .ticker-card:active::before {
        opacity: 1;
    }

    /* COMPACT HEADER ROW */
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        padding-bottom: 8px;
        margin-bottom: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .card-header-left {
        display: flex;
        align-items: center;
        gap: 6px;
        flex-wrap: wrap;
    }
    .t-link {
        font-size: 14px;
        font-weight: 700;
        color: #448aff !important;
        text-decoration: none;
        letter-spacing: 0.3px;
        transition: color 0.2s;
    }
    .t-link:hover,
    .t-link:active { color: #00e676 !important; }
    .card-company {
        font-size: 11px;
        color: #b0bec5;
        font-weight: 500;
        line-height: 1.3;
        word-break: break-word;
    }
    .header-price-block {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 2px;
    }
    .t-price {
        font-size: 15px;
        color: #fff;
        font-weight: 700;
        line-height: 1.2;
    }
    .t-pe {
        font-size: 10px;
        color: #78909c;
        font-weight: 600;
        padding: 2px 6px;
        background: rgba(120,144,156,0.1);
        border-radius: 4px;
    }
    .card-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        align-items: center;
    }

    /* BADGE */
    .new-badge {
        background: linear-gradient(135deg, #00e676, #00c853);
        color: #000;
        font-size: 9px;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 800;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        box-shadow: 0 2px 4px rgba(0,230,118,0.3);
    }
    .status-badge {
        font-size: 9px;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .status-badge.valid { background: rgba(0,230,118,0.15); color: #00e676; }
    .status-badge.strong { background: rgba(255,171,0,0.15); color: #ffab00; }

    /* COMPACT DATA GRID - 2 columns, responsive */
    .card-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 6px;
    }

    /* COMPACT STAT BLOCK */
    .stat-row {
        background: rgba(22,22,22,0.6);
        padding: 6px 8px;
        border-radius: 5px;
        border: 1px solid rgba(255,255,255,0.05);
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: background 0.2s;
        min-height: 36px;
    }
    .stat-row:hover {
        background: rgba(22,22,22,0.8);
        border-color: rgba(255,255,255,0.1);
    }

    /* TEXT HIERARCHY - COMPACT */
    .lbl {
        font-size: 9px;
        color: #78909c;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        white-space: nowrap;
        flex-shrink: 0;
    }
    .val {
        font-size: 12px;
        font-weight: 700;
        color: #e0e0e0;
        text-align: right;
        line-height: 1.2;
        white-space: nowrap;
        word-break: keep-all;
        flex-shrink: 0;
    }
    .sub {
        font-size: 10px;
        font-weight: 500;
        opacity: 0.8;
        text-align: right;
        line-height: 1.2;
        display: block;
        margin-top: 2px;
        white-space: nowrap;
    }

    /* RESPONSIVE DESIGN - Scales with screen size */
    @media (max-width: 991px) and (min-width: 769px) {
        .ticker-card { padding: 12px; }
        .t-price { font-size: 16px; }
        .t-link { font-size: 15px; }
        .val { font-size: 13px; }
        .lbl { font-size: 10px; }
        .sub { font-size: 11px; }
    }

    /* Mobile: single column grid */
    @media (max-width: 768px) {
        .ticker-card {
            padding: 12px;
            margin-bottom: 12px;
        }
        .card-grid {
            grid-template-columns: 1fr;
            gap: 8px;
        }
        .card-header {
            flex-direction: column;
            gap: 8px;
        }
        .header-price-block {
            align-items: flex-start;
            width: 100%;
        }
        .t-price { font-size: 18px; }
        .t-link { font-size: 16px; }
        .stat-row {
            padding: 8px 10px;
            min-height: 40px;
        }
        .val {
            font-size: 14px;
            white-space: normal;
            word-break: break-word;
        }
        .lbl {
            font-size: 10px;
            white-space: normal;
        }
        .sub { font-size: 11px; }
    }

    @media (max-width: 480px) {
        .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        .ticker-card { padding: 10px; }
        .t-price { font-size: 20px; }
        .t-link { font-size: 17px; }
    }

    /* REJECTED CARD */
    .rejected-card {
        background: #1a0505;
        border: 1px solid #3b1010;
        border-left: 3px solid #d32f2f;
        padding: 4px 6px;
        margin-bottom: 6px;
        border-radius: 4px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        min-height: 28px;
    }
    .rej-head { font-size: 11px; font-weight: 700; color: #b0bec5; }
    .rej-sub { font-size: 10px; color: #ff5252; font-weight: 600; text-align: right; font-family: monospace;}

    /* COLORS */
    .c-green { color: #00e676; }
    .c-red { color: #ff1744; }
    .c-blue { color: #448aff; }
    .c-gold { color: #ffab00; }

</style>
"""


def _extract_symbol_label(symbol_field: str) -> str:
    """Display ticker from TradingView URL or plain symbol string."""
    s = str(symbol_field or "").strip()
    if s.startswith("http"):
        try:
            qs = parse_qs(urlparse(s).query)
            if "symbol" in qs and qs["symbol"]:
                return qs["symbol"][0]
        except Exception:
            pass
        m = re.search(r"symbol=([^&]+)", s)
        if m:
            return m.group(1)
    return s or "—"


def _stat_row(label: str, value: str, value_class: str = "") -> str:
    cls = f"val {value_class}".strip()
    return (
        f'<div class="stat-row">'
        f'<span class="lbl">{html.escape(label)}</span>'
        f'<span class="{html.escape(cls, quote=True)}">{html.escape(value)}</span>'
        f"</div>"
    )


def _build_ticker_card(row: dict) -> str:
    symbol_url = str(row.get("Symbol", ""))
    symbol_label = _extract_symbol_label(symbol_url)
    company = html.escape(str(row.get("Company Name", "")))
    url_esc = html.escape(symbol_url, quote=True)

    badges = []
    if row.get("New"):
        badges.append('<span class="new-badge">NEW</span>')
    if row.get("Valid"):
        badges.append('<span class="status-badge valid">Valid</span>')
    if row.get("Strong"):
        badges.append('<span class="status-badge strong">Strong</span>')
    badges_html = f'<div class="card-badges">{"".join(badges)}</div>' if badges else ""

    tp = row.get("TP", "—")
    sl = row.get("SL", "—")
    rr = row.get("RR", "—")
    pos_size = row.get("Position Size (shares)", 0)
    pos_val = row.get("Position Value ($)", 0)

    rr_class = "c-green" if isinstance(rr, (int, float)) and rr >= 1.5 else ""

    stats = "".join([
        _stat_row("TP", f"{tp}"),
        _stat_row("SL", f"{sl}"),
        _stat_row("RR", f"{rr}", rr_class),
        _stat_row("Size", f"{pos_size} sh"),
        _stat_row("Value", f"${pos_val}"),
    ])

    chart_key = html.escape(str(row.get("tv_symbol", "") or ""), quote=True)

    return (
        f'<div class="ticker-card" data-chart-key="{chart_key}">'
        '<div class="card-header">'
        '<div class="card-header-left">'
        f'<a class="t-link" href="{url_esc}" target="_blank" rel="noopener">{html.escape(symbol_label)}</a>'
        f"{badges_html}"
        "</div>"
        '<div class="header-price-block">'
        f'<span class="card-company">{company}</span>'
        "</div>"
        "</div>"
        f'<div class="card-grid">{stats}</div>'
        "</div>"
    )


_MOBILE_CHART_STYLES = """
<style>
html, body { height: 100%; }
body { margin: 0; padding: 4px; background: #050505; color: #e0e0e0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; overflow-x: hidden; }
.mobile-results { padding-bottom: 8px; }
.ticker-card {
    background: linear-gradient(135deg, #0a0a0a 0%, #151515 100%);
    border: 1px solid #2a2a2a; border-radius: 8px; padding: 12px; margin-bottom: 12px;
    transition: border-color 0.2s;
    cursor: pointer;
}
.ticker-card:hover, .ticker-card.chart-active { border-color: #00e676; }
.card-header { display: flex; justify-content: space-between; align-items: flex-start; padding-bottom: 8px; margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.08); }
.card-header-left { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.t-link { font-size: 16px; font-weight: 700; color: #448aff !important; text-decoration: none; }
.card-badges { display: flex; flex-wrap: wrap; gap: 4px; }
.card-company { font-size: 11px; color: #b0bec5; }
.card-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.stat-row {
    background: rgba(22,22,22,0.6); padding: 8px 10px; border-radius: 5px;
    border: 1px solid rgba(255,255,255,0.05); display: flex; justify-content: space-between; align-items: center;
}
.lbl { font-size: 10px; color: #78909c; font-weight: 700; text-transform: uppercase; }
.val { font-size: 14px; font-weight: 700; color: #e0e0e0; }
.val.c-green { color: #00e676; }
.new-badge { background: #00e676; color: #000; font-size: 9px; padding: 2px 6px; border-radius: 4px; font-weight: 800; text-transform: uppercase; }
.status-badge { font-size: 9px; padding: 2px 6px; border-radius: 4px; font-weight: 700; text-transform: uppercase; }
.status-badge.valid { background: rgba(0,230,118,0.15); color: #00e676; }
.status-badge.strong { background: rgba(255,171,0,0.15); color: #ffab00; }

.chart-overlay {
    display: none;
    position: fixed; inset: 0;
    z-index: 99998;
    background: rgba(0,0,0,0.7);
}
.chart-overlay.visible { display: block; }

#chart-hover-popup {
    display: none; position: fixed; left: 50%; top: 50%; transform: translate(-50%, -50%);
    width: 96vw; max-width: 720px; height: 70vh; min-height: 380px;
    z-index: 99999; background: #0d0d0d; border: 2px solid #00e676;
    border-radius: 10px; box-shadow: 0 12px 40px rgba(0,0,0,0.9);
}
#chart-hover-popup.visible { display: block; }

#chart-popup-close {
    display: none;
    position: fixed; z-index: 100000;
    top: calc(50% - 35vh - 6px); right: calc(2vw + 4px);
    width: 34px; height: 34px;
    border-radius: 50%;
    background: #0d0d0d; border: 2px solid #00e676; color: #00e676;
    font-size: 20px; font-weight: 700; line-height: 28px; text-align: center;
    cursor: pointer; padding: 0;
}
#chart-popup-close.visible { display: block; }

.mobile-chart-hint-inner { color: #78909c; font-size: 12px; margin: 0 0 10px 0; padding: 0 4px; }
</style>
"""


def _mobile_chart_component_html(cards_html: str, charts_json: str, tf: str) -> str:
    tf_esc = html.escape(tf)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
{_MOBILE_CHART_STYLES}
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head><body>
<p class="mobile-chart-hint-inner">Tap or hover a card to preview the chart ({tf_esc}).</p>
<div class="mobile-results">{cards_html}</div>
<div class="chart-overlay" id="chart-overlay"></div>
<div id="chart-hover-popup"></div>
<button id="chart-popup-close" type="button" aria-label="Close chart">&times;</button>
<script>
const charts = {charts_json};
let hideTimer = null;
let pinnedKey = null;
const popup = document.getElementById('chart-hover-popup');
const overlay = document.getElementById('chart-overlay');
const closeBtn = document.getElementById('chart-popup-close');

function showChart(key) {{
  if (!key || !charts[key]) return;
  const spec = charts[key];
  popup.classList.add('visible');
  overlay.classList.add('visible');
  closeBtn.classList.add('visible');
  Plotly.react(popup, spec.data, spec.layout, {{responsive: true, displayModeBar: false}})
    .then(() => {{
      try {{ Plotly.Plots.resize(popup); }} catch (e) {{}}
    }});
}}

function hideChart() {{
  popup.classList.remove('visible');
  overlay.classList.remove('visible');
  closeBtn.classList.remove('visible');
  pinnedKey = null;
  document.querySelectorAll('.ticker-card.chart-active').forEach(el => el.classList.remove('chart-active'));
}}

function scheduleHide() {{
  clearTimeout(hideTimer);
  hideTimer = setTimeout(() => {{ if (!pinnedKey) hideChart(); }}, 220);
}}

closeBtn.addEventListener('click', (e) => {{ e.preventDefault(); e.stopPropagation(); hideChart(); }});
overlay.addEventListener('click', () => hideChart());

document.addEventListener('keydown', (e) => {{
  if (e.key === 'Escape') hideChart();
}});

window.addEventListener('resize', () => {{
  if (popup.classList.contains('visible')) {{
    try {{ Plotly.Plots.resize(popup); }} catch (e) {{}}
  }}
}});

document.querySelectorAll('.ticker-card[data-chart-key]').forEach(card => {{
  const key = card.getAttribute('data-chart-key');
  if (!key || !charts[key]) return;
  card.addEventListener('mouseenter', () => {{
    clearTimeout(hideTimer);
    document.querySelectorAll('.ticker-card.chart-active').forEach(el => el.classList.remove('chart-active'));
    card.classList.add('chart-active');
    if (!pinnedKey) showChart(key);
  }});
  card.addEventListener('mouseleave', () => {{ if (!pinnedKey) scheduleHide(); }});
  card.addEventListener('click', (e) => {{
    if (e.target.closest('a')) return;
    e.preventDefault();
    if (pinnedKey === key) {{
      pinnedKey = null;
      hideChart();
    }} else {{
      pinnedKey = key;
      showChart(key);
    }}
  }});
}});
</script>
</body></html>"""


def render_mobile_cards(
    table_rows: list[dict],
    chart_cache: dict | None = None,
    tf: str = "Daily",
) -> None:
    """Render stacked ticker cards for phone view (hidden on desktop via CSS)."""
    if not table_rows:
        return
    chart_cache = chart_cache or {}
    cards_html = "".join(_build_ticker_card(row) for row in table_rows)
    chart_json = chart_json_for_mobile(chart_cache, table_rows)

    if chart_json:
        charts_str = json.dumps(chart_json)
        comp_html = _mobile_chart_component_html(cards_html, charts_str, tf)
        est_height = max(420, min(720, 220 * len(table_rows) + 80))
        st.markdown('<div class="mobile-only-chart-component">', unsafe_allow_html=True)
        components.html(comp_html, height=est_height, scrolling=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="mobile-results">{cards_html}</div>',
            unsafe_allow_html=True,
        )


def inject_styles() -> None:
    """Inject dark-theme CSS into the Streamlit app. Call once at page load."""
    cleaned = "".join(line.strip() for line in _STYLES.splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)
