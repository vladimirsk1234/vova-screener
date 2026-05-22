"""
Terminal UI styles for the screener. Injects dark-theme CSS via Streamlit.
"""
import html
import re
from urllib.parse import parse_qs, urlparse

import streamlit as st

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

    return f"""
    <div class="ticker-card">
        <div class="card-header">
            <div class="card-header-left">
                <a class="t-link" href="{url_esc}" target="_blank" rel="noopener">{html.escape(symbol_label)}</a>
                {badges_html}
            </div>
            <div class="header-price-block">
                <span class="card-company">{company}</span>
            </div>
        </div>
        <div class="card-grid">{stats}</div>
    </div>
    """


def render_mobile_cards(table_rows: list[dict]) -> None:
    """Render stacked ticker cards for phone view (hidden on desktop via CSS)."""
    if not table_rows:
        return
    cards_html = "".join(_build_ticker_card(row) for row in table_rows)
    st.markdown(
        f'<div class="mobile-results">{cards_html}</div>',
        unsafe_allow_html=True,
    )


def inject_styles() -> None:
    """Inject dark-theme CSS into the Streamlit app. Call once at page load."""
    cleaned = "".join(line.strip() for line in _STYLES.splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)
