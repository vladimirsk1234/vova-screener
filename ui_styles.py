"""
Terminal UI styles for the screener. Injects dark-theme CSS via Streamlit.
"""
import streamlit as st

_STYLES = """
<style>
    /* GLOBAL DARK THEME */
    .stApp { background-color: #050505; }
    
    /* FIX: Top padding to prevent header overlap */
    .block-container { 
        padding-top: 4rem !important; 
        padding-left: 1rem !important; 
        padding-right: 1rem !important; 
        max-width: 100% !important;
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
    .ticker-card:hover { 
        border-color: #00e676; 
        box-shadow: 0 6px 20px rgba(0,230,118,0.3), inset 0 1px 0 rgba(255,255,255,0.1);
        transform: translateY(-1px);
    }
    .ticker-card:hover::before {
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
    .t-link:hover { color: #00e676 !important; }
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
    /* Tablets: 3 columns */
    @media (max-width: 991px) and (min-width: 769px) {
        .ticker-card {
            padding: 12px;
        }
        .t-price { font-size: 16px; }
        .t-link { font-size: 15px; }
        .val { font-size: 13px; }
        .lbl { font-size: 10px; }
        .sub { font-size: 11px; }
    }
    
    /* Mobile: 2 columns then 1 column */
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
        .val { font-size: 14px; }
        .lbl { font-size: 10px; }
        .sub { font-size: 11px; }
    }
    
    @media (max-width: 480px) {
        .ticker-card {
            padding: 10px;
        }
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


def inject_styles() -> None:
    """Inject dark-theme CSS into the Streamlit app. Call once at page load."""
    cleaned = "".join(line.strip() for line in _STYLES.splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)
