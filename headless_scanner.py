import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import textwrap
import os
import time

# ==========================================
# 1. PAGE CONFIG & STYLES (TERMINAL UI)
# ==========================================
st.set_page_config(page_title="Screener Vova (Terminal)", layout="wide", page_icon="💎")

# --- SESSION STATE INITIALIZATION ---
if 'scanning' not in st.session_state:
    st.session_state.scanning = False
if 'results' not in st.session_state:
    st.session_state.results = [] 
if 'rejected' not in st.session_state:
    st.session_state.rejected = []
if 'run_params' not in st.session_state:
    st.session_state.run_params = {} # To freeze params during scan

# --- HELPER FUNCTIONS ---
def render_html(html_string):
    """Aggressively strips whitespace to prevent Markdown code block interpretation."""
    cleaned_html = "".join([line.strip() for line in html_string.splitlines()])
    st.markdown(cleaned_html, unsafe_allow_html=True)

# --- CSS STYLING ---
render_html("""
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
""")

# ==========================================
# 2. DATA & API
# ==========================================
# List files (same folder as this script); format: EXCHANGE:SYMBOL comma-separated
TV_LIST_SMALL_CAP = "TV-LIST-SMALL_CAP_2B-10B.txt"
TV_LIST_BIG_CAP = "TV-LIST-BIG_CAP_10B.txt"
TV_LIST_ETF = "TV-LIST-ETF.txt"

def read_list_file(filename):
    """
    Read tickers from a list file (EXCHANGE:SYMBOL per entry, comma-separated).
    No cache so you can update the file and next START scan uses the new list.
    Returns list of symbol strings; empty list on missing file or error.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return []
        out = []
        for part in raw.split(","):
            part = part.strip()
            if ":" in part:
                sym = part.split(":", 1)[1].strip()
            else:
                sym = part
            if sym:
                out.append(sym.replace(".", "-"))
        return out
    except FileNotFoundError:
        st.warning(f"List file not found: {path}. Add {filename} or choose another source.")
        return []
    except Exception as e:
        st.warning(f"Could not read list file {filename}: {e}")
        return []

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        html = pd.read_html(requests.get(url, headers=headers).text, header=0)
        return [t.replace('.', '-') for t in html[0]['Symbol'].tolist()]
    except Exception as e:
        st.error(f"Error S&P500: {e}")
        return []

@st.cache_data(ttl=3600)
def get_nasdaq100_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        headers = {"User-Agent": "Mozilla/5.0"}
        tables = pd.read_html(requests.get(url, headers=headers).text)
        for tbl in tables:
            if len(tbl) < 50:
                continue
            # First column is often Ticker (or unnamed)
            col0 = tbl.columns[0] if len(tbl.columns) else None
            if col0 and ('Ticker' in str(col0) or 'Symbol' in str(col0)):
                syms = [str(t).strip().replace('.', '-') for t in tbl.iloc[:, 0].tolist() if pd.notna(t) and isinstance(t, str) and 1 <= len(str(t).strip()) <= 6 and str(t).strip().isalpha()]
                if len(syms) >= 50:
                    return syms
            # Try first column as tickers (e.g. ADBE, AMD, ...)
            syms = [str(t).strip().replace('.', '-') for t in tbl.iloc[:, 0].tolist() if pd.notna(t) and isinstance(t, str) and 2 <= len(str(t).strip()) <= 5 and str(t).strip().isalpha()]
            if len(syms) >= 50:
                return syms
        return []
    except Exception:
        return []

def get_us_stock_tickers():
    """S&P 500 + NASDAQ 100, deduplicated, to match TV's larger US watchlist (more results)."""
    sp = set(get_sp500_tickers())
    ndq = set(get_nasdaq100_tickers())
    return list(sp | ndq)

@st.cache_data(ttl=86400)  # 24h cache - list updates daily
def get_all_us_listed_tickers():
    """
    Fetch all US-listed common stock symbols from NASDAQ symbol directory (nasdaqtraded.txt).
    Includes NASDAQ, NYSE, AMEX. Filters: no ETF, no test, exclude warrants/rights/units/preferred.
    """
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        lines = r.text.strip().split("\n")
        if not lines:
            return []
        # Header: Nasdaq Traded|Symbol|Security Name|Listing Exchange|Market Category|ETF|Round Lot Size|Test Issue|...
        col = lines[0].split("|")
        try:
            sym_idx = col.index("Symbol")
            name_idx = col.index("Security Name")
            etf_idx = col.index("ETF")
            test_idx = col.index("Test Issue")
        except ValueError:
            sym_idx, name_idx, etf_idx, test_idx = 1, 2, 5, 7
        out = []
        exclude_substrings = ("warrant", " - right", " - unit", " preferred stock", " preferred share", " preferred ", " etf", " bond ", " note due ", " trust units", " depositary share", " unit ")
        for line in lines[1:]:
            parts = line.split("|")
            if len(parts) <= max(sym_idx, name_idx, etf_idx, test_idx):
                continue
            sym = (parts[sym_idx] or "").strip()
            name = (parts[name_idx] or "").lower()
            etf = (parts[etf_idx] or "").strip().upper()
            test = (parts[test_idx] or "").strip().upper()
            if not sym or "$" in sym or "|" in sym or len(sym) > 6:
                continue
            if etf == "Y" or test == "Y":
                continue
            if any(x in name for x in exclude_substrings):
                continue
            out.append(sym.replace(".", "-"))
        return out
    except Exception as e:
        st.warning(f"Could not fetch all US tickers: {e}. Use S&P 500 + NASDAQ 100.")
        return []

def get_financial_info(ticker):
    try:
        t = yf.Ticker(ticker)
        i = t.info
        return i.get('trailingPE') or i.get('forwardPE')
    except: return None

# US exchanges: Yahoo MIC codes and yfinance variants (comparison uses .upper())
US_EQUITY_EXCHANGES = {
    "NMS", "NYQ", "ASE", "BTS", "BAT", "NGM", "NYS", "PCX", "OTC", "OTN",
    "NASDAQ", "NYSE", "AMEX", "BATS", "ARCA",
    "NASDAQGS", "NASDAQCM", "NASDAQGM",  # yfinance often returns e.g. "NasdaqGS"
}

def get_ticker_info_and_filter(ticker, min_market_cap=5e9, min_avg_volume=300_000, require_mc_vol=True):
    """
    Fetch ticker info from yfinance. Apply filters: US listed common stock; optionally MC/vol (require_mc_vol=True).
    When require_mc_vol=False (e.g. TV-LIST): only filter NOT_EQUITY / NOT_US; allow None MC/vol so more symbols pass.
    Returns (passed: bool, reject_reason: str, info_dict or None). info_dict: company_name, market_cap, mc_display, pe, avg_volume.
    When filter fails but we have info, returns partial info_dict so caller doesn't need a second API call.
    """
    try:
        t = yf.Ticker(ticker)
        i = t.info
        quote_type = i.get("quoteType") or ""
        exchange = (i.get("exchange") or "").upper()
        market_cap = i.get("marketCap")
        avg_vol = i.get("averageVolume")
        if market_cap is None and i.get("sharesOutstanding") and i.get("regularMarketPrice"):
            market_cap = i["sharesOutstanding"] * i["regularMarketPrice"]
        if avg_vol is None and isinstance(t.history(period="1mo"), pd.DataFrame):
            hist = t.history(period="1mo")
            if not hist.empty and "Volume" in hist.columns:
                avg_vol = float(hist["Volume"].mean())
        company_name = i.get("longName") or i.get("shortName") or ticker
        pe = i.get("trailingPE") or i.get("forwardPE")
        if pe is not None and (np.isnan(pe) or np.isinf(pe)):
            pe = None

        if quote_type and quote_type.upper() != "EQUITY":
            partial = {"company_name": company_name, "market_cap": None, "mc_display": None, "pe": pe, "avg_volume": avg_vol}
            return False, "NOT_EQUITY", partial
        if exchange and exchange not in US_EQUITY_EXCHANGES:
            partial = {"company_name": company_name, "market_cap": market_cap, "mc_display": None, "pe": pe, "avg_volume": avg_vol}
            if market_cap is not None:
                partial["mc_display"] = market_cap / 1e9 if market_cap >= 1e9 else market_cap / 1e6
            return False, "NOT_US", partial
        if require_mc_vol:
            if market_cap is None or (min_market_cap and market_cap < min_market_cap):
                partial = {"company_name": company_name, "market_cap": market_cap, "mc_display": None, "pe": pe, "avg_volume": avg_vol}
                return False, "MC_BELOW_5B", partial
            if avg_vol is None or (min_avg_volume and avg_vol < min_avg_volume):
                partial = {"company_name": company_name, "market_cap": market_cap, "mc_display": None, "pe": pe, "avg_volume": avg_vol}
                if market_cap is not None:
                    partial["mc_display"] = market_cap / 1e9 if market_cap >= 1e9 else market_cap / 1e6
                return False, "LOW_VOL", partial

        mc_display = None
        if market_cap is not None:
            mc_display = market_cap / 1e9 if market_cap >= 1e9 else market_cap / 1e6
        info_dict = {
            "company_name": company_name,
            "market_cap": market_cap,
            "mc_display": mc_display,
            "pe": pe,
            "avg_volume": avg_vol,
        }
        return True, "", info_dict
    except Exception as e:
        return False, "INFO_ERROR", None

def _fetch_fallback_company_name(ticker):
    """Fetch company name from yfinance when get_ticker_info_and_filter returns None (INFO_ERROR)."""
    try:
        i = yf.Ticker(ticker).info
        return i.get("longName") or i.get("shortName") or ticker
    except Exception:
        return ticker

# ==========================================
# 3. INDICATOR MATH
# ==========================================
def calc_atr(df, length):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

# ==========================================
# 4. SEQUENCE VOVA (EXACT PINE PORT)
# ==========================================
def run_sequence_vova_pine(df, atr_len=14, min_rr=1.5, use_last_hl_sl=True, risk_dollars=100):
    """
    Exact port of Pine "Sequence Vova Screener". Returns dict for last bar:
    TP, SL, RR, Valid, New, Strong, position_size, position_value, last_peak, seq_low_prev (for Strong).
    """
    n = len(df)
    if n < 2:
        return None
    atr = calc_atr(df, atr_len)
    c_a = df['Close'].values
    h_a = df['High'].values
    l_a = df['Low'].values
    atr_a = atr.values

    seq_state = 0
    critical_level = np.nan
    seq_high, seq_low = h_a[0], l_a[0]
    last_confirmed_peak = np.nan
    last_confirmed_trough = np.nan
    last_peak_was_hh = False
    last_trough_was_hl = False

    # Store outputs for last bar
    last_crit = np.nan
    last_peak = np.nan
    last_valid = False
    last_new = False
    last_strong = False
    last_sl = np.nan
    last_rr = 0.0
    last_pos_size = np.nan
    last_pos_value = np.nan
    prev_bar_seq_low = l_a[0]  # seq_low of previous bar (for Strong on current bar)

    for i in range(1, n):
        c, h, l = c_a[i], h_a[i], l_a[i]
        cur_atr = atr_a[i]
        prev_state = seq_state
        prev_crit = critical_level
        prev_seq_high = seq_high
        prev_seq_low = seq_low

        is_break = False
        is_bearish_break = False
        if prev_state == 1 and not np.isnan(prev_crit):
            is_break = c < prev_crit
        elif prev_state == -1 and not np.isnan(prev_crit):
            is_break = c > prev_crit
            is_bearish_break = is_break  # bullish break (downtrend broken)

        if is_break:
            if prev_state == 1:
                if h >= seq_high:
                    seq_high = h
                is_current_peak_hh = (np.isnan(last_confirmed_peak) or seq_high > last_confirmed_peak)
                last_peak_was_hh = is_current_peak_hh
                last_confirmed_peak = seq_high
                seq_state = -1
                seq_high, seq_low = h, l
                critical_level = h
            else:
                if l <= seq_low:
                    seq_low = l
                is_current_trough_hl = (np.isnan(last_confirmed_trough) or
                    (seq_low > last_confirmed_trough) or (seq_low == last_confirmed_trough))
                last_trough_was_hl = is_current_trough_hl
                last_confirmed_trough = seq_low
                seq_state = 1
                seq_high, seq_low = h, l
                critical_level = l
        else:
            seq_state = prev_state
            if seq_state == 1:
                if h >= seq_high:
                    seq_high = h
                if h >= prev_seq_high:
                    critical_level = l
                else:
                    critical_level = prev_crit
            elif seq_state == -1:
                if l <= seq_low:
                    seq_low = l
                if l <= prev_seq_low:
                    critical_level = h
                else:
                    critical_level = prev_crit
            else:
                if c > prev_seq_high:
                    seq_state = 1
                    critical_level = l
                elif c < prev_seq_low:
                    seq_state = -1
                    critical_level = h
                else:
                    seq_high = max(prev_seq_high, h)
                    seq_low = min(prev_seq_low, l)

        struct_invalid_seq_down = (seq_state == -1 and last_trough_was_hl and
            not np.isnan(last_confirmed_trough) and seq_low < last_confirmed_trough)
        struct_ok = (last_trough_was_hl or (not np.isnan(last_confirmed_peak) and c > last_confirmed_peak and last_trough_was_hl)) and (not struct_invalid_seq_down)

        sl = c - cur_atr
        if not np.isnan(critical_level) and critical_level < c:
            sl = min(sl, critical_level)
        if use_last_hl_sl and last_trough_was_hl and not np.isnan(last_confirmed_trough) and last_confirmed_trough < c:
            sl = min(sl, last_confirmed_trough)
        risk = c - sl
        reward = last_confirmed_peak - c if not np.isnan(last_confirmed_peak) else 0.0
        rr = (reward / risk) if risk > 0 else 0.0
        position_size = (risk_dollars / risk) if (risk > 0 and risk_dollars > 0) else np.nan
        position_value = position_size * c if not np.isnan(position_size) else np.nan

        valid_signal = (seq_state == 1) and struct_ok and (rr >= min_rr) and (risk > 0) and (reward > 0)
        new_signal = valid_signal and is_bearish_break
        strong_signal = new_signal and (not np.isnan(prev_bar_seq_low)) and (l <= prev_bar_seq_low)

        prev_bar_seq_low = seq_low
        last_crit = critical_level
        last_peak = last_confirmed_peak
        last_valid = valid_signal
        last_new = new_signal
        last_strong = strong_signal
        last_sl = sl
        last_rr = rr
        last_pos_size = position_size
        last_pos_value = position_value

    close_last = c_a[-1]
    atr_last = atr_a[-1]
    return {
        "TP": last_peak,
        "SL": last_sl,
        "RR": last_rr,
        "Valid": last_valid,
        "New": last_new,
        "Strong": last_strong,
        "position_size": last_pos_size,
        "position_value": last_pos_value,
        "Close": close_last,
        "ATR": atr_last,
    }

# ==========================================
# 5. UI & SIDEBAR
# ==========================================
st.sidebar.header("⚙️ CONFIGURATION")

# Disable inputs if scanning
disabled = st.session_state.scanning

# Source: SMALL CAP, BIG CAP, ETFS, ALL, MANUAL SCAN
SOURCE_OPTIONS = ["SMALL CAP", "BIG CAP", "ETFS", "ALL", "MANUAL SCAN"]
last_src = st.session_state.get("run_params", {}).get("src", "SMALL CAP")
default_idx = SOURCE_OPTIONS.index(last_src) if last_src in SOURCE_OPTIONS else 0
src = st.sidebar.radio("SOURCE", SOURCE_OPTIONS, disabled=disabled, index=default_idx)
if src == "SMALL CAP":
    st.sidebar.caption(f"Uses {TV_LIST_SMALL_CAP}. Edit file — next START uses new tickers.")
elif src == "BIG CAP":
    st.sidebar.caption(f"Uses {TV_LIST_BIG_CAP}. Edit file — next START uses new tickers.")
elif src == "ETFS":
    st.sidebar.caption(f"Uses {TV_LIST_ETF}. Edit file — next START uses new tickers.")
elif src == "ALL":
    st.sidebar.caption("Uses SMALL CAP + BIG CAP + ETFS lists merged (no duplicates).")
man_txt = ""
if src == "MANUAL SCAN":
    man_txt = st.sidebar.text_area("TICKERS", "AAPL, TSLA, NVDA", disabled=disabled)
    st.sidebar.caption("Comma-separated symbols. Next START scans these tickers.")

# Parameters
st.sidebar.subheader("RISK MANAGEMENT")
risk_per_trade = st.sidebar.number_input("$ RISK PER TRADE", value=100, min_value=1, step=10, disabled=disabled)
min_rr_in = st.sidebar.number_input("MIN RR (>=1.5)", value=1.5, min_value=0.5, step=0.1, disabled=disabled)
use_last_hl_sl = st.sidebar.checkbox("Use last HL in SL (safety)", True, disabled=disabled)

st.sidebar.subheader("FILTERS")
tf_p = st.sidebar.selectbox("TIMEFRAME", ["Daily", "Weekly", "Monthly"], disabled=disabled)
new_p = st.sidebar.checkbox("NEW SIGNALS ONLY", True, disabled=disabled)

# Buttons
c1, c2 = st.sidebar.columns(2)
start_btn = c1.button("▶ START", type="primary", disabled=disabled, use_container_width=True)
stop_btn = c2.button("⏹ STOP", type="secondary", disabled=not disabled, use_container_width=True)

# State Management for Buttons
if start_btn:
    st.session_state.scanning = True
    st.session_state.results = []   # RESET Valid
    st.session_state.rejected = [] # RESET Rejected
    # FREEZE PARAMS
    st.session_state.run_params = {
        'src': src, 'txt': man_txt, 'risk_per_trade': risk_per_trade, 'rr': min_rr_in,
        'use_last_hl_sl': use_last_hl_sl, 'tf': tf_p, 'new': new_p
    }
    st.rerun()

if stop_btn:
    st.session_state.scanning = False
    st.rerun()

# ==========================================
# 6. SCANNER EXECUTION
# ==========================================
ATR_LEN = 14
MIN_BARS = 50  # minimum bars for sequence logic

# Results Placeholder
res_area = st.empty()

def _interval_and_period(tf):
    """Always fetch daily; Weekly/Monthly resampled from daily so current period is included."""
    return "1d", "10y" if tf != "Daily" else "2y"

def _resample_to_timeframe(df, tf):
    """Resample daily OHLCV to Weekly or Monthly. Returns df unchanged for Daily."""
    if tf == "Daily" or df is None or df.empty:
        return df
    req = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in req):
        return df
    rule = "W-FRI" if tf == "Weekly" else "M"
    res = df[req].resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    return res.dropna(subset=req)

def _fill_last_bar_ohlc(df):
    """Fill last bar NaNs in OHLC from previous bar so we never drop the reference bar (avoids missing signals)."""
    if df is None or len(df) < 2:
        return df
    ohlc = ["Open", "High", "Low", "Close"]
    if not all(c in df.columns for c in ohlc):
        return df
    last_idx = df.index[-1]
    last_row = df.loc[last_idx, ohlc]
    if last_row.isna().any():
        prev = df.iloc[-2][ohlc]
        for col in ohlc:
            if pd.isna(df.at[last_idx, col]) and not pd.isna(prev[col]):
                df.at[last_idx, col] = prev[col]
        # if still NaN (e.g. first bar of resampled period), use Close for O/H/L
        if pd.isna(df.at[last_idx, "Close"]) and not pd.isna(prev["Close"]):
            df.at[last_idx, "Close"] = prev["Close"]
        for col in ["Open", "High", "Low"]:
            if pd.isna(df.at[last_idx, col]):
                df.at[last_idx, col] = df.at[last_idx, "Close"]
    return df

def _extract_ohlcv(all_data, ticker, required_cols):
    if isinstance(all_data.columns, pd.MultiIndex):
        level0 = all_data.columns.get_level_values(0).unique()
        key = ticker
        if key not in level0 and ":" in ticker:
            key = ticker.split(":")[-1]
        if key not in level0:
            return None
        if not all((key, col) in all_data.columns for col in required_cols):
            return None
        return pd.DataFrame({
            'Open': all_data[(key, 'Open')],
            'High': all_data[(key, 'High')],
            'Low': all_data[(key, 'Low')],
            'Close': all_data[(key, 'Close')],
            'Volume': all_data[(key, 'Volume')]
        })
    if len(all_data.columns) == 0:
        return None
    if not all(col in all_data.columns for col in required_cols):
        return None
    return all_data[required_cols].copy()

if st.session_state.scanning:
    p = st.session_state.run_params
    if p['src'] == "SMALL CAP":
        tickers = read_list_file(TV_LIST_SMALL_CAP)
    elif p['src'] == "BIG CAP":
        tickers = read_list_file(TV_LIST_BIG_CAP)
    elif p['src'] == "ETFS":
        tickers = read_list_file(TV_LIST_ETF)
    elif p['src'] == "ALL":
        tickers = list(dict.fromkeys(
            read_list_file(TV_LIST_SMALL_CAP) + read_list_file(TV_LIST_BIG_CAP) + read_list_file(TV_LIST_ETF)
        ))
    else:
        tickers = [x.strip().upper() for x in p['txt'].split(',') if x.strip()]

    if not tickers:
        st.error("NO TICKERS FOUND")
        st.session_state.scanning = False
        st.stop()

    inter, fetch_period = _interval_and_period(p['tf'])
    tf = p['tf']
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    # Chunk large lists to avoid huge single download (All US listed or big watchlist file)
    CHUNK_SIZE = 350
    if len(tickers) > CHUNK_SIZE:
        batches = [tickers[i:i + CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]
    else:
        batches = [tickers]

    info_box = st.empty()
    bar = st.progress(0)
    table_rows = []
    rejected_reasons = []
    processed = 0
    reference_end_date = None  # same bar for all tickers = consistent results

    # When multiple batches: download all first, then set reference_end_date = min(end date across batches)
    # so every ticker is evaluated on the same bar and we don't lose results in later batches.
    batches_data = []  # list of (batch, all_data or None)
    for batch_idx, batch in enumerate(batches):
        if not st.session_state.scanning:
            break
        info_box.info(f"Downloading batch {batch_idx + 1}/{len(batches)} ({len(batch)} tickers)... DO NOT REFRESH.")
        try:
            all_data = yf.download(
                batch,
                period=fetch_period,
                interval=inter,
                progress=False,
                auto_adjust=False,
                group_by='ticker',
                threads=True
            )
            if all_data is None or all_data.empty:
                batches_data.append((batch, None))
            else:
                batch_end = all_data.index[-1] if hasattr(all_data.index, '__len__') and len(all_data.index) > 0 else None
                if batch_end is not None:
                    reference_end_date = batch_end if reference_end_date is None else min(reference_end_date, batch_end)
                batches_data.append((batch, all_data))
        except Exception as e:
            st.warning(f"Batch download failed: {e}")
            batches_data.append((batch, None))

    if reference_end_date is None and batches_data:
        for _, all_data in batches_data:
            if all_data is not None and not all_data.empty and len(all_data.index) > 0:
                reference_end_date = all_data.index[-1]
                break

    for batch_idx, (batch, all_data) in enumerate(batches_data):
        if not st.session_state.scanning:
            break
        if all_data is not None and not all_data.empty and reference_end_date is not None and len(all_data.index) > 0 and all_data.index[-1] > reference_end_date:
            all_data = all_data.loc[all_data.index <= reference_end_date]
        info_box.info(f"Processing batch {batch_idx + 1}/{len(batches)}... DO NOT REFRESH.")
        for i, t in enumerate(batch):
            if not st.session_state.scanning:
                break
            processed += 1
            bar.progress(0.05 + 0.95 * (processed / len(tickers)))
            try:
                require_mc_vol = False  # all five sources: match TV-style lists, no MC/vol filter
                passed, reject_reason, info_dict = get_ticker_info_and_filter(t, min_market_cap=5e9, min_avg_volume=300_000, require_mc_vol=require_mc_vol)
                # Throttle info requests when scanning many tickers (e.g. ALL) to avoid rate limits and INFO_ERROR -> empty PE/MC
                if len(tickers) > 100:
                    time.sleep(0.12)
                if not passed:
                    if p['src'] == "MANUAL SCAN":
                        rejected_reasons.append({"Symbol": t, "Reason": reject_reason})
                        continue
                    # Use partial info from first call when we have it; only fetch again on INFO_ERROR
                    if info_dict is None:
                        company_name = _fetch_fallback_company_name(t)
                        info_dict = {"company_name": company_name, "market_cap": None, "mc_display": None, "pe": None, "avg_volume": None}

                df = _extract_ohlcv(all_data, t, required_cols) if all_data is not None else None
                if df is None or df.empty:
                    try:
                        df = yf.download(t, period=fetch_period, interval=inter, progress=False, auto_adjust=False, multi_level_index=False)
                        if df is not None and not df.empty and all(col in df.columns for col in required_cols):
                            df = df[required_cols].copy()
                        else:
                            df = None
                    except Exception:
                        df = None
                if df is None or df.empty or len(df) < MIN_BARS:
                    if p['src'] == "MANUAL SCAN":
                        rejected_reasons.append({"Symbol": t, "Reason": "NO_DATA"})
                    continue

                # Use same end date for all tickers so results are consistent (same bar everywhere)
                if reference_end_date is not None and len(df.index) > 0 and df.index[-1] > reference_end_date:
                    df = df.loc[df.index <= reference_end_date]
                if reference_end_date is None and len(df.index) > 0:
                    reference_end_date = df.index[-1]

                # Resample daily to weekly/monthly so current period is included (match TV)
                df = _resample_to_timeframe(df, tf)
                if df is None or df.empty or len(df) < MIN_BARS:
                    if p['src'] == "MANUAL SCAN":
                        rejected_reasons.append({"Symbol": t, "Reason": "NO_DATA"})
                    continue

                # Keep reference bar: fill last-bar NaNs from previous bar so we don't drop it (fixes missing vs TV)
                df = _fill_last_bar_ohlc(df)
                df = df.dropna(subset=['Close', 'High', 'Low', 'Open'])
                if len(df) < MIN_BARS:
                    if p['src'] == "MANUAL SCAN":
                        rejected_reasons.append({"Symbol": t, "Reason": "INSUFFICIENT_DATA"})
                    continue

                out = run_sequence_vova_pine(
                    df,
                    atr_len=ATR_LEN,
                    min_rr=p['rr'],
                    use_last_hl_sl=p['use_last_hl_sl'],
                    risk_dollars=p['risk_per_trade']
                )
                if out is None or not out["Valid"]:
                    if p['src'] == "MANUAL SCAN":
                        rejected_reasons.append({"Symbol": t, "Reason": "NO_VALID_SIGNAL"})
                    continue
                if p['new'] and not out["New"]:
                    continue

                pos_size = out["position_size"]
                if np.isnan(pos_size) or pos_size < 1:
                    pos_size = 0
                else:
                    pos_size = int(round(pos_size))
                pos_value = out["position_value"] if not np.isnan(out["position_value"]) else 0.0

                pe_val = info_dict["pe"]
                if pe_val is not None and (np.isnan(pe_val) or np.isinf(pe_val)):
                    pe_val = None

                mc_disp = info_dict.get("mc_display")
                tv_url = f"https://www.tradingview.com/chart/?symbol={t}"
                table_rows.append({
                    "Symbol": tv_url,
                    "Company Name": info_dict["company_name"],
                    "TP": round(float(out["TP"]), 2),
                    "SL": round(float(out["SL"]), 2),
                    "RR": round(float(out["RR"]), 2),
                    "MC (B/M)": round(float(mc_disp), 2) if mc_disp is not None else None,
                    "PE": round(float(pe_val), 2) if pe_val is not None else None,
                    "Position Size (shares)": pos_size,
                    "Position Value ($)": round(float(pos_value), 2),
                    "New": 1 if out["New"] else 0,
                    "Valid": 1 if out["Valid"] else 0,
                    "Strong": 1 if out["Strong"] else 0,
                })
            except Exception:
                rejected_reasons.append({"Symbol": t, "Reason": "ERROR"})

    bar.empty()
    st.session_state.results = table_rows
    st.session_state.rejected = rejected_reasons
    st.session_state.results_as_of = reference_end_date
    st.session_state.results_tf = tf

    with res_area.container():
        if table_rows:
            res_df = pd.DataFrame(table_rows)
            col_config = {"Symbol": st.column_config.LinkColumn("Symbol", display_text=r"symbol=([^&]+)")}
            st.dataframe(res_df, use_container_width=True, hide_index=True, column_config=col_config)
            if reference_end_date is not None:
                try:
                    d = pd.Timestamp(reference_end_date)
                    as_of_str = d.strftime("%Y-%m-%d")
                    dow = d.dayofweek
                    if dow >= 5:
                        st.caption(f"**Results as of {as_of_str}** (last trading day — market closed weekend/holiday). Same bar used for all symbols for consistency.")
                    else:
                        st.caption(f"**Results as of {as_of_str}** ({tf}). Same bar used for all symbols for consistency.")
                except Exception:
                    pass
        else:
            st.info("No symbols passed the screener.")
        if p['src'] == "MANUAL SCAN" and rejected_reasons:
            with st.expander("Rejected (Manual)"):
                st.dataframe(pd.DataFrame(rejected_reasons), use_container_width=True, hide_index=True)
        elif rejected_reasons and any(r.get("Reason") == "ERROR" for r in rejected_reasons):
            with st.expander("Skipped (errors)"):
                st.dataframe(pd.DataFrame(rejected_reasons), use_container_width=True, hide_index=True)

    st.session_state.scanning = False
    info_box.success("SCAN COMPLETE")

else:
    last_src = st.session_state.run_params.get('src', "SMALL CAP")
    table_rows = st.session_state.results
    rejected_reasons = st.session_state.rejected
    as_of = st.session_state.get("results_as_of")
    as_of_tf = st.session_state.get("results_tf", "Daily")

    with res_area.container():
        if table_rows:
            res_df = pd.DataFrame(table_rows)
            col_config = {"Symbol": st.column_config.LinkColumn("Symbol", display_text=r"symbol=([^&]+)")}
            st.dataframe(res_df, use_container_width=True, hide_index=True, column_config=col_config)
            if as_of is not None:
                try:
                    d = pd.Timestamp(as_of)
                    as_of_str = d.strftime("%Y-%m-%d")
                    dow = d.dayofweek
                    if dow >= 5:
                        st.caption(f"**Results as of {as_of_str}** (last trading day — market closed weekend/holiday). Same bar used for all symbols for consistency.")
                    else:
                        st.caption(f"**Results as of {as_of_str}** ({as_of_tf}). Same bar used for all symbols for consistency.")
                except Exception:
                    pass
        else:
            st.info("Ready to scan. Click START.")
        if last_src == "MANUAL SCAN" and rejected_reasons:
            with st.expander("Rejected (Manual)"):
                st.dataframe(pd.DataFrame(rejected_reasons), use_container_width=True, hide_index=True)
        elif rejected_reasons and any(r.get("Reason") == "ERROR" for r in rejected_reasons):
            with st.expander("Skipped (errors)"):
                st.dataframe(pd.DataFrame(rejected_reasons), use_container_width=True, hide_index=True)

