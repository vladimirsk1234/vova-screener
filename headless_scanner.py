import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import textwrap

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

def get_financial_info(ticker):
    try:
        t = yf.Ticker(ticker)
        i = t.info
        return i.get('trailingPE') or i.get('forwardPE')
    except: return None

# ==========================================
# 3. INDICATOR MATH
# ==========================================
def calc_sma(s, l): return s.rolling(l).mean()
def calc_ema(s, l): return s.ewm(span=l, adjust=False).mean()
def calc_macd(s, f=12, sl=26, sig=9):
    fast = s.ewm(span=f, adjust=False).mean()
    slow = s.ewm(span=sl, adjust=False).mean()
    macd = fast - slow
    return macd - macd.ewm(span=sig, adjust=False).mean()

def calc_adx_pine(df, length):
    h, l, c = df['High'], df['Low'], df['Close']
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    
    up = h - h.shift(1); down = l.shift(1) - l
    p_dm = np.where((up > down) & (up > 0), up, 0.0)
    m_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    def rma(s, len): return s.ewm(alpha=1/len, adjust=False).mean()
    
    tr_s = rma(tr, length).replace(0, np.nan)
    p_di = 100 * (rma(pd.Series(p_dm, index=df.index), length) / tr_s)
    m_di = 100 * (rma(pd.Series(m_dm, index=df.index), length) / tr_s)
    dx = 100 * (p_di - m_di).abs() / (p_di + m_di).replace(0, np.nan)
    return rma(dx, length), p_di, m_di

def calc_atr(df, length):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

# ==========================================
# 4. VOVA STRATEGY LOGIC (PINE PARITY)
# ==========================================
def run_vova_logic(df, len_maj, len_fast, len_slow, adx_len, adx_thr, atr_len):
    # --- Indicators ---
    df['SMA'] = calc_sma(df['Close'], len_maj)
    adx, p_di, m_di = calc_adx_pine(df, adx_len)
    
    ema_f = calc_ema(df['Close'], len_fast)
    ema_s = calc_ema(df['Close'], len_slow)
    hist = calc_macd(df['Close'])
    efi = calc_ema(df['Close'].diff() * df['Volume'], len_fast)
    atr = calc_atr(df, atr_len)
    
    # --- Iterative Structure Logic ---
    n = len(df)
    c_a, h_a, l_a = df['Close'].values, df['High'].values, df['Low'].values
    
    seq_st = np.zeros(n, dtype=int)
    crit_lvl = np.full(n, np.nan)
    res_peak = np.full(n, np.nan)
    res_struct = np.zeros(n, dtype=bool)
    
    # State Variables (simulate 'var' in Pine)
    s_state = 0
    s_crit = np.nan
    s_h = h_a[0]; s_l = l_a[0]
    
    last_pk = np.nan; last_tr = np.nan
    pk_hh = False; tr_hl = False
    
    for i in range(1, n):
        c, h, l = c_a[i], h_a[i], l_a[i]
        
        # Access "Previous" values (index [1] in Pine)
        prev_st = s_state
        prev_cr = s_crit
        prev_sh = s_h
        prev_sl = s_l
        
        # Break Detection
        brk = False
        if prev_st == 1 and not np.isnan(prev_cr): brk = c < prev_cr
        elif prev_st == -1 and not np.isnan(prev_cr): brk = c > prev_cr
            
        if brk:
            if prev_st == 1: # Bearish Break (Up -> Down)
                # Did we make a HH before breaking?
                is_hh = True if np.isnan(last_pk) else (prev_sh > last_pk)
                
                # Update Memory
                pk_hh = is_hh
                last_pk = prev_sh # LAST CONFIRMED PEAK (TARGET)
                
                # Reset State
                s_state = -1
                s_h = h; s_l = l
                s_crit = h # Initial stop for downtrend
                
            else: # Bullish Break (Down -> Up)
                # Did we make a HL before breaking?
                is_hl = True if np.isnan(last_tr) else (prev_sl > last_tr)
                
                # Update Memory
                tr_hl = is_hl
                last_tr = prev_sl
                
                # Reset State
                s_state = 1
                s_h = h; s_l = l
                s_crit = l # Initial stop for uptrend
        else:
            # Continue State
            s_state = prev_st
            
            if s_state == 1: # Uptrend
                if h >= s_h: s_h = h
                
                # Trailing Logic: if high >= previous seqHigh, trail stop to low
                if h >= prev_sh: s_crit = l
                else: s_crit = prev_cr
                
            elif s_state == -1: # Downtrend
                if l <= s_l: s_l = l
                
                # Trailing Logic
                if l <= prev_sl: s_crit = h
                else: s_crit = prev_cr
                
            else: # Init state 0
                if c > prev_sh: 
                    s_state = 1; s_crit = l
                elif c < prev_sl: 
                    s_state = -1; s_crit = h
                else:
                    s_h = max(prev_sh, h); s_l = min(prev_sl, l)
        
        # Store results for this bar
        seq_st[i] = s_state
        crit_lvl[i] = s_crit
        res_peak[i] = last_pk # TP IS LAST CONFIRMED PEAK
        res_struct[i] = (pk_hh and tr_hl)

    # --- Super Trend Logic (Vectorized) ---
    adx_str = adx >= adx_thr
    
    # Bullish: ADX+DI, Elder Impulse (EMA+Hist Rising), EFI > 0
    bull = (adx_str & (p_di > m_di)) & \
           ((ema_f > ema_f.shift(1)) & (ema_s > ema_s.shift(1)) & (hist > hist.shift(1))) & \
           (efi > 0)
           
    # Bearish: ADX-DI, Elder Impulse (EMA+Hist Falling), EFI < 0
    bear = (adx_str & (m_di > p_di)) & \
           ((ema_f < ema_f.shift(1)) & (ema_s < ema_s.shift(1)) & (hist < hist.shift(1))) & \
           (efi < 0)
           
    t_st = np.zeros(n, dtype=int)
    t_st[bull] = 1
    t_st[bear] = -1
    
    # Assign to DF
    df['Seq'] = seq_st
    df['Crit'] = crit_lvl
    df['Peak'] = res_peak
    df['Struct'] = res_struct
    df['Trend'] = t_st
    df['ATR'] = atr
    
    return df

def analyze_trade(df, idx):
    r = df.iloc[idx]
    errs = []
    
    # 1. Validation Rules
    if r['Seq'] != 1: errs.append("SEQ!=1")
    if np.isnan(r['SMA']) or r['Close'] <= r['SMA']: errs.append("SMA")
    if r['Trend'] == -1: errs.append("TREND")
    if not r['Struct']: errs.append("STRUCT")
    if np.isnan(r['Peak']) or np.isnan(r['Crit']): errs.append("NO DATA")
    
    if errs: return False, {}, " ".join(errs)
    
    # 2. Key Levels
    price = r['Close']
    tp = r['Peak'] # LAST CONFIRMED PEAK (HH)
    crit = r['Crit']
    atr = r['ATR']
    
    # 3. Safer SL Selection
    # For a LONG trade, we want the stop that gives the trade more room (is lower).
    # SL_Struct = Critical Level
    # SL_ATR = Price - 1.5 ATR
    sl_struct = crit
    sl_atr = price - atr
    
    # Pick minimum (lowest price)
    final_sl = min(sl_struct, sl_atr)
    
    # 4. Geometry Check
    risk = price - final_sl
    reward = tp - price
    
    if risk <= 0: return False, {}, "BAD STOP"
    if reward <= 0: return False, {}, "AT TARGET"
    
    # 5. Calculate Real Monetary RR
    rr = reward / risk
    
    return True, {
        "P": price, "TP": tp, "SL": final_sl, 
        "RR": rr, "ATR": atr, "Crit": crit,
        "SL_Type": "STR" if abs(final_sl - crit) < 0.01 else "ATR"
    }, "OK"

# ==========================================
# 5. UI & SIDEBAR
# ==========================================
st.sidebar.header("⚙️ CONFIGURATION")

# Disable inputs if scanning
disabled = st.session_state.scanning

# Source Input
src = st.sidebar.radio("SOURCE", ["All S&P 500", "Manual Input"], disabled=disabled)
man_txt = ""
if src == "Manual Input":
    man_txt = st.sidebar.text_area("TICKERS", "AAPL, TSLA, NVDA", disabled=disabled)

# Parameters
st.sidebar.subheader("RISK MANAGEMENT")
risk_per_trade = st.sidebar.number_input("$ RISK PER TRADE", value=100, min_value=1, step=10, disabled=disabled)
min_rr_in = st.sidebar.number_input("MIN RR (>=1.25)", 1.25, step=0.05, disabled=disabled)
max_atr_in = st.sidebar.number_input("MAX ATR %", 5.0, step=0.5, disabled=disabled)

st.sidebar.subheader("FILTERS")
sma_p = st.sidebar.selectbox("SMA TREND", [100, 150, 200], index=2, disabled=disabled)
tf_p = st.sidebar.selectbox("TIMEFRAME", ["Daily", "Weekly"], disabled=disabled)
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
        'matr': max_atr_in, 'sma': sma_p, 'tf': tf_p, 'new': new_p
    }
    st.rerun()

if stop_btn:
    st.session_state.scanning = False
    st.rerun()

# ==========================================
# 6. SCANNER EXECUTION
# ==========================================
# CONSTANTS (Hidden)
EMA_F=20; EMA_S=40; ADX_L=14; ADX_T=20; ATR_L=14

# Results Placeholder
res_area = st.empty()

if st.session_state.scanning:
    # Use FROZEN params
    p = st.session_state.run_params
    
    if p['src'] == "All S&P 500":
        tickers = get_sp500_tickers()
    else:
        tickers = [x.strip().upper() for x in p['txt'].split(',') if x.strip()]
        
    if not tickers:
        st.error("NO TICKERS FOUND")
        st.session_state.scanning = False
        st.stop()

    info_box = st.empty()
    info_box.info(f"DOWNLOADING DATA FOR {len(tickers)} TICKERS... DO NOT REFRESH.")
    bar = st.progress(0)
    
    # ==========================================
    # OPTIMIZATION 1: BATCH DOWNLOAD ALL TICKERS AT ONCE (yfinance)
    # ==========================================
    inter = "1d" if p['tf'] == "Daily" else "1wk"
    fetch_period = "2y" if p['tf'] == "Daily" else "5y"
    
    try:
        # Download ALL tickers in one batch (10-50x faster than sequential)
        # yfinance returns MultiIndex DataFrame when downloading multiple tickers with group_by='ticker'
        all_data = yf.download(
            tickers, 
            period=fetch_period, 
            interval=inter, 
            progress=False, 
            auto_adjust=False, 
            group_by='ticker',
            threads=True  # Enable multi-threading in yfinance
        )
        # Validate batch download was successful
        if all_data is None or all_data.empty:
            raise ValueError("Batch download returned empty data")
        info_box.info(f"PROCESSING {len(tickers)} TICKERS... DO NOT REFRESH.")
        bar.progress(0.1)  # 10% for download
    except Exception as e:
        st.warning(f"Batch download failed: {e}. Falling back to sequential.")
        all_data = None
        bar.progress(0.05)
    
    # ==========================================
    # PROCESS EACH TICKER
    # ==========================================
    for i, t in enumerate(tickers):
        if not st.session_state.scanning: break
        bar.progress(0.1 + 0.9 * ((i+1)/len(tickers)))
        
        try:
            # Get ticker data from batch download
            if all_data is not None and not all_data.empty:
                # Handle multi-level column structure from batch download (multiple tickers)
                if isinstance(all_data.columns, pd.MultiIndex):
                    # Check if ticker exists in the batch data and has all required columns
                    ticker_level = all_data.columns.get_level_values(0).unique()
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    has_all_cols = all((t, col) in all_data.columns for col in required_cols)
                    
                    if t in ticker_level and has_all_cols:
                        try:
                            df = pd.DataFrame({
                                'Open': all_data[(t, 'Open')],
                                'High': all_data[(t, 'High')],
                                'Low': all_data[(t, 'Low')],
                                'Close': all_data[(t, 'Close')],
                                'Volume': all_data[(t, 'Volume')]
                            })
                            # Validate extracted data
                            if df.empty or df['Close'].isna().all():
                                raise ValueError("Empty or invalid data")
                        except (KeyError, ValueError):
                            # Failed to extract data, try sequential download as fallback
                            df = None
                            try:
                                df = yf.download(t, period=fetch_period, interval=inter, progress=False, auto_adjust=False, multi_level_index=False)
                                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                                if df is None or df.empty or not all(col in df.columns for col in required_cols):
                                    df = None
                            except Exception:
                                df = None
                    else:
                        df = None
                    
                    if df is None:
                        # Ticker not found in batch download or all download methods failed
                        if p['src'] == "Manual Input":
                            st.session_state.rejected.append(f"""<div class="rejected-card"><span class="rej-head">{t}</span><span class="rej-sub">NO DATA</span></div>""")
                        continue
                else:
                    # Single ticker returned as regular DataFrame (when len(tickers)==1)
                    if len(tickers) == 1 and t == tickers[0]:
                        # Validate required columns exist
                        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                        if all(col in all_data.columns for col in required_cols):
                            df = all_data[required_cols].copy()
                        else:
                            df = None
                    else:
                        df = None
                    
                    if df is None:
                        # Shouldn't happen, but fallback
                        if p['src'] == "Manual Input":
                            st.session_state.rejected.append(f"""<div class="rejected-card"><span class="rej-head">{t}</span><span class="rej-sub">NO DATA</span></div>""")
                        continue
            else:
                # Fallback: sequential download if batch failed
                try:
                    df = yf.download(t, period=fetch_period, interval=inter, progress=False, auto_adjust=False, multi_level_index=False)
                    # Validate downloaded data has required columns
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if not all(col in df.columns for col in required_cols):
                        df = None
                except Exception as e:
                    df = None
                
                if df is None:
                    if p['src'] == "Manual Input":
                        st.session_state.rejected.append(f"""<div class="rejected-card"><span class="rej-head">{t}</span><span class="rej-sub">DOWNLOAD ERROR</span></div>""")
                    continue
            
            # A. Data Validation Check
            if df is None or df.empty or len(df) < p['sma'] + 5:
                if p['src'] == "Manual Input":
                    st.session_state.rejected.append(f"""<div class="rejected-card"><span class="rej-head">{t}</span><span class="rej-sub">NO DATA</span></div>""")
                continue
            
            # Validate required columns exist and have valid data
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                if p['src'] == "Manual Input":
                    st.session_state.rejected.append(f"""<div class="rejected-card"><span class="rej-head">{t}</span><span class="rej-sub">MISSING COLUMNS</span></div>""")
                continue
            
            # Drop rows with NaN in critical columns
            df = df.dropna(subset=['Close', 'High', 'Low', 'Open'])
            if len(df) < p['sma'] + 5:
                if p['src'] == "Manual Input":
                    st.session_state.rejected.append(f"""<div class="rejected-card"><span class="rej-head">{t}</span><span class="rej-sub">INSUFFICIENT DATA</span></div>""")
                continue

            # B. Logic
            df = run_vova_logic(df, p['sma'], EMA_F, EMA_S, ADX_L, ADX_T, ATR_L)
            
            # C. Analyze
            valid, d, reason = analyze_trade(df, -1)
            
            # REJECTION HANDLING
            if not valid:
                if p['src'] == "Manual Input":
                    pr = df['Close'].iloc[-1]
                    h = f"""<div class="rejected-card"><div><span class="rej-head">{t}</span> <span style="font-size:9px;color:#555">${pr:.2f}</span></div><span class="rej-sub">{reason}</span></div>"""
                    st.session_state.rejected.append(h)
                continue
            
            # D. Filters
            # New Only
            valid_prev, _, _ = analyze_trade(df, -2)
            is_new = not valid_prev
            if p['src'] == "All S&P 500" and p['new'] and not is_new: continue
            
            # RR
            if d['RR'] < p['rr']:
                if p['src'] == "Manual Input":
                    st.session_state.rejected.append(f"""<div class="rejected-card"><span class="rej-head">{t}</span><span class="rej-sub">LOW RR {d['RR']:.2f}</span></div>""")
                continue
                
            # ATR
            atr_pct = (d['ATR']/d['P'])*100
            if atr_pct > p['matr']:
                if p['src'] == "Manual Input":
                    st.session_state.rejected.append(f"""<div class="rejected-card"><span class="rej-head">{t}</span><span class="rej-sub">HIGH VOL {atr_pct:.1f}%</span></div>""")
                continue
                
            # E. Position Sizing (using $ risk per trade directly)
            risk_per_share = d['P'] - d['SL']
            if risk_per_share <= 0: continue 
            
            # Calculate shares: risk_per_trade / risk_per_share
            shares = int(p['risk_per_trade'] / risk_per_share)
            
            if shares < 1:
                if p['src'] == "Manual Input":
                    st.session_state.rejected.append(f"""<div class="rejected-card"><span class="rej-head">{t}</span><span class="rej-sub">LOW FUNDS</span></div>""")
                continue
                
            # F. Prepare Data
            pe = get_financial_info(t)  # Fetch P/E ratio
            pe_s = f"P/E {pe:.1f}" if pe is not None and not np.isnan(pe) else ""
            tv = f"https://www.tradingview.com/chart/?symbol={t.replace('-', '.')}"
            badge = '<span class="new-badge">NEW</span>' if is_new else ""
            
            val_pos = shares * d['P']
            profit_pot = (d['TP'] - d['P']) * shares
            loss_pot = (d['P'] - d['SL']) * shares
            
            # G. Generate HTML - Compact Luxury Design
            html = f"""
            <div class="ticker-card">
                <div class="card-header">
                    <div class="card-header-left">
                        <a href="{tv}" target="_blank" class="t-link">{t}</a>
                        {badge}
                    </div>
                    <div class="header-price-block">
                        <span class="t-price">${d['P']:.2f}</span>
                        {f'<span class="t-pe">{pe_s}</span>' if pe_s else ''}
                    </div>
                </div>
                <div class="card-grid">
                    <div class="stat-row"><span class="lbl">POS</span> <div><span class="val c-gold">{shares}</span> <span class="sub c-gold">${val_pos:.0f}</span></div></div>
                    <div class="stat-row"><span class="lbl">R:R</span> <span class="val c-blue">{d['RR']:.2f}</span></div>
                    <div class="stat-row"><span class="lbl">TARGET</span> <div><span class="val c-green">${d['TP']:.2f}</span> <span class="sub c-green">+${profit_pot:.0f}</span></div></div>
                    <div class="stat-row"><span class="lbl">STOP</span> <div><span class="val c-red">${d['SL']:.2f}</span> <span class="sub c-red">-${loss_pot:.0f}</span></div></div>
                    <div class="stat-row"><span class="lbl">CRIT</span> <div><span class="val">${d['Crit']:.2f}</span></div></div>
                    <div class="stat-row"><span class="lbl">ATR(14)</span> <div><span class="val">${d['ATR']:.2f}</span> <span class="sub">{atr_pct:.1f}%</span></div></div>
                </div>
            </div>
            """
            st.session_state.results.append(html)
            
            # OPTIMIZATION 2: Update UI every 10 tickers instead of every ticker (much faster)
            if (i + 1) % 10 == 0 or i == len(tickers) - 1:
                with res_area.container():
                    current_list = st.session_state.results + (st.session_state.rejected if p['src'] == "Manual Input" else [])
                    if current_list:
                        # 6 columns for more cards per row
                        cols = st.columns(6)
                        for idx, h in enumerate(current_list):
                            with cols[idx % 6]:
                                render_html(h)
                            
        except Exception as e:
            pass

    # Final UI update
    bar.empty()
    with res_area.container():
        final_list = st.session_state.results + (st.session_state.rejected if p['src'] == "Manual Input" else [])
        if final_list:
            # 6 columns for more cards per row
            cols = st.columns(6)
            for idx, h in enumerate(final_list):
                with cols[idx % 6]:
                    render_html(h)
    
    st.session_state.scanning = False
    info_box.success("SCAN COMPLETE")

# --- PERSISTENT DISPLAY (When not scanning) ---
else:
    # Use params from last run or default for display logic
    last_src = st.session_state.run_params.get('src', "All S&P 500")
    
    final_list = st.session_state.results + (st.session_state.rejected if last_src == "Manual Input" else [])
    
    with res_area.container():
        if final_list:
            # 6 columns for more cards per row
            cols = st.columns(6)
            for idx, h in enumerate(final_list):
                with cols[idx % 6]:
                    render_html(h)
        else:
            st.info("Ready to scan. Click START.")
