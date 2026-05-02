import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

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

# --- CSS STYLING (from ui_styles.py) ---
from ui_styles import inject_styles
inject_styles()

# ==========================================
# 2. DATA & API (from ticker_data.py)
# ==========================================
from ticker_data import (
    TV_LIST_BIG_CAP,
    TV_LIST_ETF,
    TV_LIST_SMALL_CAP,
    fetch_fallback_company_name,
    get_ticker_info_and_filter,
    read_list_file,
)

# ==========================================
# 3. SEQUENCE VOVA (from sequence_vova.py)
# ==========================================
from sequence_vova import run_sequence_vova_pine

# ==========================================
# 4. UI & SIDEBAR
# ==========================================
from ticker_sources import FileListSource, MergedListSource, ManualSource

# Ticker sources: only from the 3 TXT files (SMALL CAP, BIG CAP, ETF) + ALL (merged) + MANUAL
SOURCE_OPTIONS = ["SMALL CAP", "BIG CAP", "ETFS", "ALL", "MANUAL SCAN"]
def _build_source_registry():
    return {
        "SMALL CAP": FileListSource(TV_LIST_SMALL_CAP, read_list_file),
        "BIG CAP": FileListSource(TV_LIST_BIG_CAP, read_list_file),
        "ETFS": FileListSource(TV_LIST_ETF, read_list_file),
        "ALL": MergedListSource([TV_LIST_SMALL_CAP, TV_LIST_BIG_CAP, TV_LIST_ETF], read_list_file),
    }

SOURCE_REGISTRY = _build_source_registry()

st.sidebar.header("⚙️ CONFIGURATION")

# Disable inputs if scanning
disabled = st.session_state.scanning

last_src = st.session_state.get("run_params", {}).get("src", "SMALL CAP")
default_idx = SOURCE_OPTIONS.index(last_src) if last_src in SOURCE_OPTIONS else 0
src = st.sidebar.radio("SOURCE", SOURCE_OPTIONS, disabled=disabled, index=default_idx)
if src in SOURCE_REGISTRY:
    st.sidebar.caption(SOURCE_REGISTRY[src].description())
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


@dataclass
class ScanConfig:
    """Scan parameters; built from run_params dict for type safety and centralization."""
    risk_per_trade: int
    min_rr: float
    use_last_hl_sl: bool
    tf: str
    new_only: bool
    is_manual_src: bool

    @classmethod
    def from_run_params(cls, p: dict) -> "ScanConfig":
        return cls(
            risk_per_trade=int(p.get("risk_per_trade", 100)),
            min_rr=float(p.get("rr", 1.5)),
            use_last_hl_sl=bool(p.get("use_last_hl_sl", True)),
            tf=str(p.get("tf", "Daily")),
            new_only=bool(p.get("new", True)),
            is_manual_src=(p.get("src") == "MANUAL SCAN"),
        )


ATR_LEN = 14
MIN_BARS = 50  # minimum bars for sequence logic
CHUNK_SIZE = 500  # batch size for yf.download
YF_INFO_RETRY_DELAY_SEC = 0.25  # delay before retrying get_ticker_info_and_filter on INFO_ERROR
YF_DOWNLOAD_MAX_RETRIES = 2  # retry batch download up to this many times on failure
YF_INFO_RATE_LIMIT_PER_SEC = 12  # max .info requests per second when using parallel fetch
YF_INFO_MAX_WORKERS = 8  # thread pool size for parallel get_ticker_info_and_filter

# Results Placeholder
res_area = st.empty()


def render_scan_results(table_rows, rejected_reasons, reference_end_date, tf, is_manual_src, empty_message="Ready to scan. Click START."):
    """
    Render scan results: dataframe, as-of caption, and rejected/skipped expander.
    Used by both "scan just finished" and "idle show last results" paths.
    """
    if table_rows:
        res_df = pd.DataFrame(table_rows)
        col_config = {"Symbol": st.column_config.LinkColumn("Symbol", display_text=r"symbol=([^&]+)")}
        # height="content": table grows with all rows; only the page scrolls (no nested grid scroll until Streamlit cap).
        st.dataframe(
            res_df,
            hide_index=True,
            column_config=col_config,
            width="stretch",
            height="content",
        )
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
        st.info(empty_message)
    if is_manual_src and rejected_reasons:
        with st.expander("Rejected (Manual)"):
            st.dataframe(pd.DataFrame(rejected_reasons), width="stretch", hide_index=True)
    elif rejected_reasons and any(str(r.get("Reason", "")).startswith("ERROR") for r in rejected_reasons):
        with st.expander("Skipped (errors)"):
            st.dataframe(pd.DataFrame(rejected_reasons), width="stretch", hide_index=True)


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
    rule = "W-FRI" if tf == "Weekly" else "ME"
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


def run_scan(
    tickers,
    *,
    risk_per_trade,
    min_rr,
    use_last_hl_sl,
    tf,
    new_only,
    is_manual_src,
    on_progress=None,
    on_status=None,
    is_cancelled=None,
):
    """
    Pure scanner: batch download, then per-ticker filter + OHLCV + sequence. No Streamlit calls.
    Returns (table_rows, rejected_reasons, reference_end_date).
    """
    inter, fetch_period = _interval_and_period(tf)
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if len(tickers) > CHUNK_SIZE:
        batches = [tickers[i:i + CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]
    else:
        batches = [tickers]

    table_rows = []
    rejected_reasons = []
    reference_end_date = None
    batches_data = []

    # Progress: info phase + download phase + process phase so the bar moves through the whole scan
    total_steps = len(tickers) + len(batches) + len(tickers)
    step = [0]  # use list so inner function can update

    # Pre-fetch ticker info in parallel with rate limiter (faster than per-ticker in loop)
    _rate_limiter_lock = threading.Lock()
    _rate_limiter_last = [0.0]

    def _rate_limited_info(ticker):
        with _rate_limiter_lock:
            now = time.monotonic()
            wait = (1.0 / YF_INFO_RATE_LIMIT_PER_SEC) - (now - _rate_limiter_last[0])
            if wait > 0:
                time.sleep(wait)
            passed, reason, info_dict = get_ticker_info_and_filter(ticker, min_market_cap=5e9, min_avg_volume=300_000, require_mc_vol=False)
            _rate_limiter_last[0] = time.monotonic()
        if not passed and info_dict is None:
            time.sleep(YF_INFO_RETRY_DELAY_SEC)
            with _rate_limiter_lock:
                now = time.monotonic()
                wait = (1.0 / YF_INFO_RATE_LIMIT_PER_SEC) - (now - _rate_limiter_last[0])
                if wait > 0:
                    time.sleep(wait)
                _, _, info_dict = get_ticker_info_and_filter(ticker, min_market_cap=5e9, min_avg_volume=300_000, require_mc_vol=False)
                _rate_limiter_last[0] = time.monotonic()
            if info_dict is None:
                info_dict = {"company_name": fetch_fallback_company_name(ticker), "avg_volume": None}
        return (ticker, passed, reason, info_dict)

    info_cache = {}
    if on_status:
        on_status("Fetching ticker info... DO NOT REFRESH.")
    with ThreadPoolExecutor(max_workers=YF_INFO_MAX_WORKERS) as executor:
        futures = {executor.submit(_rate_limited_info, t): t for t in tickers}
        for future in as_completed(futures):
            if is_cancelled and is_cancelled():
                break
            try:
                t, passed, reason, info_dict = future.result()
                info_cache[t] = (passed, reason, info_dict)
            except Exception:
                t = futures[future]
                info_cache[t] = (False, "INFO_ERROR", {"company_name": fetch_fallback_company_name(t), "avg_volume": None})
            step[0] += 1
            if on_progress:
                on_progress(step[0], total_steps)

    for batch_idx, batch in enumerate(batches):
        if is_cancelled and is_cancelled():
            break
        if on_status:
            on_status(f"Downloading batch {batch_idx + 1}/{len(batches)} ({len(batch)} tickers)... DO NOT REFRESH.")
        all_data = None
        for attempt in range(YF_DOWNLOAD_MAX_RETRIES):
            try:
                all_data = yf.download(
                    batch,
                    period=fetch_period,
                    interval=inter,
                    progress=False,
                    auto_adjust=False,
                    group_by='ticker',
                    threads=False,
                )
                break
            except Exception:
                if attempt < YF_DOWNLOAD_MAX_RETRIES - 1:
                    time.sleep(YF_INFO_RETRY_DELAY_SEC * (attempt + 1))
        if all_data is None or all_data.empty:
            batches_data.append((batch, None))
        else:
            batch_end = all_data.index[-1] if hasattr(all_data.index, '__len__') and len(all_data.index) > 0 else None
            if batch_end is not None:
                reference_end_date = batch_end if reference_end_date is None else min(reference_end_date, batch_end)
            batches_data.append((batch, all_data))
        step[0] += 1
        if on_progress:
            on_progress(step[0], total_steps)

    if reference_end_date is None and batches_data:
        for _, all_data in batches_data:
            if all_data is not None and not all_data.empty and len(all_data.index) > 0:
                reference_end_date = all_data.index[-1]
                break

    for batch_idx, (batch, all_data) in enumerate(batches_data):
        if is_cancelled and is_cancelled():
            break
        if on_status:
            on_status(f"Processing batch {batch_idx + 1}/{len(batches)}... DO NOT REFRESH.")
        if all_data is not None and not all_data.empty and reference_end_date is not None and len(all_data.index) > 0 and all_data.index[-1] > reference_end_date:
            all_data = all_data.loc[all_data.index <= reference_end_date]
        for i, t in enumerate(batch):
            if is_cancelled and is_cancelled():
                break
            step[0] += 1
            if on_progress:
                on_progress(step[0], total_steps)
            try:
                passed, reject_reason, info_dict = info_cache.get(t, (False, "INFO_ERROR", {"company_name": t, "avg_volume": None}))
                if info_dict is None:
                    info_dict = {"company_name": t, "avg_volume": None}
                elif not isinstance(info_dict, dict):
                    info_dict = {"company_name": str(t), "avg_volume": None}
                else:
                    info_dict.setdefault("company_name", t)
                if not passed:
                    if is_manual_src:
                        # Manual: Yahoo .info can fail (INFO_ERROR) while history still works — do not block TA scan.
                        if reject_reason != "INFO_ERROR":
                            rejected_reasons.append({"Symbol": t, "Reason": reject_reason})
                            continue
                    # info_dict from cache is never None (fallback applied in pre-fetch)

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
                    if is_manual_src:
                        rejected_reasons.append({"Symbol": t, "Reason": "NO_DATA"})
                    continue

                if reference_end_date is not None and len(df.index) > 0 and df.index[-1] > reference_end_date:
                    df = df.loc[df.index <= reference_end_date]
                if reference_end_date is None and len(df.index) > 0:
                    reference_end_date = df.index[-1]

                df = _resample_to_timeframe(df, tf)
                if df is None or df.empty or len(df) < MIN_BARS:
                    if is_manual_src:
                        rejected_reasons.append({"Symbol": t, "Reason": "NO_DATA"})
                    continue

                df = _fill_last_bar_ohlc(df)
                df = df.dropna(subset=['Close', 'High', 'Low', 'Open'])
                if len(df) < MIN_BARS:
                    if is_manual_src:
                        rejected_reasons.append({"Symbol": t, "Reason": "INSUFFICIENT_DATA"})
                    continue

                out = run_sequence_vova_pine(
                    df,
                    atr_len=ATR_LEN,
                    min_rr=min_rr,
                    use_last_hl_sl=use_last_hl_sl,
                    risk_dollars=risk_per_trade
                )
                if out is None or not out["Valid"]:
                    if is_manual_src:
                        rejected_reasons.append({"Symbol": t, "Reason": "NO_VALID_SIGNAL"})
                    continue
                if new_only and not out["New"]:
                    continue

                pos_size = out["position_size"]
                if np.isnan(pos_size) or pos_size < 1:
                    pos_size = 0
                else:
                    pos_size = int(round(pos_size))
                pos_value = out["position_value"] if not np.isnan(out["position_value"]) else 0.0

                tv_url = f"https://www.tradingview.com/chart/?symbol={t}"
                table_rows.append({
                    "Symbol": tv_url,
                    "Company Name": info_dict["company_name"],
                    "TP": round(float(out["TP"]), 2),
                    "SL": round(float(out["SL"]), 2),
                    "RR": round(float(out["RR"]), 2),
                    "Position Size (shares)": pos_size,
                    "Position Value ($)": round(float(pos_value), 2),
                    "New": 1 if out["New"] else 0,
                    "Valid": 1 if out["Valid"] else 0,
                    "Strong": 1 if out["Strong"] else 0,
                })
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                if len(msg) > 200:
                    msg = msg[:197] + "..."
                rejected_reasons.append({"Symbol": t, "Reason": f"ERROR: {msg}"})

    return (table_rows, rejected_reasons, reference_end_date)


if st.session_state.scanning:
    p = st.session_state.run_params
    cfg = ScanConfig.from_run_params(p)
    if p["src"] == "MANUAL SCAN":
        source = ManualSource(lambda: p["txt"])
    else:
        source = SOURCE_REGISTRY[p["src"]]
    tickers, err = source.get_tickers()
    if err:
        st.warning(err)
    if not tickers:
        st.error("NO TICKERS FOUND")
        st.session_state.scanning = False
        st.stop()

    info_box = st.empty()
    bar = st.progress(0)
    pct_placeholder = st.empty()

    def on_progress(processed, total):
        if total:
            bar.progress(0.05 + 0.95 * (processed / total))
            pct = round(100 * processed / total)
            pct_placeholder.markdown(f"**{pct}%**")

    def on_status(msg):
        info_box.info(msg)

    table_rows, rejected_reasons, reference_end_date = run_scan(
        tickers,
        risk_per_trade=cfg.risk_per_trade,
        min_rr=cfg.min_rr,
        use_last_hl_sl=cfg.use_last_hl_sl,
        tf=cfg.tf,
        new_only=cfg.new_only,
        is_manual_src=cfg.is_manual_src,
        on_progress=on_progress,
        on_status=on_status,
        is_cancelled=lambda: not st.session_state.scanning,
    )

    bar.empty()
    pct_placeholder.empty()
    st.session_state.results = table_rows
    st.session_state.rejected = rejected_reasons
    st.session_state.results_as_of = reference_end_date
    st.session_state.results_tf = cfg.tf

    with res_area.container():
        render_scan_results(
            table_rows, rejected_reasons, reference_end_date, cfg.tf,
            is_manual_src=cfg.is_manual_src,
            empty_message="No symbols passed the screener.",
        )

    st.session_state.scanning = False
    info_box.success("SCAN COMPLETE")

else:
    last_src = st.session_state.run_params.get('src', "SMALL CAP")
    table_rows = st.session_state.results
    rejected_reasons = st.session_state.rejected
    as_of = st.session_state.get("results_as_of")
    as_of_tf = st.session_state.get("results_tf", "Daily")

    with res_area.container():
        render_scan_results(
            table_rows, rejected_reasons, as_of, as_of_tf,
            is_manual_src=(last_src == "MANUAL SCAN"),
        )

