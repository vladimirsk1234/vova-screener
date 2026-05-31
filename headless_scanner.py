import logging
import os
import re
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

_log = logging.getLogger(__name__)

# ==========================================
# 1. PAGE CONFIG & STYLES (TERMINAL UI)
# ==========================================
st.set_page_config(
    page_title="Screener Vova (Terminal)",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="collapsed",
)

# --- SESSION STATE INITIALIZATION ---
if 'scanning' not in st.session_state:
    st.session_state.scanning = False
if 'results' not in st.session_state:
    st.session_state.results = []
if 'rejected' not in st.session_state:
    st.session_state.rejected = []
if 'run_params' not in st.session_state:
    st.session_state.run_params = {} # To freeze params during scan
if 'chart_cache' not in st.session_state:
    st.session_state.chart_cache = {}
if 'ohlc_cache' not in st.session_state:
    st.session_state.ohlc_cache = {}
if 'selected_tv_symbol' not in st.session_state:
    st.session_state.selected_tv_symbol = None

# --- CSS STYLING (from ui_styles.py) ---
from ui_styles import inject_styles
from chart_preview import (
    DEFAULT_CHART_HEIGHT,
    PLOTLY_CHART_CONFIG,
    company_description_from_payload,
    figure_from_payload,
    resolve_chart_payload,
)
from chart_settings_ui import render_chart_settings
inject_styles()

# ==========================================
# 2. DATA & API (from ticker_data.py)
# ==========================================
from ticker_data import (
    TV_LIST_BIG_CAP,
    TV_LIST_ETF,
    TV_LIST_SMALL_CAP,
    build_name_cache,
    get_ticker_info_and_filter,
    read_list_file,
    resolve_company_name,
)

# ==========================================
# 3. SEQUENCE VOVA (from sequence_vova.py)
# ==========================================
from sequence_vova import run_sequence_vova_pine
from data_utils import (
    extract_ohlcv as _extract_ohlcv,
    fill_last_bar_ohlc as _fill_last_bar_ohlc,
    interval_and_period as _interval_and_period,
    resample_to_timeframe as _resample_to_timeframe,
    split_batch_ohlcv as _split_batch_ohlcv,
)
from tradingview_embed import (
    build_chart_url,
    infer_tv_symbol,
    tf_to_tv_interval,
)

# ==========================================
# 4. UI & SIDEBAR
# ==========================================
from ticker_sources import FileListSource, ManualSource

# Ticker sources: list files + optional MANUAL symbols
SOURCE_OPTIONS = ["BIG CAP", "SMALL CAP", "ETF", "MANUAL SCAN"]
def _build_source_registry():
    return {
        "BIG CAP": FileListSource(TV_LIST_BIG_CAP, read_list_file),
        "SMALL CAP": FileListSource(TV_LIST_SMALL_CAP, read_list_file),
        "ETF": FileListSource(TV_LIST_ETF, read_list_file),
    }

SOURCE_REGISTRY = _build_source_registry()

st.sidebar.header("⚙️ CONFIGURATION")

# Disable inputs if scanning
disabled = st.session_state.scanning

last_src = st.session_state.get("run_params", {}).get("src", "BIG CAP")
default_idx = SOURCE_OPTIONS.index(last_src) if last_src in SOURCE_OPTIONS else 0
src = st.sidebar.radio("SOURCE", SOURCE_OPTIONS, disabled=disabled, index=default_idx)
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
    st.session_state.chart_cache = {}
    st.session_state.ohlc_cache = {}
    st.session_state.selected_tv_symbol = None
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
CHUNK_SIZE = 200  # batch size for yf.download (smaller chunks = more parallel downloads)
DOWNLOAD_MAX_WORKERS = 3  # parallel yf.download batches
YF_INFO_RETRY_DELAY_SEC = 0.25  # delay before retrying get_ticker_info_and_filter on INFO_ERROR
YF_DOWNLOAD_MAX_RETRIES = 2  # retry batch download up to this many times on failure
YF_DOWNLOAD_BACKOFF_SEC = 2.0  # extra delay after failed parallel batch (rate limit)
YF_INFO_RATE_LIMIT_PER_SEC = 12  # max .info requests per second when using parallel fetch
YF_INFO_MAX_WORKERS = 8  # thread pool size for parallel get_ticker_info_and_filter
TA_MAX_WORKERS = min(16, max(4, (os.cpu_count() or 8) * 2))  # parallel per-ticker TA after download

_name_resolve_lock = threading.Lock()
_name_resolve_last = [0.0]


def _rate_limited_resolve_company_name(ticker: str) -> str:
    with _name_resolve_lock:
        now = time.monotonic()
        wait = (1.0 / YF_INFO_RATE_LIMIT_PER_SEC) - (now - _name_resolve_last[0])
        if wait > 0:
            time.sleep(wait)
        name = resolve_company_name(ticker)
        _name_resolve_last[0] = time.monotonic()
    return name


def _yahoo_ticker_from_row(row: dict) -> str:
    tv = str(row.get("tv_symbol", "") or "")
    if ":" in tv:
        return tv.split(":", 1)[1].replace(".", "-").upper()
    sym = str(row.get("Symbol", "") or "")
    if "symbol=" in sym:
        m = re.search(r"symbol=([^&]+)", sym)
        if m:
            return m.group(1).replace(".", "-").upper()
    return sym.replace(".", "-").upper()


def _patch_symbol_only_company_names(table_rows: list[dict]) -> None:
    """Second pass: resolve names for rows that still show only the ticker symbol."""
    pending: list[tuple[dict, str]] = []
    for row in table_rows:
        ticker = _yahoo_ticker_from_row(row)
        company = str(row.get("Company Name", "") or "").strip()
        if not company or company.upper() == ticker.upper():
            pending.append((row, ticker))
    if not pending:
        return

    def _resolve_one(item: tuple[dict, str]) -> tuple[dict, str, str]:
        row, ticker = item
        return row, ticker, _rate_limited_resolve_company_name(ticker)

    with ThreadPoolExecutor(max_workers=YF_INFO_MAX_WORKERS) as pool:
        for row, ticker, name in pool.map(_resolve_one, pending):
            if name and name.upper() != ticker.upper():
                row["Company Name"] = name

# Results Placeholder
res_area = st.empty()


def _display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Hide internal tv_symbol from the results grid."""
    hide = {"tv_symbol"}
    cols = [c for c in df.columns if c not in hide]
    return df[cols] if cols else df


def render_scan_results(
    table_rows,
    rejected_reasons,
    reference_end_date,
    tf,
    is_manual_src,
    chart_cache=None,
    ohlc_cache=None,
    empty_message="Ready to scan. Click START.",
):
    """
    Render scan results: dataframe, as-of caption, and rejected/skipped expander.
    Used by both "scan just finished" and "idle show last results" paths.
    """
    chart_cache = chart_cache if chart_cache is not None else {}
    ohlc_cache = ohlc_cache if ohlc_cache is not None else {}
    has_charts = bool(chart_cache) or bool(ohlc_cache)

    if table_rows:
        res_df = pd.DataFrame(table_rows)
        display_df = _display_columns(res_df)
        col_config = {"Symbol": st.column_config.LinkColumn("Symbol", display_text=r"symbol=([^&]+)")}

        event = st.dataframe(
            display_df,
            hide_index=True,
            column_config=col_config,
            width="stretch",
            height="content",
            on_select="rerun",
            selection_mode="single-row",
            key="scan_results_table",
        )

        if has_charts:
            st.caption(f"Click a row for chart preview ({tf}).")
        else:
            st.caption("Run a scan to enable chart previews.")

        tv_sym = None
        if has_charts:
            selected = getattr(event, "selection", None)
            sel_rows = list(selected.rows) if selected and getattr(selected, "rows", None) else []
            if sel_rows:
                tv_sym = str(res_df.iloc[sel_rows[0]].get("tv_symbol", "") or "")
                if tv_sym:
                    st.session_state.selected_tv_symbol = tv_sym
            elif st.session_state.get("selected_tv_symbol"):
                tv_sym = st.session_state.selected_tv_symbol

            chart_params = render_chart_settings()
            payload = (
                resolve_chart_payload(tv_sym, chart_cache=chart_cache, ohlc_cache=ohlc_cache)
                if tv_sym
                else None
            )
            if payload:
                fig = figure_from_payload(
                    payload,
                    symbol=tv_sym or "",
                    params=chart_params,
                    height=DEFAULT_CHART_HEIGHT,
                )
                if fig is not None:
                    with st.container(border=True):
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config=PLOTLY_CHART_CONFIG,
                        )
                        desc = company_description_from_payload(
                            payload, symbol=tv_sym or ""
                        )
                        if desc:
                            st.markdown("**About**")
                            st.markdown(desc)
                elif tv_sym:
                    st.caption("Failed to build chart.")
            elif tv_sym and has_charts:
                st.caption("No chart cache for the selected row.")

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


def _download_batch(
    batch_idx: int,
    batch: list[str],
    fetch_period: str,
    inter: str,
) -> tuple[int, list[str], pd.DataFrame | None]:
    all_data = None
    for attempt in range(YF_DOWNLOAD_MAX_RETRIES):
        try:
            all_data = yf.download(
                batch,
                period=fetch_period,
                interval=inter,
                progress=False,
                auto_adjust=False,
                group_by="ticker",
                threads=True,
            )
            break
        except Exception:
            if attempt < YF_DOWNLOAD_MAX_RETRIES - 1:
                time.sleep(YF_INFO_RETRY_DELAY_SEC * (attempt + 1))
    if all_data is None or (hasattr(all_data, "empty") and all_data.empty):
        return batch_idx, batch, None
    return batch_idx, batch, all_data


def _process_ticker_for_scan(
    t: str,
    ticker_df: pd.DataFrame | None,
    required_cols: list[str],
    fetch_period: str,
    inter: str,
    tf: str,
    reference_end_date,
    risk_per_trade: int,
    min_rr: float,
    use_last_hl_sl: bool,
    new_only: bool,
    is_manual_src: bool,
    lazy_metadata: bool,
    info_cache_entry: tuple[bool, str, dict | None] | None,
    tv_symbol_by_ticker: dict[str, str] | None = None,
    name_cache: dict[str, str] | None = None,
) -> dict:
    """
    Pure per-ticker work for one symbol. Returns:
    {"kind": "skip"} | {"kind": "reject", "row": dict} | {"kind": "row", "row": dict}
    """
    try:
        if lazy_metadata:
            passed = True
            reject_reason = ""
            info_dict: dict = {"company_name": t, "avg_volume": None}
        else:
            passed, reject_reason, info_dict = info_cache_entry or (
                False,
                "INFO_ERROR",
                {"company_name": t, "avg_volume": None},
            )
            if info_dict is None:
                info_dict = {"company_name": t, "avg_volume": None}
            elif not isinstance(info_dict, dict):
                info_dict = {"company_name": str(t), "avg_volume": None}
            else:
                info_dict.setdefault("company_name", t)
            if not passed:
                if is_manual_src:
                    if reject_reason != "INFO_ERROR":
                        return {"kind": "reject", "row": {"Symbol": t, "Reason": reject_reason}}
                # file list: still run TA (same as previous inline behavior)

        df = ticker_df
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
                return {"kind": "reject", "row": {"Symbol": t, "Reason": "NO_DATA"}}
            return {"kind": "skip"}

        ref_end = reference_end_date
        if ref_end is not None and len(df.index) > 0 and df.index[-1] > ref_end:
            df = df.loc[df.index <= ref_end]
        if ref_end is None and len(df.index) > 0:
            pass  # reference_end_date alignment handled in main scan

        df = df.copy()
        df_daily_chart = df.copy()
        df = _resample_to_timeframe(df, tf)
        if df is None or df.empty or len(df) < MIN_BARS:
            if is_manual_src:
                return {"kind": "reject", "row": {"Symbol": t, "Reason": "NO_DATA"}}
            return {"kind": "skip"}

        df = _fill_last_bar_ohlc(df)
        df = df.dropna(subset=["Close", "High", "Low", "Open"])
        if len(df) < MIN_BARS:
            if is_manual_src:
                return {"kind": "reject", "row": {"Symbol": t, "Reason": "INSUFFICIENT_DATA"}}
            return {"kind": "skip"}

        out = run_sequence_vova_pine(
            df,
            atr_len=ATR_LEN,
            min_rr=min_rr,
            use_last_hl_sl=use_last_hl_sl,
            risk_dollars=risk_per_trade,
        )
        if out is None or not out["Valid"]:
            if is_manual_src:
                return {"kind": "reject", "row": {"Symbol": t, "Reason": "NO_VALID_SIGNAL"}}
            return {"kind": "skip"}
        if new_only and not out["New"]:
            return {"kind": "skip"}

        pos_size = out["position_size"]
        if np.isnan(pos_size) or pos_size < 1:
            pos_size = 0
        else:
            pos_size = int(round(pos_size))
        pos_value = out["position_value"] if not np.isnan(out["position_value"]) else 0.0

        tv_sym = (tv_symbol_by_ticker or {}).get(t) or infer_tv_symbol(t)
        interval = tf_to_tv_interval(tf)
        tv_url = build_chart_url(tv_sym, interval)
        if lazy_metadata and name_cache is not None:
            company_name = name_cache.get(t, t)
        else:
            company_name = info_dict.get("company_name", t)

        table_row = {
            "Symbol": tv_url,
            "tv_symbol": tv_sym,
            "Company Name": company_name,
            "TP": round(float(out["TP"]), 2),
            "SL": round(float(out["SL"]), 2),
            "RR": round(float(out["RR"]), 2),
            "Position Size (shares)": pos_size,
            "Position Value ($)": round(float(pos_value), 2),
            "New": 1 if out["New"] else 0,
            "Valid": 1 if out["Valid"] else 0,
            "Strong": 1 if out["Strong"] else 0,
        }
        ohlc_entry = {
            "df": df.copy(),
            "df_daily": df_daily_chart.copy(),
            "tf": tf,
            "symbol": tv_sym,
            "yahoo_ticker": t,
        }
        return {
            "kind": "row",
            "row": table_row,
            "chart_key": tv_sym,
            "ohlc_entry": ohlc_entry,
        }
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if len(msg) > 200:
            msg = msg[:197] + "..."
        return {"kind": "reject", "row": {"Symbol": t, "Reason": f"ERROR: {msg}"}}


def _parallel_download_batches(
    batches: list[list[str]],
    fetch_period: str,
    inter: str,
    is_cancelled=None,
    on_status=None,
    on_batch_done=None,
) -> tuple[list[tuple[list[str], pd.DataFrame | None]], object | None]:
    """Download all chunks in parallel; return batches sorted by index."""
    reference_end_date = None
    if not batches:
        return [], None

    workers = min(DOWNLOAD_MAX_WORKERS, len(batches))
    raw: list[tuple[int, list[str], pd.DataFrame | None]] = []

    if on_status:
        on_status(
            f"Downloading {len(batches)} batches in parallel (up to {workers} workers)... DO NOT REFRESH."
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_download_batch, batch_idx, batch, fetch_period, inter)
            for batch_idx, batch in enumerate(batches)
        ]
        for fut in as_completed(futures):
            if is_cancelled and is_cancelled():
                break
            try:
                batch_idx, batch, all_data = fut.result()
            except Exception:
                time.sleep(YF_DOWNLOAD_BACKOFF_SEC)
                continue
            raw.append((batch_idx, batch, all_data))
            if all_data is not None and not all_data.empty and len(all_data.index) > 0:
                batch_end = all_data.index[-1]
                reference_end_date = (
                    batch_end
                    if reference_end_date is None
                    else min(reference_end_date, batch_end)
                )
            if on_batch_done:
                on_batch_done()

    raw.sort(key=lambda x: x[0])
    return [(batch, data) for _, batch, data in raw], reference_end_date


def run_scan(
    tickers,
    *,
    risk_per_trade,
    min_rr,
    use_last_hl_sl,
    tf,
    new_only,
    is_manual_src,
    tv_symbol_by_ticker: dict[str, str] | None = None,
    company_name_by_ticker: dict[str, str] | None = None,
    on_progress=None,
    on_status=None,
    is_cancelled=None,
):
    """
    Pure scanner: optional parallel Yahoo metadata (manual only), threaded batch download,
    then parallel per-ticker OHLCV + sequence for list sources (lazy metadata: .info only for passes).
    Logs phase timings to the logger and stdout. No Streamlit calls.
    Returns (table_rows, rejected_reasons, reference_end_date, ohlc_cache).
    """
    inter, fetch_period = _interval_and_period(tf)
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if len(tickers) > CHUNK_SIZE:
        batches = [tickers[i:i + CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]
    else:
        batches = [tickers]

    table_rows = []
    rejected_reasons = []
    ohlc_cache: dict[str, dict] = {}
    reference_end_date = None
    batches_data: list[tuple[list[str], pd.DataFrame | None]] = []

    lazy_metadata = not is_manual_src
    use_embedded_names = lazy_metadata
    if lazy_metadata:
        total_steps = (len(batches) + len(tickers)) if use_embedded_names else (
            len(tickers) + len(batches) + len(tickers)
        )
    else:
        total_steps = len(tickers) + len(batches) + len(tickers)
    step = [0]

    t_scan0 = time.perf_counter()
    info_sec = 0.0
    info_cache: dict[str, tuple[bool, str, dict | None]] = {}
    name_cache: dict[str, str] = {}

    if not lazy_metadata:
        t_info0 = time.perf_counter()
        _rate_limiter_lock = threading.Lock()
        _rate_limiter_last = [0.0]

        def _rate_limited_info(ticker):
            with _rate_limiter_lock:
                now = time.monotonic()
                wait = (1.0 / YF_INFO_RATE_LIMIT_PER_SEC) - (now - _rate_limiter_last[0])
                if wait > 0:
                    time.sleep(wait)
                passed, reason, info_dict = get_ticker_info_and_filter(
                    ticker, min_market_cap=5e9, min_avg_volume=300_000, require_mc_vol=False
                )
                _rate_limiter_last[0] = time.monotonic()
            if not passed and info_dict is None:
                time.sleep(YF_INFO_RETRY_DELAY_SEC)
                with _rate_limiter_lock:
                    now = time.monotonic()
                    wait = (1.0 / YF_INFO_RATE_LIMIT_PER_SEC) - (now - _rate_limiter_last[0])
                    if wait > 0:
                        time.sleep(wait)
                    _, _, info_dict = get_ticker_info_and_filter(
                        ticker, min_market_cap=5e9, min_avg_volume=300_000, require_mc_vol=False
                    )
                    _rate_limiter_last[0] = time.monotonic()
                if info_dict is None:
                    info_dict = {"company_name": resolve_company_name(ticker), "avg_volume": None}
            return (ticker, passed, reason, info_dict)

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
                    info_cache[t] = (
                        False,
                        "INFO_ERROR",
                        {"company_name": resolve_company_name(t), "avg_volume": None},
                    )
                step[0] += 1
                if on_progress:
                    on_progress(step[0], total_steps)
        info_sec = time.perf_counter() - t_info0

    t_names_dl0 = time.perf_counter()
    if lazy_metadata:
        if use_embedded_names:
            name_cache = dict(company_name_by_ticker or {})
            for t in tickers:
                name_cache.setdefault(t, t)
            if on_status:
                on_status("Downloading OHLC... DO NOT REFRESH.")

            def _dl_batch_done_embedded():
                step[0] += 1
                if on_progress:
                    on_progress(step[0], total_steps)

            batches_data, reference_end_date = _parallel_download_batches(
                batches,
                fetch_period,
                inter,
                is_cancelled,
                on_status,
                _dl_batch_done_embedded,
            )
        else:
            if on_status:
                on_status("Loading company names (parallel with download)... DO NOT REFRESH.")

            # UI callbacks and st.session_state must stay on the main thread (NoSessionContext in workers).
            with ThreadPoolExecutor(max_workers=2) as prep_pool:
                name_future = prep_pool.submit(
                    build_name_cache,
                    tickers,
                    rate_limit_per_sec=YF_INFO_RATE_LIMIT_PER_SEC,
                    max_workers=YF_INFO_MAX_WORKERS,
                    is_cancelled=None,
                    on_one_done=None,
                )
                dl_future = prep_pool.submit(
                    _parallel_download_batches,
                    batches,
                    fetch_period,
                    inter,
                    None,
                    None,
                    None,
                )
                name_cache = name_future.result()
                batches_data, reference_end_date = dl_future.result()

            step[0] = len(tickers) + len(batches)
            if on_progress:
                on_progress(step[0], total_steps)
    else:

        def _dl_batch_done_manual():
            step[0] += 1
            if on_progress:
                on_progress(step[0], total_steps)

        batches_data, reference_end_date = _parallel_download_batches(
            batches,
            fetch_period,
            inter,
            is_cancelled,
            on_status,
            _dl_batch_done_manual,
        )

    prefetch_sec = time.perf_counter() - t_names_dl0

    if reference_end_date is None and batches_data:
        for _, all_data in batches_data:
            if all_data is not None and not all_data.empty and len(all_data.index) > 0:
                reference_end_date = all_data.index[-1]
                break

    def _merge_ticker_result(res: dict) -> None:
        if res["kind"] == "row":
            table_rows.append(res["row"])
            key = res.get("chart_key")
            entry = res.get("ohlc_entry")
            if key and entry:
                ohlc_cache[key] = entry
        elif res["kind"] == "reject":
            rejected_reasons.append(res["row"])

    t_proc0 = time.perf_counter()
    for batch_idx, (batch, all_data) in enumerate(batches_data):
        if is_cancelled and is_cancelled():
            break
        if on_status:
            on_status(f"Processing batch {batch_idx + 1}/{len(batches)}... DO NOT REFRESH.")
        slice_data = all_data
        if (
            slice_data is not None
            and not slice_data.empty
            and reference_end_date is not None
            and len(slice_data.index) > 0
            and slice_data.index[-1] > reference_end_date
        ):
            slice_data = slice_data.loc[slice_data.index <= reference_end_date]

        ticker_dfs = _split_batch_ohlcv(slice_data, batch, required_cols)
        nc = name_cache if lazy_metadata else None

        use_parallel = TA_MAX_WORKERS > 1 and len(batch) > 1
        if use_parallel:
            with ThreadPoolExecutor(max_workers=TA_MAX_WORKERS) as pool:
                pairs = []
                for t in batch:
                    if is_cancelled and is_cancelled():
                        break
                    ent = None
                    if not lazy_metadata:
                        ent = info_cache.get(t, (False, "INFO_ERROR", {"company_name": t, "avg_volume": None}))
                    fut = pool.submit(
                        _process_ticker_for_scan,
                        t,
                        ticker_dfs.get(t),
                        required_cols,
                        fetch_period,
                        inter,
                        tf,
                        reference_end_date,
                        risk_per_trade,
                        min_rr,
                        use_last_hl_sl,
                        new_only,
                        is_manual_src,
                        lazy_metadata,
                        ent,
                        tv_symbol_by_ticker,
                        nc,
                    )
                    pairs.append((t, fut))
                for t, fut in pairs:
                    if is_cancelled and is_cancelled():
                        break
                    try:
                        _merge_ticker_result(fut.result())
                    except Exception as e:
                        msg = f"{type(e).__name__}: {e}"
                        if len(msg) > 200:
                            msg = msg[:197] + "..."
                        rejected_reasons.append({"Symbol": t, "Reason": f"ERROR: {msg}"})
                    step[0] += 1
                    if on_progress:
                        on_progress(step[0], total_steps)
        else:
            for t in batch:
                if is_cancelled and is_cancelled():
                    break
                ent = None
                if not lazy_metadata:
                    ent = info_cache.get(t, (False, "INFO_ERROR", {"company_name": t, "avg_volume": None}))
                res = _process_ticker_for_scan(
                    t,
                    ticker_dfs.get(t),
                    required_cols,
                    fetch_period,
                    inter,
                    tf,
                    reference_end_date,
                    risk_per_trade,
                    min_rr,
                    use_last_hl_sl,
                    new_only,
                    is_manual_src,
                    lazy_metadata,
                    ent,
                    tv_symbol_by_ticker,
                    nc,
                )
                _merge_ticker_result(res)
                step[0] += 1
                if on_progress:
                    on_progress(step[0], total_steps)

    proc_sec = time.perf_counter() - t_proc0

    if table_rows and not use_embedded_names:
        _patch_symbol_only_company_names(table_rows)

    total_sec = time.perf_counter() - t_scan0
    timing_msg = (
        f"Screener scan timings: symbols={len(tickers)} "
        f"info={info_sec:.2f}s prefetch={prefetch_sec:.2f}s process={proc_sec:.2f}s total={total_sec:.2f}s"
    )
    _log.info(timing_msg)
    print(timing_msg, flush=True)

    return (table_rows, rejected_reasons, reference_end_date, ohlc_cache)


if st.session_state.scanning:
    p = st.session_state.run_params
    cfg = ScanConfig.from_run_params(p)
    if p["src"] == "MANUAL SCAN":
        source = ManualSource(lambda: p["txt"])
    else:
        source = SOURCE_REGISTRY[p["src"]]
    tickers, tv_symbol_by_ticker, company_names, err = source.get_tickers()
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

    table_rows, rejected_reasons, reference_end_date, ohlc_cache = run_scan(
        tickers,
        risk_per_trade=cfg.risk_per_trade,
        min_rr=cfg.min_rr,
        use_last_hl_sl=cfg.use_last_hl_sl,
        tf=cfg.tf,
        new_only=cfg.new_only,
        is_manual_src=cfg.is_manual_src,
        tv_symbol_by_ticker=tv_symbol_by_ticker,
        company_name_by_ticker=company_names,
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
    st.session_state.ohlc_cache = ohlc_cache
    st.session_state.chart_cache = {}
    st.session_state.selected_tv_symbol = None

    with res_area.container():
        render_scan_results(
            table_rows, rejected_reasons, reference_end_date, cfg.tf,
            is_manual_src=cfg.is_manual_src,
            ohlc_cache=ohlc_cache,
            empty_message="No symbols passed the screener.",
        )

    st.session_state.scanning = False
    info_box.success("SCAN COMPLETE")

else:
    last_src = st.session_state.run_params.get('src', "BIG CAP")
    table_rows = st.session_state.results
    rejected_reasons = st.session_state.rejected
    as_of = st.session_state.get("results_as_of")
    as_of_tf = st.session_state.get("results_tf", "Daily")

    with res_area.container():
        render_scan_results(
            table_rows, rejected_reasons, as_of, as_of_tf,
            is_manual_src=(last_src == "MANUAL SCAN"),
            chart_cache=st.session_state.get("chart_cache", {}),
            ohlc_cache=st.session_state.get("ohlc_cache", {}),
        )

