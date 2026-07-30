from __future__ import annotations

import html
import gc
import logging
import os
import re
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import time
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
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
if 'results_token' not in st.session_state:
    st.session_state.results_token = 0
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
try:
    from ticker_data import (
        TV_LIST_ETF,
        TV_LIST_STOCK_TICKERS,
        build_name_cache,
        get_ticker_info_and_filter,
        read_list_file,
        resolve_company_name,
    )
except ImportError as exc:
    raise ImportError(
        f"Failed to import ticker_data ({exc}). "
        "On Streamlit Cloud open Manage app -> Logs for the full traceback "
        "(often yfinance cache / dependency)."
    ) from exc

# ==========================================
# 3. SEQUENCE VOVA (from sequence_vova.py)
# ==========================================
try:
    import sequence_vova as _sequence_vova
    from sequence_vova import (
        run_sequence_vova_pine,
        run_sequence_vova_close_scan,
        run_sequence_vova_full,
    )
except ImportError as exc:
    raise ImportError(
        f"Failed to import sequence_vova ({exc}). "
        "On Streamlit Cloud open Manage app -> Logs for the full traceback."
    ) from exc

# Optional helper: never require a 4th import name for app boot (Cloud hot-reload safe).
explain_invalid_buy = getattr(_sequence_vova, "explain_invalid_buy", None)
if not callable(explain_invalid_buy):
    def explain_invalid_buy(full, *, min_rr=1.5, no_rr_req=False):  # noqa: N802 — matches sequence_vova API
        return "NO_VALID_SIGNAL"

try:
    import data_utils as _data_utils
    from data_utils import (
        extract_ohlcv as _extract_ohlcv,
        fill_last_bar_ohlc as _fill_last_bar_ohlc,
        interval_and_period as _interval_and_period_raw,
        split_batch_ohlcv as _split_batch_ohlcv,
    )
except ImportError as exc:
    raise ImportError(
        f"Failed to import data_utils ({exc}). "
        "On Streamlit Cloud open Manage app -> Logs for the full traceback."
    ) from exc

_resample_to_timeframe = getattr(_data_utils, "resample_to_timeframe", None)


def _interval_and_period(tf: str, *, scanner_id: str | None = None) -> tuple[str, str]:
    """Native Weekly/Monthly Yahoo intervals even if a stale data_utils is loaded."""
    inter, period = _interval_and_period_raw(tf, scanner_id=scanner_id)
    if tf == "Weekly" and inter in ("1d", "1day", "daily"):
        return "1wk", period or "10y"
    if tf == "Monthly" and inter in ("1d", "1day", "daily"):
        return "1mo", period or "10y"
    return inter, period


_prepare_scan_ohlc = getattr(_data_utils, "prepare_scan_ohlc", None)
if not callable(_prepare_scan_ohlc):
    def _prepare_scan_ohlc(df, tf, *, inter):
        """Fallback when Cloud hot-reload still has pre-parity data_utils."""
        if df is None or getattr(df, "empty", True):
            return None, None
        frame = df.copy()
        is_daily = inter in ("1d", "1day", "daily")
        daily = frame.copy() if is_daily else None
        if is_daily and tf != "Daily" and callable(_resample_to_timeframe):
            frame = _resample_to_timeframe(frame, tf)
        return frame, daily
from scan_memory import (
    download_max_workers,
    is_low_memory_runtime,
    scan_chunk_size,
    slim_ohlc_entry,
    ta_max_workers,
    yf_download_threads,
    yf_info_max_workers,
    yf_name_cache_rate_per_sec,
)
from tradingview_embed import (
    build_chart_url,
    infer_tv_symbol,
    tf_to_tv_interval,
)

# ==========================================
# 4. UI & SIDEBAR
# ==========================================
from ticker_sources import CombinedListSource, FileListSource, ManualSource

# Ticker sources: one combined stocks list + ETF + optional MANUAL symbols
STOCKS_SRC = "Stocks"
ETF_SRC = "ETF"
MANUAL_SRC = "MANUAL SCAN"
# Legacy session/run_params source labels map onto the combined stocks list.
_LEGACY_STOCK_SOURCES = frozenset(
    {
        "BIG CAP",
        "SMALL CAP",
        "BIG + SMALL CAP",
        "US + CANADA FULL",
        "STOCK TICKERS",
        "STOCKS",
    }
)


def _normalize_source_label(src: str | None) -> str:
    if not src:
        return STOCKS_SRC
    if src in _LEGACY_STOCK_SOURCES:
        return STOCKS_SRC
    return src


def _source_options() -> list[str]:
    return [STOCKS_SRC, ETF_SRC, MANUAL_SRC]


SOURCE_OPTIONS = _source_options()
TF_OPTIONS = ["Daily", "Weekly", "Monthly"]


def _results_table_height(n_rows: int) -> int:
    """Pixel height for st.dataframe (compatible with Streamlit < 1.52)."""
    return min(max(int(n_rows) * 36 + 44, 150), 520)


def _st_dataframe(df, **kwargs):
    """st.dataframe wrapper: avoids height='content' / width='stretch' on older Streamlit."""
    kwargs.pop("width", None)
    kwargs.pop("height", None)
    n_rows = len(df) if hasattr(df, "__len__") else 1
    return st.dataframe(
        df,
        width="stretch",
        height=_results_table_height(n_rows),
        **kwargs,
    )


def _build_stocks_source():
    """Individual-company list: STOCK-TICKERS.txt only."""
    return FileListSource(TV_LIST_STOCK_TICKERS, read_list_file)


def _build_source_registry():
    return {
        STOCKS_SRC: _build_stocks_source(),
        # CombinedListSource dedupes Yahoo symbols (ETF file can contain repeats).
        ETF_SRC: CombinedListSource(
            [FileListSource(TV_LIST_ETF, read_list_file)],
            label="ETF",
        ),
    }


SOURCE_REGISTRY = _build_source_registry()



st.sidebar.header("⚙️ CONFIGURATION")

# Disable inputs if scanning
disabled = st.session_state.scanning

scanner_id = "sequence_vova"

last_src = _normalize_source_label(
    st.session_state.get("run_params", {}).get("src", STOCKS_SRC)
)
_source_opts = _source_options()
default_idx = _source_opts.index(last_src) if last_src in _source_opts else 0
src = st.sidebar.radio("SOURCE", _source_opts, disabled=disabled, index=default_idx)
if src == STOCKS_SRC:
    _stocks_tickers, _, _, _stocks_err = SOURCE_REGISTRY[STOCKS_SRC].get_tickers()
    if _stocks_err:
        st.sidebar.warning(_stocks_err)
    elif _stocks_tickers:
        st.sidebar.caption(
            f"{len(_stocks_tickers)} unique company tickers (all stock lists merged)"
        )
man_txt = ""
if src == MANUAL_SRC:
    man_txt = st.sidebar.text_area("TICKERS", "AAPL, TSLA, NVDA", disabled=disabled)
    st.sidebar.caption("Comma-separated symbols. Next START scans these tickers.")

st.sidebar.subheader("RISK MANAGEMENT")
risk_per_trade = st.sidebar.number_input("$ RISK PER TRADE", value=100, min_value=1, step=10, disabled=disabled)
no_rr_req = st.sidebar.checkbox(
    "ANY VALID SIGNAL (NO RR REQ)",
    False,
    disabled=disabled,
)
st.sidebar.caption(
    "Skips the min RR threshold only. Still needs positive risk and reward. "
    "More rows than min RR; BUY RR≥min still match both modes."
)
min_rr_in = st.sidebar.number_input(
    "MIN RR (>=0.1)",
    value=1.5,
    min_value=0.1,
    step=0.1,
    disabled=disabled or no_rr_req,
)
st.sidebar.caption(
    "BUY: filters last-bar RR. SELL TO CLOSE: filters RR at entry only "
    "(not realized RR at close)."
)
st.sidebar.subheader("FILTERS")
SCAN_DIRECTION_OPTIONS = ["BUY TO OPEN", "SELL TO CLOSE"]
_last_dir = str(st.session_state.get("run_params", {}).get("scan_direction", "buy")).lower()
dir_default_idx = 1 if _last_dir == "sell" else 0
scan_dir = st.sidebar.radio(
    "SCAN DIRECTION",
    SCAN_DIRECTION_OPTIONS,
    disabled=disabled,
    index=dir_default_idx,
    horizontal=True,
)
is_sell_scan = scan_dir == "SELL TO CLOSE"
if not is_sell_scan:
    use_last_hl_sl = st.sidebar.checkbox("Use last HL in SL (safety)", True, disabled=disabled)
else:
    use_last_hl_sl = bool(st.session_state.get("run_params", {}).get("use_last_hl_sl", True))
tf_p = st.sidebar.selectbox("TIMEFRAME", TF_OPTIONS, disabled=disabled)
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
    st.session_state.results_token = int(st.session_state.get("results_token", 0)) + 1
    # FREEZE PARAMS
    st.session_state.run_params = {
        'src': src, 'txt': man_txt, 'risk_per_trade': risk_per_trade, 'rr': min_rr_in,
        'no_rr_req': no_rr_req,
        'use_last_hl_sl': use_last_hl_sl, 'tf': tf_p, 'new': new_p,
        'scan_direction': "sell" if is_sell_scan else "buy",
        'scanner_id': scanner_id,
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
    no_rr_req: bool
    use_last_hl_sl: bool
    tf: str
    new_only: bool
    is_manual_src: bool
    scan_direction: str
    scanner_id: str

    @classmethod
    def from_run_params(cls, p: dict) -> "ScanConfig":
        raw_dir = str(p.get("scan_direction", "buy")).lower()
        scan_direction = raw_dir if raw_dir in ("buy", "sell") else "buy"
        return cls(
            risk_per_trade=int(p.get("risk_per_trade", 100)),
            min_rr=float(p.get("rr", 1.5)),
            no_rr_req=bool(p.get("no_rr_req", False)),
            use_last_hl_sl=bool(p.get("use_last_hl_sl", True)),
            tf=str(p.get("tf", "Daily")),
            new_only=bool(p.get("new", True)),
            is_manual_src=(p.get("src") == MANUAL_SRC),
            scan_direction=scan_direction,
            scanner_id=str(p.get("scanner_id", "sequence_vova")),
        )


ATR_LEN = 14
MIN_BARS = 50  # minimum bars for sequence logic
CHUNK_SIZE = 200  # batch size for yf.download (smaller chunks = more parallel downloads)
DOWNLOAD_MAX_WORKERS = 4  # parallel yf.download batches
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

    with ThreadPoolExecutor(max_workers=yf_info_max_workers(default=YF_INFO_MAX_WORKERS)) as pool:
        for row, ticker, name in pool.map(_resolve_one, pending):
            if name and name.upper() != ticker.upper():
                row["Company Name"] = name

# Results Placeholder
res_area = st.empty()


def _scan_direction_label(scan_direction: str) -> str:
    return "SELL TO CLOSE" if str(scan_direction).lower() == "sell" else "BUY TO OPEN"


def _display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Hide internal tv_symbol from the results grid."""
    hide = {"tv_symbol", "_is_summary"}
    cols = [c for c in df.columns if c not in hide]
    return df[cols] if cols else df


def _safe_rr_floats(rows: list[dict], key: str) -> list[float]:
    """Collect numeric RR values; skip None / NaN / 'N/A' / non-numeric."""
    out: list[float] = []
    for r in rows:
        val = r.get(key)
        if val is None:
            continue
        if isinstance(val, str) and val.strip().upper() in ("N/A", "", "—", "-"):
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        if isinstance(f, float) and np.isnan(f):
            continue
        out.append(f)
    return out


def _build_sell_summary_row(table_rows: list[dict]) -> dict | None:
    """Aggregate SELL close-scan rows into a TOTAL summary line."""
    data_rows = [r for r in table_rows if not r.get("_is_summary")]
    if not data_rows:
        return None

    pnls = [float(r["P&L ($)"]) for r in data_rows]
    invested_vals = [
        float(r.get("Invested ($)", 0) or 0)
        for r in data_rows
    ]
    total_pnl = sum(pnls)
    total_invested = sum(invested_vals)
    wins = sum(1 for p in pnls if p > 0)
    winrate = (wins / len(data_rows)) * 100.0
    total_pnl_pct = (total_pnl / total_invested * 100.0) if total_invested > 0 else 0.0

    entry_rr_vals = _safe_rr_floats(data_rows, "RR at Entry")
    close_rr_vals = _safe_rr_floats(data_rows, "RR at Close")
    avg_entry_rr = sum(entry_rr_vals) / len(entry_rr_vals) if entry_rr_vals else 0.0
    avg_close_rr = sum(close_rr_vals) / len(close_rr_vals) if close_rr_vals else 0.0

    return {
        "_is_summary": True,
        "Symbol": "TOTAL",
        "Company Name": f"Win rate: {winrate:.1f}%",
        "Entry": "—",
        "Exit": "—",
        "Position Size (shares)": "—",
        "RR at Entry": round(avg_entry_rr, 2),
        "RR at Close": round(avg_close_rr, 2),
        "Invested ($)": round(total_invested, 2),
        "P&L ($)": round(total_pnl, 2),
        "P&L (%)": round(total_pnl_pct, 2),
    }


def render_scan_results(
    table_rows,
    rejected_reasons,
    reference_end_date,
    tf,
    is_manual_src,
    chart_cache=None,
    ohlc_cache=None,
    empty_message="Ready to scan. Click START.",
    scan_direction="buy",
    scanner_id: str = "sequence_vova",
):
    """
    Render scan results: dataframe, as-of caption, and rejected/skipped expander.
    Used by both "scan just finished" and "idle show last results" paths.
    """
    chart_cache = chart_cache if chart_cache is not None else {}
    ohlc_cache = ohlc_cache if ohlc_cache is not None else {}
    has_charts = bool(chart_cache) or bool(ohlc_cache)

    if table_rows:
        summary_row = None
        if scan_direction == "sell":
            summary_row = _build_sell_summary_row(table_rows)

        res_df = pd.DataFrame(table_rows)
        display_df = _display_columns(res_df)
        col_config = {
            "Symbol": st.column_config.LinkColumn("Symbol", display_text=r"symbol=([^&]+)"),
        }
        if scan_direction == "sell":
            col_config["RR at Entry"] = st.column_config.Column(
                "Entry RR (filter)",
                help="Reward/risk at open. Min RR filters this value.",
            )
            col_config["RR at Close"] = st.column_config.Column(
                "Close RR (realized)",
                help="Realized R-multiple at exit. Not used by min RR filter.",
            )

        results_token = int(st.session_state.get("results_token", 0))
        event = _st_dataframe(
            display_df,
            hide_index=True,
            column_config=col_config,
            on_select="rerun",
            selection_mode="single-row",
            key=f"scan_results_table_{results_token}",
        )

        if summary_row:
            summary_display = {k: v for k, v in summary_row.items() if k != "_is_summary"}
            st.caption("Scan totals")
            _st_dataframe(
                pd.DataFrame([summary_display]),
                hide_index=True,
                column_config={
                    "RR at Entry": st.column_config.Column("Entry RR (filter)"),
                    "RR at Close": st.column_config.Column("Close RR (realized)"),
                },
            )

        if has_charts:
            dir_label = _scan_direction_label(scan_direction)
            if scan_direction == "sell":
                st.caption(
                    f"Close signals. Min RR filters Entry RR only "
                    f"(not Close RR). Click a row for chart ({dir_label} · {tf})."
                )
            else:
                st.caption(
                    f"Click a row for chart preview ({dir_label} · {tf}). "
                    f"Min RR filters last-bar RR; NO RR REQ skips that threshold only."
                )
        else:
            st.caption("Run a scan to enable chart previews.")

        tv_sym = None
        if has_charts:
            selected = getattr(event, "selection", None)
            raw_rows = list(selected.rows) if selected and getattr(selected, "rows", None) else []
            n_rows = len(res_df)
            sel_rows = [int(i) for i in raw_rows if 0 <= int(i) < n_rows]
            if sel_rows:
                tv_sym = str(res_df.iloc[sel_rows[0]].get("tv_symbol", "") or "")
                if tv_sym:
                    st.session_state.selected_tv_symbol = tv_sym
            elif st.session_state.get("selected_tv_symbol"):
                tv_sym = st.session_state.selected_tv_symbol

            payload = (
                resolve_chart_payload(tv_sym, chart_cache=chart_cache, ohlc_cache=ohlc_cache)
                if tv_sym
                else None
            )
            if payload:
                chart_params = render_chart_settings()
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
                            width="stretch",
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
                mode_label = _scan_direction_label(scan_direction)
                if dow >= 5:
                    st.caption(
                        f"**Results as of {as_of_str}** ({mode_label} · {tf}) "
                        f"(last trading day — market closed weekend/holiday). Same bar used for all symbols for consistency."
                    )
                else:
                    st.caption(
                        f"**Results as of {as_of_str}** ({mode_label} · {tf}). "
                        f"Same bar used for all symbols for consistency."
                    )
            except Exception:
                pass
    else:
        st.info(empty_message)

    if is_manual_src and rejected_reasons:
        with st.expander("Rejected (Manual)"):
            _st_dataframe(pd.DataFrame(rejected_reasons), hide_index=True)
    elif rejected_reasons and any(str(r.get("Reason", "")).startswith("ERROR") for r in rejected_reasons):
        with st.expander("Skipped (errors)"):
            _st_dataframe(pd.DataFrame(rejected_reasons), hide_index=True)


def _download_batch(
    batch_idx: int,
    batch: list[str],
    fetch_period: str,
    inter: str,
    *,
    auto_adjust: bool = False,
) -> tuple[int, list[str], pd.DataFrame | None]:
    all_data = None
    use_threads = yf_download_threads()
    for attempt in range(YF_DOWNLOAD_MAX_RETRIES):
        try:
            all_data = yf.download(
                batch,
                period=fetch_period,
                interval=inter,
                progress=False,
                auto_adjust=auto_adjust,
                group_by="ticker",
                threads=use_threads,
            )
            break
        except Exception as exc:
            is_rate_limit = type(exc).__name__ == "YFRateLimitError" or "Rate limit" in str(exc)
            if attempt < YF_DOWNLOAD_MAX_RETRIES - 1:
                delay = YF_DOWNLOAD_BACKOFF_SEC * (attempt + 1)
                if is_rate_limit:
                    delay = max(delay, 8.0)
                time.sleep(delay)
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
    scan_direction: str = "buy",
    scanner_id: str = "sequence_vova",
    no_rr_req: bool = False,
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

        df, df_daily_chart = _prepare_scan_ohlc(df, tf, inter=inter)
        if df is None or df.empty or len(df) < MIN_BARS:
            if is_manual_src:
                return {"kind": "reject", "row": {"Symbol": t, "Reason": "NO_DATA"}}
            return {"kind": "skip"}

        df = _fill_last_bar_ohlc(df)
        df = df.dropna(subset=["Close", "High", "Low", "Open"])
        if df_daily_chart is not None:
            df_daily_chart = _fill_last_bar_ohlc(df_daily_chart)
            df_daily_chart = df_daily_chart.dropna(subset=["Close", "High", "Low", "Open"])
        if len(df) < MIN_BARS:
            if is_manual_src:
                return {"kind": "reject", "row": {"Symbol": t, "Reason": "INSUFFICIENT_DATA"}}
            return {"kind": "skip"}

        tv_sym = (tv_symbol_by_ticker or {}).get(t) or infer_tv_symbol(t)
        interval = tf_to_tv_interval(tf)
        tv_url = build_chart_url(tv_sym, interval)
        if lazy_metadata and name_cache is not None:
            company_name = name_cache.get(t, t)
        else:
            company_name = info_dict.get("company_name", t)

        out = run_sequence_vova_close_scan(
            df,
            atr_len=ATR_LEN,
            min_rr=min_rr,
            use_last_hl_sl=use_last_hl_sl,
            risk_dollars=risk_per_trade,
            no_rr_req=no_rr_req,
        ) if scan_direction == "sell" else run_sequence_vova_pine(
            df,
            atr_len=ATR_LEN,
            min_rr=min_rr,
            use_last_hl_sl=use_last_hl_sl,
            risk_dollars=risk_per_trade,
            direction="buy",
            no_rr_req=no_rr_req,
        )
        if out is None or not out["Valid"]:
            if is_manual_src:
                if scan_direction == "sell":
                    reason = "NO_CLOSE_SIGNAL"
                else:
                    full_dbg = run_sequence_vova_full(
                        df,
                        atr_len=ATR_LEN,
                        min_rr=min_rr,
                        use_last_hl_sl=use_last_hl_sl,
                        risk_dollars=risk_per_trade,
                        no_rr_req=no_rr_req,
                    )
                    reason = explain_invalid_buy(full_dbg, min_rr=min_rr, no_rr_req=no_rr_req)
                return {"kind": "reject", "row": {"Symbol": t, "Reason": reason}}
            return {"kind": "skip"}
        if new_only and not out["New"]:
            return {"kind": "skip"}
        if scan_direction != "sell":
            # Hard guard: BUY rows must have the latest confirmed peak as HH, not LH/DT.
            full = run_sequence_vova_full(df)
            last_peak_hh = bool(full.get("last_peak_was_hh", False)) if full else False
            if not last_peak_hh:
                if is_manual_src:
                    return {"kind": "reject", "row": {"Symbol": t, "Reason": "NO_HH_LAST_PEAK"}}
                return {"kind": "skip"}

        pos_size = out["position_size"]
        if np.isnan(pos_size) or pos_size < 1:
            pos_size = 0
        else:
            pos_size = int(round(pos_size))

        def _fmt_rr(val) -> float | str:
            try:
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return "N/A"
                return round(float(val), 2)
            except (TypeError, ValueError):
                return "N/A"

        if scan_direction == "sell":
            entry_px = round(float(out["entry_price"]), 2)
            invested = round(entry_px * pos_size, 2) if pos_size > 0 else 0.0
            entry_rr = out.get("entry_rr", np.nan)
            close_rr = out.get("close_rr", np.nan)
            table_row = {
                "Symbol": tv_url,
                "tv_symbol": tv_sym,
                "Company Name": company_name,
                "Entry": entry_px,
                "Exit": round(float(out["exit_price"]), 2),
                "Position Size (shares)": pos_size,
                "RR at Entry": _fmt_rr(entry_rr),
                "RR at Close": _fmt_rr(close_rr),
                "Invested ($)": invested,
                "P&L ($)": round(float(out["pnl_dollars"]), 2),
                "P&L (%)": round(float(out["pnl_pct"]), 2),
            }
        else:
            pos_value = out["position_value"] if not np.isnan(out["position_value"]) else 0.0
            table_row = {
                "Symbol": tv_url,
                "tv_symbol": tv_sym,
                "Company Name": company_name,
                "TP": round(float(out["TP"]), 2),
                "SL": round(float(out["SL"]), 2),
                "RR": _fmt_rr(out["RR"]),
                "Position Size (shares)": pos_size,
                "Position Value ($)": round(float(pos_value), 2),
                "New": 1 if out["New"] else 0,
                "Valid": 1 if out["Valid"] else 0,
                "Strong": 1 if out["Strong"] else 0,
            }
        ohlc_entry = {
            "df": df.copy(),
            "df_daily": df_daily_chart.copy() if df_daily_chart is not None else None,
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


class ScanPhaseProgressUI:
    """Multi-step scan progress: custom HTML bars + emoji status per phase."""

    STATUS_PENDING = "⏳"
    STATUS_ACTIVE = "🔄"
    STATUS_THINKING = "💭"
    STATUS_DONE = "✅"
    STATUS_CANCELLED = "⏹️"

    def __init__(self) -> None:
        self._phases: list[tuple[str, str, str]] = []
        self._state: dict[str, dict] = {}
        self._placeholder = None

    def _blank_state(self) -> dict:
        return {"pct": 0, "status": self.STATUS_PENDING, "active": False, "indeterminate": False}

    def setup(self, phases: list[tuple[str, str, str]]) -> None:
        self._phases = phases
        self._state = {phase_id: self._blank_state() for phase_id, _, _ in phases}
        self._placeholder = st.empty()
        self._render_all()

    def start_indeterminate(self, phase: str) -> None:
        state = self._state[phase]
        state["pct"] = 0
        state["status"] = self.STATUS_THINKING
        state["active"] = True
        state["indeterminate"] = True
        self._render_all()

    def on_phase_start(self, phase: str) -> None:
        if phase in self._state:
            self.start_indeterminate(phase)

    def update(self, phase: str, current: int, total: int) -> None:
        pct = 0 if total <= 0 else min(100, round(100 * current / total))
        state = self._state[phase]
        state["pct"] = pct
        state["status"] = self.STATUS_ACTIVE
        state["active"] = True
        state["indeterminate"] = False
        self._render_all()

    def complete(self, phase: str) -> None:
        state = self._state[phase]
        state["pct"] = 100
        state["status"] = self.STATUS_DONE
        state["active"] = False
        state["indeterminate"] = False
        self._render_all()

    def cancel_active(self) -> None:
        for state in self._state.values():
            if state["active"] or state["indeterminate"]:
                state["status"] = self.STATUS_CANCELLED
                state["active"] = False
                state["indeterminate"] = False
        self._render_all()

    def _render_row(
        self,
        phase_id: str,
        label: str,
        label_emoji: str,
        pct: int,
        status: str,
        is_active: bool,
        indeterminate: bool,
    ) -> str:
        row_cls = "scan-phase-row"
        if is_active:
            row_cls += " is-active"
        if indeterminate:
            row_cls += " is-indeterminate"
        pct_cls = " is-indeterminate-pct" if indeterminate else (" is-active-pct" if is_active else "")
        pct_text = "···" if indeterminate else f"{pct}%"
        safe_label = html.escape(label)
        if indeterminate:
            fill_html = '<div class="scan-phase-fill scan-phase-fill-indeterminate"></div>'
        else:
            fill_html = f'<div class="scan-phase-fill" style="width:{pct}%;"></div>'
        return (
            f'<div class="{row_cls}" data-phase="{phase_id}">'
            f'<div class="scan-phase-head">'
            f'<span class="scan-phase-label">{label_emoji} {safe_label}</span>'
            f'<span class="scan-phase-status">{status}</span>'
            f"</div>"
            f'<div class="scan-phase-track">'
            f'<div class="scan-phase-bar">{fill_html}</div>'
            f'<span class="scan-phase-pct{pct_cls}">{pct_text}</span>'
            f"</div></div>"
        )

    def _render_all(self) -> None:
        rows = "".join(
            self._render_row(
                phase_id,
                label,
                emoji,
                self._state[phase_id]["pct"],
                self._state[phase_id]["status"],
                self._state[phase_id]["active"],
                self._state[phase_id]["indeterminate"],
            )
            for phase_id, label, emoji in self._phases
        )
        self._placeholder.markdown(
            f'<div class="scan-phases">{rows}</div>',
            unsafe_allow_html=True,
        )


def _parallel_download_batches(
    batches: list[list[str]],
    fetch_period: str,
    inter: str,
    is_cancelled=None,
    on_batch_done=None,
    *,
    auto_adjust: bool = False,
    max_workers: int | None = None,
) -> tuple[list[tuple[list[str], pd.DataFrame | None]], object | None]:
    """Download all chunks in parallel; return batches sorted by index."""
    reference_end_date = None
    if not batches:
        return [], None

    workers = max_workers if max_workers is not None else min(DOWNLOAD_MAX_WORKERS, len(batches))
    raw: list[tuple[int, list[str], pd.DataFrame | None]] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_download_batch, batch_idx, batch, fetch_period, inter, auto_adjust=auto_adjust)
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
    scan_direction="buy",
    scanner_id: str = "sequence_vova",
    no_rr_req: bool = False,
    tv_symbol_by_ticker: dict[str, str] | None = None,
    company_name_by_ticker: dict[str, str] | None = None,
    on_phase_progress=None,
    on_phase_start=None,
    on_phase_complete=None,
    on_scan_cancelled=None,
    is_cancelled=None,
):
    """
    Pure scanner: optional parallel Yahoo metadata (manual only), threaded batch download,
    then parallel per-ticker OHLCV + sequence for list sources (lazy metadata: .info only for passes).
    Logs phase timings to the logger and stdout. No Streamlit calls.
    Returns (table_rows, rejected_reasons, reference_end_date, ohlc_cache).
    """
    inter, fetch_period = _interval_and_period(tf, scanner_id=scanner_id)
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    chunk = scan_chunk_size(scanner_id)
    if len(tickers) > chunk:
        batches = [tickers[i:i + chunk] for i in range(0, len(tickers), chunk)]
    else:
        batches = [tickers]

    auto_adjust_prices = False
    stream_batches = is_low_memory_runtime()
    workers_ta = ta_max_workers(scanner_id, default=TA_MAX_WORKERS)
    workers_dl = download_max_workers(scanner_id, default=DOWNLOAD_MAX_WORKERS)

    # Smaller stacks on constrained hosts so more threads can be created.
    if stream_batches:
        try:
            threading.stack_size(512 * 1024)
        except (ValueError, RuntimeError, OSError):
            pass

    table_rows = []
    rejected_reasons = []
    ohlc_cache: dict[str, dict] = {}
    reference_end_date = None
    batches_data: list[tuple[list[str], pd.DataFrame | None] | None] = []

    lazy_metadata = not is_manual_src
    use_embedded_names = lazy_metadata
    info_done = [0]
    dl_done = [0]
    proc_done = [0]
    n_tickers = len(tickers)
    n_batches = len(batches)

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

        if on_phase_start:
            on_phase_start("info")
        with ThreadPoolExecutor(max_workers=yf_info_max_workers(default=YF_INFO_MAX_WORKERS)) as executor:
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
                info_done[0] += 1
                if on_phase_progress:
                    on_phase_progress("info", info_done[0], n_tickers)
        if on_phase_complete and not (is_cancelled and is_cancelled()):
            on_phase_complete("info")
        info_sec = time.perf_counter() - t_info0

    def _merge_ticker_result(res: dict) -> None:
        if res["kind"] == "row":
            table_rows.append(res["row"])
            key = res.get("chart_key")
            entry = res.get("ohlc_entry")
            if key and entry:
                ohlc_cache[key] = slim_ohlc_entry(entry, scanner_id)
        elif res["kind"] == "reject":
            rejected_reasons.append(res["row"])

    def _process_one_ticker(
        t: str,
        ticker_df,
        nc: dict[str, str] | None,
    ) -> dict:
        ent = None
        if not lazy_metadata:
            ent = info_cache.get(t, (False, "INFO_ERROR", {"company_name": t, "avg_volume": None}))
        return _process_ticker_for_scan(
            t,
            ticker_df,
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
            scan_direction,
            scanner_id,
            no_rr_req,
        )

    def _finish_ticker_result(t: str, res_or_exc) -> None:
        try:
            if isinstance(res_or_exc, Exception):
                raise res_or_exc
            _merge_ticker_result(res_or_exc)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if len(msg) > 200:
                msg = msg[:197] + "..."
            rejected_reasons.append({"Symbol": t, "Reason": f"ERROR: {msg}"})
        proc_done[0] += 1
        if on_phase_progress:
            on_phase_progress("process", proc_done[0], n_tickers)

    def _process_batch_slice(
        batch: list[str],
        all_data: pd.DataFrame | None,
        *,
        pool: ThreadPoolExecutor | None = None,
    ) -> None:
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

        def _run_serial(symbols: list[str]) -> None:
            for t in symbols:
                if is_cancelled and is_cancelled():
                    break
                try:
                    res = _process_one_ticker(t, ticker_dfs.get(t), nc)
                except Exception as e:
                    res = e
                _finish_ticker_result(t, res)

        use_parallel = pool is not None and workers_ta > 1 and len(batch) > 1
        if use_parallel:
            window = max(workers_ta * 2, workers_ta)
            pending: dict = {}
            tickers_iter = iter(batch)
            degrade_serial = False

            def _submit_next() -> bool:
                """Submit one more ticker. False = no more / cancelled / degraded."""
                nonlocal degrade_serial
                if is_cancelled and is_cancelled():
                    return False
                try:
                    t = next(tickers_iter)
                except StopIteration:
                    return False
                try:
                    fut = pool.submit(_process_one_ticker, t, ticker_dfs.get(t), nc)
                    pending[fut] = t
                    return True
                except (RuntimeError, OSError) as exc:
                    _log.warning(
                        "Thread pool submit failed (%s); finishing batch serially",
                        exc,
                    )
                    degrade_serial = True
                    _run_serial([t] + list(tickers_iter))
                    return False

            while len(pending) < window and _submit_next():
                pass

            while pending:
                if is_cancelled and is_cancelled():
                    break
                done, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    t = pending.pop(fut)
                    try:
                        res = fut.result()
                    except Exception as e:
                        res = e
                    _finish_ticker_result(t, res)
                if not degrade_serial:
                    while len(pending) < window and _submit_next():
                        pass
        else:
            _run_serial(batch)
        del ticker_dfs

    ta_pool: ThreadPoolExecutor | None = None
    if workers_ta > 1:
        try:
            ta_pool = ThreadPoolExecutor(max_workers=workers_ta)
        except (RuntimeError, OSError) as exc:
            _log.warning("Could not create TA thread pool (%s); using serial TA", exc)
            ta_pool = None

    t_names_dl0 = time.perf_counter()
    try:
        if stream_batches:
            if lazy_metadata:
                if use_embedded_names:
                    name_cache = dict(company_name_by_ticker or {})
                    for t in tickers:
                        name_cache.setdefault(t, t)
                else:
                    if on_phase_start:
                        on_phase_start("download")
                    name_cache = build_name_cache(
                        tickers,
                        rate_limit_per_sec=yf_name_cache_rate_per_sec(default=YF_INFO_RATE_LIMIT_PER_SEC),
                        max_workers=yf_info_max_workers(default=YF_INFO_MAX_WORKERS),
                        is_cancelled=is_cancelled,
                        on_one_done=None,
                    )
            if on_phase_start:
                on_phase_start("download")
            if on_phase_start:
                on_phase_start("process")
            # Pipeline: keep N downloads in flight; process batches as they finish
            # (not strictly ordered) so a slow Yahoo batch does not stall TA.
            dl_inflight = max(1, workers_dl)
            with ThreadPoolExecutor(max_workers=dl_inflight) as dl_pool:
                pending_dl: dict[object, int] = {}
                next_submit = 0
                processed = 0
                n_to_process = len(batches)

                def _submit_downloads() -> None:
                    nonlocal next_submit
                    while (
                        next_submit < n_to_process
                        and len(pending_dl) < dl_inflight
                        and not (is_cancelled and is_cancelled())
                    ):
                        bi = next_submit
                        next_submit += 1
                        fut = dl_pool.submit(
                            _download_batch,
                            bi,
                            batches[bi],
                            fetch_period,
                            inter,
                            auto_adjust=auto_adjust_prices,
                        )
                        pending_dl[fut] = bi

                _submit_downloads()
                while pending_dl and processed < n_to_process:
                    if is_cancelled and is_cancelled():
                        break
                    done, _ = wait(tuple(pending_dl.keys()), return_when=FIRST_COMPLETED)
                    for fut in done:
                        pending_dl.pop(fut, None)
                        try:
                            _, batch, all_data = fut.result()
                        except Exception:
                            time.sleep(YF_DOWNLOAD_BACKOFF_SEC)
                            processed += 1
                            dl_done[0] += 1
                            if on_phase_progress:
                                on_phase_progress("download", dl_done[0], n_batches)
                            _submit_downloads()
                            continue
                        if all_data is not None and not all_data.empty and len(all_data.index) > 0:
                            batch_end = all_data.index[-1]
                            reference_end_date = (
                                batch_end
                                if reference_end_date is None
                                else min(reference_end_date, batch_end)
                            )
                        dl_done[0] += 1
                        if on_phase_progress:
                            on_phase_progress("download", dl_done[0], n_batches)
                        _process_batch_slice(batch, all_data, pool=ta_pool)
                        del all_data
                        processed += 1
                        if processed % 4 == 0:
                            gc.collect()
                        _submit_downloads()
            if on_phase_complete and not (is_cancelled and is_cancelled()):
                on_phase_complete("download")
            if on_phase_complete and not (is_cancelled and is_cancelled()):
                on_phase_complete("process")


        elif lazy_metadata:
            if use_embedded_names:
                name_cache = dict(company_name_by_ticker or {})
                for t in tickers:
                    name_cache.setdefault(t, t)

                def _dl_batch_done_embedded():
                    dl_done[0] += 1
                    if on_phase_progress:
                        on_phase_progress("download", dl_done[0], n_batches)

                if on_phase_start:
                    on_phase_start("download")
                batches_data, reference_end_date = _parallel_download_batches(
                    batches,
                    fetch_period,
                    inter,
                    is_cancelled,
                    _dl_batch_done_embedded,
                    max_workers=workers_dl,
                )
                if on_phase_complete and not (is_cancelled and is_cancelled()):
                    on_phase_complete("download")
            else:
                # UI callbacks and st.session_state must stay on the main thread (NoSessionContext in workers).
                if on_phase_start:
                    on_phase_start("download")
                _name_rate = yf_name_cache_rate_per_sec(default=YF_INFO_RATE_LIMIT_PER_SEC)
                _name_workers = yf_info_max_workers(default=YF_INFO_MAX_WORKERS)
                if is_low_memory_runtime():
                    name_cache = build_name_cache(
                        tickers,
                        rate_limit_per_sec=_name_rate,
                        max_workers=_name_workers,
                        is_cancelled=None,
                        on_one_done=None,
                    )
                    batches_data, reference_end_date = _parallel_download_batches(
                        batches,
                        fetch_period,
                        inter,
                        is_cancelled,
                        None,
                        max_workers=workers_dl,
                    )
                else:
                    with ThreadPoolExecutor(max_workers=2) as prep_pool:
                        name_future = prep_pool.submit(
                            build_name_cache,
                            tickers,
                            rate_limit_per_sec=_name_rate,
                            max_workers=_name_workers,
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
                            max_workers=workers_dl,
                        )
                        name_cache = name_future.result()
                        batches_data, reference_end_date = dl_future.result()

                if on_phase_progress:
                    on_phase_progress("download", n_batches, n_batches)
                if on_phase_complete:
                    on_phase_complete("download")
        else:

            def _dl_batch_done_manual():
                dl_done[0] += 1
                if on_phase_progress:
                    on_phase_progress("download", dl_done[0], n_batches)

            if on_phase_start:
                on_phase_start("download")
            batches_data, reference_end_date = _parallel_download_batches(
                batches,
                fetch_period,
                inter,
                is_cancelled,
                _dl_batch_done_manual,
                max_workers=workers_dl,
            )
            if on_phase_complete and not (is_cancelled and is_cancelled()):
                on_phase_complete("download")

        prefetch_sec = time.perf_counter() - t_names_dl0
        t_proc0 = time.perf_counter()

        if not stream_batches:
            if reference_end_date is None and batches_data:
                for _, all_data in batches_data:
                    if all_data is not None and not all_data.empty and len(all_data.index) > 0:
                        reference_end_date = all_data.index[-1]
                        break

            if on_phase_start:
                on_phase_start("process")
            for _batch_idx, item in enumerate(batches_data):
                if item is None:
                    continue
                if is_cancelled and is_cancelled():
                    break
                batch, all_data = item
                _process_batch_slice(batch, all_data, pool=ta_pool)
                batches_data[_batch_idx] = None
                if _batch_idx % 2 == 0:
                    gc.collect()
            if on_phase_complete and not (is_cancelled and is_cancelled()):
                on_phase_complete("process")

        proc_sec = time.perf_counter() - t_proc0

        if is_cancelled and is_cancelled():
            if on_scan_cancelled:
                on_scan_cancelled()

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
    finally:
        if ta_pool is not None:
            ta_pool.shutdown(wait=True)



if st.session_state.scanning:
    p = st.session_state.run_params
    cfg = ScanConfig.from_run_params(p)

    info_box = st.empty()

    phases: list[tuple[str, str, str]] = []
    if cfg.is_manual_src:
        phases.append(("info", "Fetching ticker info", "📋"))
    phases.append(("download", "Downloading OHLC", "📥"))
    phases.append(("process", "Processing symbols", "⚙️"))

    progress_ui = ScanPhaseProgressUI()
    progress_ui.setup(phases)

    if p["src"] == MANUAL_SRC:
        source = ManualSource(lambda: p["txt"])
    else:
        source = SOURCE_REGISTRY[_normalize_source_label(p["src"])]
    tickers, tv_symbol_by_ticker, company_names, err = source.get_tickers()
    if err:
        st.warning(err)
    if not tickers:
        st.error("NO TICKERS FOUND")
        st.session_state.scanning = False
        st.stop()

    def on_phase_start(phase):
        progress_ui.on_phase_start(phase)

    def on_phase_progress(phase, current, total):
        progress_ui.update(phase, current, total)

    def on_phase_complete(phase):
        progress_ui.complete(phase)

    def on_scan_cancelled():
        progress_ui.cancel_active()

    table_rows, rejected_reasons, reference_end_date, ohlc_cache = run_scan(
        tickers,
        risk_per_trade=cfg.risk_per_trade,
        min_rr=cfg.min_rr,
        use_last_hl_sl=cfg.use_last_hl_sl,
        tf=cfg.tf,
        new_only=cfg.new_only,
        is_manual_src=cfg.is_manual_src,
        scan_direction=cfg.scan_direction,
        scanner_id=cfg.scanner_id,
        no_rr_req=cfg.no_rr_req,
        tv_symbol_by_ticker=tv_symbol_by_ticker,
        company_name_by_ticker=company_names,
        on_phase_progress=on_phase_progress,
        on_phase_start=on_phase_start,
        on_phase_complete=on_phase_complete,
        on_scan_cancelled=on_scan_cancelled,
        is_cancelled=lambda: not st.session_state.scanning,
    )

    st.session_state.results = table_rows
    st.session_state.rejected = rejected_reasons
    st.session_state.results_as_of = reference_end_date
    st.session_state.results_tf = cfg.tf
    st.session_state.results_direction = cfg.scan_direction
    st.session_state.results_scanner = cfg.scanner_id
    st.session_state.ohlc_cache = ohlc_cache
    st.session_state.chart_cache = {}
    st.session_state.selected_tv_symbol = None

    with res_area.container():
        render_scan_results(
            table_rows, rejected_reasons, reference_end_date, cfg.tf,
            is_manual_src=cfg.is_manual_src,
            ohlc_cache=ohlc_cache,
            empty_message="No symbols passed the screener.",
            scan_direction=cfg.scan_direction,
            scanner_id=cfg.scanner_id,
        )

    st.session_state.scanning = False
    info_box.success("SCAN COMPLETE ✅")

else:
    last_src = _normalize_source_label(
        st.session_state.run_params.get("src", STOCKS_SRC)
    )
    table_rows = st.session_state.results
    rejected_reasons = st.session_state.rejected
    as_of = st.session_state.get("results_as_of")
    as_of_tf = st.session_state.get("results_tf", "Daily")
    as_of_dir = st.session_state.get("results_direction", "buy")
    as_of_scanner = st.session_state.get("results_scanner", "sequence_vova")

    with res_area.container():
        render_scan_results(
            table_rows, rejected_reasons, as_of, as_of_tf,
            is_manual_src=(last_src == MANUAL_SRC),
            chart_cache=st.session_state.get("chart_cache", {}),
            ohlc_cache=st.session_state.get("ohlc_cache", {}),
            scan_direction=as_of_dir,
            scanner_id=as_of_scanner,
        )

