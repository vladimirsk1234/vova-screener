"""
Memory helpers for Streamlit Cloud / large-universe FAST Graph scans.
"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd

# ~16y weekly bars / ~8y daily closes — enough for 10Y CAGR + Normal P/E.
FG_MAX_WEEKLY_BARS = 52 * 16
FG_MAX_DAILY_BARS = 252 * 8
FG_CHUNK_SIZE = 40
FG_CHUNK_SIZE_LOCAL = 80


def is_low_memory_runtime() -> bool:
    """True on Streamlit Community Cloud and when SCREENER_LOW_MEMORY=1."""
    env = os.environ.get("SCREENER_LOW_MEMORY", "auto").lower()
    if env in ("1", "true", "yes"):
        return True
    if env in ("0", "false", "no"):
        return False
    return bool(os.environ.get("STREAMLIT_SERVER_PORT"))


def scan_chunk_size(scanner_id: str) -> int:
    if scanner_id == "fast_graphs":
        return FG_CHUNK_SIZE if is_low_memory_runtime() else FG_CHUNK_SIZE_LOCAL
    if is_low_memory_runtime():
        return 50
    return 200


def ta_max_workers(scanner_id: str, *, default: int) -> int:
    if is_low_memory_runtime():
        return 2 if scanner_id == "fast_graphs" else 4
    return default


def download_max_workers(scanner_id: str, *, default: int) -> int:
    if is_low_memory_runtime():
        return 1
    return default


def yf_download_threads() -> bool:
    """Parallel symbol fetch inside yf.download — off on Streamlit Cloud."""
    return not is_low_memory_runtime()


def yf_info_max_workers(*, default: int) -> int:
    if is_low_memory_runtime():
        return 2
    return default


def yf_name_cache_rate_per_sec(*, default: float) -> float:
    if is_low_memory_runtime():
        return 4.0
    return default


def trim_price_frame(
    df: pd.DataFrame | None,
    *,
    max_bars: int,
) -> pd.DataFrame | None:
    """Keep Close only (float32) for chart / Normal P/E cache."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if "Close" not in df.columns:
        return df
    out = pd.DataFrame(
        {"Close": pd.to_numeric(df["Close"], errors="coerce").astype("float32")},
        index=df.index,
    )
    out = out.dropna(subset=["Close"])
    if len(out) > max_bars:
        out = out.iloc[-max_bars:]
    return out


def slim_fast_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Drop heavy bundle payload; keep chart + panel essentials."""
    if not metrics:
        return metrics
    slim = {k: v for k, v in metrics.items() if k != "bundle"}
    bundle = metrics.get("bundle") or {}
    if not slim.get("earnings_estimates"):
        slim["earnings_estimates"] = bundle.get("earnings_estimates") or {}
    if not slim.get("annual_dividends"):
        slim["annual_dividends"] = bundle.get("annual_dividends") or {}
    info = bundle.get("info") or {}
    slim["chart_info"] = {
        "currency": info.get("currency"),
        "dividend_rate": info.get("dividend_rate"),
    }
    return slim


def slim_ohlc_entry(entry: dict[str, Any], scanner_id: str) -> dict[str, Any]:
    if scanner_id != "fast_graphs" or not entry:
        return entry
    out = dict(entry)
    fm = out.get("fast_metrics")
    if isinstance(fm, dict):
        out["fast_metrics"] = slim_fast_metrics(fm)
    out["df"] = trim_price_frame(out.get("df"), max_bars=FG_MAX_WEEKLY_BARS)
    out["df_daily"] = trim_price_frame(out.get("df_daily"), max_bars=FG_MAX_DAILY_BARS)
    return out
