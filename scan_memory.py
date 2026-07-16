"""
Memory helpers for Streamlit Cloud / large-universe TA scans.
"""
from __future__ import annotations

import os
from typing import Any


def is_low_memory_runtime() -> bool:
    """True on Streamlit Community Cloud and when SCREENER_LOW_MEMORY=1."""
    env = os.environ.get("SCREENER_LOW_MEMORY", "auto").lower()
    if env in ("1", "true", "yes"):
        return True
    if env in ("0", "false", "no"):
        return False
    return bool(os.environ.get("STREAMLIT_SERVER_PORT"))


def scan_chunk_size(scanner_id: str = "sequence_vova") -> int:
    if is_low_memory_runtime():
        return 50
    return 200


def ta_max_workers(scanner_id: str = "sequence_vova", *, default: int) -> int:
    if is_low_memory_runtime():
        return 4
    return default


def download_max_workers(scanner_id: str = "sequence_vova", *, default: int) -> int:
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


def slim_ohlc_entry(entry: dict[str, Any], scanner_id: str = "sequence_vova") -> dict[str, Any]:
    return entry
