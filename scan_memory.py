"""
Memory helpers for Streamlit Cloud / large-universe TA scans.
"""
from __future__ import annotations

import os
from typing import Any

# Soft ceiling (bytes): cgroup memory max below this => treat as low-memory host.
_LOW_MEMORY_CGROUP_BYTES = int(2.5 * 1024 * 1024 * 1024)

_cached_is_streamlit_cloud: bool | None = None
_cached_is_low_memory: bool | None = None


def is_streamlit_cloud() -> bool:
    """True on Streamlit Community Cloud (and similar managed hosts).

    Do NOT key off STREAMLIT_SERVER_PORT alone — local `streamlit run` sets it too.
    """
    global _cached_is_streamlit_cloud
    if _cached_is_streamlit_cloud is not None:
        return _cached_is_streamlit_cloud
    _cached_is_streamlit_cloud = bool(
        os.path.isdir("/mount/src")
        or os.environ.get("HOME", "").startswith("/home/appuser")
    )
    return _cached_is_streamlit_cloud


def _cgroup_memory_limit_bytes() -> int | None:
    """Return cgroup memory max in bytes, or None if unknown / unlimited."""
    for path in (
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    ):
        try:
            raw = open(path, encoding="utf-8").read().strip()
        except OSError:
            continue
        if not raw or raw.lower() == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            return None
        # Ignore absurd "unlimited" placeholders used by some kernels.
        if value >= (1 << 62):
            return None
        return value
    return None


def is_low_memory_runtime() -> bool:
    """True on Streamlit Community Cloud and when SCREENER_LOW_MEMORY=1."""
    global _cached_is_low_memory
    if _cached_is_low_memory is not None:
        return _cached_is_low_memory

    env = os.environ.get("SCREENER_LOW_MEMORY", "auto").lower()
    if env in ("1", "true", "yes"):
        _cached_is_low_memory = True
        return True
    if env in ("0", "false", "no"):
        _cached_is_low_memory = False
        return False

    if is_streamlit_cloud():
        _cached_is_low_memory = True
        return True

    limit = _cgroup_memory_limit_bytes()
    if limit is not None and limit < _LOW_MEMORY_CGROUP_BYTES:
        _cached_is_low_memory = True
        return True

    _cached_is_low_memory = False
    return False


def scan_chunk_size(scanner_id: str = "sequence_vova") -> int:
    # Near full-speed batch size; peak RAM still bounded by the download pipeline.
    if is_low_memory_runtime():
        return 150
    return 200


def ta_max_workers(scanner_id: str = "sequence_vova", *, default: int) -> int:
    if is_low_memory_runtime():
        return 12
    return default


def download_max_workers(scanner_id: str = "sequence_vova", *, default: int) -> int:
    # Match local parallel download width; pipeline frees each batch after TA.
    if is_low_memory_runtime():
        return 3
    return default


def yf_download_threads() -> bool:
    """Parallel symbol fetch inside yf.download.

    Safe on Cloud when batch size is modest (scan_chunk_size); the OOM was from
    holding every batch + hundreds of TA futures, not from yfinance threads.
    """
    return True


def yf_info_max_workers(*, default: int) -> int:
    if is_low_memory_runtime():
        return 6
    return default


def yf_name_cache_rate_per_sec(*, default: float) -> float:
    if is_low_memory_runtime():
        return 10.0
    return default


def slim_ohlc_entry(entry: dict[str, Any], scanner_id: str = "sequence_vova") -> dict[str, Any]:
    return entry
