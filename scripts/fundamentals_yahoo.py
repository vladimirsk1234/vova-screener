#!/usr/bin/env python3
"""
Shared Yahoo .info fundamentals filter for stock-list builders.

Keeps EQUITY on major US/CA exchanges with trailingEps > 0; drops OTC / non-tradable.
Resumable cache + rate limit (same pattern as the former gap_scan PE filter).
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ticker_data import (
    _eps_from_yf_info,
    is_major_us_ca_exchange,
    is_otc_yahoo_exchange,
    name_suggests_non_common,
)

EPS_RATE_LIMIT_PER_SEC = 4.0
EPS_RETRY_DELAY_SEC = 1.0
EPS_RETRY_REASONS = frozenset(
    {"EPS_RATE_LIMIT", "EPS_INFO_ERROR", "NON_MAJOR_EXCHANGE"}
)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _float_or_none(val) -> float | None:
    if val is None:
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def check_positive_eps(yahoo: str, *, name_hint: str = "") -> tuple[bool, str, float | None]:
    """
    Return (ok, reason, trailing_eps).

    Rejects: NOT_EQUITY, OTC_EXCHANGE, NON_MAJOR_EXCHANGE, NON_COMMON_NAME,
    NO_TRAILING_EPS, NON_POSITIVE_EPS, EPS_RATE_LIMIT, EPS_INFO_ERROR.
    """
    try:
        info = yf.Ticker(yahoo).info or {}
    except Exception as exc:
        msg = str(exc)
        if "RateLimit" in msg or "Too Many Requests" in msg:
            return False, "EPS_RATE_LIMIT", None
        return False, "EPS_INFO_ERROR", None

    quote_type = str(info.get("quoteType") or "").upper()
    if quote_type and quote_type != "EQUITY":
        return False, "NOT_EQUITY", None

    exchange = info.get("exchange") or info.get("fullExchangeName")
    if is_otc_yahoo_exchange(str(exchange) if exchange is not None else None):
        return False, "OTC_EXCHANGE", None
    if exchange and not is_major_us_ca_exchange(str(exchange)):
        return False, "NON_MAJOR_EXCHANGE", None

    display_name = (
        str(info.get("longName") or info.get("shortName") or name_hint or "").strip()
    )
    if display_name and name_suggests_non_common(display_name):
        return False, "NON_COMMON_NAME", None

    trailing_eps, _forward = _eps_from_yf_info(info)
    trailing_eps = _float_or_none(trailing_eps)
    if trailing_eps is None:
        return False, "NO_TRAILING_EPS", None
    if trailing_eps <= 0:
        return False, "NON_POSITIVE_EPS", trailing_eps
    return True, "PASS", trailing_eps


def filter_positive_eps(
    entries: list[tuple[str, str, str]],
    *,
    cache_path: Path,
    resume: bool = True,
    retry_errors: bool = False,
) -> tuple[list[tuple[str, str, str]], Counter]:
    """Filter (tv, yahoo, name) entries to those with trailingEps > 0."""
    cache = _load_json(cache_path) if resume else {}
    checked: dict = cache.setdefault("checked", {})
    if retry_errors:
        for key in list(checked):
            if checked[key].get("reason") in EPS_RETRY_REASONS:
                del checked[key]

    passed: list[tuple[str, str, str]] = []
    rejects: Counter = Counter()
    pending: list[tuple[str, str, str]] = []

    for tv, yahoo, name in entries:
        prev = checked.get(yahoo)
        if prev:
            if prev.get("reason") == "PASS":
                passed.append((tv, yahoo, name))
            else:
                rejects[prev.get("reason", "EPS_INFO_ERROR")] += 1
        else:
            pending.append((tv, yahoo, name))

    print(f"EPS check pending: {len(pending)} (cached: {len(entries) - len(pending)})")
    min_interval = 1.0 / EPS_RATE_LIMIT_PER_SEC
    last_t = 0.0

    for i, (tv, yahoo, name) in enumerate(pending, start=1):
        now = time.monotonic()
        wait = min_interval - (now - last_t)
        if wait > 0:
            time.sleep(wait)
        last_t = time.monotonic()

        ok, reason, eps = check_positive_eps(yahoo, name_hint=name)
        if reason == "EPS_RATE_LIMIT":
            time.sleep(EPS_RETRY_DELAY_SEC * 4)
            ok, reason, eps = check_positive_eps(yahoo, name_hint=name)

        checked[yahoo] = {
            "reason": reason,
            "tv_part": tv,
            "name": name,
            "trailingEps": eps,
        }
        if ok:
            passed.append((tv, yahoo, name))
        else:
            rejects[reason] += 1

        if i % 25 == 0 or i == len(pending):
            _save_json(cache_path, {"checked": checked})
            print(f"  EPS checked {i}/{len(pending)} - passed so far: {len(passed)}")

    _save_json(cache_path, {"checked": checked})
    passed.sort(key=lambda e: e[0])
    return passed, rejects
