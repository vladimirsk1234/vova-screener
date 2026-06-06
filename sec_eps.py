"""
SEC EDGAR annual EPS via Company Facts API (free, no API key).
Falls back gracefully when SEC is unreachable or ticker is non-US.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "sec_eps"
_TICKER_CACHE = _CACHE_DIR / "company_tickers.json"
_EPS_TTL_SEC = 86400.0
_TICKER_TTL_SEC = 86400.0 * 7
_MIN_REQUEST_INTERVAL_SEC = 0.12

_LAST_REQUEST_AT = 0.0

_EPS_TAGS = (
    "EarningsPerShareDiluted",
    "EarningsPerShareBasic",
)


def _sec_user_agent() -> str:
    return os.environ.get("SEC_USER_AGENT", "vova-screener screener@example.com")


def _throttle() -> None:
    global _LAST_REQUEST_AT
    now = time.time()
    wait = _MIN_REQUEST_INTERVAL_SEC - (now - _LAST_REQUEST_AT)
    if wait > 0:
        time.sleep(wait)
    _LAST_REQUEST_AT = time.time()


def _sec_get(url: str, *, timeout: float = 30.0) -> requests.Response | None:
    try:
        _throttle()
        resp = requests.get(
            url,
            headers={"User-Agent": _sec_user_agent(), "Accept": "application/json"},
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp
    except Exception:
        pass
    return None


def _load_ticker_index() -> dict[str, int]:
    if _TICKER_CACHE.is_file():
        try:
            mtime = _TICKER_CACHE.stat().st_mtime
            if time.time() - mtime <= _TICKER_TTL_SEC:
                with open(_TICKER_CACHE, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return {str(k).upper(): int(v) for k, v in data.items()}
        except Exception:
            pass

    resp = _sec_get("https://www.sec.gov/files/company_tickers.json")
    if resp is None:
        return {}
    try:
        raw = resp.json()
        out: dict[str, int] = {}
        for item in raw.values():
            ticker = str(item.get("ticker", "")).upper()
            cik = int(item.get("cik_str", 0))
            if ticker and cik:
                out[ticker] = cik
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_TICKER_CACHE, "w", encoding="utf-8") as f:
            json.dump(out, f)
        return out
    except Exception:
        return {}


def _cik_for_ticker(ticker: str) -> str | None:
    sym = ticker.strip().upper().split(".")[0]
    cik = _load_ticker_index().get(sym)
    if not cik:
        return None
    return str(cik).zfill(10)


def _eps_cache_path(ticker: str) -> Path:
    safe = ticker.replace("/", "_").upper()
    return _CACHE_DIR / f"{safe}.json"


def _load_eps_cache(ticker: str) -> dict[int, float] | None:
    path = _eps_cache_path(ticker)
    if not path.is_file():
        return None
    try:
        if time.time() - path.stat().st_mtime > _EPS_TTL_SEC:
            return None
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return None
        return {int(k): float(v) for k, v in raw.items()}
    except Exception:
        return None


def _save_eps_cache(ticker: str, eps_map: dict[int, float]) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_eps_cache_path(ticker), "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in eps_map.items()}, f)
    except Exception:
        pass


def _parse_annual_eps_from_facts(facts: dict[str, Any]) -> dict[int, float]:
    gaap = facts.get("facts", {}).get("us-gaap", {})
    by_fy: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for tag in _EPS_TAGS:
        tag_data = gaap.get(tag)
        if not tag_data:
            continue
        units = tag_data.get("units", {})
        for unit_rows in units.values():
            if not isinstance(unit_rows, list):
                continue
            for row in unit_rows:
                if row.get("form") != "10-K" or row.get("fp") != "FY":
                    continue
                try:
                    fy = int(row["fy"])
                    val = float(row["val"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not (val == val and val > 0):  # skip NaN / non-positive
                    continue
                by_fy[fy].append(row)
        if by_fy:
            break

    out: dict[int, float] = {}
    for fy, rows in by_fy.items():
        best = max(rows, key=lambda r: str(r.get("filed", "")))
        out[fy] = float(best["val"])
    return out


def get_sec_annual_eps(ticker: str, *, use_cache: bool = True) -> dict[int, float]:
    """
    Return {fiscal_year: diluted/basic EPS} from SEC 10-K filings.
    Empty dict when ticker has no SEC CIK or fetch fails.
    """
    if use_cache:
        hit = _load_eps_cache(ticker)
        if hit is not None:
            return hit

    cik = _cik_for_ticker(ticker)
    if not cik:
        return {}

    resp = _sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
    if resp is None:
        return {}

    try:
        eps_map = _parse_annual_eps_from_facts(resp.json())
    except Exception:
        return {}

    if eps_map:
        _save_eps_cache(ticker, eps_map)
    return eps_map
