#!/usr/bin/env python3
"""FMP EPS filter + PE/PEG valuation helpers. Premium cap: 750 HTTP calls/min (we use 12/s)."""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ticker_data import TV_LIST_STOCK_TICKERS, read_list_file, write_list_file

BASE = "https://financialmodelingprep.com/stable"
CACHE_PATH = ROOT / ".cache" / "fmp_eps.json"
VALUATION_CACHE_PATH = ROOT / ".cache" / "fmp_valuation.json"
RETRY_REASONS = frozenset({"FMP_HTTP", "FMP_RATE_LIMIT", "FMP_ERROR"})
GROWTH_PE_FLOOR = 15.0
GROWTH_LOOKBACK_YEARS = 5
FMP_HTTP_RATE_PER_SEC = 12.0
FMP_SCAN_WORKERS = 8
_last_http_at = 0.0
_http_lock = threading.Lock()


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
        f.flush()
    tmp.replace(path)


def _throttle_fmp_http(rate_per_sec: float = FMP_HTTP_RATE_PER_SEC) -> None:
    """Thread-safe global pacer so concurrent workers still respect the FMP cap."""
    global _last_http_at
    interval = 1.0 / max(rate_per_sec, 0.1)
    with _http_lock:
        now = time.monotonic()
        wait = interval - (now - _last_http_at)
        if wait > 0:
            time.sleep(wait)
        _last_http_at = time.monotonic()


def yahoo_to_fmp(yahoo: str) -> str:
    s = (yahoo or "").strip().upper()
    if not s:
        return s
    if s.endswith((".TO", ".V", ".NE", ".CN")):
        return s
    if len(s) >= 3 and s[-2] == "-" and s[-1].isalpha():
        return f"{s[:-2]}.{s[-1]}"
    return s.replace("-", ".")


def fmp_symbol_candidates(yahoo: str) -> list[str]:
    """
    Symbol forms to try, in order. FMP serves US class shares under the dash
    form (BRK-B); the dotted form answers HTTP 402, so keep it only as fallback.
    """
    raw = (yahoo or "").strip().upper()
    forms = [raw] if raw else []
    mapped = yahoo_to_fmp(yahoo)
    if mapped and mapped not in forms:
        forms.append(mapped)
    return forms


def _num(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        n = float(v)
    except (TypeError, ValueError):
        return None
    return n if math.isfinite(n) else None


def _str(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def load_fmp_api_key() -> str:
    api_key = (os.environ.get("FMP_API_KEY") or "").strip()
    if api_key:
        return api_key
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if t.startswith("FMP_API_KEY="):
                return t.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _fmp_get(path: str, api_key: str, params: dict[str, str]) -> object:
    q = urllib.parse.urlencode({"apikey": api_key, **params})
    url = f"{BASE}{path}?{q}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    last_err: Exception | None = None
    for attempt in range(5):
        _throttle_fmp_http()
        try:
            with urllib.request.urlopen(req, timeout=60) as res:
                data = json.loads(res.read().decode("utf-8"))
            if isinstance(data, dict) and data.get("Error Message"):
                raise RuntimeError(str(data.get("Error Message")))
            return data
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            # Transient network/read timeout: retry with backoff instead of
            # dropping a whole screener exchange or a ticker on the first hiccup.
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
            continue
    if last_err:
        raise last_err
    raise RuntimeError(f"FMP request failed for {path}")


def _year_of(row: dict) -> int | None:
    y = _num(row.get("calendarYear"))
    if y is not None:
        return int(y)
    date = _str(row.get("date")) or ""
    if len(date) >= 4 and date[:4].isdigit():
        return int(date[:4])
    return None


def cagr_pct(first: float, last: float, years: float) -> float | None:
    if years <= 0 or first <= 0 or last <= 0:
        return None
    return ((last / first) ** (1.0 / years) - 1.0) * 100.0


def trailing_eps_cagr(
    points: list[tuple[int, float]],
    lookback_years: int = GROWTH_LOOKBACK_YEARS,
) -> float | None:
    with_m = sorted([(y, e) for y, e in points if e > 0], key=lambda p: p[0])
    if len(with_m) < 2:
        return None
    last_y, last_e = with_m[-1]
    target_year = last_y - lookback_years
    first_y, first_e = with_m[0]
    for y, e in with_m:
        if y <= target_year:
            first_y, first_e = y, e
    span = last_y - first_y
    if span < 2:
        return None
    return cagr_pct(first_e, last_e, span)


def fair_value_rule(growth_pct: float | None) -> str | None:
    if growth_pct is None or not math.isfinite(growth_pct):
        return None
    if growth_pct < GROWTH_PE_FLOOR:
        return "pe15"
    return "lynch_peg"


def lynch_peg(pe: float | None, growth_pct: float | None) -> float | None:
    if pe is None or growth_pct is None or growth_pct == 0:
        return None
    if not math.isfinite(pe) or not math.isfinite(growth_pct):
        return None
    return pe / growth_pct


def is_undervalued(pe: float | None, growth_pct: float | None) -> bool:
    if pe is None or pe <= 0 or growth_pct is None or not math.isfinite(growth_pct):
        return False
    rule = fair_value_rule(growth_pct)
    if rule == "pe15":
        return pe < GROWTH_PE_FLOOR
    if rule == "lynch_peg":
        peg = lynch_peg(pe, growth_pct)
        return peg is not None and peg < 1.0
    return False


def fetch_eps_pe_growth(
    fmp_symbol: str,
    api_key: str,
) -> tuple[bool, str, float | None, float | None, float | None]:
    try:
        ttm_raw = _fmp_get("/key-metrics-ttm", api_key, {"symbol": fmp_symbol})
        ttm_rows = ttm_raw if isinstance(ttm_raw, list) else []
        ttm = ttm_rows[0] if ttm_rows else {}
        inc_raw = _fmp_get(
            "/income-statement",
            api_key,
            {"symbol": fmp_symbol, "period": "annual", "limit": "8"},
        )
        inc_rows = inc_raw if isinstance(inc_raw, list) else []
        eps = _num(ttm.get("netIncomePerShareTTM")) or _num(ttm.get("epsTTM"))
        if eps is None and inc_rows:
            inc0 = inc_rows[0] if isinstance(inc_rows[0], dict) else {}
            eps = _num(inc0.get("epsdiluted")) or _num(inc0.get("eps"))
        pe = _num(ttm.get("peRatioTTM")) or _num(ttm.get("priceToEarningsRatioTTM"))
        if pe is None:
            try:
                rat_raw = _fmp_get("/ratios-ttm", api_key, {"symbol": fmp_symbol})
                rat_rows = rat_raw if isinstance(rat_raw, list) else []
                rat = rat_rows[0] if rat_rows else {}
                pe = _num(rat.get("priceToEarningsRatioTTM")) or _num(rat.get("peRatioTTM"))
            except Exception:
                pe = None
        points: list[tuple[int, float]] = []
        for row in inc_rows:
            if not isinstance(row, dict):
                continue
            year = _year_of(row)
            e = _num(row.get("epsdiluted")) or _num(row.get("eps"))
            if year is None or e is None:
                continue
            points.append((year, e))
        growth = trailing_eps_cagr(points)
        if eps is None:
            return False, "NO_TRAILING_EPS", None, pe, growth
        if eps <= 0:
            return False, "NON_POSITIVE_EPS", eps, pe, growth
        return True, "PASS", eps, pe, growth
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            return False, "FMP_RATE_LIMIT", None, None, None
        return False, "FMP_HTTP", None, None, None
    except Exception:
        return False, "FMP_ERROR", None, None, None


def check_positive_eps_fmp(yahoo: str, api_key: str) -> tuple[bool, str, float | None]:
    symbol = yahoo_to_fmp(yahoo)
    try:
        profile_raw = _fmp_get("/profile", api_key, {"symbol": symbol})
        rows = profile_raw if isinstance(profile_raw, list) else []
        profile = rows[0] if rows else {}
        if profile.get("isEtf") or profile.get("isFund"):
            return False, "NOT_EQUITY", None
        if profile.get("isActivelyTrading") is False:
            return False, "NOT_TRADING", None
        ttm_raw = _fmp_get("/key-metrics-ttm", api_key, {"symbol": symbol})
        ttm_rows = ttm_raw if isinstance(ttm_raw, list) else []
        ttm = ttm_rows[0] if ttm_rows else {}
        eps = (
            _num(ttm.get("netIncomePerShareTTM"))
            or _num(ttm.get("epsTTM"))
            or _num(profile.get("eps"))
        )
        if eps is None:
            inc_raw = _fmp_get(
                "/income-statement",
                api_key,
                {"symbol": symbol, "period": "annual", "limit": "1"},
            )
            inc_rows = inc_raw if isinstance(inc_raw, list) else []
            inc = inc_rows[0] if inc_rows else {}
            eps = _num(inc.get("epsdiluted")) or _num(inc.get("eps"))
        if eps is None:
            return False, "NO_TRAILING_EPS", None
        if eps <= 0:
            return False, "NON_POSITIVE_EPS", eps
        return True, "PASS", eps
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            return False, "FMP_RATE_LIMIT", None
        return False, "FMP_HTTP", None
    except Exception:
        return False, "FMP_ERROR", None


def filter_positive_eps_fmp(
    entries: list[tuple[str, str, str]],
    *,
    api_key: str,
    cache_path: Path = CACHE_PATH,
    resume: bool = True,
    retry_errors: bool = False,
    rate_per_sec: float = 4.0,
) -> tuple[list[tuple[str, str, str]], Counter]:
    cache = _load_json(cache_path) if resume else {}
    checked: dict = cache.setdefault("checked", {})
    if retry_errors:
        for key in list(checked):
            if checked[key].get("reason") in RETRY_REASONS:
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
                rejects[prev.get("reason", "FMP_ERROR")] += 1
        else:
            pending.append((tv, yahoo, name))
    print(f"FMP EPS pending: {len(pending)} (cached: {len(entries) - len(pending)})")
    _ = rate_per_sec
    for i, (tv, yahoo, name) in enumerate(pending, start=1):
        ok, reason, eps = check_positive_eps_fmp(yahoo, api_key)
        if reason == "FMP_RATE_LIMIT":
            time.sleep(4.0)
            ok, reason, eps = check_positive_eps_fmp(yahoo, api_key)
        checked[yahoo] = {
            "reason": reason,
            "tv_part": tv,
            "name": name,
            "trailingEps": eps,
            "fmp": yahoo_to_fmp(yahoo),
        }
        if ok:
            passed.append((tv, yahoo, name))
        else:
            rejects[reason] += 1
        if i % 100 == 0 or i == len(pending):
            _save_json(cache_path, cache)
            print(f"  {i}/{len(pending)}  passed={len(passed)}  last={yahoo} {reason}")
    _save_json(cache_path, cache)
    return passed, rejects


def _valuation_row(
    tv: str,
    yahoo: str,
    name: str,
    eps: float | None,
    pe: float | None,
    growth: float | None,
    reason: str,
) -> dict:
    exchange = tv.split(":", 1)[0] if ":" in tv else ""
    rule = fair_value_rule(growth) if reason == "PASS" else None
    peg = lynch_peg(pe, growth)
    passed = bool(reason == "PASS" and is_undervalued(pe, growth))
    return {
        "yahoo": yahoo,
        "tv": tv,
        "name": name,
        "exchange": exchange,
        "epsTtm": eps,
        "peTtm": pe,
        "growth5y": growth,
        "rule": rule or "",
        "pegLynch": peg,
        "pass": passed,
        "epsReason": reason,
    }


def scan_eps_and_valuation(
    entries: list[tuple[str, str, str]],
    *,
    api_key: str,
    eps_cache_path: Path = CACHE_PATH,
    val_cache_path: Path = VALUATION_CACHE_PATH,
    resume: bool = True,
    retry_errors: bool = False,
    rate_per_sec: float = 4.0,
    workers: int = FMP_SCAN_WORKERS,
) -> tuple[list[tuple[str, str, str]], Counter, list[dict]]:
    eps_cache = _load_json(eps_cache_path) if resume else {}
    val_cache = _load_json(val_cache_path) if resume else {}
    eps_checked: dict = eps_cache.setdefault("checked", {})
    val_checked: dict = val_cache.setdefault("checked", {})
    if retry_errors:
        for key in list(eps_checked):
            if eps_checked[key].get("reason") in RETRY_REASONS:
                del eps_checked[key]
                val_checked.pop(key, None)
        for key in list(val_checked):
            if val_checked[key].get("reason") in RETRY_REASONS:
                del val_checked[key]
    passed: list[tuple[str, str, str]] = []
    rejects: Counter = Counter()
    rows: list[dict] = []
    pending: list[tuple[str, str, str]] = []
    for tv, yahoo, name in entries:
        prev_eps = eps_checked.get(yahoo)
        prev_val = val_checked.get(yahoo)
        have_val = bool(prev_val) and prev_val.get("reason") not in RETRY_REASONS
        if prev_eps and have_val:
            reason = prev_eps.get("reason", "FMP_ERROR")
            eps = _num(prev_eps.get("trailingEps"))
            pe = _num(prev_val.get("peTtm"))
            growth = _num(prev_val.get("growth5y"))
            if reason == "PASS":
                passed.append((tv, yahoo, name))
                rows.append(_valuation_row(tv, yahoo, name, eps, pe, growth, reason))
            else:
                rejects[reason] += 1
        else:
            pending.append((tv, yahoo, name))
    workers = max(1, int(workers))
    print(
        f"FMP EPS+valuation pending: {len(pending)} "
        f"(cached: {len(entries) - len(pending)}, workers={workers})"
    )
    _ = rate_per_sec

    def _persist() -> None:
        try:
            _save_json(val_cache_path, val_cache)
            _save_json(eps_cache_path, eps_cache)
        except OSError as exc:
            print(f"  cache save failed ({exc}); continuing in memory", file=sys.stderr)

    def _scan_one(item: tuple[str, str, str]) -> tuple[tuple[str, str, str], str, str, float | None, float | None, float | None]:
        tv, yahoo, name = item
        symbol = yahoo_to_fmp(yahoo)
        reason = "FMP_ERROR"
        eps = pe = growth = None
        for candidate in fmp_symbol_candidates(yahoo):
            symbol = candidate
            _ok, reason, eps, pe, growth = fetch_eps_pe_growth(symbol, api_key)
            if reason == "FMP_RATE_LIMIT":
                time.sleep(4.0)
                _ok, reason, eps, pe, growth = fetch_eps_pe_growth(symbol, api_key)
            if reason not in ("FMP_HTTP", "FMP_ERROR"):
                break
        return item, symbol, reason, eps, pe, growth

    # Concurrent fetch: the global _throttle_fmp_http lock still caps total
    # request rate, but overlapping network latency instead of paying it per
    # ticker cuts a full uncached pass from hours to ~20 min.
    results: dict[str, tuple[str, str, float | None, float | None, float | None]] = {}
    pending_by_yahoo = {yahoo: (tv, yahoo, name) for tv, yahoo, name in pending}
    done = 0

    def _write_cache(item: tuple[str, str, str], res: tuple[str, str, float | None, float | None, float | None]) -> None:
        tv_i, yahoo_i, name_i = item
        symbol_i, reason_i, eps_i, pe_i, growth_i = res
        eps_checked[yahoo_i] = {"reason": reason_i, "tv_part": tv_i, "name": name_i, "trailingEps": eps_i, "fmp": symbol_i}
        val_checked[yahoo_i] = {"reason": reason_i, "tv_part": tv_i, "name": name_i, "epsTtm": eps_i, "peTtm": pe_i, "growth5y": growth_i, "fmp": symbol_i}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_scan_one, item): item for item in pending}
        for fut in as_completed(futures):
            item, symbol, reason, eps, pe, growth = fut.result()
            _tv, yahoo, _name = item
            results[yahoo] = (symbol, reason, eps, pe, growth)
            done += 1
            if done % 100 == 0 or done == len(pending):
                # Persist incrementally so an interrupt keeps a warm cache.
                for y, res in results.items():
                    _write_cache(pending_by_yahoo[y], res)
                _persist()
                print(f"  {done}/{len(pending)}  last={yahoo} {reason}")

    # Materialize in original (sorted) pending order for deterministic output.
    for tv, yahoo, name in pending:
        res = results.get(yahoo)
        if res is None:
            continue
        symbol, reason, eps, pe, growth = res
        eps_checked[yahoo] = {
            "reason": reason,
            "tv_part": tv,
            "name": name,
            "trailingEps": eps,
            "fmp": symbol,
        }
        val_checked[yahoo] = {
            "reason": reason,
            "tv_part": tv,
            "name": name,
            "epsTtm": eps,
            "peTtm": pe,
            "growth5y": growth,
            "fmp": symbol,
        }
        if reason == "PASS":
            passed.append((tv, yahoo, name))
            rows.append(_valuation_row(tv, yahoo, name, eps, pe, growth, reason))
        else:
            rejects[reason] += 1
    _persist()
    return passed, rejects, rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter STOCK-TICKERS.txt via FMP TTM EPS > 0")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    api_key = load_fmp_api_key()
    if not api_key:
        print("FMP_API_KEY is not set.", file=sys.stderr)
        return 1
    tickers, tv_map, name_map, err = read_list_file(TV_LIST_STOCK_TICKERS)
    if err:
        print(err, file=sys.stderr)
        return 1
    entries = [(tv_map.get(y) or y, y, name_map.get(y) or "") for y in tickers]
    if args.limit > 0:
        entries = entries[: args.limit]
    passed, rejects = filter_positive_eps_fmp(
        entries,
        api_key=api_key,
        resume=not args.no_resume,
        retry_errors=args.retry_errors,
    )
    print()
    print(f"In {len(entries)} → EPS>0 {len(passed)}")
    for reason, n in sorted(rejects.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {n}")
    if not args.write:
        print(f"Dry-run: would write {len(passed)} lines -> {TV_LIST_STOCK_TICKERS}")
        return 0
    write_list_file(TV_LIST_STOCK_TICKERS, passed)
    print(f"Wrote {len(passed)} lines -> {TV_LIST_STOCK_TICKERS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
