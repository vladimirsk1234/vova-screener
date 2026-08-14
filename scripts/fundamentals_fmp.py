#!/usr/bin/env python3
"""
Filter STOCK-TICKERS.txt to names with FMP TTM EPS (or net income per share) > 0.

Does not rewrite Yahoo OHLC validity. Requires FMP_API_KEY (Premium for CA + long history).
Default is dry-run; pass --write to overwrite STOCK-TICKERS.txt.

Usage:
  set FMP_API_KEY=...
  python scripts/fundamentals_fmp.py
  python scripts/fundamentals_fmp.py --write
  python scripts/fundamentals_fmp.py --limit 30 --write
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ticker_data import TV_LIST_STOCK_TICKERS, read_list_file, write_list_file

BASE = "https://financialmodelingprep.com/stable"
CACHE_PATH = ROOT / ".cache" / "fmp_eps.json"
RETRY_REASONS = frozenset({"FMP_HTTP", "FMP_RATE_LIMIT", "FMP_ERROR"})


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


def yahoo_to_fmp(yahoo: str) -> str:
    s = (yahoo or "").strip().upper()
    if not s:
        return s
    if s.endswith((".TO", ".V", ".NE", ".CN")):
        return s
    if len(s) >= 3 and s[-2] == "-" and s[-1].isalpha():
        return f"{s[:-2]}.{s[-1]}"
    return s.replace("-", ".")


def _num(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        n = float(v)
    except (TypeError, ValueError):
        return None
    return n if math.isfinite(n) else None


def _fmp_get(path: str, api_key: str, params: dict[str, str]) -> object:
    q = urllib.parse.urlencode({"apikey": api_key, **params})
    url = f"{BASE}{path}?{q}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as res:
        return json.loads(res.read().decode("utf-8"))


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
        if exc.code in (401, 403):
            return False, "FMP_HTTP", None
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
    min_interval = 1.0 / max(rate_per_sec, 0.1)
    last_t = 0.0

    for i, (tv, yahoo, name) in enumerate(pending, start=1):
        now = time.monotonic()
        wait = min_interval - (now - last_t)
        if wait > 0:
            time.sleep(wait)
        last_t = time.monotonic()

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

        if i % 25 == 0 or i == len(pending):
            _save_json(cache_path, cache)
            print(f"  {i}/{len(pending)}  passed={len(passed)}  last={yahoo} {reason}")

    _save_json(cache_path, cache)
    return passed, rejects


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter STOCK-TICKERS.txt via FMP TTM EPS > 0")
    parser.add_argument("--write", action="store_true", help="Overwrite STOCK-TICKERS.txt")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()

    api_key = (os.environ.get("FMP_API_KEY") or "").strip()
    if not api_key:
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                t = line.strip()
                if t.startswith("FMP_API_KEY="):
                    api_key = t.split("=", 1)[1].strip().strip('"').strip("'")
                    break
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
