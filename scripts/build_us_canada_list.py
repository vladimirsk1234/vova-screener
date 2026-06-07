#!/usr/bin/env python3
"""
Build TV-LIST-US-CANADA-FULL.txt — US + Canada common stocks, positive trailing EPS.

Hard excludes: OTC, ETFs, preferreds, warrants, units, funds.
Resumable via .cache/us_ca_list_build.json

Usage:
  python scripts/build_us_canada_list.py              # full build
  python scripts/build_us_canada_list.py --limit 50   # smoke sample
  python scripts/build_us_canada_list.py --resume     # continue after interrupt
  python scripts/build_us_canada_list.py --us-only
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yfinance as yf

from ticker_data import (
    TV_LIST_US_CANADA_FULL,
    is_major_us_ca_exchange,
    is_otc_yahoo_exchange,
    name_suggests_non_common,
    tv_part_to_yahoo,
    write_list_file,
)

NASDAQ_TRADED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
ADANOS_LISTINGS_URL = (
    "https://raw.githubusercontent.com/adanos-software/free-ticker-database/main/data/listings.csv"
)
TSX_JSON_URL = "https://www.tsx.com/json/company-directory/search/tsx/%5E/all/%5E/all"
TSXV_JSON_URL = "https://www.tsx.com/json/company-directory/search/tsxv/%5E/all/%5E/all"

CACHE_PATH = ROOT / ".cache" / "us_ca_list_build.json"
OUT_PATH = ROOT / TV_LIST_US_CANADA_FULL

YAHOO_DELAY_SEC = 0.12
NON_COMMON_SYMBOL_RE = re.compile(
    r"(\$|\.PR|/P[A-Z]?$|-P[A-Z]$|-PA$|-PB$|-PC$|-PD$|-PE$|-PF$|-PG$|-PH$|-PI$|"
    r"-WT$|\.WS$|-W$|-UN$|\.UN$|-U$)",
    re.IGNORECASE,
)

US_LISTING_TO_TV: dict[str, str] = {
    "N": "NASDAQ",
    "Q": "NASDAQ",
    "A": "AMEX",
    "P": "AMEX",
    "Z": "NASDAQ",
    "B": "NASDAQ",
    "V": "NASDAQ",
    "C": "NASDAQ",
}

OTHER_LISTING_TO_TV: dict[str, str] = {
    "N": "NYSE",
    "A": "AMEX",
    "P": "AMEX",
    "Z": "NASDAQ",
    "B": "NASDAQ",
    "V": "NASDAQ",
}

COMMON_STOCK_MARKERS = (
    "common stock",
    "ordinary shares",
    "class a common",
    "class b common",
    "class c common",
    "common shares",
)


@dataclass(frozen=True)
class Candidate:
    tv_part: str
    yahoo: str
    name_hint: str
    region: str


def _fetch_text(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "vova-screener/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _looks_otc_exchange_name(name: str) -> bool:
    n = name.upper()
    return "OTC" in n or "PINK" in n or "GREY" in n or "OTCBB" in n


def _is_common_stock_name(name: str) -> bool:
    n = name.lower()
    if name_suggests_non_common(name):
        return False
    if any(m in n for m in ("preferred", "warrant", " unit", "units", " etf", " fund", "debenture")):
        return False
    if any(m in n for m in COMMON_STOCK_MARKERS):
        return True
    # NYSE/Nasdaq rows without explicit "Common Stock" but also no red flags
    bad = ("note", "bond", "trust unit", "lp ", " l.p.", "reit", " depositary")
    return not any(b in n for b in bad)


def _symbol_non_common(symbol: str) -> bool:
    return bool(NON_COMMON_SYMBOL_RE.search(symbol))


def _load_us_candidates() -> list[Candidate]:
    out: list[Candidate] = []
    seen: set[str] = set()

    def add(tv_ex: str, symbol: str, name: str) -> None:
        sym = symbol.strip().upper()
        if not sym or sym in seen:
            return
        if _symbol_non_common(sym):
            return
        tv_part = f"{tv_ex}:{sym}"
        yahoo = tv_part_to_yahoo(tv_part)
        if not yahoo:
            return
        seen.add(sym)
        out.append(Candidate(tv_part=tv_part, yahoo=yahoo, name_hint=name.strip(), region="US"))

    # NASDAQ Trader combined file
    raw = _fetch_text(NASDAQ_TRADED_URL)
    reader = csv.DictReader(io.StringIO(raw), delimiter="|")
    for row in reader:
        if (row.get("ETF") or "").strip().upper() == "Y":
            continue
        if (row.get("Test Issue") or "").strip().upper() == "Y":
            continue
        if (row.get("NextShares") or "").strip().upper() == "Y":
            continue
        ex_name = row.get("Exchange Name") or ""
        if _looks_otc_exchange_name(ex_name):
            continue
        name = row.get("Security Name") or ""
        if not _is_common_stock_name(name):
            continue
        sym = (row.get("Symbol") or row.get("NASDAQ Symbol") or "").strip()
        if not sym:
            continue
        listing = (row.get("Listing Exchange") or "").strip().upper()
        tv_ex = US_LISTING_TO_TV.get(listing, "NASDAQ")
        add(tv_ex, sym, name)

    # NYSE / AMEX / Arca via otherlisted.txt
    raw2 = _fetch_text(OTHER_LISTED_URL)
    reader2 = csv.DictReader(io.StringIO(raw2), delimiter="|")
    for row in reader2:
        if (row.get("ETF") or "").strip().upper() == "Y":
            continue
        if (row.get("Test Issue") or "").strip().upper() == "Y":
            continue
        ex_name = row.get("Exchange Name") or ""
        if _looks_otc_exchange_name(ex_name):
            continue
        name = row.get("Security Name") or ""
        if not _is_common_stock_name(name):
            continue
        sym = (row.get("ACT Symbol") or row.get("CQS Symbol") or "").strip()
        if not sym:
            continue
        listing = (row.get("Listing Exchange") or "").strip().upper()
        tv_ex = OTHER_LISTING_TO_TV.get(listing, "NYSE")
        add(tv_ex, sym, name)

    return out


def _load_tsx_json(url: str, tv_ex: str) -> list[Candidate]:
    out: list[Candidate] = []
    try:
        raw = _fetch_text(url)
        data = json.loads(raw)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        print(f"Warning: could not load {tv_ex} directory: {exc}", file=sys.stderr)
        return out

    rows = data.get("results") or data.get("data") or []
    if isinstance(rows, dict):
        rows = list(rows.values())

    for item in rows:
        if not isinstance(item, dict):
            continue
        sym = str(item.get("symbol") or item.get("ticker") or "").strip().upper()
        name = str(item.get("name") or item.get("companyName") or "").strip()
        if not sym or _symbol_non_common(sym):
            continue
        if sym.endswith(".UN") or sym.endswith("-UN"):
            continue
        if name_suggests_non_common(name):
            continue
        tv_part = f"{tv_ex}:{sym}"
        yahoo = tv_part_to_yahoo(tv_part)
        if not yahoo:
            continue
        out.append(Candidate(tv_part=tv_part, yahoo=yahoo, name_hint=name, region="CA"))
    return out


def _load_ca_candidates_from_adanos() -> list[Candidate]:
    """Canadian common stocks from free-ticker-database listings.csv (TSX + TSXV)."""
    out: list[Candidate] = []
    try:
        raw = _fetch_text(ADANOS_LISTINGS_URL, timeout=90)
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"Warning: could not load Canadian listings CSV: {exc}", file=sys.stderr)
        return out

    reader = csv.DictReader(io.StringIO(raw))
    for row in reader:
        exchange = (row.get("exchange") or "").strip().upper()
        if exchange not in ("TSX", "TSXV"):
            continue
        if (row.get("country_code") or "").strip().upper() != "CA":
            continue
        asset_type = (row.get("asset_type") or "").strip()
        if asset_type != "Stock":
            continue
        if (row.get("etf_category") or "").strip():
            continue
        sym = (row.get("ticker") or "").strip().upper()
        name = (row.get("name") or "").strip()
        if not sym or _symbol_non_common(sym):
            continue
        if sym.endswith(".P") or sym.endswith(".PR") or sym.endswith("-H"):
            continue
        if name_suggests_non_common(name):
            continue
        tv_ex = "TSX" if exchange == "TSX" else "TSXV"
        tv_part = f"{tv_ex}:{sym}"
        yahoo = tv_part_to_yahoo(tv_part)
        if not yahoo:
            continue
        out.append(Candidate(tv_part=tv_part, yahoo=yahoo, name_hint=name, region="CA"))
    return out


def _load_ca_candidates() -> list[Candidate]:
    out = _load_ca_candidates_from_adanos()
    if out:
        print(f"  Canada (adanos listings.csv): {len(out)} Layer-1 candidates")
        seen: set[str] = set()
        deduped: list[Candidate] = []
        for c in out:
            if c.yahoo in seen:
                continue
            seen.add(c.yahoo)
            deduped.append(c)
        return deduped

    print("  Falling back to TSX JSON directory (legacy API)...", file=sys.stderr)
    tsx = _load_tsx_json(TSX_JSON_URL, "TSX")
    tsxv = _load_tsx_json(TSXV_JSON_URL, "TSXV")
    seen: set[str] = set()
    merged: list[Candidate] = []
    for c in tsx + tsxv:
        if c.yahoo in seen:
            continue
        seen.add(c.yahoo)
        merged.append(c)
    return merged


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {"checked": {}, "passed": []}
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"checked": {}, "passed": []}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _validate_yahoo(candidate: Candidate) -> tuple[bool, str, str]:
    """Return (keep, reject_reason, company_name)."""
    try:
        info = yf.Ticker(candidate.yahoo).info or {}
    except Exception:
        return False, "NO_DATA", candidate.name_hint

    if not info or info.get("regularMarketPrice") is None and info.get("symbol") is None:
        return False, "NO_DATA", candidate.name_hint

    quote_type = str(info.get("quoteType") or "").upper()
    if quote_type == "ETF":
        return False, "ETF", candidate.name_hint
    if quote_type and quote_type != "EQUITY":
        return False, "NOT_EQUITY", candidate.name_hint

    exchange = info.get("exchange")
    if is_otc_yahoo_exchange(exchange):
        return False, "OTC", candidate.name_hint
    if exchange and not is_major_us_ca_exchange(exchange):
        return False, "OTC", candidate.name_hint

    country = str(info.get("country") or "")
    if country not in ("United States", "Canada"):
        return False, "NOT_US_CA", candidate.name_hint

    long_name = str(info.get("longName") or info.get("shortName") or candidate.name_hint or "")
    if name_suggests_non_common(long_name):
        return False, "NOT_COMMON", candidate.name_hint

    trailing_eps = info.get("trailingEps")
    try:
        eps = float(trailing_eps) if trailing_eps is not None else None
    except (TypeError, ValueError):
        eps = None
    if eps is None or eps <= 0:
        return False, "NEGATIVE_EPS", long_name or candidate.name_hint

    return True, "PASS", long_name or candidate.name_hint


def build_list(
    *,
    us_only: bool = False,
    ca_only: bool = False,
    limit: int = 0,
    resume: bool = True,
) -> tuple[list[tuple[str, str, str]], Counter]:
    """Return (entries for write_list_file, reject_stats)."""
    candidates: list[Candidate] = []
    if not ca_only:
        print("Loading US symbol directories...")
        candidates.extend(_load_us_candidates())
    if not us_only:
        print("Loading Canadian symbol directories...")
        candidates.extend(_load_ca_candidates())

    # Dedupe by Yahoo symbol
    by_yahoo: dict[str, Candidate] = {}
    for c in candidates:
        by_yahoo.setdefault(c.yahoo, c)
    candidates = list(by_yahoo.values())
    candidates.sort(key=lambda c: (c.region, c.yahoo))

    if limit > 0:
        candidates = candidates[:limit]

    print(f"Candidates after Layer 1 filters: {len(candidates)}")

    cache = _load_cache() if resume else {"checked": {}, "passed": []}
    checked: dict = cache.setdefault("checked", {})
    passed: list[list[str]] = list(cache.get("passed") or [])
    passed_yahoo = {p[1] for p in passed if len(p) >= 2}
    rejects: Counter = Counter()

    for i, cand in enumerate(candidates, 1):
        if cand.yahoo in checked:
            prev = checked[cand.yahoo]
            reason = prev.get("reason", "NO_DATA")
            if reason == "PASS":
                if cand.yahoo not in passed_yahoo:
                    passed.append([prev.get("tv_part", cand.tv_part), cand.yahoo, prev.get("name", cand.name_hint)])
                    passed_yahoo.add(cand.yahoo)
            else:
                rejects[reason] += 1
            continue

        time.sleep(YAHOO_DELAY_SEC)
        ok, reason, name = _validate_yahoo(cand)
        checked[cand.yahoo] = {
            "reason": reason,
            "tv_part": cand.tv_part,
            "name": name,
            "region": cand.region,
        }
        if ok:
            passed.append([cand.tv_part, cand.yahoo, name])
            passed_yahoo.add(cand.yahoo)
        else:
            rejects[reason] += 1

        if i % 25 == 0:
            _save_cache({"checked": checked, "passed": passed})
            print(f"  Yahoo validated {i}/{len(candidates)} — passed so far: {len(passed)}")

    _save_cache({"checked": checked, "passed": passed})

    entries: list[tuple[str, str, str]] = []
    seen_tv: set[str] = set()
    for tv_part, yahoo, name in passed:
        if tv_part in seen_tv:
            continue
        seen_tv.add(tv_part)
        entries.append((tv_part, yahoo, name))
    entries.sort(key=lambda e: e[0])
    return entries, rejects


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build TV-LIST-US-CANADA-FULL.txt (common stocks, positive EPS, no OTC/ETF)",
    )
    parser.add_argument("--us-only", action="store_true", help="US symbols only")
    parser.add_argument("--ca-only", action="store_true", help="Canada symbols only")
    parser.add_argument("--limit", type=int, default=0, help="Max candidates to Yahoo-check (0=all)")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore .cache/us_ca_list_build.json and re-check all symbols",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_PATH,
        help=f"Output list path (default: {TV_LIST_US_CANADA_FULL})",
    )
    args = parser.parse_args()

    if args.us_only and args.ca_only:
        print("Choose at most one of --us-only / --ca-only", file=sys.stderr)
        return 1

    entries, rejects = build_list(
        us_only=args.us_only,
        ca_only=args.ca_only,
        limit=args.limit,
        resume=not args.no_resume,
    )

    if not entries:
        print("No symbols passed all filters.", file=sys.stderr)
        return 1

    if args.output.resolve() == OUT_PATH.resolve():
        write_list_file(TV_LIST_US_CANADA_FULL, entries)
        out_display = OUT_PATH
    else:
        lines = []
        for tv_sym, _yahoo, company_name in entries:
            name = (company_name or "").strip()
            lines.append(f"{tv_sym}|{name}" if name else tv_sym)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out_display = args.output

    us_n = sum(1 for tv, _, _ in entries if tv.startswith(("NASDAQ:", "NYSE:", "AMEX:")))
    ca_n = len(entries) - us_n
    print(f"\nWrote {len(entries)} lines to {out_display}")
    print(f"  US: {us_n}  Canada: {ca_n}")
    if rejects:
        print("Reject breakdown (this run + resumed cache misses):")
        for reason, count in rejects.most_common(12):
            print(f"  {reason}: {count}")
    print("\nNext: restart RUN_SCREENER.bat and select US + CANADA FULL as source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
