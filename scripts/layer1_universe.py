#!/usr/bin/env python3
"""
Layer-1 US + Canada symbol universe from exchange directories.

Used by scripts/build_full_us_tsx_ohlc_list.py (no Yahoo .info / EPS filter).
"""
from __future__ import annotations

import csv
import io
import json
import re
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

try:
    import certifi
except ImportError:
    certifi = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ticker_data import (
    name_suggests_non_common,
    tv_part_to_yahoo,
)

NASDAQ_TRADED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
ADANOS_LISTINGS_URL = (
    "https://raw.githubusercontent.com/adanos-software/free-ticker-database/main/data/listings.csv"
)
TSX_JSON_URL = "https://www.tsx.com/json/company-directory/search/tsx/%5E/all/%5E/all"
TSXV_JSON_URL = "https://www.tsx.com/json/company-directory/search/tsxv/%5E/all/%5E/all"

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
    contexts: list[ssl.SSLContext | None] = []
    if certifi is not None:
        contexts.append(ssl.create_default_context(cafile=certifi.where()))
    contexts.append(None)
    contexts.append(ssl._create_unverified_context())

    last_err: BaseException | None = None
    for i, ctx in enumerate(contexts):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                if i == len(contexts) - 1:
                    print(f"Warning: fetched {url} with SSL verify disabled", file=sys.stderr)
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            last_err = exc
            if "CERTIFICATE_VERIFY_FAILED" not in str(exc) and "certificate verify failed" not in str(exc).lower():
                raise
            continue
    if last_err:
        raise last_err
    raise RuntimeError(f"Could not fetch {url}")


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


ADANOS_US_TV: dict[str, str] = {
    "NYSE": "NYSE",
    "NASDAQ": "NASDAQ",
    "AMEX": "AMEX",
}


def _load_us_candidates_from_adanos() -> list[Candidate]:
    """US common stocks from adanos listings.csv (NYSE + NASDAQ; AMEX if present)."""
    out: list[Candidate] = []
    try:
        raw = _fetch_text(ADANOS_LISTINGS_URL, timeout=90)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        print(f"Warning: could not load US listings CSV: {exc}", file=sys.stderr)
        return out

    reader = csv.DictReader(io.StringIO(raw))
    for row in reader:
        exchange = (row.get("exchange") or "").strip().upper()
        if exchange not in ADANOS_US_TV:
            continue
        if (row.get("country_code") or "").strip().upper() != "US":
            continue
        if (row.get("asset_type") or "").strip() != "Stock":
            continue
        if (row.get("etf_category") or "").strip():
            continue
        sym = (row.get("ticker") or "").strip().upper()
        name = (row.get("name") or "").strip()
        if not sym or _symbol_non_common(sym):
            continue
        if name_suggests_non_common(name):
            continue
        tv_ex = ADANOS_US_TV[exchange]
        tv_part = f"{tv_ex}:{sym}"
        yahoo = tv_part_to_yahoo(tv_part)
        if not yahoo:
            continue
        out.append(Candidate(tv_part=tv_part, yahoo=yahoo, name_hint=name, region="US"))
    print(f"  US (adanos listings.csv): {len(out)} Layer-1 candidates", file=sys.stderr)
    return out


def _load_us_candidates_resilient() -> list[Candidate]:
    """NasdaqTrader directories when reachable; otherwise adanos CSV fallback."""
    try:
        out = _load_us_candidates()
        if out:
            print(f"  US (NasdaqTrader): {len(out)} Layer-1 candidates")
            return out
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        print(f"Warning: NasdaqTrader US dirs failed ({exc})", file=sys.stderr)
    return _load_us_candidates_from_adanos()


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


def _load_tsx_only_candidates() -> list[Candidate]:
    """TSX main board only (.TO) — no TSXV."""
    out = [c for c in _load_ca_candidates_from_adanos() if c.tv_part.startswith("TSX:")]
    if out:
        print(f"  Canada TSX only (adanos listings.csv): {len(out)} Layer-1 candidates")
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
    seen: set[str] = set()
    deduped: list[Candidate] = []
    for c in tsx:
        if c.yahoo in seen:
            continue
        seen.add(c.yahoo)
        deduped.append(c)
    return deduped


def load_layer1_candidates(
    *,
    us_only: bool = False,
    ca_only: bool = False,
    tsx_only: bool = True,
) -> list[Candidate]:
    """
    Layer-1 universe from exchange directories (no Yahoo .info).
    tsx_only=True: Canada = TSX main board only (default for full OHLC build).
    """
    candidates: list[Candidate] = []
    if not ca_only:
        print("Loading US symbol directories...")
        candidates.extend(_load_us_candidates_resilient())
    if not us_only:
        print("Loading Canadian symbol directories...")
        if tsx_only:
            candidates.extend(_load_tsx_only_candidates())
        else:
            candidates.extend(_load_ca_candidates())

    by_yahoo: dict[str, Candidate] = {}
    for c in candidates:
        by_yahoo.setdefault(c.yahoo, c)
    result = list(by_yahoo.values())
    result.sort(key=lambda c: (c.region, c.yahoo))
    return result
