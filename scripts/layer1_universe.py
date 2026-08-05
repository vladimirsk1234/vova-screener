#!/usr/bin/env python3
"""
Layer-1 US + Canada symbol universe from exchange directories.

Used by scripts/build_full_us_tsx_ohlc_list.py and gap_scan_us_ca.py.
Also provides dual-list dedup (prefer US when listed on both markets).
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


# Corporate suffixes stripped before dual-list name matching (prefer US listing).
_CORP_SUFFIX_RE = re.compile(
    r"\b(INCORPORATED|CORPORATION|COMPANY|LIMITED|INC|CORP|LTD|CO|PLC|SA|AG|NV|LLC|LP|LLP)\b\.?",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[^\w\s]")
_CA_YAHOO_SUFFIXES = (".TO", ".V", ".NE", ".CN")


def normalize_company_name(name: str) -> str:
    """Normalize company name for US/CA dual-list matching."""
    n = str(name or "").upper().strip()
    if not n:
        return ""
    n = _CORP_SUFFIX_RE.sub("", n)
    n = _PUNCT_RE.sub(" ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _ca_yahoo_base(yahoo: str) -> str | None:
    y = str(yahoo or "").strip().upper()
    for suf in _CA_YAHOO_SUFFIXES:
        if y.endswith(suf):
            return y[: -len(suf)]
    return None


def _names_similar(a: str, b: str) -> bool:
    """True when two company names likely refer to the same issuer."""
    na = normalize_company_name(a)
    nb = normalize_company_name(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if len(na) >= 4 and len(nb) >= 4 and (na in nb or nb in na):
        return True
    ta = [t for t in na.split() if len(t) >= 4]
    tb = [t for t in nb.split() if len(t) >= 4]
    if ta and tb and ta[0] == tb[0]:
        return True
    return False


def dedupe_dual_listed(
    candidates: list[Candidate],
) -> tuple[list[Candidate], list[tuple[Candidate, Candidate]]]:
    """
    Drop Canadian listings when the same company also trades in the US.

    Matching:
      1) Same Yahoo base (SHOP.TO / SHOP.V vs SHOP) AND similar company names
      2) Same normalized company name across US + CA

    Base-only matches without name similarity are ignored (BBD.TO != NYSE:BBD).

    Policy: prefer US. Returns (kept, dropped_pairs) where each pair is (us, dropped_ca).
    """
    us = [c for c in candidates if c.region == "US"]
    ca = [c for c in candidates if c.region == "CA"]
    other = [c for c in candidates if c.region not in ("US", "CA")]

    us_by_yahoo = {c.yahoo.upper(): c for c in us}

    dropped_yahoo: set[str] = set()
    dropped_pairs: list[tuple[Candidate, Candidate]] = []

    # Pass 1: CA base matches a US ticker AND names look like the same company.
    for c in ca:
        base = _ca_yahoo_base(c.yahoo)
        if not base or base not in us_by_yahoo:
            continue
        us_c = us_by_yahoo[base]
        if not _names_similar(c.name_hint, us_c.name_hint):
            continue
        dropped_yahoo.add(c.yahoo)
        dropped_pairs.append((us_c, c))

    # Pass 2: same normalized name, US + CA both present.
    name_to_us: dict[str, Candidate] = {}
    for c in us:
        key = normalize_company_name(c.name_hint)
        if key and key not in name_to_us:
            name_to_us[key] = c

    for c in ca:
        if c.yahoo in dropped_yahoo:
            continue
        key = normalize_company_name(c.name_hint)
        if not key or key not in name_to_us:
            continue
        # Require a reasonably specific name (avoid empty / very short collisions).
        if len(key) < 4:
            continue
        us_c = name_to_us[key]
        dropped_yahoo.add(c.yahoo)
        dropped_pairs.append((us_c, c))

    kept_ca = [c for c in ca if c.yahoo not in dropped_yahoo]
    kept = us + kept_ca + other
    kept.sort(key=lambda c: (c.region, c.yahoo))

    if dropped_pairs:
        print(
            f"Dual-list dedup: dropped {len(dropped_pairs)} CA listings (prefer US)",
            file=sys.stderr,
        )
        for us_c, ca_c in dropped_pairs[:20]:
            print(
                f"  drop {ca_c.tv_part} ({ca_c.yahoo}) keep {us_c.tv_part} ({us_c.yahoo}) "
                f"| {ca_c.name_hint or us_c.name_hint}",
                file=sys.stderr,
            )
        if len(dropped_pairs) > 20:
            print(f"  ... and {len(dropped_pairs) - 20} more", file=sys.stderr)

    return kept, dropped_pairs


def _fetch_text(url: str, timeout: int = 60) -> str:
    # NasdaqTrader blocks bare/bot UAs with 403; use a normal browser UA.
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/plain,*/*",
        },
    )
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
    # Hard rejects even when the name also mentions "common shares" (e.g. ADS).
    hard_reject = (
        "preferred",
        "warrant",
        " unit",
        "units",
        " etf",
        " fund",
        "debenture",
        " rights",
        " right,",
        "depositary",
        "depository",
        " american depositary",
    )
    if any(m in n for m in hard_reject):
        return False
    if any(m in n for m in COMMON_STOCK_MARKERS):
        return True
    # NYSE/Nasdaq rows without explicit "Common Stock" but also no red flags
    bad = ("note", "bond", "trust unit", "lp ", " l.p.", "reit", "ordinary share")
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
