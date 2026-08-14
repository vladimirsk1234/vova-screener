#!/usr/bin/env python3
"""
Layer-1 US + Canada common-stock universe from FMP company-screener.

Maps FMP symbols onto TradingView + Yahoo tickers used by STOCK-TICKERS.txt.
"""
from __future__ import annotations

import re
import sys
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from fundamentals_fmp import _fmp_get, _str  # noqa: E402
from layer1_universe import Candidate, _symbol_non_common, dedupe_dual_listed  # noqa: E402
from ticker_data import tv_part_to_yahoo  # noqa: E402

# FMP exchangeShortName / exchange → (TV prefix, region)
US_EXCHANGES: dict[str, str] = {
    "NASDAQ": "NASDAQ",
    "NASDAQGS": "NASDAQ",
    "NASDAQGM": "NASDAQ",
    "NASDAQCM": "NASDAQ",
    "NMS": "NASDAQ",
    "NGM": "NASDAQ",
    "NCM": "NASDAQ",
    "NYSE": "NYSE",
    "NYQ": "NYSE",
    "NYS": "NYSE",
    "AMEX": "AMEX",
    "ASE": "AMEX",
    "NYSEAMERICAN": "AMEX",
    "NYSEAMER": "AMEX",
    "NYSEMKT": "AMEX",
}

CA_EXCHANGES: dict[str, tuple[str, str]] = {
    "TSX": ("TSX", ".TO"),
    "TOR": ("TSX", ".TO"),
    "TSXV": ("TSXV", ".V"),
    "CVE": ("TSXV", ".V"),
    "VAN": ("TSXV", ".V"),
    "NEO": ("NEO", ".NE"),
    "NEOE": ("NEO", ".NE"),
    "CSE": ("CSE", ".CN"),
    "CNQ": ("CSE", ".CN"),
}

SCREENER_EXCHANGES = (
    "NASDAQ",
    "NYSE",
    "AMEX",
    "TSX",
    "TSXV",
    "CNQ",
    "NEO",
    "CSE",
)

_CA_SUFFIXES = (".TO", ".V", ".NE", ".CN")
_OTC_MARKERS = ("OTC", "PINK", "GREY", "GRAY", "PNK", "OTCQX", "OTCQB", "OOTC")
_SPACE_RE = re.compile(r"\s+")
# Trailing spaces so " unit " does not match UnitedHealth / United Airlines.
_NAME_REJECT = (
    " preferred ",
    " warrant ",
    " warrants ",
    " etf ",
    " etn ",
    " fund ",
    " units ",
    " unit trust ",
    " trust unit ",
    " debenture ",
    " rights ",
    " depositary ",
    " depository ",
    " american depositary ",
    " ads ",
    " adr ",
)


def _name_looks_non_common(name: str) -> bool:
    if not name:
        return False
    n = f" {name.lower().replace(',', ' ').replace('.', ' ').replace('-', ' ')} "
    n = _SPACE_RE.sub(" ", n)
    return any(m in n for m in _NAME_REJECT)


def _truthy_false(v: object) -> bool:
    if v is False or v == 0:
        return True
    s = str(v or "").strip().lower()
    return s in ("false", "0", "no")


def _truthy_true(v: object) -> bool:
    if v is True or v == 1:
        return True
    s = str(v or "").strip().lower()
    return s in ("true", "1", "yes")


def _looks_otc(exchange: str) -> bool:
    n = (exchange or "").upper()
    return any(m in n for m in _OTC_MARKERS)


def _norm_exchange(row: dict) -> str:
    short = _str(row.get("exchangeShortName")) or ""
    full = _str(row.get("exchange")) or ""
    return (short or full).upper().replace(" ", "")


def _classify_exchange(row: dict) -> tuple[str, str] | None:
    """Return (TV prefix, region) or None if not a major US/CA equity venue."""
    key = _norm_exchange(row)
    if key in US_EXCHANGES:
        return US_EXCHANGES[key], "US"
    if key in CA_EXCHANGES:
        return CA_EXCHANGES[key][0], "CA"
    blob = " ".join(
        x
        for x in (
            _str(row.get("exchangeShortName")),
            _str(row.get("exchange")),
        )
        if x
    ).upper()
    if _looks_otc(blob):
        return None
    if "NASDAQ" in blob:
        return "NASDAQ", "US"
    if "AMEX" in blob or "NYSE AMERICAN" in blob:
        return "AMEX", "US"
    if "NYSE" in blob:
        return "NYSE", "US"
    if "VENTURE" in blob or "TSXV" in blob:
        return "TSXV", "CA"
    if "NEO" in blob:
        return "NEO", "CA"
    if "CSE" in blob or "CANADIAN SECURITIES" in blob:
        return "CSE", "CA"
    if "TSX" in blob or "TORONTO" in blob:
        return "TSX", "CA"
    return None


def _strip_ca_suffix(sym: str) -> str:
    s = (sym or "").strip().upper()
    for suf in _CA_SUFFIXES:
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


def _tv_symbol_part(raw: str) -> str:
    """FMP BRK.B / ACO.X → TV/Yahoo class style BRK-B / ACO-X."""
    s = _strip_ca_suffix(raw).replace("/", "-")
    return s.replace(".", "-")


def row_to_candidate(row: dict) -> Candidate | None:
    if not isinstance(row, dict):
        return None
    if _truthy_true(row.get("isEtf")) or _truthy_true(row.get("isFund")):
        return None
    if _truthy_false(row.get("isActivelyTrading")):
        return None

    fmp_sym = (_str(row.get("symbol")) or "").upper()
    if not fmp_sym:
        return None

    ex_raw = _str(row.get("exchangeShortName")) or _str(row.get("exchange")) or ""
    if _looks_otc(ex_raw) or _looks_otc(_norm_exchange(row)):
        return None

    classified = _classify_exchange(row)
    name = _str(row.get("companyName")) or _str(row.get("name")) or ""
    if _symbol_non_common(fmp_sym) or _symbol_non_common(_tv_symbol_part(fmp_sym)):
        return None
    if _name_looks_non_common(name):
        return None
    if classified is None:
        return None
    tv_ex, region = classified
    tv_sym = _tv_symbol_part(fmp_sym)

    if not tv_sym or _symbol_non_common(tv_sym):
        return None

    tv_part = f"{tv_ex}:{tv_sym}"
    yahoo = tv_part_to_yahoo(tv_part)
    if not yahoo:
        return None
    return Candidate(tv_part=tv_part, yahoo=yahoo, name_hint=name.strip(), region=region)


def fetch_screener_rows(
    api_key: str,
    *,
    exchanges: tuple[str, ...] = SCREENER_EXCHANGES,
    limit: int = 10000,
    rate_per_sec: float = 12.0,
) -> list[dict]:
    """Pull company-screener pages per exchange. Dedupes by FMP symbol+exchange."""
    _ = rate_per_sec  # HTTP throttle lives in _fmp_get (Premium 750/min).
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for exchange in exchanges:
        page = 0
        while page <= 20:
            params = {
                "exchange": exchange,
                "isEtf": "false",
                "isActivelyTrading": "true",
                "limit": str(limit),
            }
            if page > 0:
                params["page"] = str(page)
            try:
                raw = _fmp_get("/company-screener", api_key, params)
            except urllib.error.HTTPError as exc:
                print(f"  screener {exchange} page={page} HTTP {exc.code}", file=sys.stderr)
                break
            except Exception as exc:
                print(f"  screener {exchange} page={page} error: {exc}", file=sys.stderr)
                break
            rows = raw if isinstance(raw, list) else []
            if not rows:
                break
            new_n = 0
            for row in rows:
                if not isinstance(row, dict):
                    continue
                sym = (_str(row.get("symbol")) or "").upper()
                ex = (_str(row.get("exchangeShortName")) or exchange).upper()
                key = (sym, ex)
                if not sym or key in seen:
                    continue
                seen.add(key)
                out.append(row)
                new_n += 1
            print(f"  screener {exchange} page={page} rows={len(rows)} new={new_n}")
            if len(rows) < limit or new_n == 0:
                break
            page += 1

    print(f"FMP screener unique rows: {len(out)}")
    return out


def load_fmp_candidates(
    api_key: str,
    *,
    us_only: bool = False,
    ca_only: bool = False,
    limit: int = 0,
) -> tuple[list[Candidate], list[tuple[Candidate, Candidate]]]:
    exchanges = SCREENER_EXCHANGES
    if us_only:
        exchanges = tuple(e for e in exchanges if e in ("NASDAQ", "NYSE", "AMEX"))
    elif ca_only:
        exchanges = tuple(e for e in exchanges if e not in ("NASDAQ", "NYSE", "AMEX"))

    rows = fetch_screener_rows(api_key, exchanges=exchanges)
    candidates: list[Candidate] = []
    seen_yahoo: set[str] = set()
    skipped = 0
    for row in rows:
        c = row_to_candidate(row)
        if c is None:
            skipped += 1
            continue
        y = c.yahoo.upper()
        if y in seen_yahoo:
            skipped += 1
            continue
        seen_yahoo.add(y)
        candidates.append(c)

    print(f"Mapped common stocks: {len(candidates)} (skipped={skipped})")
    kept, dropped_pairs = dedupe_dual_listed(candidates)
    us_n = sum(1 for c in kept if c.region == "US")
    ca_n = sum(1 for c in kept if c.region == "CA")
    print(
        f"After dual-list dedup: {len(kept)} (US={us_n}, CA={ca_n}, dropped_ca={len(dropped_pairs)})"
    )
    kept.sort(key=lambda c: (c.region, c.yahoo))
    if limit > 0:
        kept = kept[:limit]
        print(f"Limited to first {limit} candidates")
    return kept, dropped_pairs


def candidates_to_entries(candidates: list[Candidate]) -> list[tuple[str, str, str]]:
    return [(c.tv_part, c.yahoo, c.name_hint) for c in candidates]
