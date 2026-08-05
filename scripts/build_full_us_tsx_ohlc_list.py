#!/usr/bin/env python3
"""
Build STOCK-TICKERS.txt from full US (NYSE/NASDAQ/AMEX) + TSX/TSXV universe.

Layer 1: exchange symbol directories (no OTC; common stock)
Dual-list: drop CA when same company also lists in US (prefer US)
Layer 2: Yahoo OHLC validation for Daily / Weekly / Monthly (>=50 bars)
Layer 3: Yahoo trailingEps > 0 + EQUITY + not OTC

Resumable via .cache/full_us_tsx_ohlc_build.json and .cache/full_us_tsx_eps.json

Usage:
  python scripts/build_full_us_tsx_ohlc_list.py --limit 30 --dry-run
  python scripts/build_full_us_tsx_ohlc_list.py
  python scripts/build_full_us_tsx_ohlc_list.py --resume
  python scripts/build_full_us_tsx_ohlc_list.py --retry-no-data
  python scripts/build_full_us_tsx_ohlc_list.py --us-only
  python scripts/build_full_us_tsx_ohlc_list.py --ca-only
  python scripts/build_full_us_tsx_ohlc_list.py --tsx-only
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from fundamentals_yahoo import filter_positive_eps
from layer1_universe import Candidate, dedupe_dual_listed, load_layer1_candidates
from ohlc_yahoo import validate_ohlc_candidates
from ticker_data import TV_LIST_STOCK_TICKERS, read_list_file, write_list_file

OHLC_CACHE_PATH = ROOT / ".cache" / "full_us_tsx_ohlc_build.json"
EPS_CACHE_PATH = ROOT / ".cache" / "full_us_tsx_eps.json"


def _candidates_to_entries(candidates) -> list[tuple[str, str, str]]:
    return [(c.tv_part, c.yahoo, c.name_hint) for c in candidates]


def _existing_file_candidates() -> list[Candidate]:
    """Seed from current STOCK-TICKERS.txt so a Layer-1 gap (e.g. NasdaqTrader 403) does not drop names."""
    tickers, tv_map, name_map, err = read_list_file(TV_LIST_STOCK_TICKERS)
    if err:
        print(f"Warning: could not read existing list: {err}", file=sys.stderr)
        return []
    out: list[Candidate] = []
    for yahoo in tickers:
        tv = tv_map.get(yahoo) or yahoo
        name = name_map.get(yahoo) or ""
        y = yahoo.upper()
        region = "CA" if y.endswith((".TO", ".V", ".NE", ".CN")) else "US"
        out.append(Candidate(tv_part=tv, yahoo=yahoo, name_hint=name, region=region))
    return out


def _merge_candidates(primary: list[Candidate], extra: list[Candidate]) -> list[Candidate]:
    """Union by yahoo; primary wins on conflicts."""
    by_yahoo: dict[str, Candidate] = {c.yahoo: c for c in extra}
    for c in primary:
        by_yahoo[c.yahoo] = c
    result = list(by_yahoo.values())
    result.sort(key=lambda c: (c.region, c.yahoo))
    return result


def build_list(
    *,
    us_only: bool = False,
    ca_only: bool = False,
    tsx_only: bool = False,
    limit: int = 0,
    resume: bool = True,
    retry_no_data: bool = False,
    skip_eps: bool = False,
) -> tuple[list[tuple[str, str, str]], Counter, dict]:
    t0 = time.perf_counter()
    # Default includes TSX + TSXV so a full rebuild does not drop venture names
    # already present in STOCK-TICKERS.txt.
    layer1 = load_layer1_candidates(us_only=us_only, ca_only=ca_only, tsx_only=tsx_only)
    us_n = sum(1 for c in layer1 if c.region == "US")
    ca_n = sum(1 for c in layer1 if c.region == "CA")
    ca_label = "TSX" if tsx_only else "TSX+TSXV"
    print(f"Layer-1 candidates: {len(layer1)} (US={us_n}, {ca_label}={ca_n})")

    existing = _existing_file_candidates()
    if existing:
        before = len(layer1)
        layer1 = _merge_candidates(layer1, existing)
        print(
            f"Merged existing {TV_LIST_STOCK_TICKERS}: "
            f"{len(existing)} file + {before} layer1 -> {len(layer1)} unique"
        )

    layer1, dual_pairs = dedupe_dual_listed(layer1)
    us_after = sum(1 for c in layer1 if c.region == "US")
    ca_after = sum(1 for c in layer1 if c.region == "CA")
    print(
        f"After dual-list dedup: {len(layer1)} "
        f"(US={us_after}, {ca_label}={ca_after}, dropped_ca={len(dual_pairs)})"
    )

    entries_in = _candidates_to_entries(layer1)
    if limit > 0:
        entries_in = entries_in[:limit]
        print(f"Limited to first {limit} candidates")

    ohlc_entries, ohlc_rejects = validate_ohlc_candidates(
        entries_in,
        cache_path=OHLC_CACHE_PATH,
        resume=resume,
        retry_no_data=retry_no_data,
    )
    print(f"OHLC passed: {len(ohlc_entries)}")

    eps_rejects: Counter = Counter()
    if skip_eps:
        entries = ohlc_entries
        print("Skipping EPS filter (--skip-eps)")
    else:
        entries, eps_rejects = filter_positive_eps(
            ohlc_entries,
            cache_path=EPS_CACHE_PATH,
            resume=resume,
            retry_errors=retry_no_data,
        )
        print(f"Positive EPS passed: {len(entries)}")

    rejects = Counter(ohlc_rejects)
    rejects.update(eps_rejects)

    elapsed = time.perf_counter() - t0
    print(f"Elapsed: {elapsed / 60:.1f} min")

    stats = {
        "layer1_count": us_n + ca_n,
        "layer1_us": us_n,
        "layer1_ca": ca_n,
        "existing_seed": len(existing),
        "after_dual": len(layer1) if limit <= 0 else len(entries_in),
        "dual_dropped_ca": len(dual_pairs),
        "ohlc_passed": len(ohlc_entries),
        "eps_passed": len(entries),
        "ohlc_rejects": dict(ohlc_rejects),
        "eps_rejects": dict(eps_rejects),
        "elapsed_sec": round(elapsed, 1),
    }
    return entries, rejects, stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full US + TSX universe with OHLC + EPS > 0 -> STOCK-TICKERS.txt",
    )
    parser.add_argument("--limit", type=int, default=0, help="Smoke: first N tickers")
    parser.add_argument("--us-only", action="store_true", help="US exchanges only")
    parser.add_argument("--ca-only", action="store_true", help="Canada only (TSX+TSXV unless --tsx-only)")
    parser.add_argument(
        "--tsx-only",
        action="store_true",
        help="Canada = TSX main board only (exclude TSXV)",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true", help="Ignore OHLC/EPS caches")
    parser.add_argument(
        "--retry-no-data",
        action="store_true",
        help="Re-check OHLC NO_DAILY/RATE_LIMIT and EPS info errors",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Alias for --retry-no-data (OHLC + EPS retryable errors)",
    )
    parser.add_argument(
        "--skip-eps",
        action="store_true",
        help="Skip trailingEps > 0 filter (OHLC only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report only; do not write STOCK-TICKERS.txt",
    )
    args = parser.parse_args()

    if args.us_only and args.ca_only:
        print("Cannot use --us-only and --ca-only together", file=sys.stderr)
        return 1

    retry = args.retry_no_data or args.retry_errors
    entries, rejects, stats = build_list(
        us_only=args.us_only,
        ca_only=args.ca_only,
        tsx_only=args.tsx_only,
        limit=args.limit,
        resume=not args.no_resume,
        retry_no_data=retry,
        skip_eps=args.skip_eps,
    )

    print()
    print("=== DONE ===")
    print(
        f"Layer1 {stats['layer1_us']}+{stats['layer1_ca']} -> "
        f"dual_drop {stats['dual_dropped_ca']} -> "
        f"OHLC {stats['ohlc_passed']} -> "
        f"EPS {stats['eps_passed']}"
    )
    if stats.get("ohlc_rejects"):
        print("OHLC rejects:")
        for reason, n in sorted(stats["ohlc_rejects"].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {n}")
    if stats.get("eps_rejects"):
        print("EPS rejects:")
        for reason, n in sorted(stats["eps_rejects"].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {n}")

    if args.dry_run:
        print(f"Dry-run: would write {len(entries)} lines -> {TV_LIST_STOCK_TICKERS}")
        return 0

    write_list_file(TV_LIST_STOCK_TICKERS, entries)
    print(f"Wrote {len(entries)} lines -> {TV_LIST_STOCK_TICKERS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
