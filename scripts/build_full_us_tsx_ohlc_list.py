#!/usr/bin/env python3
"""
Build STOCK-TICKERS.txt from full US (NYSE/NASDAQ/AMEX) + TSX/TSXV universe.

Layer 1: exchange symbol directories (no Yahoo .info, no EPS filter).
Layer 2: Yahoo OHLC validation for Daily / Weekly / Monthly (>=50 bars).

Resumable via .cache/full_us_tsx_ohlc_build.json

Usage:
  python scripts/build_full_us_tsx_ohlc_list.py --limit 30
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

from layer1_universe import load_layer1_candidates
from ohlc_yahoo import validate_ohlc_candidates
from ticker_data import TV_LIST_STOCK_TICKERS, write_list_file

CACHE_PATH = ROOT / ".cache" / "full_us_tsx_ohlc_build.json"


def _candidates_to_entries(candidates) -> list[tuple[str, str, str]]:
    return [(c.tv_part, c.yahoo, c.name_hint) for c in candidates]


def build_list(
    *,
    us_only: bool = False,
    ca_only: bool = False,
    tsx_only: bool = False,
    limit: int = 0,
    resume: bool = True,
    retry_no_data: bool = False,
) -> tuple[list[tuple[str, str, str]], Counter]:
    t0 = time.perf_counter()
    # Default includes TSX + TSXV so a full rebuild does not drop venture names
    # already present in STOCK-TICKERS.txt.
    layer1 = load_layer1_candidates(us_only=us_only, ca_only=ca_only, tsx_only=tsx_only)
    us_n = sum(1 for c in layer1 if c.region == "US")
    ca_n = sum(1 for c in layer1 if c.region == "CA")
    ca_label = "TSX" if tsx_only else "TSX+TSXV"
    print(f"Layer-1 candidates: {len(layer1)} (US={us_n}, {ca_label}={ca_n})")

    entries_in = _candidates_to_entries(layer1)
    if limit > 0:
        entries_in = entries_in[:limit]
        print(f"Limited to first {limit} candidates")

    entries, rejects = validate_ohlc_candidates(
        entries_in,
        cache_path=CACHE_PATH,
        resume=resume,
        retry_no_data=retry_no_data,
    )
    elapsed = time.perf_counter() - t0
    print(f"Elapsed: {elapsed / 60:.1f} min")
    return entries, rejects


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full US + TSX universe with OHLC D/W/M -> STOCK-TICKERS.txt",
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
    parser.add_argument("--no-resume", action="store_true", help="Ignore OHLC cache")
    parser.add_argument(
        "--retry-no-data",
        action="store_true",
        help="Re-check NO_DAILY / RATE_LIMIT from cache",
    )
    args = parser.parse_args()

    if args.us_only and args.ca_only:
        print("Cannot use --us-only and --ca-only together", file=sys.stderr)
        return 1

    entries, rejects = build_list(
        us_only=args.us_only,
        ca_only=args.ca_only,
        tsx_only=args.tsx_only,
        limit=args.limit,
        resume=not args.no_resume,
        retry_no_data=args.retry_no_data,
    )
    write_list_file(TV_LIST_STOCK_TICKERS, entries)

    print()
    print("=== DONE ===")
    print(f"Wrote {len(entries)} lines -> {TV_LIST_STOCK_TICKERS}")
    if rejects:
        print("Rejects:")
        for reason, n in rejects.most_common():
            print(f"  {reason}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
