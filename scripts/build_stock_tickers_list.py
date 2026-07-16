#!/usr/bin/env python3
"""
Merge BIG CAP + SMALL CAP + US/CANADA FULL into STOCK-TICKERS.txt.

- Dedupe by Yahoo ticker (priority: BIG -> SMALL -> FULL)
- Optional OHLC validation for Daily / Weekly / Monthly (scanner-compatible)
- Yahoo-friendly pacing via scripts/ohlc_yahoo.py

Usage:
  python scripts/build_stock_tickers_list.py
  python scripts/build_stock_tickers_list.py --limit 50
  python scripts/build_stock_tickers_list.py --merge-only
  python scripts/build_stock_tickers_list.py --retry-no-data
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from ohlc_yahoo import validate_ohlc_candidates
from ticker_data import (
    TV_LIST_BIG_CAP,
    TV_LIST_SMALL_CAP,
    TV_LIST_STOCK_TICKERS,
    TV_LIST_US_CANADA_FULL,
    read_list_file,
    write_list_file,
)

CACHE_PATH = ROOT / ".cache" / "stock_tickers_ohlc_build.json"

SOURCE_FILES = (
    TV_LIST_BIG_CAP,
    TV_LIST_SMALL_CAP,
    TV_LIST_US_CANADA_FULL,
)


def _merge_sources() -> tuple[list[tuple[str, str, str]], dict[str, int]]:
    """Return (entries, source_counts). First wins BIG -> SMALL -> FULL."""
    seen: set[str] = set()
    merged: list[tuple[str, str, str]] = []
    source_counts: dict[str, int] = {}

    for filename in SOURCE_FILES:
        tickers, tv_map, name_map, err = read_list_file(filename)
        if err:
            raise SystemExit(err)
        source_counts[filename] = len(tickers)
        for yahoo in tickers:
            if yahoo in seen:
                continue
            seen.add(yahoo)
            tv_sym = tv_map.get(yahoo, yahoo)
            name = name_map.get(yahoo, "")
            merged.append((tv_sym, yahoo, name))

    return merged, source_counts


def build_list(
    *,
    limit: int = 0,
    validate_ohlc: bool = True,
    resume: bool = True,
    retry_no_data: bool = False,
) -> tuple[list[tuple[str, str, str]], Counter, dict[str, int]]:
    candidates, source_counts = _merge_sources()
    if limit > 0:
        candidates = candidates[:limit]

    print(f"Merged unique Yahoo tickers: {len(candidates)}")
    for fn, n in source_counts.items():
        print(f"  {fn}: {n} lines")

    if not validate_ohlc:
        entries = sorted(candidates, key=lambda e: e[0])
        return entries, Counter(), source_counts

    entries, rejects = validate_ohlc_candidates(
        candidates,
        cache_path=CACHE_PATH,
        resume=resume,
        retry_no_data=retry_no_data,
    )
    return entries, rejects, source_counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge BIG/SMALL/FULL -> STOCK-TICKERS.txt (optional OHLC validation)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Smoke: first N tickers")
    parser.add_argument("--merge-only", action="store_true", help="Skip OHLC validation")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true", help="Ignore OHLC cache")
    parser.add_argument(
        "--retry-no-data",
        action="store_true",
        help="Re-check NO_DAILY / RATE_LIMIT from cache",
    )
    args = parser.parse_args()

    entries, rejects, _ = build_list(
        limit=args.limit,
        validate_ohlc=not args.merge_only,
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
