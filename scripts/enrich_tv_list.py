#!/usr/bin/env python3
"""One-time: enrich TV-LIST with company names from Yahoo (uses disk cache when available)."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ticker_data import (
    TV_LIST_BIG_CAP,
    build_name_cache,
    read_list_file,
    write_list_file,
)


def main() -> int:
    tickers, tv_map, existing_names, err = read_list_file(TV_LIST_BIG_CAP)
    if err:
        print(err, file=sys.stderr)
        return 1
    if not tickers:
        print("No tickers in list file.", file=sys.stderr)
        return 1

    print(f"Resolving names for {len(tickers)} tickers...")
    names = build_name_cache(tickers, rate_limit_per_sec=12.0, max_workers=8)
    names.update({k: v for k, v in existing_names.items() if v.strip()})

    entries: list[tuple[str, str, str]] = []
    for t in tickers:
        tv_sym = tv_map.get(t, t)
        entries.append((tv_sym, t, names.get(t, t)))

    write_list_file(TV_LIST_BIG_CAP, entries)
    named = sum(
        1 for t in tickers
        if (names.get(t, t) or "").strip().upper() != t.upper()
    )
    print(f"Wrote {len(entries)} lines to {TV_LIST_BIG_CAP} ({named} with company names).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
