#!/usr/bin/env python3
"""
Phase 2 placeholder: generate TV-LIST-US-CANADA-FULL.txt for full US+Canada universe.

When the list file exists, register it in headless_scanner._build_source_registry():
    from ticker_data import TV_LIST_US_CANADA_FULL
    "US + CANADA FULL": FileListSource(TV_LIST_US_CANADA_FULL, read_list_file),

This script is intentionally minimal until an exchange directory source is chosen.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "TV-LIST-US-CANADA-FULL.txt"


def main() -> int:
    print(f"Output path: {OUT}")
    print("Not implemented yet. Use BIG + SMALL CAP source for ~1,800 tickers today.")
    print("To add symbols manually, create TV-LIST-US-CANADA-FULL.txt with lines:")
    print("  EXCHANGE:SYMBOL|Company Name")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
