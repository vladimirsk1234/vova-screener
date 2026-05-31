#!/usr/bin/env python3
"""Verify _split_batch_ohlcv matches _extract_ohlcv."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yfinance as yf

from data_utils import extract_ohlcv, split_batch_ohlcv

REQUIRED = ["Open", "High", "Low", "Close", "Volume"]
BATCH = ["MSFT", "AAPL", "NVDA"]


def main() -> int:
    data = yf.download(
        BATCH,
        period="6mo",
        interval="1d",
        progress=False,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
    )
    split = split_batch_ohlcv(data, BATCH, REQUIRED)
    for t in BATCH:
        a = extract_ohlcv(data, t, REQUIRED)
        b = split.get(t)
        if a is None and b is None:
            print(f"OK: {t} both None")
            continue
        if a is None or b is None:
            print(f"FAIL: {t} mismatch None")
            return 1
        if not a["Close"].equals(b["Close"]):
            print(f"FAIL: {t} Close differs")
            return 1
        print(f"OK: {t} Close last={a['Close'].iloc[-1]}")
    print("OK: split matches extract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
