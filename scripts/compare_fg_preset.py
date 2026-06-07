#!/usr/bin/env python3
"""Compare pass counts: FG Undervalued Quality vs CPFS-G Strict on a ticker sample."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yfinance as yf

from data_utils import resample_to_timeframe
from fast_graph_metrics import FastGraphFilterConfig, passes_fast_graph_filters
from fast_graph_scanner import run_fast_graph_scan
from ticker_data import read_list_file


def _load_tickers(path: Path, limit: int) -> list[str]:
    tickers, _, _, err = read_list_file(str(path))
    if err:
        print(f"Warning: {err}")
    all_tickers = tickers or []
    if limit <= 0:
        return all_tickers
    return all_tickers[:limit]


def _scan_preset(tickers: list[str], cfg: FastGraphFilterConfig) -> tuple[list[str], Counter]:
    passed: list[str] = []
    rejects: Counter = Counter()
    for ticker in tickers:
        try:
            daily = yf.Ticker(ticker).history(period="5y", auto_adjust=True)
            if daily is None or daily.empty:
                rejects["NO_DATA"] += 1
                continue
            weekly = resample_to_timeframe(daily, "Weekly")
            metrics = run_fast_graph_scan(
                weekly,
                ticker=ticker,
                df_daily=daily,
                filter_cfg=cfg,
            )
            if not metrics:
                rejects["NO_METRICS"] += 1
                continue
            ok, reason = passes_fast_graph_filters(metrics, cfg)
            if ok:
                passed.append(ticker)
            else:
                rejects[reason or "FILTER_FAIL"] += 1
        except Exception:
            rejects["ERROR"] += 1
    return passed, rejects


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare FAST Graph filter presets")
    parser.add_argument("--limit", type=int, default=0, help="Max tickers (0 = full list)")
    parser.add_argument(
        "--list",
        type=Path,
        default=ROOT / "TV-LIST-BIG_CAP_10B.txt",
        help="Ticker list file",
    )
    args = parser.parse_args()

    tickers = _load_tickers(args.list, args.limit)
    if not tickers:
        print(f"No tickers loaded from {args.list}")
        return 1

    fg_cfg = FastGraphFilterConfig.fg_undervalued_quality()
    cpfs_cfg = FastGraphFilterConfig.cpfs_strict()

    print(f"Scanning {len(tickers)} tickers from {args.list.name}\n")

    fg_pass, fg_reject = _scan_preset(tickers, fg_cfg)
    cpfs_pass, cpfs_reject = _scan_preset(tickers, cpfs_cfg)

    print(f"FG Undervalued Quality: {len(fg_pass)} passed")
    for reason, count in fg_reject.most_common(10):
        print(f"  {reason}: {count}")
    print(f"\nCPFS-G Strict: {len(cpfs_pass)} passed")
    for reason, count in cpfs_reject.most_common(10):
        print(f"  {reason}: {count}")

    fg_only = sorted(set(fg_pass) - set(cpfs_pass))
    if fg_only:
        print(f"\nFG-only passes ({len(fg_only)}): {', '.join(fg_only[:20])}")
        if len(fg_only) > 20:
            print(f"  ... +{len(fg_only) - 20} more")

    if len(fg_pass) < len(cpfs_pass):
        print("\nWarning: FG preset passed fewer names than CPFS strict (unexpected).")
    elif len(fg_pass) >= 5 and len(cpfs_pass) <= max(1, len(fg_pass) // 5):
        print(f"\nSanity OK: FG preset ({len(fg_pass)}) >> CPFS strict ({len(cpfs_pass)}).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
