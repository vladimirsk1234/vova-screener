#!/usr/bin/env python3
"""
Finviz S: UNDERVALUED parity — Option C:
  Pass A: validate 40 Finviz ground-truth tickers on Yahoo filters
  Pass B: scan universe list and compare overlap with Finviz 40
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finviz_metrics import (
    FINVIZ_S_UNDERVALUED_40,
    FinvizUndervaluedConfig,
    build_finviz_metrics,
    passes_finviz_filters,
)
from ticker_data import read_list_file

FINVIZ_40_FILE = ROOT / "data" / "finviz_s_undervalued_40.txt"
DEFAULT_LIST = ROOT / "TV-LIST-US-CANADA-FULL.txt"
REPORT_DIR = ROOT / "reports"
YAHOO_DELAY_SEC = 0.08

OUTSIDE_UNIVERSE_ADRS = frozenset({"INFY", "NTES", "HLN", "VIV", "XP", "BIRK", "DLO", "DRD", "GFI", "TFPM"})


def _load_finviz_tickers(path: Path) -> list[str]:
    if not path.is_file():
        return list(FINVIZ_S_UNDERVALUED_40)
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().split("#")[0].strip()
        if line:
            out.append(line.upper())
    return out


def _check_ticker(ticker: str, cfg: FinvizUndervaluedConfig) -> dict:
    try:
        metrics = build_finviz_metrics(ticker)
        ok, reason = passes_finviz_filters(metrics, cfg)
        return {
            "ticker": ticker,
            "passed": ok,
            "reject": reason,
            **metrics,
        }
    except Exception as exc:
        return {
            "ticker": ticker,
            "passed": False,
            "reject": f"ERROR:{type(exc).__name__}",
            "company_name": "",
            "country": "",
        }


def pass_a(finviz_tickers: list[str], cfg: FinvizUndervaluedConfig) -> list[dict]:
    print(f"\n=== Pass A: Finviz 40 validation ({len(finviz_tickers)} tickers) ===\n")
    rows: list[dict] = []
    for i, t in enumerate(finviz_tickers, 1):
        row = _check_ticker(t, cfg)
        rows.append(row)
        status = "PASS" if row["passed"] else f"FAIL ({row['reject']})"
        print(f"  {t:6} {status}")
        if i < len(finviz_tickers):
            time.sleep(YAHOO_DELAY_SEC)

    passed = sum(1 for r in rows if r["passed"])
    print(f"\nPass A recall: {passed}/{len(finviz_tickers)} ({100*passed/len(finviz_tickers):.0f}%)")
    fails = Counter(r["reject"] for r in rows if not r["passed"])
    if fails:
        print("Reject breakdown:")
        for reason, count in fails.most_common():
            print(f"  {reason}: {count}")
    return rows


def pass_b(
    list_path: Path,
    finviz_tickers: list[str],
    cfg: FinvizUndervaluedConfig,
    *,
    limit: int = 0,
) -> tuple[list[dict], set[str]]:
    print(f"\n=== Pass B: Universe scan ({list_path.name}) ===\n")
    tickers, _, _, err = read_list_file(str(list_path.name))
    if err:
        print(f"Warning: {err}")
    if not tickers:
        print("No tickers in list — run: python scripts/build_us_canada_list.py")
        return [], set()

    if limit > 0:
        tickers = tickers[:limit]

    finviz_set = set(finviz_tickers)
    yahoo_pass: set[str] = set()
    rows: list[dict] = []

    for i, t in enumerate(tickers, 1):
        row = _check_ticker(t, cfg)
        if row["passed"]:
            yahoo_pass.add(t.upper())
        rows.append(row)
        if i % 50 == 0:
            print(f"  scanned {i}/{len(tickers)} — Yahoo passes: {len(yahoo_pass)}")
        time.sleep(YAHOO_DELAY_SEC)

    overlap = finviz_set & yahoo_pass
    finviz_only = finviz_set - yahoo_pass
    yahoo_only = yahoo_pass - finviz_set
    outside = finviz_only & OUTSIDE_UNIVERSE_ADRS
    finviz_only_in_list = finviz_only - outside

    print(f"\nUniverse scanned: {len(tickers)}")
    print(f"Yahoo passes: {len(yahoo_pass)}")
    print(f"Finviz & Yahoo overlap: {len(overlap)} - {', '.join(sorted(overlap)[:20])}{'...' if len(overlap) > 20 else ''}")
    print(f"Finviz only (failed Yahoo): {len(finviz_only_in_list)}")
    if finviz_only_in_list:
        print(f"  {', '.join(sorted(finviz_only_in_list))}")
    print(f"Finviz outside US+CA list (ADRs): {len(outside)} - {', '.join(sorted(outside))}")
    print(f"Yahoo only (not in Finviz 40): {len(yahoo_only)}")

    return rows, yahoo_pass


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Finviz S: UNDERVALUED vs Yahoo parity")
    parser.add_argument("--pass-a-only", action="store_true", help="Only validate Finviz 40")
    parser.add_argument("--pass-b-only", action="store_true", help="Only scan universe list")
    parser.add_argument("--finviz-file", type=Path, default=FINVIZ_40_FILE)
    parser.add_argument("--list", type=Path, default=DEFAULT_LIST, help="Universe list for Pass B")
    parser.add_argument("--limit", type=int, default=0, help="Max tickers for Pass B (0=all)")
    parser.add_argument("--csv", type=Path, default=REPORT_DIR / "finviz_yahoo_parity.csv")
    args = parser.parse_args()

    finviz_tickers = _load_finviz_tickers(args.finviz_file)
    cfg = FinvizUndervaluedConfig.s_undervalued()

    all_rows: list[dict] = []

    if not args.pass_b_only:
        rows_a = pass_a(finviz_tickers, cfg)
        for r in rows_a:
            r["pass"] = "A"
        all_rows.extend(rows_a)

    if not args.pass_a_only:
        rows_b, _ = pass_b(args.list, finviz_tickers, cfg, limit=args.limit)
        for r in rows_b:
            r["pass"] = "B"
        all_rows.extend(rows_b)

    if all_rows and args.csv:
        _write_csv(args.csv, all_rows)
        print(f"\nWrote report: {args.csv}")

    passed_a = sum(1 for r in all_rows if r.get("pass") == "A" and r.get("passed"))
    if not args.pass_b_only and finviz_tickers:
        target = max(1, int(len(finviz_tickers) * 0.8))
        if passed_a >= target:
            print(f"\nSanity OK: Pass A recall {passed_a}/{len(finviz_tickers)} >= 80% target ({target})")
        else:
            print(f"\nWarning: Pass A recall {passed_a}/{len(finviz_tickers)} below 80% target ({target})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
