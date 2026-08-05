#!/usr/bin/env python3
"""
Gap-scan: find US/CA tickers missing from STOCK-TICKERS.txt.

Same universe filters as the full build:
  Layer 1: NasdaqTrader / adanos directories (no OTC; common stock; NYSE/NASDAQ/AMEX + TSX/TSXV)
  Dual-list: drop CA when same company also lists in US (prefer US; vs file + candidates)
  Diff:    only symbols not already in STOCK-TICKERS.txt
  Layer 2: Yahoo OHLC Daily/Weekly/Monthly >= 50 bars
  Layer 3: Yahoo trailingEps > 0 + EQUITY + not OTC

Usage:
  python scripts/gap_scan_us_ca.py --limit 40
  python scripts/gap_scan_us_ca.py
  python scripts/gap_scan_us_ca.py --apply
  python scripts/gap_scan_us_ca.py --resume --retry-no-data
"""
from __future__ import annotations

import argparse
import json
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
from ticker_data import (
    TV_LIST_STOCK_TICKERS,
    read_list_file,
    write_list_file,
)

OHLC_CACHE_PATH = ROOT / ".cache" / "gap_scan_ohlc.json"
EPS_CACHE_PATH = ROOT / ".cache" / "gap_scan_eps.json"
REPORT_PATH = ROOT / ".cache" / "gap_scan_report.json"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _existing_yahoo_set() -> set[str]:
    tickers, _tv, _names, err = read_list_file(TV_LIST_STOCK_TICKERS)
    if err:
        raise RuntimeError(err)
    return set(tickers)


def _existing_entries() -> list[tuple[str, str, str]]:
    tickers, tv_map, name_map, err = read_list_file(TV_LIST_STOCK_TICKERS)
    if err:
        raise RuntimeError(err)
    out: list[tuple[str, str, str]] = []
    for yahoo in tickers:
        tv = tv_map.get(yahoo) or yahoo
        name = name_map.get(yahoo) or ""
        out.append((tv, yahoo, name))
    return out


def _existing_as_candidates() -> list[Candidate]:
    """Map current file entries to Candidates so dual-list sees US already in the list."""
    tickers, tv_map, name_map, err = read_list_file(TV_LIST_STOCK_TICKERS)
    if err:
        raise RuntimeError(err)
    out: list[Candidate] = []
    for yahoo in tickers:
        tv = tv_map.get(yahoo) or yahoo
        name = name_map.get(yahoo) or ""
        y = yahoo.upper()
        region = "CA" if y.endswith((".TO", ".V", ".NE", ".CN")) else "US"
        out.append(Candidate(tv_part=tv, yahoo=yahoo, name_hint=name, region=region))
    return out


def gap_scan(
    *,
    us_only: bool = False,
    ca_only: bool = False,
    limit: int = 0,
    resume: bool = True,
    retry_no_data: bool = False,
    skip_eps: bool = False,
) -> dict:
    t0 = time.perf_counter()
    existing = _existing_yahoo_set()
    print(f"Current {TV_LIST_STOCK_TICKERS}: {len(existing)} tickers")

    layer1 = load_layer1_candidates(us_only=us_only, ca_only=ca_only, tsx_only=False)
    us_n = sum(1 for c in layer1 if c.region == "US")
    ca_n = sum(1 for c in layer1 if c.region == "CA")
    print(f"Layer-1 candidates: {len(layer1)} (US={us_n}, TSX+TSXV={ca_n})")

    # Dual-list against existing file + Layer-1 so we do not add CA when US is present.
    combined = _existing_as_candidates() + layer1
    combined, dual_pairs = dedupe_dual_listed(combined)
    kept_yahoo = {c.yahoo for c in combined}
    layer1 = [c for c in layer1 if c.yahoo in kept_yahoo]
    print(f"Dual-list: dropped {len(dual_pairs)} CA (prefer US); Layer-1 kept {len(layer1)}")

    missing = [c for c in layer1 if c.yahoo not in existing]
    missing.sort(key=lambda c: (c.region, c.yahoo))
    print(f"Missing vs file: {len(missing)}")

    if limit > 0:
        missing = missing[:limit]
        print(f"Limited to first {limit} missing candidates")

    entries_in = [(c.tv_part, c.yahoo, c.name_hint) for c in missing]
    ohlc_pass, ohlc_rejects = validate_ohlc_candidates(
        entries_in,
        cache_path=OHLC_CACHE_PATH,
        resume=resume,
        retry_no_data=retry_no_data,
    )
    print(f"OHLC passed: {len(ohlc_pass)}")

    eps_rejects: Counter = Counter()
    if skip_eps:
        accepted = ohlc_pass
        print("Skipping EPS filter (--skip-eps)")
    else:
        accepted, eps_rejects = filter_positive_eps(
            ohlc_pass,
            cache_path=EPS_CACHE_PATH,
            resume=resume,
            retry_errors=retry_no_data,
        )
        print(f"Positive EPS passed: {len(accepted)}")

    elapsed = time.perf_counter() - t0
    report = {
        "existing_count": len(existing),
        "layer1_count": len(layer1),
        "layer1_us": us_n,
        "layer1_ca": ca_n,
        "dual_dropped_ca": len(dual_pairs),
        "missing_count": len(entries_in) if limit > 0 else len([c for c in layer1 if c.yahoo not in existing]),
        "ohlc_passed": len(ohlc_pass),
        "accepted_count": len(accepted),
        "ohlc_rejects": dict(ohlc_rejects),
        "eps_rejects": dict(eps_rejects),
        # Keep pe_rejects alias empty for older report consumers.
        "pe_rejects": {},
        "accepted": [
            {"tv": tv, "yahoo": yahoo, "name": name} for tv, yahoo, name in accepted
        ],
        "elapsed_sec": round(elapsed, 1),
    }
    _save_json(REPORT_PATH, report)
    print(f"Report -> {REPORT_PATH} ({elapsed / 60:.1f} min)")
    return report


def apply_accepted(accepted: list[tuple[str, str, str]]) -> int:
    if not accepted:
        print("Nothing to apply")
        return 0
    existing = _existing_entries()
    by_yahoo = {yahoo: (tv, yahoo, name) for tv, yahoo, name in existing}
    added = 0
    for tv, yahoo, name in accepted:
        if yahoo in by_yahoo:
            continue
        by_yahoo[yahoo] = (tv, yahoo, name)
        added += 1
    merged = sorted(by_yahoo.values(), key=lambda e: e[0])
    write_list_file(TV_LIST_STOCK_TICKERS, merged)
    print(f"Applied {added} new tickers -> {TV_LIST_STOCK_TICKERS} (total {len(merged)})")
    return added


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gap-scan missing US/CA tickers into STOCK-TICKERS.txt",
    )
    parser.add_argument("--limit", type=int, default=0, help="Smoke: first N missing")
    parser.add_argument("--us-only", action="store_true")
    parser.add_argument("--ca-only", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--retry-no-data",
        action="store_true",
        help="Re-check OHLC NO_DAILY/RATE_LIMIT and EPS info errors",
    )
    parser.add_argument(
        "--skip-eps",
        action="store_true",
        help="Skip trailingEps > 0 filter (OHLC only)",
    )
    parser.add_argument(
        "--skip-pe",
        action="store_true",
        help="Deprecated alias for --skip-eps",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Append accepted tickers to STOCK-TICKERS.txt",
    )
    parser.add_argument(
        "--apply-report",
        action="store_true",
        help="Apply accepted from existing .cache/gap_scan_report.json without re-scan",
    )
    args = parser.parse_args()

    if args.us_only and args.ca_only:
        print("Cannot use --us-only and --ca-only together", file=sys.stderr)
        return 1

    if args.apply_report:
        report = _load_json(REPORT_PATH)
        accepted = [
            (row["tv"], row["yahoo"], row.get("name") or "")
            for row in report.get("accepted") or []
        ]
        apply_accepted(accepted)
        return 0

    report = gap_scan(
        us_only=args.us_only,
        ca_only=args.ca_only,
        limit=args.limit,
        resume=not args.no_resume,
        retry_no_data=args.retry_no_data,
        skip_eps=args.skip_eps or args.skip_pe,
    )

    print()
    print("=== GAP SCAN DONE ===")
    print(f"Missing checked: {report['missing_count']}")
    print(f"OHLC passed: {report['ohlc_passed']}")
    print(f"Accepted (positive EPS): {report['accepted_count']}")
    if report.get("ohlc_rejects"):
        print("OHLC rejects:")
        for reason, n in sorted(report["ohlc_rejects"].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {n}")
    if report.get("eps_rejects"):
        print("EPS rejects:")
        for reason, n in sorted(report["eps_rejects"].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {n}")

    if args.apply:
        accepted = [
            (row["tv"], row["yahoo"], row.get("name") or "")
            for row in report.get("accepted") or []
        ]
        apply_accepted(accepted)
    else:
        print("Dry-run only. Re-run with --apply to append accepted tickers.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
