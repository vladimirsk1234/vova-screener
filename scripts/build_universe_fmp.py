#!/usr/bin/env python3
"""
Build US+Canada STOCK-TICKERS.txt from FMP (common stock, TTM EPS > 0)
and write a PE/PEG undervaluation CSV.

Layer 1: FMP company-screener (NYSE/NASDAQ/AMEX + TSX/TSXV/NEO/CSE)
Dual-list: drop CA when the same company lists in the US
Layer 2: FMP TTM EPS > 0
Valuation: 5y EPS CAGR; PE < 15 if growth < 15%, else Lynch PEG < 1

Yahoo OHLC is not used. Requires FMP_API_KEY (Premium for CA + history).

Usage:
  python scripts/build_universe_fmp.py
  python scripts/build_universe_fmp.py --write
  python scripts/build_universe_fmp.py --limit 30 --write
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from fundamentals_fmp import (  # noqa: E402
    load_fmp_api_key,
    scan_eps_and_valuation,
)
from fmp_universe import candidates_to_entries, load_fmp_candidates  # noqa: E402
from ticker_data import TV_LIST_STOCK_TICKERS, write_list_file  # noqa: E402

CSV_PATH = ROOT / "reports" / "undervalued-pe-peg.csv"
CSV_FIELDS = (
    "yahoo",
    "tv",
    "name",
    "exchange",
    "epsTtm",
    "peTtm",
    "growth5y",
    "rule",
    "pegLynch",
    "pass",
)


def _fmt_num(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def write_valuation_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    "yahoo": row.get("yahoo") or "",
                    "tv": row.get("tv") or "",
                    "name": row.get("name") or "",
                    "exchange": row.get("exchange") or "",
                    "epsTtm": _fmt_num(row.get("epsTtm")),
                    "peTtm": _fmt_num(row.get("peTtm")),
                    "growth5y": _fmt_num(row.get("growth5y")),
                    "rule": row.get("rule") or "",
                    "pegLynch": _fmt_num(row.get("pegLynch")),
                    "pass": "true" if row.get("pass") else "false",
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FMP US+CA common stocks: EPS>0 list + PE/PEG undervalued CSV",
    )
    parser.add_argument("--write", action="store_true", help="Overwrite STOCK-TICKERS.txt")
    parser.add_argument("--limit", type=int, default=0, help="Smoke: first N mapped candidates")
    parser.add_argument("--us-only", action="store_true")
    parser.add_argument("--ca-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write reports/undervalued-pe-peg.csv",
    )
    parser.add_argument(
        "--csv",
        default=str(CSV_PATH),
        help="CSV output path (all EPS>0 rows; pass=true are undervalued)",
    )
    args = parser.parse_args()

    if args.us_only and args.ca_only:
        print("Cannot use --us-only and --ca-only together", file=sys.stderr)
        return 1

    api_key = load_fmp_api_key()
    if not api_key:
        print("FMP_API_KEY is not set.", file=sys.stderr)
        return 1

    candidates, dropped_pairs = load_fmp_candidates(
        api_key,
        us_only=args.us_only,
        ca_only=args.ca_only,
        limit=args.limit,
    )
    entries = candidates_to_entries(candidates)
    print(f"Universe candidates: {len(entries)} (dual-dropped CA={len(dropped_pairs)})")

    passed, rejects, rows = scan_eps_and_valuation(
        entries,
        api_key=api_key,
        resume=not args.no_resume,
        retry_errors=args.retry_errors,
    )
    undervalued = [r for r in rows if r.get("pass")]
    print()
    print(f"Mapped {len(entries)} → EPS>0 {len(passed)} → undervalued {len(undervalued)}")
    for reason, n in sorted(rejects.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {n}")

    if not args.no_csv:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = ROOT / csv_path
        write_valuation_csv(csv_path, rows)
        print(f"Wrote {len(rows)} valuation rows ({len(undervalued)} pass) -> {csv_path}")

    if not args.write:
        print(f"Dry-run: would write {len(passed)} lines -> {TV_LIST_STOCK_TICKERS}")
        return 0

    write_list_file(TV_LIST_STOCK_TICKERS, passed)
    print(f"Wrote {len(passed)} lines -> {TV_LIST_STOCK_TICKERS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
