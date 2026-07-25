#!/usr/bin/env python3
"""
Startup smoke: import modules in Streamlit Cloud order, then Ford Weekly parity.

Does not import headless_scanner.py (that calls st.set_page_config at import time).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    # Cloud-ish import order: chart_preview pulls sequence_vova + watermark_status first.
    from chart_preview import resolve_chart_payload  # noqa: F401
    from watermark_status import build_dwm_lines, build_trade_line  # noqa: F401
    import sequence_vova as sv
    from sequence_vova import (
        run_sequence_vova_pine,
        run_sequence_vova_close_scan,
        run_sequence_vova_full,
    )
    from data_utils import fill_last_bar_ohlc, interval_and_period, prepare_scan_ohlc

    for name in (
        "run_sequence_vova_pine",
        "run_sequence_vova_close_scan",
        "run_sequence_vova_full",
    ):
        if not callable(globals().get(name) or getattr(sv, name, None)):
            print(f"FAIL: missing {name}")
            return 1

    # Boot must not require explain_invalid_buy; helper may still exist.
    explain = getattr(sv, "explain_invalid_buy", None)
    print(f"OK: imports (explain_invalid_buy={'yes' if callable(explain) else 'fallback'})")

    inter, period = interval_and_period("Weekly")
    if inter != "1wk":
        print(f"FAIL: Weekly interval expected 1wk, got {inter!r}")
        return 1

    import yfinance as yf

    raw = yf.download(
        "F",
        period=period,
        interval=inter,
        progress=False,
        auto_adjust=False,
        multi_level_index=False,
    )
    if raw is None or raw.empty:
        print("FAIL: no Yahoo weekly data for F")
        return 1

    cols = ["Open", "High", "Low", "Close", "Volume"]
    df, daily = prepare_scan_ohlc(raw[cols].copy(), "Weekly", inter=inter)
    if daily is not None:
        print("FAIL: native 1wk must not invent daily companion")
        return 1
    df = fill_last_bar_ohlc(df)
    df = df.dropna(subset=cols[:4])
    out = run_sequence_vova_pine(
        df, atr_len=14, min_rr=1.5, use_last_hl_sl=True, risk_dollars=100, direction="buy"
    )
    if out is None:
        print("FAIL: pine returned None")
        return 1

    print(
        f"F Weekly Close={float(out['Close']):.2f} RR={float(out['RR']):.2f} "
        f"Valid={bool(out['Valid'])} New={bool(out['New'])}"
    )
    # Parity with direct native bars (same frame) — must agree.
    direct = run_sequence_vova_pine(
        fill_last_bar_ohlc(raw[cols].copy()),
        atr_len=14,
        min_rr=1.5,
        use_last_hl_sl=True,
        risk_dollars=100,
        direction="buy",
    )
    if bool(out["Valid"]) != bool(direct["Valid"]) or bool(out["New"]) != bool(direct["New"]):
        print("FAIL: prepare_scan_ohlc path disagrees with native 1wk")
        return 1

    print("OK: smoke_app_import")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
