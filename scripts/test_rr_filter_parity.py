#!/usr/bin/env python3
"""Regression: min-RR vs no-RR scan parity for BUY and SELL close-scan."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sequence_vova import (  # noqa: E402
    run_sequence_vova_close_scan,
    run_sequence_vova_pine,
)


def _ohlc_from_arrays(c, h, l, o=None) -> pd.DataFrame:
    n = len(c)
    if o is None:
        o = c
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": o, "High": h, "Low": l, "Close": c},
        index=idx,
    )


def test_buy_high_rr_no_rr_implies_min_rr():
    """Every no_rr BUY Valid with RR >= 1.5 must also be Valid under min_rr=1.5."""
    rng = np.random.default_rng(42)
    violations = []
    for seed in range(400):
        rng = np.random.default_rng(seed)
        n = 120
        close = 100 + np.cumsum(rng.normal(0.12, 1.0, n))
        high = close + rng.uniform(0.5, 2.2, n)
        low = close - rng.uniform(0.5, 2.2, n)
        df = _ohlc_from_arrays(close, high, low)
        a = run_sequence_vova_pine(df, min_rr=1.5, no_rr_req=False)
        b = run_sequence_vova_pine(df, min_rr=1.5, no_rr_req=True)
        if not (b and b["Valid"]):
            continue
        rr = b["RR"]
        if rr != rr or float(rr) < 1.5:
            continue
        if not (a and a["Valid"]):
            violations.append((seed, float(rr), a))
        elif b["New"] and not a["New"]:
            violations.append((seed, float(rr), "New mismatch", a))
    assert not violations, f"BUY parity violations: {violations[:5]}"


def test_buy_no_rr_requires_positive_risk_reward():
    """no_rr_req must not mark Valid when reward/risk are non-positive."""
    bad = []
    any_valid = False
    for seed in range(500):
        rng = np.random.default_rng(seed)
        n = 100
        close = 100 + np.cumsum(rng.normal(0.05, 1.1, n))
        high = close + rng.uniform(0.4, 2.0, n)
        low = close - rng.uniform(0.4, 2.0, n)
        df = _ohlc_from_arrays(close, high, low)
        out = run_sequence_vova_pine(df, min_rr=1.5, no_rr_req=True)
        if not (out and out["Valid"]):
            continue
        any_valid = True
        rr = float(out["RR"]) if out["RR"] == out["RR"] else float("nan")
        if not (rr == rr and rr > 0):
            bad.append((seed, rr))
        if not (float(out["TP"]) > float(out["Close"]) > float(out["SL"])):
            bad.append((seed, "risk/reward", out["TP"], out["Close"], out["SL"]))
    assert any_valid, "expected at least one Valid no_rr BUY in sample"
    assert not bad, f"Valid no_rr rows with non-positive risk/reward: {bad[:5]}"


def test_sell_min_rr_filters_entry_not_close_rr():
    """
    Crafted path: under no_rr_req a close can show close_rr >= 1.5 with entry_rr < 1.5,
    and that row must not appear when min_rr=1.5.
    """
    found = None
    for seed in range(2500):
        rng = np.random.default_rng(seed)
        n = 200
        close = 100 + np.cumsum(rng.normal(0.05, 1.2, n))
        high = close + rng.uniform(0.3, 3.0, n)
        low = close - rng.uniform(0.3, 3.0, n)
        df = _ohlc_from_arrays(close, high, low)
        no_rr = run_sequence_vova_close_scan(df, min_rr=1.5, no_rr_req=True)
        with_rr = run_sequence_vova_close_scan(df, min_rr=1.5, no_rr_req=False)
        if not (no_rr and no_rr["Valid"]):
            continue
        entry = float(no_rr["entry_rr"])
        close_rr = float(no_rr["close_rr"]) if no_rr["close_rr"] == no_rr["close_rr"] else float("nan")
        if entry < 1.5 and close_rr == close_rr and close_rr >= 1.5:
            found = (seed, entry, close_rr, bool(with_rr and with_rr["Valid"]))
            # Prefer cases where min_rr does not Valid; if it does, entry must be >= 1.5 path
            if not (with_rr and with_rr["Valid"]):
                break
    assert found is not None, "expected a fixture with entry_rr<1.5 and close_rr>=1.5 under no_rr"
    seed, entry, close_rr, min_valid = found
    assert not min_valid, (
        f"seed={seed}: entry_rr={entry:.4f} close_rr={close_rr:.4f} "
        f"must not be Valid under min_rr=1.5 (got Valid={min_valid})"
    )


def test_sell_high_entry_rr_under_no_rr_is_not_spurious():
    """
    Document: close_rr >= 1.5 alone does not imply survival under min_rr.
    When no_rr entry_rr >= 1.5, min_rr should also produce a Valid close on that series
    only if an entry with RR>=min was open — we assert the filter key is entry_rr.
    """
    saw_high_entry = False
    for seed in range(1500):
        rng = np.random.default_rng(seed)
        n = 180
        close = 100 + np.cumsum(rng.normal(0.08, 1.0, n))
        high = close + rng.uniform(0.4, 2.5, n)
        low = close - rng.uniform(0.4, 2.5, n)
        df = _ohlc_from_arrays(close, high, low)
        no_rr = run_sequence_vova_close_scan(df, min_rr=1.5, no_rr_req=True)
        with_rr = run_sequence_vova_close_scan(df, min_rr=1.5, no_rr_req=False)
        if not (no_rr and no_rr["Valid"]):
            continue
        entry = float(no_rr["entry_rr"])
        if entry < 1.5:
            # close_rr may still be high; min_rr may or may not Valid (different trade)
            continue
        saw_high_entry = True
        # Same open bar qualifies under min_rr when flat at that bar; if min_rr Valid,
        # its entry_rr must also be >= 1.5.
        if with_rr and with_rr["Valid"]:
            assert float(with_rr["entry_rr"]) >= 1.5
    assert saw_high_entry, "expected at least one no_rr close with entry_rr >= 1.5"


def main() -> int:
    test_buy_high_rr_no_rr_implies_min_rr()
    print("OK: BUY high-RR no_rr => min_rr parity")
    test_buy_no_rr_requires_positive_risk_reward()
    print("OK: BUY no_rr requires positive risk/reward when Valid")
    test_sell_min_rr_filters_entry_not_close_rr()
    print("OK: SELL min_rr filters entry_rr (close_rr alone insufficient)")
    test_sell_high_entry_rr_under_no_rr_is_not_spurious()
    print("OK: SELL entry_rr is the min_rr filter key")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
