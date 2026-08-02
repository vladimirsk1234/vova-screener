"""
Does the app's CLOSED list say what the Streamlit SELL TO CLOSE table says, on today's market?

Two things have to be true and they fail for different reasons, so they are checked separately.

  logic  — given the same bars, `evaluateClose` and `run_sequence_vova_close_scan` must name the
           same symbols and the same numbers. A break in this is an engine bug.
  bars   — the series the app fetches from the Yahoo chart API must be the series Streamlit gets
           from `yf.download`. A break in this is a data-plumbing bug, and it looks exactly like
           an engine bug from the outside: the same code reads a different last bar and the close
           list comes out empty or full of symbols Streamlit never mentions.

Feed it the file `packages/engine/scripts/export_live_close_scan.ts` writes.

    npx tsx packages/engine/scripts/export_live_close_scan.ts --tf Daily --limit 400
    PYTHONPATH=. python3 scripts/check_live_close_parity.py reports/live_close_daily.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

from sequence_vova import run_sequence_vova_close_scan

TOL = 0.01
# Reading every symbol back from Yahoo a second time is the slow half, and one bad series is
# enough to explain a wrong close list, so the bar check samples rather than sweeps.
BAR_SAMPLE = 40


def frame(bars: list[dict]) -> pd.DataFrame:
    idx = pd.to_datetime([b["date"] for b in bars])
    return pd.DataFrame(
        {
            "Open": [b["open"] for b in bars],
            "High": [b["high"] for b in bars],
            "Low": [b["low"] for b in bars],
            "Close": [b["close"] for b in bars],
            "Volume": [b["volume"] for b in bars],
        },
        index=idx,
    )


def same(a, b, tol: float = TOL) -> bool:
    a_empty = a is None or (isinstance(a, float) and math.isnan(a))
    b_empty = b is None or (isinstance(b, float) and math.isnan(b))
    if a_empty or b_empty:
        return a_empty and b_empty
    return abs(float(a) - float(b)) <= tol


def check_logic(payload: dict) -> int:
    """Same bars into both engines: the close lists must be identical, symbol and number."""
    atr_len = int(payload.get("atr_len", 14))
    params = payload["params"]
    kwargs = dict(
        atr_len=atr_len,
        min_rr=float(params["minRr"]),
        use_last_hl_sl=bool(params["useLastHlSl"]),
        risk_dollars=float(params["riskPerTrade"]),
        no_rr_req=bool(params["noRrReq"]),
    )

    ts_closes: dict[str, dict] = {}
    py_closes: dict[str, dict] = {}
    failures = 0

    for ticker, entry in payload["symbols"].items():
        bars = entry["bars"]
        if len(bars) < 50:
            continue
        out = run_sequence_vova_close_scan(frame(bars), **kwargs)
        if out and bool(out["Valid"]):
            py_closes[ticker] = out
        if entry.get("close"):
            ts_closes[ticker] = entry["close"]

    only_ts = sorted(set(ts_closes) - set(py_closes))
    only_py = sorted(set(py_closes) - set(ts_closes))
    for ticker in only_ts:
        print(f"FAIL {ticker}: the app closes it, Streamlit does not")
        failures += 1
    for ticker in only_py:
        print(f"FAIL {ticker}: Streamlit closes it, the app does not")
        failures += 1

    for ticker in sorted(set(ts_closes) & set(py_closes)):
        got, exp = ts_closes[ticker], py_closes[ticker]
        for field, key in (
            ("entry", "entry_price"),
            ("exit", "exit_price"),
            ("rrAtEntry", "entry_rr"),
            ("rrAtClose", "close_rr"),
            ("pnlUsd", "pnl_dollars"),
            ("pnlPct", "pnl_pct"),
        ):
            if not same(got.get(field), exp.get(key)):
                print(f"FAIL {ticker}.{field}: app={got.get(field)} streamlit={exp.get(key)}")
                failures += 1

    print(
        f"logic: {len(payload['symbols'])} symbols, "
        f"{len(py_closes)} closes in Streamlit, {len(ts_closes)} in the app, {failures} mismatch(es)"
    )
    return failures


def check_bars(payload: dict) -> int:
    """The app's series against the one Streamlit downloads, for a sample of the same symbols."""
    try:
        import yfinance as yf

        from headless_scanner import _fill_last_bar_ohlc, _prepare_scan_ohlc
    except Exception as exc:  # noqa: BLE001
        print(f"bars: skipped ({type(exc).__name__}: {exc})")
        return 0

    tf = payload["tf"]
    inter = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}[tf]
    period = {"Daily": "2y", "Weekly": "5y", "Monthly": "10y"}[tf]
    tickers = sorted(payload["symbols"])[:BAR_SAMPLE]
    failures = 0
    compared = 0

    for ticker in tickers:
        bars = payload["symbols"][ticker]["bars"]
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=inter,
                progress=False,
                auto_adjust=False,
                multi_level_index=False,
            )
        except Exception:  # noqa: BLE001
            continue
        if df is None or df.empty:
            continue
        df, _ = _prepare_scan_ohlc(df, tf, inter=inter)
        if df is None or df.empty:
            continue
        df = _fill_last_bar_ohlc(df).dropna(subset=["Close", "High", "Low", "Open"])
        compared += 1

        last_app = bars[-1]
        last_py = df.index[-1].strftime("%Y-%m-%d")
        if last_app["date"] != last_py:
            print(f"FAIL {ticker}: app ends {last_app['date']}, Streamlit ends {last_py}")
            failures += 1
            continue
        for field, col in (("close", "Close"), ("high", "High"), ("low", "Low")):
            got, exp = float(last_app[field]), float(df[col].iloc[-1])
            if abs(got - exp) > max(TOL, abs(exp) * 0.001):
                print(f"FAIL {ticker}.{field}: app={got} streamlit={exp}")
                failures += 1

    print(f"bars: {compared} series compared, {failures} mismatch(es)")
    return failures


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/live_close_daily.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"{path} — {payload['tf']}, scanned {payload['scannedAt']}")
    failures = check_logic(payload) + check_bars(payload)
    if failures:
        sys.exit(1)
    print("Live close parity OK")


if __name__ == "__main__":
    main()
