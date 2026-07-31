"""
Export a reject-reason parity fixture from the Python oracle.

Pins the regression where the TS port answered `RR_TOO_LOW` for a symbol whose real
blocker was a down sequence (YMM Monthly): the reason ordering of
`sequence_vova.explain_invalid_buy` is what the app must reproduce.

Bars are baked into the fixture so the TS check stays offline and deterministic.

Run: python scripts/export_reject_reason_fixture.py [YAHOO_TICKER]
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_utils import fill_last_bar_ohlc, interval_and_period
from sequence_vova import explain_invalid_buy, run_sequence_vova_full, run_sequence_vova_pine

OUT = ROOT / "packages" / "engine" / "fixtures" / "reject_reasons_parity.json"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
ATR_LEN = 14

# (label, close override or None, min_rr, leading bars to drop) — sequence gate, RR gate,
# and the history-window sensitivity that made a 5y fetch answer RR_TOO_LOW.
CASES = [
    ("as-is close below critical level", None, 1.5, 0),
    ("close above critical level, RR under a high bar", 9.60, 2.5, 0),
    ("close above critical level, RR clears the bar", 9.60, 1.5, 0),
    ("history truncated by one bar moves the confirmed trough", None, 1.5, 1),
]


def fetch_bars(ticker: str, tf: str) -> pd.DataFrame:
    interval, period = interval_and_period(tf)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval={interval}&range={period}&includePrePost=false&events=div%2Csplit"
    )
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as res:
        payload = json.load(res)
    result = payload["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    index = [datetime.fromtimestamp(t, tz=timezone.utc).date() for t in result["timestamp"]]
    df = pd.DataFrame(
        {
            "Open": quote["open"],
            "High": quote["high"],
            "Low": quote["low"],
            "Close": quote["close"],
            "Volume": [v if v is not None else 0 for v in quote["volume"]],
        },
        index=pd.DatetimeIndex(index),
    )
    df = fill_last_bar_ohlc(df)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return collapse_in_progress_period(df, tf)


def collapse_in_progress_period(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Yahoo appends a mid-period stamp on top of the native Weekly/Monthly candle."""
    if tf == "Daily" or len(df) < 2:
        return df
    freq = "W" if tf == "Weekly" else "M"
    keys = df.index.to_period(freq)
    rows: list[dict] = []
    index: list[pd.Timestamp] = []
    for key, group in df.groupby(keys, sort=False):
        rows.append(
            {
                "Open": float(group["Open"].iloc[0]),
                "High": float(group["High"].max()),
                "Low": float(group["Low"].min()),
                "Close": float(group["Close"].iloc[-1]),
                "Volume": float(group["Volume"].sum()),
            }
        )
        index.append(group.index[0])
        del key
    return pd.DataFrame(rows, index=pd.DatetimeIndex(index))


def bars_payload(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "date": ts.strftime("%Y-%m-%d"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
        }
        for ts, row in df.iterrows()
    ]


def num(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if value != value else value


def expectation(df: pd.DataFrame, min_rr: float) -> dict:
    pine = run_sequence_vova_pine(
        df, atr_len=ATR_LEN, min_rr=min_rr, use_last_hl_sl=True, risk_dollars=100, direction="buy"
    )
    full = run_sequence_vova_full(
        df, atr_len=ATR_LEN, min_rr=min_rr, use_last_hl_sl=True, risk_dollars=100
    )
    return {
        "reason": explain_invalid_buy(full, min_rr=min_rr, no_rr_req=False),
        "valid": bool(pine["Valid"]),
        "seq_state": int(full.get("seq_state_final", 0) or 0),
        "critical_level": num(full.get("critical_level")),
        "rr": num(pine["RR"]),
        "sl": num(pine["SL"]),
        "tp": num(pine["TP"]),
    }


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else "YMM"
    tf = "Monthly"
    df = fetch_bars(ticker, tf)
    cases = []
    for label, close_override, min_rr, drop_leading in CASES:
        frame = df.iloc[drop_leading:].copy()
        if close_override is not None:
            last = frame.index[-1]
            frame.loc[last, "Close"] = close_override
            frame.loc[last, "High"] = max(float(frame.at[last, "High"]), close_override)
        cases.append(
            {
                "label": label,
                "closeOverride": close_override,
                "min_rr": min_rr,
                "dropLeadingBars": drop_leading,
                "expect": expectation(frame, min_rr),
            }
        )

    payload = {
        "ticker": ticker,
        "tf": tf,
        "opts": {"atr_len": ATR_LEN, "use_last_hl_sl": True, "risk_dollars": 100},
        "bars": bars_payload(df),
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} ({len(payload['bars'])} bars)")
    for case in cases:
        print(f"  {case['label']}: {case['expect']['reason']} (RR={case['expect']['rr']})")


if __name__ == "__main__":
    main()
