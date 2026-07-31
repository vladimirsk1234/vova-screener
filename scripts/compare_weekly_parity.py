#!/usr/bin/env python3
"""Compare yfinance vs Yahoo chart API weekly bars + pine Valid/New for Streamlit hits."""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_utils import fill_last_bar_ohlc, interval_and_period
from sequence_vova import run_sequence_vova_pine

TICKERS = ["AAL", "AYI", "BOOT", "CPA", "SNEX", "SWKS", "SPHR", "ULS", "MRE.TO"]
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def chart_api_bars(ticker: str, interval: str = "1wk", range_: str = "5y") -> list[dict]:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval={interval}&range={range_}&includePrePost=false&events=div%2Csplit"
    )
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as res:
        data = json.loads(res.read().decode("utf-8"))
    result = (data.get("chart") or {}).get("result") or [None]
    result = result[0]
    if not result or not result.get("timestamp"):
        return []
    quote = (result.get("indicators") or {}).get("quote") or [{}]
    quote = quote[0]
    ts = result["timestamp"]
    bars = []
    for i, sec in enumerate(ts):
        from datetime import datetime, timezone

        d = datetime.fromtimestamp(sec, tz=timezone.utc).strftime("%Y-%m-%d")
        o = quote.get("open", [None] * len(ts))[i]
        h = quote.get("high", [None] * len(ts))[i]
        l = quote.get("low", [None] * len(ts))[i]
        c = quote.get("close", [None] * len(ts))[i]
        v = quote.get("volume", [None] * len(ts))[i]
        if o is None or h is None or l is None or c is None:
            continue
        bars.append(
            {
                "date": d,
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(v or 0),
            }
        )
    # fill last bar like TS
    if len(bars) >= 2:
        last, prev = bars[-1], bars[-2]
        for k in ("close", "open", "high", "low"):
            if last[k] != last[k]:  # nan
                last[k] = prev["close"] if k == "close" else last.get("close", prev["close"])
    return bars


def yf_bars(ticker: str, period: str = "5y") -> list[dict]:
    df = yf.download(
        ticker,
        period=period,
        interval="1wk",
        progress=False,
        auto_adjust=False,
        multi_level_index=False,
    )
    if df is None or df.empty:
        return []
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[cols].copy()
    df = fill_last_bar_ohlc(df)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    out = []
    for ts, row in df.iterrows():
        t = ts.tz_localize(None) if getattr(ts, "tzinfo", None) or getattr(ts, "tz", None) else ts
        try:
            d = t.strftime("%Y-%m-%d")
        except Exception:
            d = str(t)[:10]
        out.append(
            {
                "date": d,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]),
            }
        )
    return out


def bars_to_df(bars: list[dict]):
    import pandas as pd

    if not bars:
        return None
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


def pine_summary(df) -> dict:
    if df is None or len(df) < 50:
        return {"ok": False, "n": 0 if df is None else len(df)}
    out = run_sequence_vova_pine(
        df, atr_len=14, min_rr=1.5, use_last_hl_sl=True, risk_dollars=100, direction="buy"
    )
    return {
        "ok": True,
        "n": len(df),
        "asOf": df.index[-1].strftime("%Y-%m-%d"),
        "close": float(df["Close"].iloc[-1]),
        "high": float(df["High"].iloc[-1]),
        "low": float(df["Low"].iloc[-1]),
        "Valid": bool(out["Valid"]),
        "New": bool(out["New"]),
        "RR": None if out["RR"] != out["RR"] else round(float(out["RR"]), 2),
    }


def main() -> int:
    inter, period = interval_and_period("Weekly")
    # Force Cloud-like 5y for apples-to-apples with Nest
    period = "5y"
    print(f"interval={inter} period={period}")
    print(
        f"{'TICKER':8} {'SRC':6} {'n':>4} {'asOf':12} {'Close':>8} {'Valid':5} {'New':3} {'RR':>6}"
    )
    payload = {"tickers": {}, "opts": {"atr_len": 14, "min_rr": 1.5, "use_last_hl_sl": True, "risk_dollars": 100}}
    for t in TICKERS:
        ybars = yf_bars(t, period)
        cbars = chart_api_bars(t, inter, period)
        ydf = bars_to_df(ybars)
        cdf = bars_to_df(cbars)
        ys = pine_summary(ydf)
        cs = pine_summary(cdf)
        for src, s in (("yf", ys), ("chart", cs)):
            if not s.get("ok"):
                print(f"{t:8} {src:6} n={s.get('n')} FAIL")
                continue
            print(
                f"{t:8} {src:6} {s['n']:4} {s['asOf']:12} {s['close']:8.2f} "
                f"{str(s['Valid']):5} {str(s['New']):3} {s['RR']}"
            )
        # last-3 bar diff
        if ybars and cbars:
            for i in range(-3, 0):
                yb = ybars[i] if abs(i) <= len(ybars) else None
                cb = cbars[i] if abs(i) <= len(cbars) else None
                if not yb or not cb:
                    continue
                same = (
                    abs(yb["close"] - cb["close"]) < 1e-4
                    and abs(yb["high"] - cb["high"]) < 1e-4
                    and abs(yb["low"] - cb["low"]) < 1e-4
                )
                if not same or yb["date"] != cb["date"]:
                    print(
                        f"  DIFF[{i}] yf={yb['date']} C={yb['close']:.4f} H={yb['high']:.4f} L={yb['low']:.4f}"
                        f" | chart={cb['date']} C={cb['close']:.4f} H={cb['high']:.4f} L={cb['low']:.4f}"
                    )
        payload["tickers"][t] = {
            "yf": {"bars": ybars, "pine": ys},
            "chart": {"bars": cbars, "pine": cs},
        }

    out = ROOT / "packages" / "engine" / "fixtures" / "weekly_hits_parity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Keep fixture lean: only chart bars (Nest path) + pine expectations from Python on those bars
    lean = {
        "opts": payload["opts"],
        "tickers": {
            t: {
                "bars": payload["tickers"][t]["chart"]["bars"],
                "pine": payload["tickers"][t]["chart"]["pine"],
                "yf_pine": payload["tickers"][t]["yf"]["pine"],
            }
            for t in TICKERS
            if t in payload["tickers"]
        },
    }
    out.write_text(json.dumps(lean), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
