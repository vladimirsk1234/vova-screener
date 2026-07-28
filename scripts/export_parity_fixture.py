"""Export a small OHLC + pine-result fixture for mobile TS parity checks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sequence_vova import run_sequence_vova_close_scan, run_sequence_vova_pine

OUT = Path(__file__).resolve().parents[1] / "mobile" / "fixtures" / "parity_sample.json"


def _synthetic(n: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1.2, n))
    high = close + rng.uniform(0.2, 1.5, n)
    low = close - rng.uniform(0.2, 1.5, n)
    open_ = close + rng.normal(0, 0.3, n)
    vol = rng.integers(1_000_000, 5_000_000, n)
    idx = pd.date_range("2023-01-01", periods=n, freq="W-FRI")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def _bars(df: pd.DataFrame) -> list[dict]:
    out = []
    for ts, row in df.iterrows():
        out.append(
            {
                "date": ts.strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]),
            }
        )
    return out


def _clean(d: dict | None) -> dict | None:
    if d is None:
        return None
    out = {}
    for k, v in d.items():
        if isinstance(v, (float, np.floating)):
            out[k] = None if np.isnan(v) else float(v)
        elif isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def main() -> None:
    df = _synthetic()
    pine = run_sequence_vova_pine(df, atr_len=14, min_rr=1.5, use_last_hl_sl=True, risk_dollars=100)
    close = run_sequence_vova_close_scan(df, atr_len=14, min_rr=1.5, use_last_hl_sl=True, risk_dollars=100)
    payload = {
        "bars": _bars(df),
        "pine": _clean(pine),
        "close": _clean(close),
        "opts": {"atr_len": 14, "min_rr": 1.5, "use_last_hl_sl": True, "risk_dollars": 100},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
