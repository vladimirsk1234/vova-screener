"""
OHLCV helpers shared by the scanner and the preview chart.
Pure pandas, no Streamlit or Yahoo I/O.
"""
from __future__ import annotations

import pandas as pd


def interval_and_period(tf: str) -> tuple[str, str]:
    """Always fetch daily; Weekly/Monthly resampled from daily so current period is included."""
    return "1d", ("10y" if tf != "Daily" else "2y")


def resample_to_timeframe(df: pd.DataFrame | None, tf: str) -> pd.DataFrame | None:
    """Resample daily OHLCV to Weekly or Monthly. Returns df unchanged for Daily."""
    if tf == "Daily" or df is None or df.empty:
        return df
    req = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in req):
        return df
    rule = "W-FRI" if tf == "Weekly" else "ME"
    res = df[req].resample(rule).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return res.dropna(subset=req)


def fill_last_bar_ohlc(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Fill last bar NaNs in OHLC from previous bar so the reference bar is never dropped."""
    if df is None or len(df) < 2:
        return df
    ohlc = ["Open", "High", "Low", "Close"]
    if not all(c in df.columns for c in ohlc):
        return df
    last_idx = df.index[-1]
    last_row = df.loc[last_idx, ohlc]
    if last_row.isna().any():
        prev = df.iloc[-2][ohlc]
        for col in ohlc:
            if pd.isna(df.at[last_idx, col]) and not pd.isna(prev[col]):
                df.at[last_idx, col] = prev[col]
        if pd.isna(df.at[last_idx, "Close"]) and not pd.isna(prev["Close"]):
            df.at[last_idx, "Close"] = prev["Close"]
        for col in ["Open", "High", "Low"]:
            if pd.isna(df.at[last_idx, col]):
                df.at[last_idx, col] = df.at[last_idx, "Close"]
    return df
