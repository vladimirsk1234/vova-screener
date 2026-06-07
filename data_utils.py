"""
OHLCV helpers shared by the scanner and the preview chart.
Pure pandas, no Streamlit or Yahoo I/O.
"""
from __future__ import annotations

import pandas as pd


def interval_and_period(tf: str, *, scanner_id: str | None = None) -> tuple[str, str]:
    """Always fetch daily; Weekly/Monthly resampled from daily so current period is included."""
    if scanner_id == "fast_graphs":
        # 15y is enough for 10Y CAGR + charts; "max" OOMs Streamlit Cloud on large lists.
        return "1d", "15y"
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


def extract_ohlcv(all_data: pd.DataFrame | None, ticker: str, required_cols: list[str]) -> pd.DataFrame | None:
    """Extract one symbol from yf.download batch (MultiIndex or single-ticker frame)."""
    if all_data is None or (hasattr(all_data, "empty") and all_data.empty):
        return None
    if isinstance(all_data.columns, pd.MultiIndex):
        level0 = all_data.columns.get_level_values(0).unique()
        key = ticker
        if key not in level0 and ":" in ticker:
            key = ticker.split(":")[-1]
        if key not in level0:
            return None
        if not all((key, col) in all_data.columns for col in required_cols):
            return None
        return pd.DataFrame({
            "Open": all_data[(key, "Open")],
            "High": all_data[(key, "High")],
            "Low": all_data[(key, "Low")],
            "Close": all_data[(key, "Close")],
            "Volume": all_data[(key, "Volume")],
        })
    if len(all_data.columns) == 0:
        return None
    if not all(col in all_data.columns for col in required_cols):
        return None
    return all_data[required_cols].copy()


def split_batch_ohlcv(
    all_data: pd.DataFrame | None,
    batch: list[str],
    required_cols: list[str],
) -> dict[str, pd.DataFrame]:
    """One pass per batch: same semantics as per-ticker extract_ohlcv."""
    if all_data is None or (hasattr(all_data, "empty") and all_data.empty):
        return {}
    out: dict[str, pd.DataFrame] = {}
    for t in batch:
        df = extract_ohlcv(all_data, t, required_cols)
        if df is not None and not df.empty:
            out[t] = df
    return out


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
