"""
Plotly chart preview for Sequence Vova scan results.
Uses resampled OHLC (same timeframe as the scan) and full indicator overlays.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

from sequence_vova import run_sequence_vova_full

CACHE_TRIM_BARS = 150
DEFAULT_MAX_BARS = 120

BG = "#0d0d0d"
PAPER = "#0d0d0d"
GRID = "#1e1e1e"
UP = "#00e676"
DOWN = "#ff1744"
CRIT_UP = "#00e676"
CRIT_DOWN = "#ff5252"
PEAK_COLOR = "#448aff"
TROUGH_COLOR = "#ffab00"
TP_COLOR = "#00e676"
SL_COLOR = "#ff5252"
FIB_COLOR = "#78909c"


def _to_list(arr) -> list:
    if arr is None:
        return []
    if isinstance(arr, np.ndarray):
        return [None if (isinstance(x, float) and np.isnan(x)) else float(x) for x in arr]
    return list(arr)


def _serialize_full(full: dict) -> dict:
    """Convert numpy arrays in full result to JSON/session-safe lists."""
    out = {}
    for k, v in full.items():
        if k in ("critical_level_series", "seq_state_series"):
            out[k] = _to_list(v)
        elif k == "peaks" or k == "troughs":
            out[k] = list(v)
        elif k == "fib" and v is not None:
            out[k] = dict(v)
        elif isinstance(v, (np.floating, float)):
            out[k] = None if (v is not None and np.isnan(v)) else float(v)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        elif isinstance(v, (np.bool_, bool)):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


def build_chart_payload(
    df: pd.DataFrame,
    tf: str,
    *,
    atr_len: int = 14,
    min_rr: float = 1.5,
    use_last_hl_sl: bool = True,
    risk_dollars: float = 100,
    trim_bars: int = CACHE_TRIM_BARS,
) -> dict | None:
    """Run full indicator on scan-time df; cache trimmed OHLC + serialized overlays."""
    full = run_sequence_vova_full(
        df,
        atr_len=atr_len,
        min_rr=min_rr,
        use_last_hl_sl=use_last_hl_sl,
        risk_dollars=risk_dollars,
    )
    if full is None:
        return None
    trim = min(len(df), trim_bars)
    offset = len(df) - trim
    return {
        "df": df.iloc[offset:].copy(),
        "full": _serialize_full(full),
        "tf": tf,
        "bar_offset": offset,
    }


def _chart_dates(index: pd.Index) -> list[str]:
    return [pd.Timestamp(x).strftime("%Y-%m-%d") for x in index]


def build_sequence_vova_figure(
    df: pd.DataFrame,
    full: dict,
    *,
    title: str = "",
    tf: str = "Daily",
    max_bars: int = DEFAULT_MAX_BARS,
    bar_offset: int = 0,
) -> go.Figure:
    """Candlestick chart with Sequence Vova overlays (scan timeframe bars)."""
    plot_df = df.iloc[-max_bars:].copy() if len(df) > max_bars else df.copy()
    extra_offset = max(0, len(df) - len(plot_df))
    display_offset = bar_offset + extra_offset
    dates = _chart_dates(plot_df.index)
    n = len(dates)

    crit = np.array(full.get("critical_level_series") or [], dtype=float)
    state = np.array(full.get("seq_state_series") or [], dtype=int)
    if len(crit) > display_offset:
        crit = crit[display_offset : display_offset + n]
        state = state[display_offset : display_offset + n]
    else:
        crit = np.full(n, np.nan)
        state = np.zeros(n, dtype=int)
    if len(crit) < n:
        crit = np.pad(crit, (0, n - len(crit)), constant_values=np.nan)
        state = np.pad(state, (0, n - len(state)), constant_values=0)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="OHLC",
            increasing_line_color=UP,
            increasing_fillcolor=UP,
            decreasing_line_color=DOWN,
            decreasing_fillcolor=DOWN,
        )
    )

    crit_up = np.where(state == 1, crit, np.nan)
    crit_dn = np.where(state == -1, crit, np.nan)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=crit_up,
            mode="lines",
            name="Critical (up)",
            line=dict(color=CRIT_UP, width=1.5),
            connectgaps=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=crit_dn,
            mode="lines",
            name="Critical (down)",
            line=dict(color=CRIT_DOWN, width=1.5),
            connectgaps=False,
        )
    )

    peak_x, peak_y, peak_txt = [], [], []
    for p in full.get("peaks") or []:
        idx = int(p["idx"])
        if idx < display_offset or idx - display_offset >= n:
            continue
        xi = idx - display_offset
        peak_x.append(dates[xi])
        peak_y.append(float(p["price"]))
        peak_txt.append(str(p.get("label", "")))
    if peak_x:
        fig.add_trace(
            go.Scatter(
                x=peak_x,
                y=peak_y,
                mode="markers+text",
                name="Peaks",
                text=peak_txt,
                textposition="top center",
                marker=dict(color=PEAK_COLOR, size=8, symbol="triangle-down"),
                textfont=dict(size=9, color=PEAK_COLOR),
            )
        )

    trough_x, trough_y, trough_txt = [], [], []
    for t in full.get("troughs") or []:
        idx = int(t["idx"])
        if idx < display_offset or idx - display_offset >= n:
            continue
        xi = idx - display_offset
        trough_x.append(dates[xi])
        trough_y.append(float(t["price"]))
        trough_txt.append(str(t.get("label", "")))
    if trough_x:
        fig.add_trace(
            go.Scatter(
                x=trough_x,
                y=trough_y,
                mode="markers+text",
                name="Troughs",
                text=trough_txt,
                textposition="bottom center",
                marker=dict(color=TROUGH_COLOR, size=8, symbol="triangle-up"),
                textfont=dict(size=9, color=TROUGH_COLOR),
            )
        )

    tp = full.get("TP")
    sl = full.get("SL")
    if tp is not None and not (isinstance(tp, float) and np.isnan(tp)):
        fig.add_hline(
            y=float(tp),
            line_dash="dash",
            line_color=TP_COLOR,
            annotation_text="TP",
            annotation_position="right",
        )
    if sl is not None and not (isinstance(sl, float) and np.isnan(sl)):
        fig.add_hline(
            y=float(sl),
            line_dash="dash",
            line_color=SL_COLOR,
            annotation_text="SL",
            annotation_position="right",
        )

    fib = full.get("fib")
    if fib:
        for key, label in (
            ("fib_382", "Fib 38.2%"),
            ("fib_500", "Fib 50%"),
            ("fib_618", "Fib 61.8%"),
        ):
            val = fib.get(key)
            if val is not None:
                fig.add_hline(
                    y=float(val),
                    line_dash="dot",
                    line_color=FIB_COLOR,
                    annotation_text=label,
                    annotation_position="left",
                )

    chart_title = title or f"Sequence Vova - {tf}"
    fig.update_layout(
        title=dict(text=chart_title, font=dict(color="#e0e0e0", size=14)),
        template="plotly_dark",
        paper_bgcolor=PAPER,
        plot_bgcolor=BG,
        xaxis=dict(
            gridcolor=GRID,
            rangeslider=dict(visible=False),
            type="category",
        ),
        yaxis=dict(gridcolor=GRID, side="right"),
        margin=dict(l=8, r=48, t=40, b=24),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=9),
        ),
        height=420,
    )
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    return fig


def figure_from_payload(payload: dict, *, symbol: str = "") -> go.Figure | None:
    """Build figure from cached chart_payload."""
    if not payload:
        return None
    df = payload.get("df")
    full = payload.get("full")
    if df is None or full is None:
        return None
    if isinstance(df, dict):
        df = pd.DataFrame(df)
    tf = str(payload.get("tf", "Daily"))
    title = f"{symbol} - {tf}" if symbol else f"Sequence Vova - {tf}"
    return build_sequence_vova_figure(
        df,
        full,
        title=title,
        tf=tf,
        bar_offset=int(payload.get("bar_offset", 0)),
    )


def figure_to_plotly_json(fig: go.Figure) -> dict[str, Any]:
    """Serialize figure for Plotly.js in mobile HTML component."""
    return json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))


def chart_json_for_mobile(chart_cache: dict, table_rows: list[dict]) -> dict[str, dict]:
    """Pre-build plotly JSON per tv_symbol for hover popup."""
    out: dict[str, dict] = {}
    for row in table_rows:
        key = str(row.get("tv_symbol", "") or "")
        if not key or key in out:
            continue
        payload = chart_cache.get(key)
        if not payload:
            continue
        fig = figure_from_payload(payload, symbol=key)
        if fig is not None:
            out[key] = figure_to_plotly_json(fig)
    return out
