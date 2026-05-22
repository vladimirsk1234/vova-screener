"""
Plotly chart builder mirroring the Sequence Vova Pine indicator:
candles (Elder Impulse coloring), EMA20/EMA40/SMA200, Bollinger Bands,
Elder Envelope, HH/HL labels + connecting lines, critical level step,
Fibonacci levels, TP/SL horizontal markers, and a volume subplot.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    macd_line = _ema(close, fast) - _ema(close, slow)
    sig_line = _ema(macd_line, sig)
    return macd_line - sig_line


def compute_chart_layers(
    df: pd.DataFrame,
    *,
    len_fast: int = 20,
    len_slow: int = 40,
    len_major: int = 200,
    bb_len: int = 20,
    bb_mult: float = 2.0,
    env_lookback: int = 100,
    env_mult: float = 2.0,
) -> dict:
    close = df["Close"]
    ema_fast = _ema(close, len_fast)
    ema_slow = _ema(close, len_slow)
    sma_major = _sma(close, len_major)
    macd_hist = _macd_hist(close)

    bb_basis = _sma(close, bb_len)
    std = close.rolling(bb_len, min_periods=bb_len).std(ddof=0)
    bb_upper = bb_basis + bb_mult * std
    bb_lower = bb_basis - bb_mult * std

    # Elder AutoEnvelope: sqrt(SMA(|close-emaSlow|^2, lookback)) * mult around emaSlow
    dev = (close - ema_slow).abs()
    mymov = np.sqrt((dev * dev).rolling(env_lookback, min_periods=env_lookback).mean())
    newmax = mymov.rolling(6, min_periods=1).max()
    env_upper = ema_slow + newmax * env_mult
    env_lower = ema_slow - newmax * env_mult

    return {
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "sma_major": sma_major,
        "bb_basis": bb_basis,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "env_upper": env_upper,
        "env_lower": env_lower,
        "macd_hist": macd_hist,
    }


def elder_bar_colors(
    df: pd.DataFrame,
    ema_fast: pd.Series,
    macd_hist: pd.Series,
    *,
    bull_color: str = "#00c853",
    bear_color: str = "#ff1744",
    neutral_color: str = "#4eadfc",
) -> tuple[list[str], list[str]]:
    """Returns (increasing_colors, decreasing_colors) per bar following Elder Impulse rules."""
    ema_d = ema_fast.diff()
    hist_d = macd_hist.diff()
    bulls = (ema_d > 0) & (hist_d > 0)
    bears = (ema_d < 0) & (hist_d < 0)
    n = len(df)
    inc = [neutral_color] * n
    dec = [neutral_color] * n
    for i in range(n):
        if bool(bulls.iloc[i]):
            inc[i] = bull_color
            dec[i] = bull_color
        elif bool(bears.iloc[i]):
            inc[i] = bear_color
            dec[i] = bear_color
    return inc, dec


def _idx_to_x(df: pd.DataFrame, i: int):
    if i < 0 or i >= len(df):
        return None
    return df.index[i]


def _add_peak_trough_labels(fig: go.Figure, df: pd.DataFrame, seq_full: dict) -> None:
    for ev in seq_full.get("peaks", []):
        x = _idx_to_x(df, int(ev["idx"]))
        if x is None:
            continue
        fig.add_annotation(
            x=x, y=float(ev["price"]),
            text=str(ev["label"]),
            showarrow=True,
            arrowhead=2,
            arrowcolor="#cfd8dc",
            ax=0, ay=-22,
            font=dict(size=11, color="#eceff1"),
            bgcolor="rgba(20,20,20,0.65)",
            bordercolor="#37474f",
            borderwidth=1,
            row=1, col=1,
        )
    for ev in seq_full.get("troughs", []):
        x = _idx_to_x(df, int(ev["idx"]))
        if x is None:
            continue
        fig.add_annotation(
            x=x, y=float(ev["price"]),
            text=str(ev["label"]),
            showarrow=True,
            arrowhead=2,
            arrowcolor="#cfd8dc",
            ax=0, ay=22,
            font=dict(size=11, color="#eceff1"),
            bgcolor="rgba(20,20,20,0.65)",
            bordercolor="#37474f",
            borderwidth=1,
            row=1, col=1,
        )


def _add_peak_lines(fig: go.Figure, df: pd.DataFrame, events: list[dict], color: str) -> None:
    pts = []
    for ev in events:
        x = _idx_to_x(df, int(ev["idx"]))
        if x is None:
            continue
        pts.append((x, float(ev["price"])))
    if len(pts) < 2:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=color, width=1, dash="dot"),
            name="Structure",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1, col=1,
    )


def _add_critical_step(fig: go.Figure, df: pd.DataFrame, seq_full: dict) -> None:
    crit = seq_full.get("critical_level_series")
    state = seq_full.get("seq_state_series")
    if crit is None or state is None:
        return
    crit = np.asarray(crit, dtype=float)
    state = np.asarray(state, dtype=int)
    if len(crit) != len(df):
        return
    up_mask = state == 1
    dn_mask = state == -1
    crit_up = np.where(up_mask, crit, np.nan)
    crit_dn = np.where(dn_mask, crit, np.nan)
    fig.add_trace(
        go.Scatter(
            x=df.index, y=crit_up, mode="lines",
            line=dict(color="#00e676", width=2, dash="dot", shape="hv"),
            name="Critical (Up)",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=crit_dn, mode="lines",
            line=dict(color="#ff1744", width=2, dash="dot", shape="hv"),
            name="Critical (Down)",
        ),
        row=1, col=1,
    )


def _add_fib(fig: go.Figure, df: pd.DataFrame, fib: dict | None) -> None:
    if not fib:
        return
    high_idx = int(fib.get("high_idx", -1))
    x0 = _idx_to_x(df, high_idx) if high_idx >= 0 else df.index[0]
    x1 = df.index[-1]
    for key, label, color in (
        ("fib_382", "0.382", "#ffd54f"),
        ("fib_500", "0.5", "#ffb74d"),
        ("fib_618", "0.618", "#ff8a65"),
    ):
        val = fib.get(key)
        if val is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=[x0, x1], y=[val, val], mode="lines",
                line=dict(color=color, width=1.5, dash="dash"),
                name=f"Fib {label} ({val:.2f})",
            ),
            row=1, col=1,
        )


def _add_tp_sl(fig: go.Figure, df: pd.DataFrame, tp: float | None, sl: float | None) -> None:
    if tp is not None and not (isinstance(tp, float) and math.isnan(tp)):
        fig.add_hline(
            y=float(tp), line=dict(color="#69f0ae", width=1.5, dash="dashdot"),
            annotation_text=f"TP {float(tp):.2f}", annotation_position="right",
            annotation_font_color="#69f0ae",
            row=1, col=1,
        )
    if sl is not None and not (isinstance(sl, float) and math.isnan(sl)):
        fig.add_hline(
            y=float(sl), line=dict(color="#ff5252", width=1.5, dash="dashdot"),
            annotation_text=f"SL {float(sl):.2f}", annotation_position="right",
            annotation_font_color="#ff5252",
            row=1, col=1,
        )


def build_sequence_vova_figure(
    df: pd.DataFrame,
    seq_full: dict | None,
    layers: dict,
    *,
    tp: float | None = None,
    sl: float | None = None,
    title: str = "",
    height: int = 800,
) -> go.Figure:
    """Two-row Plotly chart (candles + volume) with full Sequence Vova overlays."""
    inc_colors, dec_colors = elder_bar_colors(df, layers["ema_fast"], layers["macd_hist"])

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.78, 0.22],
        vertical_spacing=0.02,
        specs=[[{"type": "xy"}], [{"type": "xy"}]],
    )

    # Bollinger fill
    fig.add_trace(
        go.Scatter(
            x=df.index, y=layers["bb_upper"], mode="lines",
            line=dict(color="rgba(120,120,120,0.55)", width=1),
            name="BB Upper", hoverinfo="skip", showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=layers["bb_lower"], mode="lines",
            line=dict(color="rgba(120,120,120,0.55)", width=1),
            fill="tonexty", fillcolor="rgba(120,120,120,0.10)",
            name="BB Lower", hoverinfo="skip", showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=layers["bb_basis"], mode="lines",
            line=dict(color="#448aff", width=1.2), name="BB Basis",
        ),
        row=1, col=1,
    )

    # Elder Envelope
    fig.add_trace(
        go.Scatter(
            x=df.index, y=layers["env_upper"], mode="lines",
            line=dict(color="rgba(180,180,180,0.55)", width=1, dash="dot"),
            name="Env Upper", hoverinfo="skip",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=layers["env_lower"], mode="lines",
            line=dict(color="rgba(180,180,180,0.55)", width=1, dash="dot"),
            name="Env Lower", hoverinfo="skip",
        ),
        row=1, col=1,
    )

    # EMAs and SMA
    fig.add_trace(
        go.Scatter(
            x=df.index, y=layers["ema_fast"], mode="lines",
            line=dict(color="#2962ff", width=1.5), name="EMA 20",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=layers["ema_slow"], mode="lines",
            line=dict(color="#e53935", width=1.5), name="EMA 40",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=layers["sma_major"], mode="lines",
            line=dict(color="#ffa726", width=2), name="SMA 200",
        ),
        row=1, col=1,
    )

    # Candles with Elder coloring (per-bar via marker arrays)
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            increasing=dict(line=dict(color="#00c853"), fillcolor="#00c853"),
            decreasing=dict(line=dict(color="#ff1744"), fillcolor="#ff1744"),
            name="Price",
            showlegend=False,
            whiskerwidth=0.4,
        ),
        row=1, col=1,
    )

    # Sequence Vova structural overlays
    if seq_full is not None:
        _add_critical_step(fig, df, seq_full)
        _add_peak_lines(fig, df, seq_full.get("peaks", []), color="#cfd8dc")
        _add_peak_lines(fig, df, seq_full.get("troughs", []), color="#cfd8dc")
        _add_peak_trough_labels(fig, df, seq_full)
        _add_fib(fig, df, seq_full.get("fib"))

    _add_tp_sl(fig, df, tp, sl)

    # Volume bars (colored by close vs open)
    vol_colors = np.where(df["Close"] >= df["Open"], "rgba(0,200,83,0.55)", "rgba(255,23,68,0.55)")
    fig.add_trace(
        go.Bar(
            x=df.index, y=df["Volume"],
            marker=dict(color=vol_colors.tolist()),
            name="Volume", showlegend=False,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1116",
        plot_bgcolor="#0e1116",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text=title, x=0.01, font=dict(size=14, color="#cfd8dc")),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(size=10, color="#cfd8dc")),
        hovermode="x unified",
        dragmode="pan",
    )
    fig.update_xaxes(showgrid=False, color="#90a4ae")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(120,120,120,0.15)", color="#90a4ae")

    return fig
