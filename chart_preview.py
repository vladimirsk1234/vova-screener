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

from indicator_params import IndicatorParams
from sequence_vova import run_sequence_vova_full
from ticker_data import get_chart_fundamentals
from watermark_status import (
    build_dwm_lines,
    build_trade_line,
    build_watermark_text,
)

CACHE_TRIM_BARS = 200
DEFAULT_MAX_BARS = 120
DEFAULT_CHART_HEIGHT = 720

PLOTLY_CHART_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": True,
    "doubleClick": "reset",
}

DAY_MS = 86_400_000


def _max_bars_for_tf(tf: str) -> int:
    return {"Daily": 180, "Weekly": 80, "Monthly": 36}.get(tf, DEFAULT_MAX_BARS)


def _xperiod_ms(tf: str) -> int:
    return {
        "Daily": DAY_MS,
        "Weekly": 7 * DAY_MS,
        "Monthly": 30 * DAY_MS,
    }.get(tf, DAY_MS)


def _bar_align_kwargs(xperiod: int) -> dict:
    return {"xperiod": xperiod, "xperiodalignment": "middle"}


def _to_list(arr) -> list:
    if arr is None:
        return []
    if isinstance(arr, np.ndarray):
        return [None if (isinstance(x, float) and np.isnan(x)) else x for x in arr]
    if isinstance(arr, pd.Series):
        return _to_list(arr.values)
    return list(arr)


def _slice_series(series: pd.Series | np.ndarray, offset: int, n: int) -> np.ndarray:
    if isinstance(series, pd.Series):
        arr = series.values
    else:
        arr = np.asarray(series)
    if len(arr) > offset:
        s = arr[offset : offset + n]
    else:
        s = np.array([])
    if len(s) < n:
        s = np.pad(s, (0, n - len(s)), constant_values=np.nan)
    return s.astype(float)


def build_chart_payload(
    df: pd.DataFrame,
    tf: str,
    *,
    symbol: str = "",
    yahoo_ticker: str = "",
    df_daily: pd.DataFrame | None = None,
    trim_bars: int = CACHE_TRIM_BARS,
    **legacy: Any,
) -> dict | None:
    """Cache trimmed OHLC only; indicator recomputed at display time.

    Accepts legacy kwargs (atr_len, min_rr, use_last_hl_sl, risk_dollars) from
    older callers; those are ignored because indicator params apply at display time.
    """
    _ = legacy  # backward-compatible no-op
    if df is None or df.empty:
        return None
    trim = min(len(df), trim_bars)
    offset = len(df) - trim
    out_df = df.iloc[offset:].copy()
    daily_out = None
    if df_daily is not None and not df_daily.empty:
        dtrim = min(len(df_daily), trim_bars)
        doff = len(df_daily) - dtrim
        daily_out = df_daily.iloc[doff:].copy()
    return {
        "df": out_df,
        "df_daily": daily_out,
        "tf": tf,
        "symbol": symbol,
        "yahoo_ticker": yahoo_ticker,
    }


def _extend_right_y(y0: float, y1: float, x0_idx: int, x1_idx: int, x_end_idx: int) -> float:
    """Extend line segment to x_end_idx at the same slope (Pine extend.right)."""
    if x1_idx == x0_idx:
        return y1
    slope = (y1 - y0) / (x1_idx - x0_idx)
    return y1 + slope * (x_end_idx - x1_idx)


def _interp_y(x: int, x0: int, y0: float, x1: int, y1: float) -> float:
    if x1 == x0:
        return y1
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _compute_y_range(
    plot_df: pd.DataFrame,
    full: dict,
    params: IndicatorParams,
    display_offset: int,
    n: int,
) -> tuple[float, float]:
    """Y-axis from OHLC + nearby overlays (ignore off-screen shape extremes)."""
    lo = float(plot_df["Low"].min())
    hi = float(plot_df["High"].max())
    extras: list[float] = []

    if params.show_crit_level:
        crit = full.get("critical_level")
        if crit is not None and not np.isnan(crit):
            extras.append(float(crit))

    if params.show_tp_sl:
        for key in ("TP", "SL"):
            val = full.get(key)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                extras.append(float(val))

    if params.show_fib:
        fib = full.get("fib") or {}
        for key in ("fib_382", "fib_500", "fib_618"):
            val = fib.get(key)
            if val is not None:
                extras.append(float(val))

    for p in full.get("peaks") or []:
        idx = int(p["idx"]) - display_offset
        if 0 <= idx < n:
            extras.append(float(p["price"]))
    for t in full.get("troughs") or []:
        idx = int(t["idx"]) - display_offset
        if 0 <= idx < n:
            extras.append(float(t["price"]))

    if extras:
        lo = min(lo, min(extras))
        hi = max(hi, max(extras))

    span = hi - lo
    pad = max(span * 0.12, hi * 0.03)
    return lo - pad, hi + pad


def _add_extension_lines(
    fig: go.Figure,
    extension_lines: list[dict],
    x_index: pd.DatetimeIndex,
    display_offset: int,
    n: int,
    color: str,
) -> None:
    x_end = n - 1
    for seg in extension_lines or []:
        x0 = int(seg["x0_idx"]) - display_offset
        x1 = int(seg["x1_idx"]) - display_offset
        if x0 < 0 and x1 < 0:
            continue
        if x0 >= n and x1 >= n:
            continue
        x0c = max(0, min(x0, n - 1))
        x1c = max(0, min(x1, n - 1))
        y0 = float(seg["y0"])
        y1 = float(seg["y1"])
        # Slope must use true bar indices; clamped x0c/x1c alone steepen lines off-screen.
        y_start = _interp_y(x0c, x0, y0, x1, y1)
        y_at_x1 = _interp_y(x1c, x0, y0, x1, y1)
        y_end = _extend_right_y(y0, y1, x0, x1, x_end)
        fig.add_shape(
            type="line",
            x0=x_index[x0c],
            y0=y_start,
            x1=x_index[x1c],
            y1=y_at_x1,
            line=dict(color=color, width=2),
            xref="x",
            yref="y",
        )
        if x1c < x_end:
            fig.add_shape(
                type="line",
                x0=x_index[x1c],
                y0=y_at_x1,
                x1=x_index[x_end],
                y1=y_end,
                line=dict(color=color, width=2),
                xref="x",
                yref="y",
            )


def build_sequence_vova_figure(
    df: pd.DataFrame,
    full: dict,
    params: IndicatorParams,
    *,
    title: str = "",
    tf: str = "Daily",
    max_bars: int | None = None,
    height: int = DEFAULT_CHART_HEIGHT,
    fundamentals: dict | None = None,
    df_daily: pd.DataFrame | None = None,
    df_chart: pd.DataFrame | None = None,
    yahoo_ticker: str = "",
) -> go.Figure:
    """Candlestick chart with Sequence Vova overlays.

    `df` is the visible plot window; indicator arrays in `full` must match its length.
    `df_chart` optional longer series for watermark HTF (defaults to `df`).
    """
    if max_bars is None:
        max_bars = _max_bars_for_tf(tf)

    plot_df = df.iloc[-max_bars:].copy() if len(df) > max_bars else df.copy()
    display_offset = 0
    x_index = pd.DatetimeIndex(plot_df.index)
    n = len(x_index)
    xperiod = _xperiod_ms(tf)

    crit = _slice_series(full.get("critical_level_series", []), display_offset, n)
    state = _slice_series(full.get("seq_state_series", []), display_offset, n).astype(int)
    overlays = full.get("overlays") or {}

    fig = go.Figure()

    # Optional impulse tint on candles (overlay scatter bars — simplified as marker-less)
    if params.show_elder_impulse:
        impulse = full.get("impulse_colors")
        if impulse is not None and len(impulse) > display_offset:
            ic = impulse[display_offset : display_offset + n]
            for i in range(n):
                if i < len(ic) and ic[i]:
                    fig.add_vrect(
                        x0=x_index[i],
                        x1=x_index[i] + pd.Timedelta(milliseconds=xperiod),
                        fillcolor=str(ic[i]),
                        opacity=0.12,
                        layer="below",
                        line_width=0,
                    )

    fig.add_trace(
        go.Candlestick(
            x=x_index,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="OHLC",
            increasing_line_color=params.candle_border,
            increasing_fillcolor=params.candle_up,
            decreasing_line_color=params.candle_border,
            decreasing_fillcolor=params.candle_down,
            whiskerwidth=0,
            line=dict(width=1.5),
            xperiod=xperiod,
            xperiodalignment="middle",
        )
    )

    if params.show_bb and overlays:
        bb_u = _slice_series(overlays.get("bb_upper", []), display_offset, n)
        bb_l = _slice_series(overlays.get("bb_lower", []), display_offset, n)
        bb_b = _slice_series(overlays.get("bb_basis", []), display_offset, n)
        fig.add_trace(
            go.Scatter(
                x=x_index, y=bb_u, mode="lines", name="BB Upper",
                line=dict(color=params.bb_upper_color, width=1.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_index, y=bb_l, mode="lines", name="BB Lower",
                line=dict(color=params.bb_lower_color, width=1.5),
                fill="tonexty" if params.show_bb_background else None,
                fillcolor=params.bb_fill_color if params.show_bb_background else None,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_index, y=bb_b, mode="lines", name="BB Basis",
                line=dict(color=params.bb_basis_color, width=1.5),
            )
        )

    if params.show_elder_envelope and overlays:
        eu = _slice_series(overlays.get("env_upper", []), display_offset, n)
        el = _slice_series(overlays.get("env_lower", []), display_offset, n)
        fig.add_trace(
            go.Scatter(
                x=x_index, y=eu, mode="lines", name="Env Upper",
                line=dict(color=params.env_upper_color, width=1.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_index, y=el, mode="lines", name="Env Lower",
                line=dict(color=params.env_lower_color, width=1.5),
            )
        )

    if params.show_center_ema and overlays:
        ema_s = _slice_series(overlays.get("ema_slow", []), display_offset, n)
        fig.add_trace(
            go.Scatter(
                x=x_index, y=ema_s, mode="lines", name="Center EMA",
                line=dict(color=params.center_ema_color, width=2),
            )
        )

    if params.show_short_ema and overlays:
        ema_f = _slice_series(overlays.get("ema_fast", []), display_offset, n)
        fig.add_trace(
            go.Scatter(
                x=x_index, y=ema_f, mode="lines", name="Short EMA",
                line=dict(color=params.short_ema_color, width=2),
            )
        )

    if params.show_sma_major and overlays:
        sma = _slice_series(overlays.get("sma_major", []), display_offset, n)
        fig.add_trace(
            go.Scatter(
                x=x_index, y=sma, mode="lines", name=f"SMA {params.length_major}",
                line=dict(color=params.sma_major_color, width=3),
            )
        )

    if params.show_crit_level:
        crit_up = np.where(state == 1, crit, np.nan)
        crit_dn = np.where(state == -1, crit, np.nan)
        fig.add_trace(
            go.Scatter(
                x=x_index,
                y=crit_up,
                mode="lines",
                name="Critical (up)",
                line=dict(color=params.crit_stop_color_up, width=2, dash="dot", shape="hv"),
                connectgaps=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_index,
                y=crit_dn,
                mode="lines",
                name="Critical (down)",
                line=dict(color=params.crit_stop_color_down, width=2, dash="dot", shape="hv"),
                connectgaps=False,
            )
        )
        last_crit = full.get("critical_level")
        if last_crit is not None and n > 0:
            line_color = (
                params.crit_stop_color_up if full.get("seq_state_final") == 1 else params.crit_stop_color_down
            )
            fig.add_shape(
                type="line",
                x0=x_index[-1],
                y0=float(last_crit),
                x1=x_index[-1] + pd.Timedelta(days=3 if tf == "Daily" else 14),
                y1=float(last_crit),
                line=dict(color=line_color, width=2, dash="dash"),
            )
            fig.add_annotation(
                x=x_index[-1],
                y=float(last_crit),
                text=f"Critical Level = {last_crit:.2f}",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font=dict(color=params.crit_custom_color, size=11),
            )

    if params.show_extension_lines:
        _add_extension_lines(
            fig,
            full.get("extension_lines") or [],
            x_index,
            display_offset,
            n,
            params.hhll_color,
        )

    if params.show_hhll:
        peak_x, peak_y, peak_txt = [], [], []
        for p in full.get("peaks") or []:
            idx = int(p["idx"])
            if idx < display_offset or idx - display_offset >= n:
                continue
            xi = idx - display_offset
            lbl = str(p.get("label", ""))
            peak_x.append(x_index[xi])
            peak_y.append(float(p["price"]))
            peak_txt.append(f"{lbl}<br>↓")
        if peak_x:
            fig.add_trace(
                go.Scatter(
                    x=peak_x,
                    y=peak_y,
                    mode="markers+text",
                    name="Peaks",
                    text=peak_txt,
                    textposition="top center",
                    marker=dict(color=params.hhll_color, size=10, symbol="triangle-down"),
                    textfont=dict(size=params.hhll_label_size, color=params.hhll_color),
                    **_bar_align_kwargs(xperiod),
                )
            )

        trough_x, trough_y, trough_txt = [], [], []
        for t in full.get("troughs") or []:
            idx = int(t["idx"])
            if idx < display_offset or idx - display_offset >= n:
                continue
            xi = idx - display_offset
            lbl = str(t.get("label", ""))
            trough_x.append(x_index[xi])
            trough_y.append(float(t["price"]))
            trough_txt.append(f"↑<br>{lbl}")
        if trough_x:
            fig.add_trace(
                go.Scatter(
                    x=trough_x,
                    y=trough_y,
                    mode="markers+text",
                    name="Troughs",
                    text=trough_txt,
                    textposition="bottom center",
                    marker=dict(color=params.hhll_color, size=10, symbol="triangle-up"),
                    textfont=dict(size=params.hhll_label_size, color=params.hhll_color),
                    **_bar_align_kwargs(xperiod),
                )
            )

    if params.show_breaks:
        bull = full.get("bullish_break")
        bear = full.get("bearish_break")
        if bull is not None and len(bull) > display_offset:
            bslice = bull[display_offset : display_offset + n]
            bx, by = [], []
            for i, flag in enumerate(bslice):
                if flag and i < n:
                    bx.append(x_index[i])
                    by.append(float(plot_df["High"].iloc[i]) * 1.002)
            if bx:
                fig.add_trace(
                    go.Scatter(
                        x=bx, y=by, mode="markers", name="Bear break",
                        marker=dict(symbol="triangle-down", size=8, color="#000"),
                        **_bar_align_kwargs(xperiod),
                    )
                )
        if bear is not None and len(bear) > display_offset:
            bslice = bear[display_offset : display_offset + n]
            bx, by = [], []
            for i, flag in enumerate(bslice):
                if flag and i < n:
                    bx.append(x_index[i])
                    by.append(float(plot_df["Low"].iloc[i]) * 0.998)
            if bx:
                fig.add_trace(
                    go.Scatter(
                        x=bx, y=by, mode="markers", name="Bull break",
                        marker=dict(symbol="triangle-up", size=8, color="#000"),
                        **_bar_align_kwargs(xperiod),
                    )
                )

    if params.show_tp_sl:
        tp = full.get("TP")
        sl = full.get("SL")
        if tp is not None and not (isinstance(tp, float) and np.isnan(tp)):
            fig.add_hline(
                y=float(tp), line_dash="dash", line_color=params.candle_up,
                annotation_text="TP", annotation_position="right",
            )
        if sl is not None and not (isinstance(sl, float) and np.isnan(sl)):
            fig.add_hline(
                y=float(sl), line_dash="dash", line_color=params.candle_down,
                annotation_text="SL", annotation_position="right",
            )

    if params.show_fib:
        fib = full.get("fib")
        if fib:
            fib_start_idx = int(fib.get("high_idx", 0)) - display_offset
            fib_start_idx = max(0, min(fib_start_idx, n - 1))
            fib_x0 = x_index[fib_start_idx]
            fib_x1 = x_index[-1] + pd.Timedelta(days=3 if tf == "Daily" else 14)
            for key, label in (
                ("fib_382", "0.382"),
                ("fib_500", "0.5"),
                ("fib_618", "0.618"),
            ):
                val = fib.get(key)
                if val is not None:
                    fig.add_shape(
                        type="line",
                        x0=fib_x0,
                        y0=float(val),
                        x1=fib_x1,
                        y1=float(val),
                        line=dict(color=params.fib_color, width=params.fib_width, dash="dash"),
                        xref="x",
                        yref="y",
                    )
                    fig.add_annotation(
                        x=fib_x1,
                        y=float(val),
                        text=f"{label} ({val:.2f})",
                        showarrow=False,
                        xanchor="left",
                        yanchor="middle",
                        font=dict(color=params.fib_color, size=10),
                    )

    chart_title = title or f"Sequence Vova - {tf}"
    xaxis_kwargs = dict(
        showgrid=False,
        zeroline=False,
        tickfont=dict(color="#d1d4dc"),
        linecolor="#2a2e39",
        rangeslider=dict(visible=False),
        type="date",
        showspikes=True,
        spikemode="across",
        spikecolor="#444",
        spikethickness=1,
    )
    if tf == "Daily":
        xaxis_kwargs["rangebreaks"] = [dict(bounds=["sat", "mon"])]

    x_pad = pd.Timedelta(days=7 if tf == "Daily" else (21 if tf == "Weekly" else 45))
    xaxis_kwargs["range"] = [x_index[0], x_index[-1] + x_pad]
    xaxis_kwargs["autorange"] = False

    y_lo, y_hi = _compute_y_range(plot_df, full, params, display_offset, n)

    fig.update_layout(
        title=dict(text=chart_title, font=dict(color="#d1d4dc", size=15)),
        template="plotly_dark",
        paper_bgcolor=params.paper_color,
        plot_bgcolor=params.bg_color,
        dragmode="pan",
        xaxis=xaxis_kwargs,
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(color="#d1d4dc"),
            linecolor="#2a2e39",
            side="right",
            range=[y_lo, y_hi],
            autorange=False,
        ),
        margin=dict(l=12, r=56, t=44, b=72),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(size=10, color="#d1d4dc"),
        ),
        height=height,
        hovermode="x unified",
    )

    fund = fundamentals or {}
    chart_df = df_chart if df_chart is not None else plot_df
    dwm = build_dwm_lines(chart_df, df_daily, params, chart_tf=tf)
    trade = build_trade_line(full, params, len(plot_df) - 1)
    wm = build_watermark_text(
        fundamentals=fund,
        full=full,
        params=params,
        dwm_lines=dwm,
        chart_tf=tf,
        ticker=yahoo_ticker or title,
        trade_line=trade,
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.99,
        xanchor="left",
        yanchor="top",
        text=wm,
        showarrow=False,
        align="left",
        font=dict(size=params.wm_font_size, color=params.wm_text_color),
        bgcolor="rgba(42,46,57,0.75)",
    )

    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    return fig


def figure_from_payload(
    payload: dict,
    *,
    symbol: str = "",
    params: IndicatorParams | None = None,
    height: int = DEFAULT_CHART_HEIGHT,
) -> go.Figure | None:
    """Build figure from cached OHLC payload + live indicator params."""
    if not payload:
        return None
    df = payload.get("df")
    if df is None:
        return None
    if isinstance(df, dict):
        df = pd.DataFrame(df)
    p = params or IndicatorParams()
    tf = str(payload.get("tf", "Daily"))
    max_bars = _max_bars_for_tf(tf)
    plot_df = df.iloc[-max_bars:].copy() if len(df) > max_bars else df.copy()
    full = run_sequence_vova_full(plot_df, params=p)
    if full is None:
        return None

    title = f"{symbol} - {tf}" if symbol else f"Sequence Vova - {tf}"
    yahoo = str(payload.get("yahoo_ticker", "") or "")
    df_daily = payload.get("df_daily")
    if isinstance(df_daily, dict):
        df_daily = pd.DataFrame(df_daily)

    fundamentals = None
    if yahoo:
        prev_close = None
        if df_daily is not None and len(df_daily) >= 2:
            prev_close = float(df_daily["Close"].iloc[-2])
        elif len(df) >= 2:
            prev_close = float(df["Close"].iloc[-2])
        fundamentals = get_chart_fundamentals(
            yahoo,
            close=float(plot_df["Close"].iloc[-1]),
            prev_daily_close=prev_close,
        )

    return build_sequence_vova_figure(
        plot_df,
        full,
        p,
        title=title,
        tf=tf,
        height=height,
        fundamentals=fundamentals,
        df_daily=df_daily,
        df_chart=df,
        yahoo_ticker=yahoo or symbol,
    )


def figure_to_plotly_json(fig: go.Figure) -> dict[str, Any]:
    return json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))


def chart_json_for_mobile(
    chart_cache: dict,
    table_rows: list[dict],
    *,
    height: int = 460,
    params: IndicatorParams | None = None,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    p = params or IndicatorParams()
    for row in table_rows:
        key = str(row.get("tv_symbol", "") or "")
        if not key or key in out:
            continue
        payload = chart_cache.get(key)
        if not payload:
            continue
        fig = figure_from_payload(payload, symbol=key, params=p, height=height)
        if fig is not None:
            out[key] = figure_to_plotly_json(fig)
    return out
