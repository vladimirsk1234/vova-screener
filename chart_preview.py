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

DAY_MS = 86_400_000


def _max_bars_for_tf(tf: str) -> int:
    return {"Daily": 180, "Weekly": 80, "Monthly": 36}.get(tf, DEFAULT_MAX_BARS)


def _xperiod_ms(tf: str) -> int:
    return {
        "Daily": DAY_MS,
        "Weekly": 7 * DAY_MS,
        "Monthly": 30 * DAY_MS,
    }.get(tf, DAY_MS)


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
) -> dict | None:
    """Cache trimmed OHLC only; indicator recomputed at display time."""
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
        "bar_offset": offset,
        "symbol": symbol,
        "yahoo_ticker": yahoo_ticker,
    }


def _add_extension_lines(
    fig: go.Figure,
    extension_lines: list[dict],
    x_index: pd.DatetimeIndex,
    display_offset: int,
    n: int,
    color: str,
) -> None:
    for seg in extension_lines or []:
        x0 = int(seg["x0_idx"]) - display_offset
        x1 = int(seg["x1_idx"]) - display_offset
        if x0 < 0 and x1 < 0:
            continue
        if x0 >= n and x1 >= n:
            continue
        x0c = max(0, min(x0, n - 1))
        x1c = max(0, min(x1, n - 1))
        fig.add_shape(
            type="line",
            x0=x_index[x0c],
            y0=float(seg["y0"]),
            x1=x_index[x1c],
            y1=float(seg["y1"]),
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
    bar_offset: int = 0,
    height: int = 580,
    fundamentals: dict | None = None,
    df_daily: pd.DataFrame | None = None,
    yahoo_ticker: str = "",
) -> go.Figure:
    """Candlestick chart with Sequence Vova overlays."""
    if max_bars is None:
        max_bars = _max_bars_for_tf(tf)

    plot_df = df.iloc[-max_bars:].copy() if len(df) > max_bars else df.copy()
    extra_offset = max(0, len(df) - len(plot_df))
    display_offset = bar_offset + extra_offset
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
            increasing_line_color=params.candle_up,
            increasing_fillcolor=params.candle_up,
            decreasing_line_color=params.candle_down,
            decreasing_fillcolor=params.candle_down,
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
                x=x_index[min(n - 1, max(0, n - params.crit_lbl_offset))],
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
            for key, label in (
                ("fib_382", "0.382"),
                ("fib_500", "0.5"),
                ("fib_618", "0.618"),
            ):
                val = fib.get(key)
                if val is not None:
                    fig.add_hline(
                        y=float(val),
                        line_dash="dash",
                        line_color=params.fib_color,
                        line_width=params.fib_width,
                        annotation_text=f"{label} ({val:.2f})",
                        annotation_position="left",
                    )

    chart_title = title or f"Sequence Vova - {tf}"
    xaxis_kwargs = dict(
        gridcolor=params.grid_color,
        rangeslider=dict(visible=False),
        type="date",
        showspikes=True,
        spikemode="across",
        spikecolor="#444",
        spikethickness=1,
    )
    if tf == "Daily":
        xaxis_kwargs["rangebreaks"] = [dict(bounds=["sat", "mon"])]

    fig.update_layout(
        title=dict(text=chart_title, font=dict(color="#e0e0e0", size=15)),
        template="plotly_dark",
        paper_bgcolor=params.paper_color,
        plot_bgcolor=params.bg_color,
        xaxis=xaxis_kwargs,
        yaxis=dict(gridcolor=params.grid_color, side="right"),
        margin=dict(l=12, r=56, t=44, b=72),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        height=height,
        hovermode="x unified",
    )

    if params.show_watermark:
        fund = fundamentals or {}
        dwm = build_dwm_lines(df, df_daily, params, chart_tf=tf)
        trade = build_trade_line(full, params, len(df) - 1)
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
            bgcolor="rgba(0,0,0,0.35)",
        )

    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    return fig


def figure_from_payload(
    payload: dict,
    *,
    symbol: str = "",
    params: IndicatorParams | None = None,
    height: int = 580,
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
    full = run_sequence_vova_full(df, params=p)
    if full is None:
        return None

    tf = str(payload.get("tf", "Daily"))
    title = f"{symbol} - {tf}" if symbol else f"Sequence Vova - {tf}"
    yahoo = str(payload.get("yahoo_ticker", "") or "")
    df_daily = payload.get("df_daily")
    if isinstance(df_daily, dict):
        df_daily = pd.DataFrame(df_daily)

    fundamentals = None
    if p.show_watermark and yahoo:
        prev_close = None
        if df_daily is not None and len(df_daily) >= 2:
            prev_close = float(df_daily["Close"].iloc[-2])
        elif len(df) >= 2:
            prev_close = float(df["Close"].iloc[-2])
        fundamentals = get_chart_fundamentals(
            yahoo,
            close=float(df["Close"].iloc[-1]),
            prev_daily_close=prev_close,
        )

    return build_sequence_vova_figure(
        df,
        full,
        p,
        title=title,
        tf=tf,
        bar_offset=int(payload.get("bar_offset", 0)),
        height=height,
        fundamentals=fundamentals,
        df_daily=df_daily,
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
