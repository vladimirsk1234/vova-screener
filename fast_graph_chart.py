"""
FAST Graphs–style Plotly charts (Historical + Forecasting). No Streamlit.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fast_graph_metrics import (
    _estimate_eps_chain,
    compute_forecast_growth_pct,
    compute_historical_growth_rate_pct,
    resolve_fair_pe,
)
from eps_yield import avg_historical_pe_5y

# FAST Graphs palette (matches premium UI: orange fair, blue normal, green earnings).
FG_COLORS = {
    "fair_line": "#FF9500",
    "fair_marker_fill": "#FF9500",
    "fair_marker_line": "#FFFFFF",
    "fair_fill": "rgba(255, 149, 0, 0.22)",
    "normal_line": "#4DA3FF",
    "normal_marker_fill": "#4DA3FF",
    "normal_marker_line": "#FFFFFF",
    "earnings_fill": "rgba(22, 72, 22, 0.78)",
    "earnings_line": "#3d8b37",
    "dividend_fill": "rgba(76, 160, 55, 0.42)",
    "dividend_line": "#2E7D32",
    "div_por": "#E8EAED",
    "price_line": "#FFFFFF",
    "price_marker": "#FFFFFF",
    "grid": "rgba(255, 255, 255, 0.08)",
    "paper_bg": "#0d1117",
    "plot_bg": "#131722",
    "growth_accent": "#6BAF4A",
    "ror_accent": "#6BAF4A",
}

# FAST Graphs uses calendar fiscal labels like 12/24 (December year-end).
_FISCAL_MONTH = 12
_FISCAL_DAY = 31


def _fy_label(year: int, *, is_estimate: bool = False) -> str:
    suffix = "E" if is_estimate else ""
    return f"{_FISCAL_MONTH:02d}/{year % 100:02d}{suffix}"


def _fiscal_timestamp(year: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=_FISCAL_MONTH, day=_FISCAL_DAY)


def _chart_eps_points(
    annual_eps: dict[int, float],
    estimates: dict[str, Any],
    *,
    mode: str,
    hist_growth: float | None,
    forecast_growth: float | None,
) -> list[tuple[int, float, bool]]:
    """
    EPS points for chart.
    Historical (FAST-style): reported years + analyst forward estimates on the graph,
    but caller applies historical P/E multiples.
    Forecast: full estimate chain with projected years.
    """
    years_ahead = 4
    eps_growth = forecast_growth if mode == "forecast" else hist_growth
    return _estimate_eps_chain(
        annual_eps,
        estimates,
        years_ahead=years_ahead,
        growth_rate=eps_growth,
    )


_TABLE_ROW_Y = {
    "fy": 3.0,
    "eps": 2.0,
    "chg": 1.0,
    "div": 0.0,
    "analysts": -1.0,
}

_TABLE_LEFT_LABELS: list[tuple[str, str]] = [
    ("FY Date", "fy"),
    ("EPS", "eps"),
    ("Chg/Yr", "chg"),
    ("Div", "div"),
]


def _add_aligned_fy_table(
    eps_table: pd.DataFrame,
    eps_points: list[tuple[int, float, bool]],
    annual_dividends: dict[int, float],
    *,
    fallback_div: float | None,
    is_forecast: bool,
) -> list[dict]:
    """
    Build FAST-style table annotations: one column per fiscal year, x-aligned to chart.
    """
    annotations: list[dict] = []
    row_labels = list(_TABLE_LEFT_LABELS)
    if is_forecast and not eps_table.empty and eps_table["analysts"].notna().any():
        row_labels.append(("# Analysts", "analysts"))

    for label, key in row_labels:
        annotations.append(dict(
            x=0.045,
            xref="paper",
            yref="y2",
            y=_TABLE_ROW_Y[key],
            text=f"<b>{label}</b>",
            showarrow=False,
            xanchor="left",
            font=dict(size=10, color="#9aa0a6"),
        ))

    if eps_table.empty:
        return annotations

    for i, (_, row) in enumerate(eps_table.iterrows()):
        year = eps_points[i][0]
        x = _fiscal_timestamp(year)
        fy = str(row["fy"])
        eps_s = f"{row['eps']:.2f}"
        if row["is_est"] and not eps_s.endswith("E"):
            eps_s += "E"
        chg = row["chg_yr"]
        chg_s = f"{chg:.0f}%" if chg is not None and chg == chg else "—"
        chg_color = "#ff6b6b" if chg is not None and chg < 0 else "#e8eaed"
        dps = annual_dividends.get(year)
        if dps is None and fallback_div:
            dps = fallback_div
        div_s = f"{dps:.2f}" if dps is not None and dps > 0 else "—"

        cells: list[tuple[str, str, str]] = [
            (fy, "fy", "#e8eaed"),
            (eps_s, "eps", "#e8eaed"),
            (chg_s, "chg", chg_color),
            (div_s, "div", "#e8eaed"),
        ]
        if is_forecast and eps_table["analysts"].notna().any():
            a = row["analysts"]
            a_s = str(int(a)) if a is not None and a == a else "—"
            cells.append((a_s, "analysts", "#e8eaed"))

        for text, key, color in cells:
            annotations.append(dict(
                x=x,
                xref="x",
                yref="y2",
                y=_TABLE_ROW_Y[key],
                text=text,
                showarrow=False,
                xanchor="center",
                font=dict(size=10, color=color),
            ))

    return annotations


def _annual_eps_table_rows(
    eps_points: list[tuple[int, float, bool]],
    estimates: dict[str, Any],
    *,
    include_estimates: bool,
) -> pd.DataFrame:
    """FY / EPS / Chg/Yr table."""
    if not eps_points:
        return pd.DataFrame(columns=["fy", "eps", "chg_yr", "analysts", "is_est"])

    est_0y = (estimates or {}).get("0y", {})
    est_1y = (estimates or {}).get("+1y", {})

    rows = []
    prev: float | None = None
    for year, eps, is_est in eps_points:
        if is_est and not include_estimates:
            continue
        chg = None
        if prev is not None and prev != 0:
            chg = round((eps - prev) / abs(prev) * 100.0, 2)
        analysts = None
        if is_est and include_estimates:
            if year == eps_points[-1][0] and est_1y.get("numberOfAnalysts"):
                analysts = est_1y.get("numberOfAnalysts")
            elif est_0y.get("numberOfAnalysts"):
                analysts = est_0y.get("numberOfAnalysts")
        rows.append({
            "fy": _fy_label(year, is_estimate=is_est),
            "eps": eps,
            "chg_yr": chg,
            "analysts": analysts,
            "is_est": is_est,
        })
        prev = eps
    return pd.DataFrame(rows)


def _build_chart_timeline(
    price_index: pd.DatetimeIndex,
    eps_points: list[tuple[int, float, bool]],
) -> pd.DatetimeIndex:
    """Price history plus weekly extension through last projected fiscal year-end."""
    if pe_index_empty(price_index) or not eps_points:
        return price_index
    last_fy = _fiscal_timestamp(eps_points[-1][0])
    if price_index.max() >= last_fy:
        return price_index.sort_values()
    extra = pd.date_range(
        price_index.max() + pd.Timedelta(weeks=1),
        last_fy,
        freq="W-FRI",
    )
    return price_index.union(extra).sort_values()


def _extend_sloped_to_timeline(
    boundaries: list[tuple[pd.Timestamp, float]],
    timeline: pd.DatetimeIndex,
) -> tuple[list, list]:
    """
    FAST Graphs ramps: linear interpolation between fiscal year-end valuation points.
    Before first FY: hold first value; after last FY: hold last value.
    """
    if not boundaries or pe_index_empty(timeline):
        return [], []

    bounds = sorted(boundaries, key=lambda x: x[0])
    x_out: list = []
    y_out: list = []

    for ts in timeline:
        if ts <= bounds[0][0]:
            y = bounds[0][1]
        elif ts >= bounds[-1][0]:
            y = bounds[-1][1]
        else:
            y = bounds[-1][1]
            for i in range(len(bounds) - 1):
                t0, v0 = bounds[i]
                t1, v1 = bounds[i + 1]
                if t0 <= ts <= t1:
                    span = (t1 - t0).total_seconds()
                    if span <= 0:
                        y = v1
                    else:
                        frac = (ts - t0).total_seconds() / span
                        y = v0 + (v1 - v0) * frac
                    break
        x_out.append(ts)
        y_out.append(y)
    return x_out, y_out


def _extend_to_price_end(
    eps_points: list[tuple[int, float, bool]],
    price_index: pd.DatetimeIndex,
    *,
    pe_multiple: float,
) -> tuple[list, list]:
    """EPS × P/E sloped between fiscal year-ends on the chart timeline."""
    if not eps_points or pe_index_empty(price_index) or pe_multiple <= 0:
        return [], []

    boundaries = [
        (_fiscal_timestamp(year), eps * pe_multiple)
        for year, eps, _ in eps_points
    ]
    timeline = _build_chart_timeline(price_index, eps_points)
    return _extend_sloped_to_timeline(boundaries, timeline)


def pe_index_empty(price_index: pd.DatetimeIndex) -> bool:
    return price_index is None or len(price_index) == 0


def _annual_marker_series(
    eps_points: list[tuple[int, float, bool]],
    *,
    pe_multiple: float,
) -> tuple[list, list, list[bool]]:
    """Sparse annual points for markers (lines+markers overlay)."""
    xs, ys, est_flags = [], [], []
    for year, eps, is_est in eps_points:
        xs.append(_fiscal_timestamp(year))
        ys.append(eps * pe_multiple)
        est_flags.append(is_est)
    return xs, ys, est_flags


def _split_by_estimate(
    xs: list,
    ys: list,
    est_flags: list[bool],
) -> tuple[list, list, list, list]:
    """Split marker coordinates into reported vs estimate years."""
    act_x, act_y, est_x, est_y = [], [], [], []
    for x, y, is_est in zip(xs, ys, est_flags):
        if is_est:
            est_x.append(x)
            est_y.append(y)
        else:
            act_x.append(x)
            act_y.append(y)
    return act_x, act_y, est_x, est_y


def _add_marker_traces(
    fig: go.Figure,
    *,
    act_x: list,
    act_y: list,
    est_x: list,
    est_y: list,
    fill_color: str,
    line_color: str,
    symbol: str,
    size: int,
    hover_label: str,
) -> None:
    """Reported = filled markers; estimates = hollow markers."""
    if act_x:
        fig.add_trace(
            go.Scatter(
                x=act_x, y=act_y,
                mode="markers",
                marker=dict(
                    size=size, symbol=symbol,
                    color=fill_color,
                    line=dict(width=1.5, color=line_color),
                ),
                showlegend=False,
                hovertemplate=f"FY %{{x|%b %Y}}<br>{hover_label}: %{{y:.2f}}<extra></extra>",
            ),
            row=1, col=1,
        )
    if est_x:
        fig.add_trace(
            go.Scatter(
                x=est_x, y=est_y,
                mode="markers",
                marker=dict(
                    size=size, symbol=symbol,
                    color="rgba(0,0,0,0)",
                    line=dict(width=2, color=fill_color),
                ),
                showlegend=False,
                hovertemplate=f"FY %{{x|%b %Y}}E<br>{hover_label}: %{{y:.2f}}<extra></extra>",
            ),
            row=1, col=1,
        )


def _extend_dividend_to_price_end(
    eps_points: list[tuple[int, float, bool]],
    price_index: pd.DatetimeIndex,
    annual_dividends: dict[int, float],
    *,
    fair_pe: float,
    fallback_rate: float | None = None,
) -> tuple[list, list]:
    """Per-year DPS × fair P/E sloped between fiscal year-ends."""
    if not eps_points or pe_index_empty(price_index) or fair_pe <= 0:
        return [], []

    boundaries: list[tuple[pd.Timestamp, float]] = []
    for year, _eps, _ in eps_points:
        dps = annual_dividends.get(year)
        if dps is None or dps <= 0:
            dps = fallback_rate or 0.0
        boundaries.append((_fiscal_timestamp(year), float(dps) * fair_pe))

    timeline = _build_chart_timeline(price_index, eps_points)
    return _extend_sloped_to_timeline(boundaries, timeline)


def _compute_fast_graph_y_range(
    closes: pd.Series,
    fair_y: list[float],
    norm_y: list[float],
    div_y: list[float] | None = None,
) -> tuple[float, float]:
    """Y-axis spans price and full sloped valuation lines (FAST Graphs shows full mountain)."""
    price_lo = float(closes.min())
    price_hi = float(closes.max())
    span = max(price_hi - price_lo, price_hi * 0.05)
    pad = max(span * 0.08, price_hi * 0.03)
    y_lo = max(0.0, price_lo - pad * 0.35)

    y_hi = price_hi + pad
    all_vals = [y for y in (fair_y or []) + (norm_y or []) + (div_y or []) if y > 0]
    if all_vals:
        val_max = max(all_vals)
        y_hi = max(y_hi, val_max + pad * 0.12)
    return y_lo, y_hi


def _last_reported_fy_timestamp(
    eps_points: list[tuple[int, float, bool]],
) -> pd.Timestamp | None:
    for year, _, is_est in reversed(eps_points):
        if not is_est:
            return _fiscal_timestamp(year)
    return None


def _split_line_at_timestamp(
    x: list,
    y: list,
    split_ts: pd.Timestamp | None,
) -> tuple[tuple[list, list], tuple[list, list]]:
    """Split a sloped line into (before, after) at split_ts inclusive on after segment."""
    if not x or split_ts is None:
        return (x, y), ([], [])
    before_x, before_y, after_x, after_y = [], [], [], []
    for xi, yi in zip(x, y):
        ts = pd.Timestamp(xi)
        if ts <= split_ts:
            before_x.append(xi)
            before_y.append(yi)
        else:
            after_x.append(xi)
            after_y.append(yi)
    if before_x and after_x:
        after_x.insert(0, before_x[-1])
        after_y.insert(0, before_y[-1])
    return (before_x, before_y), (after_x, after_y)


def _fair_pe_config(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics.get("sidebar_fair_pe") or 15.0),
        float(metrics.get("growth_threshold") or 10.0),
        float(metrics.get("growth_cap_pct") or 100.0),
    )


def _mode_pe(
    metrics: dict[str, Any],
    mode: str,
    df_daily,
    annual_eps,
) -> tuple[float, float, float | None]:
    """Return (fair_pe, norm_pe, growth_rate) for chart mode."""
    sidebar_pe, growth_thr, growth_cap = _fair_pe_config(metrics)

    if mode == "forecast":
        growth = (
            metrics.get("chart_forecast_growth_rate")
            or metrics.get("forecast_growth_rate")
            or metrics.get("est_eps_growth")
        )
        fair_pe = metrics.get("forecast_fair_pe") or resolve_fair_pe(
            growth,
            sidebar_fair_pe=sidebar_pe,
            growth_threshold=growth_thr,
            growth_cap_pct=growth_cap,
        )
        norm_pe = metrics.get("forecast_normal_pe") or metrics.get("normal_pe")
    else:
        growth = (
            metrics.get("chart_historical_growth_rate")
            or metrics.get("historical_growth_rate")
            or metrics.get("growth_rate")
        )
        if growth is None:
            growth = compute_historical_growth_rate_pct(annual_eps)
        fair_pe = metrics.get("historical_fair_pe") or metrics.get("fair_pe") or resolve_fair_pe(
            growth,
            sidebar_fair_pe=sidebar_pe,
            growth_threshold=growth_thr,
            growth_cap_pct=growth_cap,
        )
        norm_pe = metrics.get("historical_normal_pe") or metrics.get("normal_pe")

    if norm_pe is None:
        norm_pe = avg_historical_pe_5y(df_daily, annual_eps)
    if norm_pe is None:
        norm_pe = fair_pe
    return float(fair_pe), float(norm_pe), growth


def _metric_box_annotation(
    *,
    y: float,
    label: str,
    value: str,
    color: str,
) -> dict:
    return dict(
        x=0.98, y=y, xref="paper", yref="paper",
        text=f"<b>{label}</b><br>{value}",
        showarrow=False,
        font=dict(size=11, color=color),
        bgcolor="rgba(0,0,0,0.65)",
        bordercolor=color,
        borderwidth=1,
        borderpad=4,
        xanchor="right",
    )


def build_fast_graph_figure(
    *,
    df_weekly: pd.DataFrame,
    df_daily: pd.DataFrame | None,
    metrics: dict[str, Any],
    bundle: dict[str, Any] | None = None,
    mode: str = "historical",
    height: int = 520,
) -> go.Figure | None:
    """
    Build FAST Graphs chart.
    mode: 'historical' | 'forecast'
    """
    if df_weekly is None or df_weekly.empty:
        return None

    bundle = bundle or metrics.get("bundle") or {}
    info = bundle.get("info") or {}
    currency = info.get("currency") or "USD"
    cur_sym = f"{currency} " if currency else ""

    annual_eps = metrics.get("annual_eps") or {}
    if isinstance(annual_eps, dict) and annual_eps and isinstance(next(iter(annual_eps.keys())), str):
        annual_eps = {int(k): float(v) for k, v in annual_eps.items()}

    estimates = bundle.get("earnings_estimates") or metrics.get("earnings_estimates") or {}
    fair_pe, norm_pe, hist_growth = _mode_pe(metrics, mode, df_daily, annual_eps)
    forecast_growth = metrics.get("forecast_growth_rate") or metrics.get("est_eps_growth")
    if forecast_growth is None:
        forecast_growth = compute_forecast_growth_pct(estimates, annual_eps, hist_growth)
    proj_growth = forecast_growth if mode == "forecast" else hist_growth

    price_df = df_weekly.copy()
    price_df.index = pd.to_datetime(price_df.index)
    closes = pd.to_numeric(price_df["Close"], errors="coerce").dropna()

    if mode == "forecast":
        eps_points = _chart_eps_points(
            annual_eps, estimates, mode=mode,
            hist_growth=hist_growth, forecast_growth=proj_growth,
        )
    else:
        eps_points = _chart_eps_points(
            annual_eps, estimates, mode=mode,
            hist_growth=hist_growth, forecast_growth=forecast_growth,
        )

    if not eps_points:
        return None

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.76, 0.24],
        vertical_spacing=0.02,
        specs=[[{"type": "xy"}], [{"type": "xy"}]],
    )

    annual_dividends_raw = bundle.get("annual_dividends") or metrics.get("annual_dividends") or {}
    annual_dividends: dict[int, float] = {}
    for k, v in annual_dividends_raw.items():
        try:
            annual_dividends[int(k)] = float(v)
        except (TypeError, ValueError):
            continue

    dividend_rate = info.get("dividend_rate") or metrics.get("dividend_rate")
    fallback_div = float(dividend_rate) if dividend_rate and dividend_rate > 0 else None
    div_step_x, div_step_y = _extend_dividend_to_price_end(
        eps_points,
        closes.index,
        annual_dividends,
        fair_pe=fair_pe,
        fallback_rate=fallback_div,
    )
    fair_step_x, fair_step_y = _extend_to_price_end(eps_points, closes.index, pe_multiple=fair_pe)
    norm_step_x, norm_step_y = _extend_to_price_end(eps_points, closes.index, pe_multiple=norm_pe)

    # Layer 1: earnings evaluation — dark green mountain (0 → sloped fair value).
    if fair_step_x and fair_step_y:
        fig.add_trace(
            go.Scatter(
                x=fair_step_x, y=fair_step_y,
                fill="tozeroy",
                fillcolor=FG_COLORS["earnings_fill"],
                line=dict(width=0),
                name="Earnings",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1, col=1,
        )

    # Layer 2: dividend zone — lighter green band at bottom (0 → sloped dividend line).
    if div_step_x and any(y > 0 for y in div_step_y):
        fig.add_trace(
            go.Scatter(
                x=div_step_x, y=div_step_y,
                fill="tozeroy",
                fillcolor=FG_COLORS["dividend_fill"],
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=div_step_x, y=div_step_y,
                name="Dividends POR",
                line=dict(color=FG_COLORS["div_por"], width=1.5, dash="dash"),
                mode="lines",
            ),
            row=1, col=1,
        )

    fair_mk_x, fair_mk_y, fair_est = _annual_marker_series(eps_points, pe_multiple=fair_pe)
    norm_mk_x, norm_mk_y, norm_est = _annual_marker_series(eps_points, pe_multiple=norm_pe)
    fair_act_x, fair_act_y, fair_est_x, fair_est_y = _split_by_estimate(fair_mk_x, fair_mk_y, fair_est)
    norm_act_x, norm_act_y, norm_est_x, norm_est_y = _split_by_estimate(norm_mk_x, norm_mk_y, norm_est)

    is_forecast = mode == "forecast"
    last_actual_ts = _last_reported_fy_timestamp(eps_points)
    fair_hist, fair_est_seg = _split_line_at_timestamp(fair_step_x, fair_step_y, last_actual_ts)

    if fair_step_x:
        fair_label = f"Fair Value Ratio ({fair_pe:.2f}x)"
        if fair_hist[0]:
            fig.add_trace(
                go.Scatter(
                    x=fair_hist[0], y=fair_hist[1],
                    name=fair_label,
                    line=dict(color=FG_COLORS["fair_line"], width=2.5, dash="solid"),
                    mode="lines",
                ),
                row=1, col=1,
            )
        if fair_est_seg[0]:
            fig.add_trace(
                go.Scatter(
                    x=fair_est_seg[0], y=fair_est_seg[1],
                    name=fair_label if not fair_hist[0] else None,
                    line=dict(color=FG_COLORS["fair_line"], width=2.5, dash="dash"),
                    mode="lines",
                    showlegend=not fair_hist[0],
                ),
                row=1, col=1,
            )
        _add_marker_traces(
            fig,
            act_x=fair_act_x, act_y=fair_act_y,
            est_x=fair_est_x, est_y=fair_est_y,
            fill_color=FG_COLORS["fair_marker_fill"],
            line_color=FG_COLORS["fair_marker_line"],
            symbol="triangle-up",
            size=8,
            hover_label="Fair",
        )

    if norm_step_x:
        fig.add_trace(
            go.Scatter(
                x=norm_step_x, y=norm_step_y,
                name=f"Normal P/E Ratio ({norm_pe:.2f}x)",
                line=dict(color=FG_COLORS["normal_line"], width=2, dash="dash"),
                mode="lines",
            ),
            row=1, col=1,
        )
        _add_marker_traces(
            fig,
            act_x=norm_act_x, act_y=norm_act_y,
            est_x=norm_est_x, est_y=norm_est_y,
            fill_color=FG_COLORS["normal_marker_fill"],
            line_color=FG_COLORS["normal_marker_line"],
            symbol="circle",
            size=7,
            hover_label="Normal",
        )

    fig.add_trace(
        go.Scatter(
            x=closes.index, y=closes.values,
            name="Price",
            line=dict(color=FG_COLORS["price_line"], width=2),
            mode="lines",
        ),
        row=1, col=1,
    )

    y_lo, y_hi = _compute_fast_graph_y_range(closes, fair_step_y, norm_step_y, div_step_y)

    eps_table = _annual_eps_table_rows(
        eps_points,
        estimates,
        include_estimates=True,
    )
    fy_tickvals = [_fiscal_timestamp(y) for y, _, _ in eps_points]
    fy_ticktext = [_fy_label(y, is_estimate=est) for y, _, est in eps_points]
    x_range = [fair_step_x[0], fair_step_x[-1]] if fair_step_x else None

    table_annotations = _add_aligned_fy_table(
        eps_table,
        eps_points,
        annual_dividends,
        fallback_div=fallback_div,
        is_forecast=is_forecast,
    )

    title_suffix = "Forecasting" if is_forecast else "Historical"
    company = info.get("company_name") or metrics.get("company_name") or ""
    fig.update_layout(
        title=dict(text=f"{company} — FAST Graph ({title_suffix})", font=dict(color="#e8eaed", size=14)),
        template="plotly_dark",
        paper_bgcolor=FG_COLORS["paper_bg"],
        plot_bgcolor=FG_COLORS["plot_bg"],
        height=height,
        margin=dict(l=58, r=30, t=50, b=36),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color="#c0c0c0")),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=FG_COLORS["grid"],
        range=x_range,
        tickvals=fy_tickvals,
        ticktext=fy_ticktext,
        tickangle=0,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        matches="x",
        showticklabels=False,
        showgrid=False,
        range=x_range,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title=dict(text=f"Price ({currency})", font=dict(color="#9aa0a6")),
        tickfont=dict(color="#9aa0a6"),
        showgrid=True,
        gridcolor=FG_COLORS["grid"],
        range=[y_lo, y_hi],
        autorange=False,
        zeroline=True,
        zerolinecolor="rgba(255,255,255,0.15)",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        range=[-1.4, 3.6],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        row=2,
        col=1,
    )
    # Invisible anchor trace so the FY table subplot shares the chart x-axis.
    if fy_tickvals:
        fig.add_trace(
            go.Scatter(
                x=fy_tickvals,
                y=[0.0] * len(fy_tickvals),
                mode="markers",
                marker=dict(size=0.1, opacity=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )

    display_growth = (
        metrics.get("chart_forecast_growth_rate") or forecast_growth
        if is_forecast
        else metrics.get("chart_historical_growth_rate") or hist_growth
    )

    annotations: list[dict] = []
    if display_growth is not None:
        annotations.append(_metric_box_annotation(
            y=0.98,
            label="Growth Rate",
            value=f"{display_growth:.2f}%",
            color=FG_COLORS["growth_accent"],
        ))
    annotations.append(_metric_box_annotation(
        y=0.91,
        label="Fair Value Ratio",
        value=f"{fair_pe:.2f}x",
        color=FG_COLORS["fair_line"],
    ))
    annotations.append(_metric_box_annotation(
        y=0.84,
        label="Normal P/E Ratio",
        value=f"{norm_pe:.2f}x",
        color=FG_COLORS["normal_line"],
    ))

    if is_forecast:
        ror = metrics.get("est_annual_ror")
        if ror is not None:
            annotations.append(_metric_box_annotation(
                y=0.77,
                label="Est. Annual ROR",
                value=f"{ror:.2f}%",
                color=FG_COLORS["ror_accent"],
            ))
        fp = metrics.get("future_price")
        if fp is not None:
            annotations.append(_metric_box_annotation(
                y=0.70,
                label="Future Price",
                value=f"{cur_sym}{fp:.2f}",
                color=FG_COLORS["fair_line"],
            ))

        if closes.index.size:
            today = pd.Timestamp.now().normalize()
            if closes.index.min() <= today <= closes.index.max() + pd.Timedelta(days=7):
                fig.add_shape(
                    type="line",
                    x0=today,
                    x1=today,
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="y domain",
                    line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"),
                    row=1,
                    col=1,
                )

    annotations.extend(table_annotations)
    if annotations:
        fig.update_layout(annotations=annotations)

    return fig


def build_fg_radar_figure(fg_axes: dict[str, float], *, height: int = 320) -> go.Figure | None:
    """Pentagon radar chart for FG score axes."""
    if not fg_axes:
        return None
    labels = [
        "Cash Flow Generation",
        "Financial Strength",
        "Growth",
        "Predictability",
        "Profitability",
    ]
    values = [fg_axes.get(l, 50) for l in labels]
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(77, 163, 255, 0.30)",
        line=dict(color=FG_COLORS["normal_line"], width=2),
        name="FG Score",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.1)"),
            bgcolor=FG_COLORS["plot_bg"],
        ),
        template="plotly_dark",
        paper_bgcolor=FG_COLORS["paper_bg"],
        height=height,
        margin=dict(l=40, r=40, t=30, b=30),
        showlegend=False,
        title=dict(
            text=f"FG Score: {fg_axes.get('FG Score', '—')}/100 (Yahoo approx.)",
            font=dict(size=12, color="#9aa0a6"),
        ),
    )
    return fig
