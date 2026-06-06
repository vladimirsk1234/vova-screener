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
    "earnings_fill": "rgba(76, 160, 55, 0.50)",
    "earnings_line": "#52A844",
    "dividend_fill": "rgba(30, 95, 30, 0.75)",
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
    years_ahead = 4 if mode == "forecast" else 4
    # Forward EPS from analysts; projection growth only fills beyond +1y.
    eps_growth = forecast_growth if mode == "forecast" else forecast_growth
    return _estimate_eps_chain(
        annual_eps,
        estimates,
        years_ahead=years_ahead,
        growth_rate=eps_growth,
    )


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


def _extend_to_price_end(
    eps_points: list[tuple[int, float, bool]],
    price_index: pd.DatetimeIndex,
    *,
    pe_multiple: float,
) -> tuple[list, list]:
    """
    FAST Graphs gaps_off: EPS × P/E held from each fiscal year-end until the next,
    then extend the last value through the end of the price series.
    """
    if not eps_points or pe_index_empty(price_index) or pe_multiple <= 0:
        return [], []

    boundaries = [
        (_fiscal_timestamp(year), eps * pe_multiple)
        for year, eps, _ in eps_points
    ]
    boundaries.sort(key=lambda x: x[0])

    x_out: list = []
    y_out: list = []
    idx = 0
    current_val = boundaries[0][1]

    for ts in price_index:
        while idx + 1 < len(boundaries) and ts >= boundaries[idx + 1][0]:
            idx += 1
            current_val = boundaries[idx][1]
        x_out.append(ts)
        y_out.append(current_val)
    return x_out, y_out


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


def _compute_fast_graph_y_range(
    closes: pd.Series,
    fair_y: list[float],
    norm_y: list[float],
    *,
    price_max_multiplier: float = 2.5,
) -> tuple[float, float]:
    """Price-first y-axis; extend slightly to show valuation lines near price."""
    price_lo = float(closes.min())
    price_hi = float(closes.max())
    span = max(price_hi - price_lo, price_hi * 0.05)
    pad = max(span * 0.15, price_hi * 0.05)
    y_lo = max(0.0, price_lo - pad * 0.5)

    y_hi = price_hi + pad
    if fair_y or norm_y:
        in_window = []
        for y in fair_y + norm_y:
            if y <= price_hi * price_max_multiplier:
                in_window.append(y)
        if in_window:
            val_max = max(in_window)
            cap = price_hi * price_max_multiplier
            y_hi = max(y_hi, min(val_max + pad * 0.3, cap))
    return y_lo, y_hi


def _mode_pe(
    metrics: dict[str, Any],
    mode: str,
    df_daily,
    annual_eps,
) -> tuple[float, float, float | None]:
    """Return (fair_pe, norm_pe, growth_rate) for chart mode."""
    if mode == "forecast":
        growth = metrics.get("forecast_growth_rate") or metrics.get("est_eps_growth")
        fair_pe = metrics.get("forecast_fair_pe") or resolve_fair_pe(growth)
        norm_pe = metrics.get("forecast_normal_pe") or metrics.get("normal_pe")
    else:
        growth = metrics.get("historical_growth_rate") or metrics.get("growth_rate")
        if growth is None:
            growth = compute_historical_growth_rate_pct(annual_eps)
        fair_pe = metrics.get("historical_fair_pe") or metrics.get("fair_pe") or resolve_fair_pe(growth)
        norm_pe = metrics.get("historical_normal_pe") or metrics.get("normal_pe")

    if norm_pe is None:
        norm_pe = avg_historical_pe_5y(df_daily, annual_eps)
    if norm_pe is None:
        norm_pe = fair_pe
    return float(fair_pe), float(norm_pe), growth


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
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
    )

    dividend_rate = info.get("dividend_rate") or metrics.get("dividend_rate")
    div_step_x, div_step_y = _extend_to_price_end(
        eps_points,
        closes.index,
        pe_multiple=(float(dividend_rate) * fair_pe if dividend_rate and dividend_rate > 0 else 0.0),
    )
    fair_step_x, fair_step_y = _extend_to_price_end(eps_points, closes.index, pe_multiple=fair_pe)
    norm_step_x, norm_step_y = _extend_to_price_end(eps_points, closes.index, pe_multiple=norm_pe)

    # Layer 1: dark green dividend band (0 → div×fair_pe).
    if div_step_x and any(y > 0 for y in div_step_y):
        fig.add_trace(
            go.Scatter(x=div_step_x, y=[0.0] * len(div_step_x), line=dict(width=0), showlegend=False, hoverinfo="skip"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=div_step_x, y=div_step_y, fill="tonexty",
                fillcolor=FG_COLORS["dividend_fill"],
                line=dict(width=0), name="Dividends", hoverinfo="skip",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=div_step_x, y=div_step_y,
                line=dict(color=FG_COLORS["dividend_line"], width=1),
                mode="lines", showlegend=False, hoverinfo="skip",
            ),
            row=1, col=1,
        )

    # Layer 2: earnings green slope (div top → fair value, or 0 → fair if no div).
    has_div = div_step_x and any(y > 0 for y in div_step_y)
    base_y = div_step_y if has_div else [0.0] * len(fair_step_x)
    if fair_step_x and fair_step_y:
        fig.add_trace(
            go.Scatter(x=fair_step_x, y=base_y, line=dict(width=0), showlegend=False, hoverinfo="skip"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=fair_step_x, y=fair_step_y, fill="tonexty",
                fillcolor=FG_COLORS["earnings_fill"],
                line=dict(color=FG_COLORS["earnings_line"], width=1),
                name="Earnings", showlegend=False, hoverinfo="skip",
            ),
            row=1, col=1,
        )

    fair_mk_x, fair_mk_y, _fair_est = _annual_marker_series(eps_points, pe_multiple=fair_pe)
    norm_mk_x, norm_mk_y, _norm_est = _annual_marker_series(eps_points, pe_multiple=norm_pe)

    is_forecast = mode == "forecast"
    fair_dash = "dash" if is_forecast else "solid"
    norm_dash = "dot"

    if fair_step_x:
        fig.add_trace(
            go.Scatter(
                x=fair_step_x, y=fair_step_y,
                name=f"Fair Value ({fair_pe:.2f}x)",
                line=dict(color=FG_COLORS["fair_line"], width=2.5, dash=fair_dash),
                mode="lines",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=fair_mk_x, y=fair_mk_y,
                mode="markers",
                marker=dict(
                    size=8, symbol="triangle-up",
                    color=FG_COLORS["fair_marker_fill"],
                    line=dict(width=1.5, color=FG_COLORS["fair_marker_line"]),
                ),
                showlegend=False,
                hovertemplate="FY %{x|%b %Y}<br>Fair: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    if norm_step_x:
        fig.add_trace(
            go.Scatter(
                x=norm_step_x, y=norm_step_y,
                name=f"Normal P/E ({norm_pe:.2f}x)",
                line=dict(color=FG_COLORS["normal_line"], width=2, dash=norm_dash),
                mode="lines",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=norm_mk_x, y=norm_mk_y,
                mode="markers",
                marker=dict(
                    size=7, symbol="circle",
                    color=FG_COLORS["normal_marker_fill"],
                    line=dict(width=1.5, color=FG_COLORS["normal_marker_line"]),
                ),
                showlegend=False,
                hovertemplate="FY %{x|%b %Y}<br>Normal: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
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

    y_lo, y_hi = _compute_fast_graph_y_range(closes, fair_step_y, norm_step_y)

    # Horizontal table: columns = fiscal years (FAST Graphs layout).
    eps_table = _annual_eps_table_rows(
        eps_points,
        estimates,
        include_estimates=True,
    )
    if not eps_table.empty:
        fy_labels = list(eps_table["fy"])
        table_cols: list[list[str]] = [["EPS", "Chg/Yr %"]]
        for _, row in eps_table.iterrows():
            chg = row["chg_yr"]
            chg_s = f"{chg:.1f}" if chg is not None and chg == chg else "—"
            eps_s = f"{row['eps']:.2f}"
            if row["is_est"] and not eps_s.endswith("E"):
                eps_s += "E"
            table_cols.append([eps_s, chg_s])

        if is_forecast and eps_table["analysts"].notna().any():
            table_cols[0].append("# Analysts")
            for i, (_, row) in enumerate(eps_table.iterrows(), start=1):
                a = row["analysts"]
                table_cols[i].append(str(int(a)) if a is not None and a == a else "—")

        fig.add_trace(
            go.Table(
                header=dict(
                    values=["FY"] + fy_labels,
                    fill_color="#1a1a1a",
                    font=dict(color="#e8eaed", size=11),
                    align="center",
                ),
                cells=dict(
                    values=table_cols,
                    fill_color="#0a0a0a",
                    font=dict(color="#c0c0c0", size=10),
                    align="center",
                ),
            ),
            row=2,
            col=1,
        )

    title_suffix = "Forecasting" if is_forecast else "Historical"
    company = info.get("company_name") or metrics.get("company_name") or ""
    fig.update_layout(
        title=dict(text=f"{company} — FAST Graph ({title_suffix})", font=dict(color="#e8eaed", size=14)),
        template="plotly_dark",
        paper_bgcolor=FG_COLORS["paper_bg"],
        plot_bgcolor=FG_COLORS["plot_bg"],
        height=height,
        margin=dict(l=50, r=30, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color="#c0c0c0")),
        xaxis=dict(showgrid=True, gridcolor=FG_COLORS["grid"]),
        yaxis=dict(
            title=dict(text=f"Price ({currency})", font=dict(color="#9aa0a6")),
            tickfont=dict(color="#9aa0a6"),
            showgrid=True,
            gridcolor=FG_COLORS["grid"],
            range=[y_lo, y_hi],
            autorange=False,
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.15)",
        ),
    )

    annotations: list[dict] = []
    if is_forecast:
        ror = metrics.get("est_annual_ror")
        if ror is not None:
            annotations.append(dict(
                x=0.98, y=0.95, xref="paper", yref="paper",
                text=f"Est. Annual ROR: {ror:.2f}%",
                showarrow=False,
                font=dict(size=13, color=FG_COLORS["ror_accent"]),
                bgcolor="rgba(0,0,0,0.6)",
                bordercolor=FG_COLORS["ror_accent"],
            ))
        fp = metrics.get("future_price")
        if fp is not None:
            annotations.append(dict(
                x=0.98, y=0.88, xref="paper", yref="paper",
                text=f"Future Price: {cur_sym}{fp:.2f}",
                showarrow=False,
                font=dict(size=11, color=FG_COLORS["fair_line"]),
                bgcolor="rgba(0,0,0,0.5)",
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
