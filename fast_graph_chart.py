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
    compute_historical_growth_rate_pct,
    compute_forecast_growth_pct,
    resolve_fair_pe,
)
from eps_yield import avg_historical_pe_5y


def _annual_eps_table_rows(
    annual_eps: dict[int, float],
    eps_points: list[tuple[int, float, bool]],
    estimates: dict[str, Any],
) -> pd.DataFrame:
    """FY / EPS / Chg/Yr table including estimate years."""
    if not eps_points:
        return pd.DataFrame(columns=["year", "eps", "chg_yr", "analysts"])

    rows = []
    prev: float | None = None
    est_0y = (estimates or {}).get("0y", {})
    est_1y = (estimates or {}).get("+1y", {})

    for year, eps, is_est in eps_points:
        chg = None
        if prev is not None and prev != 0:
            chg = round((eps - prev) / abs(prev) * 100.0, 2)
        analysts = None
        if is_est:
            if year == eps_points[-1][0] and est_1y.get("numberOfAnalysts"):
                analysts = est_1y.get("numberOfAnalysts")
            elif est_0y.get("numberOfAnalysts"):
                analysts = est_0y.get("numberOfAnalysts")
        rows.append({
            "year": year,
            "eps": eps,
            "chg_yr": chg,
            "analysts": analysts,
            "is_est": is_est,
        })
        prev = eps
    return pd.DataFrame(rows)


def _step_series_from_eps_points(
    eps_points: list[tuple[int, float, bool]],
    price_index: pd.DatetimeIndex,
    *,
    pe_multiple: float,
    fiscal_month: int = 11,
    fiscal_day: int = 30,
) -> tuple[list, list]:
    """Step-interpolated valuation line (FY gaps_off) across the price timeline."""
    if not eps_points or pe_multiple <= 0:
        return [], []

    boundaries: list[tuple[pd.Timestamp, float]] = []
    for year, eps, _ in eps_points:
        ts = pd.Timestamp(year=year, month=fiscal_month, day=fiscal_day)
        boundaries.append((ts, eps * pe_multiple))
    boundaries.sort(key=lambda x: x[0])

    x_out: list = []
    y_out: list = []
    idx = 0
    current_val = boundaries[0][1] if boundaries else 0.0

    for ts in price_index:
        while idx + 1 < len(boundaries) and ts >= boundaries[idx + 1][0]:
            idx += 1
            current_val = boundaries[idx][1]
        x_out.append(ts)
        y_out.append(current_val)
    return x_out, y_out


def _compute_fast_graph_y_range(
    closes: pd.Series,
    fair_y: list[float],
    norm_y: list[float],
    *,
    price_max_multiplier: float = 3.0,
) -> tuple[float, float]:
    """Price-first y-axis with capped extension for valuation lines."""
    price_lo = float(closes.min())
    price_hi = float(closes.max())
    span = max(price_hi - price_lo, price_hi * 0.05)
    pad = max(span * 0.12, price_hi * 0.03)
    y_lo = max(0.0, price_lo - pad)

    y_hi = price_hi + pad
    if fair_y or norm_y:
        val_max = max(fair_y + norm_y) if (fair_y or norm_y) else price_hi
        cap = price_hi * price_max_multiplier
        y_hi = max(y_hi, min(val_max, cap))
    return y_lo, y_hi


def _dividend_scaled_series(
    eps_points: list[tuple[int, float, bool]],
    price_index: pd.DatetimeIndex,
    *,
    fair_pe: float,
    dividend_rate: float | None,
    fiscal_month: int = 11,
    fiscal_day: int = 30,
) -> tuple[list, list]:
    """Annual dividend × fair_pe stepped across timeline (Pine-style green band top)."""
    if not eps_points or fair_pe <= 0:
        return [], []

    div_per_share = float(dividend_rate) if dividend_rate and dividend_rate > 0 else 0.0
    if div_per_share <= 0:
        return [price_index[0], price_index[-1]], [0.0, 0.0]

    scaled = div_per_share * fair_pe
    boundaries: list[tuple[pd.Timestamp, float]] = []
    for year, _, _ in eps_points:
        ts = pd.Timestamp(year=year, month=fiscal_month, day=fiscal_day)
        boundaries.append((ts, scaled))
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


def _mode_pe(metrics: dict[str, Any], mode: str, df_daily, annual_eps) -> tuple[float, float | None, float | None]:
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
    return float(fair_pe), norm_pe, growth


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

    years_ahead = 4 if mode == "forecast" else 2
    eps_points = _estimate_eps_chain(
        annual_eps,
        estimates,
        years_ahead=years_ahead,
        growth_rate=proj_growth,
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.06,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
    )

    fair_x, fair_y = _step_series_from_eps_points(eps_points, closes.index, pe_multiple=fair_pe)
    norm_x, norm_y = _step_series_from_eps_points(eps_points, closes.index, pe_multiple=float(norm_pe or fair_pe))

    dividend_rate = info.get("dividend_rate") or metrics.get("dividend_rate")
    div_x, div_y = _dividend_scaled_series(
        eps_points,
        closes.index,
        fair_pe=fair_pe,
        dividend_rate=dividend_rate,
    )

    if div_x:
        fig.add_trace(
            go.Scatter(
                x=div_x,
                y=[0.0] * len(div_x),
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=div_x,
                y=div_y,
                fill="tonexty",
                fillcolor="rgba(76, 175, 80, 0.35)",
                line=dict(width=0),
                name="Dividends",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    dash = "dash" if mode == "forecast" else "solid"
    if fair_x:
        fig.add_trace(
            go.Scatter(
                x=fair_x,
                y=fair_y,
                name=f"Fair Value ({fair_pe:.1f}x)",
                line=dict(color="#ffb74d", width=2, dash=dash),
                mode="lines",
            ),
            row=1,
            col=1,
        )
    if norm_x:
        fig.add_trace(
            go.Scatter(
                x=norm_x,
                y=norm_y,
                name=f"Normal P/E ({float(norm_pe):.1f}x)",
                line=dict(color="#64b5f6", width=2, dash=dash),
                mode="lines",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=closes.index,
            y=closes.values,
            name="Price",
            line=dict(color="#ffffff", width=2.5),
            mode="lines",
        ),
        row=1,
        col=1,
    )

    y_lo, y_hi = _compute_fast_graph_y_range(closes, fair_y, norm_y)

    eps_table = _annual_eps_table_rows(annual_eps, eps_points, estimates)
    if not eps_table.empty:
        fy_labels = []
        eps_labels = []
        for _, row in eps_table.iterrows():
            suffix = "E" if row["is_est"] else ""
            fy_labels.append(f"{int(row['year']) % 100:02d}/{str(int(row['year']))[-2:]}{suffix}")
            eps_labels.append(f"{row['eps']:.2f}{suffix}")

        header = ["FY", "EPS", "Chg/Yr %"]
        cell_rows = [fy_labels, eps_labels, [
            f"{x:.1f}" if x is not None and x == x else "—"
            for x in eps_table["chg_yr"]
        ]]
        if mode == "forecast" and eps_table["analysts"].notna().any():
            header.append("# Analysts")
            cell_rows.append([
                str(int(a)) if a is not None and a == a else "—"
                for a in eps_table["analysts"]
            ])

        fig.add_trace(
            go.Table(
                header=dict(
                    values=header,
                    fill_color="#1a1a1a",
                    font=dict(color="#e8eaed", size=11),
                    align="center",
                ),
                cells=dict(
                    values=cell_rows,
                    fill_color="#0a0a0a",
                    font=dict(color="#c0c0c0", size=10),
                    align="center",
                ),
            ),
            row=2,
            col=1,
        )

    title_suffix = "Forecasting" if mode == "forecast" else "Historical"
    company = info.get("company_name") or metrics.get("company_name") or ""
    fig.update_layout(
        title=dict(text=f"{company} — FAST Graph ({title_suffix})", font=dict(color="#e8eaed", size=14)),
        template="plotly_dark",
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#111111",
        height=height,
        margin=dict(l=50, r=30, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(
            title=f"Price ({currency})",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            range=[y_lo, y_hi],
            autorange=False,
        ),
    )

    annotations: list[dict] = []
    if mode == "forecast":
        ror = metrics.get("est_annual_ror")
        if ror is not None:
            annotations.append(dict(
                x=0.98, y=0.95, xref="paper", yref="paper",
                text=f"Est. Annual ROR: {ror:.2f}%",
                showarrow=False,
                font=dict(size=13, color="#7cb342"),
                bgcolor="rgba(0,0,0,0.6)",
                bordercolor="#7cb342",
            ))
        fp = metrics.get("future_price")
        if fp is not None:
            annotations.append(dict(
                x=0.98, y=0.88, xref="paper", yref="paper",
                text=f"Future Price: {cur_sym}{fp:.2f}",
                showarrow=False,
                font=dict(size=11, color="#ffb74d"),
                bgcolor="rgba(0,0,0,0.5)",
            ))

    if closes.index.size:
        today = pd.Timestamp.now().normalize()
        if mode == "forecast" and closes.index.min() <= today <= closes.index.max():
            annotations.append(dict(
                x=today,
                y=y_hi,
                xref="x",
                yref="y",
                text="Today",
                showarrow=False,
                yanchor="top",
                font=dict(size=9, color="#9aa0a6"),
            ))
            fig.add_vline(
                x=today,
                line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"),
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
        fillcolor="rgba(100, 181, 246, 0.35)",
        line=dict(color="#64b5f6", width=2),
        name="FG Score",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.1)"),
            bgcolor="#111111",
        ),
        template="plotly_dark",
        paper_bgcolor="#0a0a0a",
        height=height,
        margin=dict(l=40, r=40, t=30, b=30),
        showlegend=False,
        title=dict(
            text=f"FG Score: {fg_axes.get('FG Score', '—')}/100 (Yahoo approx.)",
            font=dict(size=12, color="#9aa0a6"),
        ),
    )
    return fig
