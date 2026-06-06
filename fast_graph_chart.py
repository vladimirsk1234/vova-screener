"""
FAST Graphs–style Plotly charts (Historical + Forecasting). No Streamlit.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fast_graph_metrics import eps_cagr_pct, project_future_eps, resolve_fair_pe
from eps_yield import avg_historical_pe_5y


def _annual_eps_series(annual_eps: dict[int, float]) -> pd.DataFrame:
    if not annual_eps:
        return pd.DataFrame(columns=["year", "eps", "chg_yr"])
    years = sorted(annual_eps.keys())
    rows = []
    prev = None
    for y in years:
        eps = float(annual_eps[y])
        chg = None
        if prev is not None and prev != 0:
            chg = round((eps - prev) / abs(prev) * 100.0, 2)
        rows.append({"year": y, "eps": eps, "chg_yr": chg})
        prev = eps
    return pd.DataFrame(rows)


def _projected_eps_points(
    annual_eps: dict[int, float],
    estimates: dict[str, Any],
    *,
    years_ahead: int = 3,
    growth_rate: float | None,
) -> list[tuple[int, float, bool]]:
    """Return list of (year, eps, is_estimate)."""
    points: list[tuple[int, float, bool]] = []
    for y, e in sorted(annual_eps.items()):
        points.append((y, e, False))

    est_0y = (estimates or {}).get("0y", {})
    est_1y = (estimates or {}).get("+1y", {})
    last_year = max(annual_eps.keys()) if annual_eps else pd.Timestamp.now().year

    if est_0y.get("avg"):
        points.append((last_year + 1, float(est_0y["avg"]), True))
    if est_1y.get("avg"):
        points.append((last_year + 2, float(est_1y["avg"]), True))

    base = est_1y.get("avg") or est_0y.get("avg")
    if base and years_ahead > 2:
        for i in range(3, years_ahead + 1):
            proj = project_future_eps(float(base), years=i - 2, growth_rate=growth_rate)
            if proj:
                points.append((last_year + i, proj, True))
    return points


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
    annual_eps = metrics.get("annual_eps") or {}
    if isinstance(annual_eps, dict) and annual_eps and isinstance(next(iter(annual_eps.keys())), str):
        annual_eps = {int(k): float(v) for k, v in annual_eps.items()}

    estimates = bundle.get("earnings_estimates") or metrics.get("earnings_estimates") or {}
    growth_rate = metrics.get("growth_rate") or eps_cagr_pct(annual_eps)
    fair_pe = metrics.get("fair_pe") or resolve_fair_pe(growth_rate)
    norm_pe = metrics.get("normal_pe") or avg_historical_pe_5y(df_daily, annual_eps) or fair_pe

    price_df = df_weekly.copy()
    price_df.index = pd.to_datetime(price_df.index)
    closes = pd.to_numeric(price_df["Close"], errors="coerce").dropna()

    eps_points = _projected_eps_points(
        annual_eps,
        estimates,
        years_ahead=4 if mode == "forecast" else 2,
        growth_rate=metrics.get("est_eps_growth") or growth_rate,
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.06,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
    )

    fig.add_trace(
        go.Scatter(
            x=closes.index,
            y=closes.values,
            name="Price",
            line=dict(color="#ffffff", width=1.5),
            mode="lines",
        ),
        row=1,
        col=1,
    )

    fair_x, fair_y, norm_x, norm_y = [], [], [], []
    for year, eps, is_est in eps_points:
        ts = pd.Timestamp(year=year, month=11, day=30)
        fair_x.append(ts)
        fair_y.append(eps * fair_pe)
        norm_x.append(ts)
        norm_y.append(eps * norm_pe)

    if fair_x:
        dash = "dash" if mode == "forecast" else "solid"
        fig.add_trace(
            go.Scatter(
                x=fair_x,
                y=fair_y,
                name=f"Fair Value ({fair_pe:.1f}x)",
                line=dict(color="#ffb74d", width=2, dash=dash),
                mode="lines+markers",
                marker=dict(size=6, symbol="triangle-up", color="#ffb74d"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=norm_x,
                y=norm_y,
                name=f"Normal P/E ({norm_pe:.1f}x)" if norm_pe else "Normal P/E",
                line=dict(color="#64b5f6", width=2, dash=dash),
                mode="lines+markers",
                marker=dict(size=5, color="#64b5f6"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=fair_x,
                y=[0] * len(fair_x),
                name="EPS base",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=fair_x,
                y=fair_y,
                fill="tonexty",
                fillcolor="rgba(76, 175, 80, 0.25)",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    eps_table = _annual_eps_series(annual_eps)
    if not eps_table.empty:
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["FY", "EPS", "Chg/Yr %"],
                    fill_color="#1a1a1a",
                    font=dict(color="#e8eaed", size=11),
                    align="center",
                ),
                cells=dict(
                    values=[
                        eps_table["year"].astype(str),
                        eps_table["eps"].map(lambda x: f"{x:.2f}"),
                        eps_table["chg_yr"].map(lambda x: f"{x:.1f}" if x is not None else "—"),
                    ],
                    fill_color="#0a0a0a",
                    font=dict(color="#c0c0c0", size=10),
                    align="center",
                ),
            ),
            row=2,
            col=1,
        )

    title_suffix = "Forecasting" if mode == "forecast" else "Historical"
    company = (bundle.get("info") or {}).get("company_name") or metrics.get("company_name") or ""
    fig.update_layout(
        title=dict(text=f"{company} — FAST Graph ({title_suffix})", font=dict(color="#e8eaed", size=14)),
        template="plotly_dark",
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#111111",
        height=height,
        margin=dict(l=50, r=30, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Price", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
    )

    if mode == "forecast":
        annotations = []
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
                text=f"Future Price: ${fp:.2f}",
                showarrow=False,
                font=dict(size=11, color="#ffb74d"),
                bgcolor="rgba(0,0,0,0.5)",
            ))
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
