"""
Extended FAST Graphs panel: ROR highlights, FG score bars, analyst beat summary.
"""
from __future__ import annotations

import html
from typing import Any

import streamlit as st

from fundamentals_panel_ui import render_fast_graph_panel


def _esc(text: str) -> str:
    return html.escape(str(text or ""), quote=True)


def _score_bar(label: str, score: float) -> str:
    pct = max(0, min(100, float(score)))
    if pct >= 70:
        color = "#7cb342"
    elif pct >= 50:
        color = "#ffb74d"
    else:
        color = "#e57373"
    return (
        f'<div class="fg-score-row">'
        f'<span class="fg-score-label">{_esc(label)}</span>'
        f'<div class="fg-score-track"><div class="fg-score-fill" style="width:{pct}%;background:{color};"></div></div>'
        f'<span class="fg-score-val">{pct:.0f}</span>'
        f"</div>"
    )


def render_fast_graph_extended_panel(
    panel_data: dict[str, Any] | None,
    metrics: dict[str, Any] | None = None,
    *,
    chart_mode: str = "historical",
) -> None:
    """Render base FAST panel plus ROR/FG/analyst extensions."""
    if panel_data:
        render_fast_graph_panel(panel_data)

    if not metrics:
        return

    currency = (panel_data or {}).get("currency") or "USD"
    cur_prefix = f"{currency} " if currency else "$"

    extras = []
    ror = metrics.get("est_annual_ror")
    if ror is not None and chart_mode == "forecast":
        extras.append(
            f'<div class="fg-highlight fg-highlight-growth">'
            f'<div class="fg-highlight-label">Est. Annual ROR</div>'
            f'<div class="fg-highlight-value">{ror:.2f}%</div></div>'
        )
    fp = metrics.get("future_price")
    if fp is not None and chart_mode == "forecast":
        extras.append(
            f'<div class="fg-highlight fg-highlight-fair">'
            f'<div class="fg-highlight-label">Future Price</div>'
            f'<div class="fg-highlight-value">{cur_prefix}{fp:.2f}</div></div>'
        )
    if extras:
        st.markdown(
            f'<div class="fg-highlights">{"".join(extras)}</div>',
            unsafe_allow_html=True,
        )

    fg_axes = metrics.get("fg_axes") or {}
    if fg_axes:
        bars = []
        for label in (
            "Profitability",
            "Growth",
            "Financial Strength",
            "Cash Flow Generation",
            "Predictability",
        ):
            if label in fg_axes:
                bars.append(_score_bar(label, fg_axes[label]))
        if bars:
            st.markdown("**FG Score (Yahoo approximation)**")
            st.markdown(f'<div class="fg-score-bars">{"".join(bars)}</div>', unsafe_allow_html=True)

    beat_pct = metrics.get("analyst_beat_pct")
    history = (metrics.get("bundle") or {}).get("earnings_history") or []
    if beat_pct is not None or history:
        with st.expander("Analyst earnings accuracy", expanded=False):
            if beat_pct is not None:
                st.caption(f"Beat rate (recent quarters): {beat_pct:.1f}%")
            if history:
                rows = []
                for h in history[:8]:
                    rows.append({
                        "Date": h.get("date", ""),
                        "Estimate": h.get("eps_estimate"),
                        "Actual": h.get("eps_actual"),
                        "Beat": "Yes" if h.get("beat") else "No",
                    })
                st.dataframe(rows, hide_index=True, width="stretch")
