"""
FAST Graphs–style fundamentals panel for Streamlit (HTML + expander).
"""
from __future__ import annotations

import html
from typing import Any

import streamlit as st


def _esc(text: str) -> str:
    return html.escape(str(text or ""), quote=True)


def render_fast_graph_panel(data: dict[str, Any] | None) -> None:
    """Render highlight boxes + detail table + Yahoo metrics expander."""
    if not data:
        return

    company = _esc(data.get("company_name", data.get("ticker", "")))
    ticker = _esc(data.get("ticker", ""))

    highlight_html = []
    for item in data.get("highlights") or []:
        css = _esc(item.get("css", "growth"))
        label = _esc(item.get("label", ""))
        value = _esc(item.get("value", "N/A"))
        highlight_html.append(
            f'<div class="fg-highlight fg-highlight-{css}">'
            f'<div class="fg-highlight-label">{label}</div>'
            f'<div class="fg-highlight-value">{value}</div>'
            f"</div>"
        )

    rows_html = []
    for label, value in data.get("details") or []:
        rows_html.append(
            f'<tr><td class="fg-detail-label">{_esc(label)}</td>'
            f'<td class="fg-detail-value">{_esc(value)}</td></tr>'
        )

    panel_html = f"""
    <div class="fg-panel">
        <div class="fg-panel-title">{company} <span class="fg-panel-ticker">({ticker})</span></div>
        <div class="fg-highlights">{"".join(highlight_html)}</div>
        <table class="fg-details-table">
            <tbody>{"".join(rows_html)}</tbody>
        </table>
    </div>
    """

    st.markdown("**Fundamentals (Yahoo / FAST-style)**")
    st.markdown(panel_html, unsafe_allow_html=True)

    warnings = data.get("warnings") or []
    if warnings:
        st.caption(" · ".join(str(w) for w in warnings))

    extended = data.get("extended") or []
    if extended:
        with st.expander("Все метрики Yahoo", expanded=False):
            st.dataframe(
                {"Metric": [label for label, _ in extended], "Value": [val for _, val in extended]},
                hide_index=True,
                width="stretch",
            )
