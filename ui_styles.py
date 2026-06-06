"""
Terminal UI styles for the screener. Injects dark-theme CSS via Streamlit.
"""
import streamlit as st

_STYLES = """
<style>
    /* GLOBAL DARK THEME */
    .stApp { background-color: #050505; }
    section[data-testid="stMain"] {
        min-height: calc(100vh - 6rem);
    }

    .block-container {
        padding-top: max(4rem, env(safe-area-inset-top, 0px)) !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
    }

    @media (max-width: 480px) {
        .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
    }

    /* Results table: horizontal scroll on narrow screens */
    div[data-testid="stDataFrame"] {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }

    /* Sidebar: touch-friendly tap targets (iPhone) */
    @media (max-width: 768px) {
        section[data-testid="stSidebar"] button,
        section[data-testid="stSidebar"] [role="radiogroup"] label {
            min-height: 44px;
        }
    }

    /* Scan phase progress (web + mobile) */
    @keyframes scanPulse {
        0%, 100% { box-shadow: 0 0 8px rgba(0, 255, 136, 0.25); }
        50% { box-shadow: 0 0 16px rgba(0, 255, 136, 0.55); }
    }

    @keyframes scanShimmer {
        0% { transform: translateX(-120%); }
        100% { transform: translateX(320%); }
    }

    @keyframes scanThink {
        0%, 100% { opacity: 0.35; }
        50% { opacity: 1; }
    }

    .scan-phases {
        display: flex;
        flex-direction: column;
        gap: 0.85rem;
        padding: 0.5rem 0 1rem;
        max-width: 720px;
    }

    .scan-phase-row {
        display: flex;
        flex-direction: column;
        gap: 0.45rem;
    }

    .scan-phase-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem;
    }

    .scan-phase-label {
        color: #e8e8e8;
        font-size: 0.92rem;
        font-weight: 500;
        letter-spacing: 0.02em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .scan-phase-status {
        font-size: 1.15rem;
        line-height: 1;
        min-width: 1.5rem;
        text-align: center;
        flex-shrink: 0;
    }

    .scan-phase-track {
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }

    .scan-phase-bar {
        flex: 1;
        height: 8px;
        background: #111;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 999px;
        overflow: hidden;
    }

    .scan-phase-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #00d4aa, #00ff88);
        box-shadow: 0 0 12px rgba(0, 255, 136, 0.35);
        transition: width 0.35s ease;
    }

    .scan-phase-row.is-active:not(.is-indeterminate) .scan-phase-fill {
        animation: scanPulse 1.5s ease-in-out infinite;
    }

    .scan-phase-row.is-indeterminate .scan-phase-bar {
        position: relative;
    }

    .scan-phase-fill-indeterminate {
        width: 45% !important;
        animation: scanShimmer 1.4s ease-in-out infinite;
    }

    .scan-phase-row.is-indeterminate .scan-phase-status {
        animation: scanThink 1.2s ease-in-out infinite;
    }

    .scan-phase-pct.is-indeterminate-pct {
        color: #00d4aa;
        animation: scanThink 1.2s ease-in-out infinite;
        letter-spacing: 0.12em;
    }

    .scan-phase-pct {
        color: #666;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.78rem;
        min-width: 2.5rem;
        text-align: right;
        flex-shrink: 0;
    }

    .scan-phase-pct.is-active-pct {
        color: #00ff88;
    }

    @media (max-width: 768px) {
        .scan-phases {
            gap: 0.75rem;
            max-width: 100%;
        }

        .scan-phase-label {
            font-size: 0.85rem;
        }

        .scan-phase-bar {
            height: 10px;
        }

        .scan-phase-pct:not(.is-active-pct):not(.is-indeterminate-pct) {
            display: none;
        }
    }

    @media (max-width: 480px) {
        .scan-phase-head {
            flex-wrap: wrap;
        }

        .scan-phase-label {
            flex: 1;
            min-width: 0;
        }
    }

    /* FAST Graphs–style fundamentals panel */
    .fg-panel {
        margin-top: 0.75rem;
        padding: 0.85rem 1rem;
        background: #0a0a0a;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    }

    .fg-panel-title {
        color: #f0f0f0;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }

    .fg-panel-ticker {
        color: #888;
        font-weight: 400;
        font-size: 0.85rem;
    }

    .fg-highlights {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 0.85rem;
    }

    .fg-highlight {
        flex: 1 1 140px;
        min-width: 120px;
        padding: 0.5rem 0.65rem;
        border-radius: 4px;
        text-align: center;
    }

    .fg-highlight-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: rgba(0, 0, 0, 0.75);
        margin-bottom: 0.2rem;
    }

    .fg-highlight-value {
        font-size: 1.05rem;
        font-weight: 700;
        color: #111;
    }

    .fg-highlight-growth {
        background: #6BAF4A;
    }

    .fg-highlight-fair {
        background: #FF9500;
    }

    .fg-highlight-normal {
        background: #4DA3FF;
    }

    .fg-details-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
    }

    .fg-details-table td {
        padding: 0.35rem 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        vertical-align: top;
    }

    .fg-detail-label {
        color: #9aa0a6;
        width: 42%;
        font-weight: 500;
    }

    .fg-detail-value {
        color: #e8eaed;
        text-align: right;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }

    .fg-highlight-ror {
        background: #6BAF4A;
    }

    .fg-score-bars {
        margin: 0.5rem 0 0.75rem;
        padding: 0.5rem 0.65rem;
        background: #0a0a0a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
    }

    .fg-score-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.35rem;
        font-size: 0.78rem;
    }

    .fg-score-label {
        flex: 0 0 140px;
        color: #9aa0a6;
    }

    .fg-score-track {
        flex: 1;
        height: 8px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 4px;
        overflow: hidden;
    }

    .fg-score-fill {
        height: 100%;
        border-radius: 4px;
    }

    .fg-score-val {
        flex: 0 0 28px;
        text-align: right;
        color: #e8eaed;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }

    @media (max-width: 480px) {
        .fg-highlights {
            flex-direction: column;
        }

        .fg-highlight {
            flex: 1 1 auto;
        }

        .fg-score-label {
            flex: 0 0 100px;
        }
    }
</style>
"""


def inject_styles() -> None:
    """Inject dark-theme CSS into the Streamlit app. Call once at page load."""
    cleaned = "".join(line.strip() for line in _STYLES.splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)
