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
</style>
"""


def inject_styles() -> None:
    """Inject dark-theme CSS into the Streamlit app. Call once at page load."""
    cleaned = "".join(line.strip() for line in _STYLES.splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)
