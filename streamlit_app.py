"""
Streamlit Community Cloud entry point.

Cloud defaults to streamlit_app.py; the app implementation lives in headless_scanner.py.
"""
from __future__ import annotations

from pathlib import Path
import runpy

_APP = Path(__file__).resolve().parent / "headless_scanner.py"
runpy.run_path(str(_APP), run_name="__main__")
