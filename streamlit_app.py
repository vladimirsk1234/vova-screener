"""
Streamlit Community Cloud entry point.

Cloud expects streamlit_app.py in the repo root. The full app lives in
headless_scanner.py; we execute it on every Streamlit rerun via runpy.
"""
from __future__ import annotations

import runpy
import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_APP = _ROOT / "headless_scanner.py"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if not _APP.is_file():
    raise FileNotFoundError(f"Missing app module: {_APP.name}")

try:
    runpy.run_path(str(_APP), run_name="__main__")
except Exception as exc:
    # Surface startup failures in Cloud logs (generic "Oh no" hides the traceback).
    traceback.print_exc()
    # Re-raise with chained cause so Cloud UI shows the real ImportError/module error.
    raise RuntimeError(f"App startup failed: {type(exc).__name__}: {exc}") from exc
