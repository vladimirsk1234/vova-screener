"""HTTP client for NestJS @vova/api (same Mongo / scan logic as React).

Base URL from (first match):
  1. Streamlit secrets VOVA_API_URL
  2. env VOVA_API_URL
  3. None → caller should use local Yahoo path

Examples:
  http://127.0.0.1:3001/api
  https://screener.example.com/api
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable

import requests

DEFAULT_TIMEOUT = 60


def get_api_base_url() -> str | None:
    url: str | None = None
    try:
        import streamlit as st  # type: ignore

        try:
            raw = st.secrets.get("VOVA_API_URL")  # type: ignore[attr-defined]
            if raw:
                url = str(raw).strip()
        except Exception:
            pass
    except Exception:
        pass
    if not url:
        url = (os.environ.get("VOVA_API_URL") or "").strip() or None
    if not url:
        return None
    return url.rstrip("/")


def api_enabled() -> bool:
    return get_api_base_url() is not None


class VovaApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = DEFAULT_TIMEOUT):
        self.base_url = (base_url or get_api_base_url() or "").rstrip("/")
        if not self.base_url:
            raise ValueError("VOVA_API_URL is not set")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json", "Content-Type": "application/json"})

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", self.timeout)
        res = self.session.request(method, self._url(path), **kwargs)
        if not res.ok:
            text = res.text[:500] if res.text else ""
            raise RuntimeError(f"{res.status_code} {res.reason}: {text}")
        if res.status_code == 204 or not res.content:
            return None
        return res.json()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def start_scan(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/scans", json=params)

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/scans/{run_id}")

    def cancel_scan(self, run_id: str) -> dict[str, Any]:
        return self._request("POST", f"/scans/{run_id}/cancel")

    def list_signals(self, run_id: str, *, limit: int = 500, only_new: bool = False) -> dict[str, Any]:
        q = f"limit={limit}"
        if only_new:
            q += "&onlyNew=true"
        return self._request("GET", f"/scans/{run_id}/signals?{q}")

    def list_rejections(self, run_id: str, *, limit: int = 2000) -> dict[str, Any]:
        return self._request("GET", f"/scans/{run_id}/rejections?limit={limit}")

    def wait_for_run(
        self,
        run_id: str,
        *,
        poll_sec: float = 1.0,
        is_cancelled: Callable[[], bool] | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        timeout_sec: float = 6 * 60 * 60,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_sec
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            if is_cancelled and is_cancelled():
                try:
                    self.cancel_scan(run_id)
                except Exception:
                    pass
                last = self.get_run(run_id)
                last["_cancelled_locally"] = True
                return last
            last = self.get_run(run_id)
            if on_progress:
                on_progress(last)
            status = str(last.get("status") or "")
            if status in ("completed", "cancelled", "failed"):
                return last
            time.sleep(poll_sec)
        raise TimeoutError(f"Scan {run_id} did not finish within {timeout_sec}s")


def map_source_label(src: str) -> str:
    s = (src or "").strip()
    if s in ("Stocks", "ETF", "MANUAL SCAN"):
        return s
    return "Stocks"


def build_scan_params(run_params: dict[str, Any]) -> dict[str, Any]:
    src = map_source_label(str(run_params.get("src", "Stocks")))
    direction = str(run_params.get("scan_direction", "buy")).lower()
    if direction not in ("buy", "sell"):
        direction = "buy"
    return {
        "source": src,
        "manualTickers": str(run_params.get("txt") or ""),
        "tf": str(run_params.get("tf") or "Daily"),
        "direction": direction,
        "minRr": float(run_params.get("rr") or 1.5),
        "riskPerTrade": float(run_params.get("risk_per_trade") or 100),
        "noRrReq": bool(run_params.get("no_rr_req", False)),
        "useLastHlSl": bool(run_params.get("use_last_hl_sl", True)),
        "newOnly": bool(run_params.get("new", True)),
    }


def _fmt_rr(val: Any) -> float | str:
    if val is None:
        return "N/A"
    try:
        return round(float(val), 2)
    except (TypeError, ValueError):
        return "N/A"


def signal_to_table_row(sig: dict[str, Any]) -> dict[str, Any]:
    tv_url = sig.get("tvUrl") or ""
    tv_sym = sig.get("tvSymbol") or sig.get("symbol") or ""
    company = sig.get("companyName") or sig.get("symbol") or ""
    kind = sig.get("kind") or "buy"
    if kind == "sell":
        shares = int(sig.get("shares") or 0)
        return {
            "Symbol": tv_url,
            "tv_symbol": tv_sym,
            "Company Name": company,
            "Entry": round(float(sig.get("entry") or 0), 2),
            "Exit": round(float(sig.get("exit") or 0), 2),
            "Position Size (shares)": shares,
            "RR at Entry": _fmt_rr(sig.get("rrAtEntry")),
            "RR at Close": _fmt_rr(sig.get("rrAtClose")),
            "Invested ($)": round(float(sig.get("invested") or 0), 2),
            "P&L ($)": round(float(sig.get("pnlUsd") or 0), 2),
            "P&L (%)": round(float(sig.get("pnlPct") or 0), 2),
        }
    shares = int(sig.get("shares") or 0)
    return {
        "Symbol": tv_url,
        "tv_symbol": tv_sym,
        "Company Name": company,
        "TP": round(float(sig.get("tp") or 0), 2),
        "SL": round(float(sig.get("sl") or 0), 2),
        "RR": _fmt_rr(sig.get("rr")),
        "Position Size (shares)": shares,
        "Position Value ($)": round(float(sig.get("positionValue") or 0), 2),
        "New": 1 if sig.get("isNew") else 0,
        "Valid": 1,
        "Strong": 1 if sig.get("isStrong") else 0,
    }


def run_scan_via_api(
    run_params: dict[str, Any],
    *,
    is_cancelled: Callable[[], bool] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    client: VovaApiClient | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any, dict[str, Any]]:
    """Run a scan on NestJS/Mongo and return Streamlit-shaped results.

    Returns (table_rows, rejected_reasons, as_of, ohlc_cache).
    ohlc_cache is empty — charts still use Yahoo/local path in Streamlit.
    """
    api = client or VovaApiClient()
    params = build_scan_params(run_params)
    started = api.start_scan(params)
    run_id = started.get("runId")
    if not run_id:
        raise RuntimeError(f"API start_scan missing runId: {started}")

    run = api.wait_for_run(run_id, is_cancelled=is_cancelled, on_progress=on_progress)
    status = str(run.get("status") or "")
    if status == "failed":
        raise RuntimeError(run.get("error") or f"Scan failed ({run_id})")

    only_new = bool(params.get("newOnly"))
    sig_payload = api.list_signals(run_id, only_new=only_new)
    rej_payload = api.list_rejections(run_id)

    rows = [signal_to_table_row(s) for s in (sig_payload.get("rows") or [])]
    rejected = [
        {"Symbol": r.get("symbol") or "", "Reason": r.get("reason") or ""}
        for r in (rej_payload.get("rows") or [])
    ]
    as_of = run.get("asOf")
    return rows, rejected, as_of, {}
