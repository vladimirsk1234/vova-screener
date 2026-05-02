"""
Ticker list I/O and Yahoo Finance info/filtering. No UI or Streamlit.
"""
import json
import math
import os
import threading
import time
import pandas as pd
import yfinance as yf

_INFO_CACHE_TTL_SEC = 86400.0  # 24h disk cache for Yahoo metadata
_info_cache_lock = threading.Lock()

# List files (same folder as this package); format: EXCHANGE:SYMBOL comma-separated
TV_LIST_BIG_CAP = "TV-LIST-BIG_CAP_10B.txt"


def read_list_file(filename: str) -> tuple[list[str], str | None]:
    """
    Read tickers from a list file (EXCHANGE:SYMBOL per entry, comma-separated).
    No cache so you can update the file and next START scan uses the new list.
    Returns (tickers, error_message). error_message is None on success.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return [], None
        out = []
        for part in raw.split(","):
            part = part.strip()
            if ":" in part:
                sym = part.split(":", 1)[1].strip()
            else:
                sym = part
            if sym:
                out.append(sym.replace(".", "-"))
        return out, None
    except FileNotFoundError:
        return [], f"List file not found: {path}. Add {filename} or choose another source."
    except Exception as e:
        return [], f"Could not read list file {filename}: {e}"


# US exchanges: Yahoo MIC codes and yfinance variants (comparison uses .upper())
US_EQUITY_EXCHANGES = {
    "NMS", "NYQ", "ASE", "BTS", "BAT", "NGM", "NYS", "PCX", "OTC", "OTN",
    "NASDAQ", "NYSE", "AMEX", "BATS", "ARCA",
    "NASDAQGS", "NASDAQCM", "NASDAQGM",  # yfinance often returns e.g. "NasdaqGS"
}


def _info_cache_base_dir() -> str:
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "yf_info")
    os.makedirs(d, exist_ok=True)
    return d


def _info_cache_path(ticker: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in ticker)[:200]
    return os.path.join(_info_cache_base_dir(), f"{safe}.json")


def _load_info_cache(ticker: str) -> tuple[bool, str, dict] | None:
    path = _info_cache_path(ticker)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            row = json.load(f)
        if time.time() - float(row.get("saved_at", 0)) > _INFO_CACHE_TTL_SEC:
            return None
        info = row.get("info_dict")
        if not isinstance(info, dict):
            return None
        return (bool(row["passed"]), str(row.get("reason", "")), info)
    except Exception:
        return None


def _save_info_cache(ticker: str, passed: bool, reason: str, info_dict: dict) -> None:
    try:
        payload = {
            "saved_at": time.time(),
            "passed": passed,
            "reason": reason,
            "info_dict": info_dict,
        }
        path = _info_cache_path(ticker)
        with _info_cache_lock:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, separators=(",", ":"))
    except Exception:
        pass


def _apply_yahoo_info_filters(
    i: dict,
    ticker: str,
    ticker_obj: yf.Ticker,
    min_market_cap: float,
    min_avg_volume: float,
    require_mc_vol: bool,
) -> tuple[bool, str, dict | None]:
    """Shared filter logic from a Yahoo-style info dict (full .info or fast_info-derived)."""
    quote_type = i.get("quoteType") or ""
    exchange = (i.get("exchange") or "").upper()
    avg_vol = i.get("averageVolume")
    if avg_vol is None and require_mc_vol:
        hist = ticker_obj.history(period="1mo")
        if isinstance(hist, pd.DataFrame) and not hist.empty and "Volume" in hist.columns:
            avg_vol = float(hist["Volume"].mean())
    company_name = i.get("longName") or i.get("shortName") or ticker
    trailing_eps, forward_eps = _eps_from_yf_info(i)

    if quote_type and quote_type.upper() != "EQUITY":
        partial = {
            "company_name": company_name,
            "avg_volume": avg_vol,
            "trailingEps": trailing_eps,
            "forwardEps": forward_eps,
        }
        return False, "NOT_EQUITY", partial
    if exchange and exchange not in US_EQUITY_EXCHANGES:
        partial = {
            "company_name": company_name,
            "avg_volume": avg_vol,
            "trailingEps": trailing_eps,
            "forwardEps": forward_eps,
        }
        return False, "NOT_US", partial
    if require_mc_vol and (avg_vol is None or (min_avg_volume and avg_vol < min_avg_volume)):
        partial = {
            "company_name": company_name,
            "avg_volume": avg_vol,
            "trailingEps": trailing_eps,
            "forwardEps": forward_eps,
        }
        return False, "LOW_VOL", partial

    info_dict = {
        "company_name": company_name,
        "avg_volume": avg_vol,
        "trailingEps": trailing_eps,
        "forwardEps": forward_eps,
    }
    return True, "", info_dict


def _eps_from_yf_info(i: dict) -> tuple[float | None, float | None]:
    """Parse trailingEps / forwardEps from yfinance .info (TTM proxy for screening)."""
    te = i.get("trailingEps")
    fe = i.get("forwardEps")
    out_te, out_fe = None, None
    if te is not None:
        try:
            out_te = float(te)
        except (TypeError, ValueError):
            pass
    if fe is not None:
        try:
            out_fe = float(fe)
        except (TypeError, ValueError):
            pass
    return out_te, out_fe


def get_ticker_info_and_filter(
    ticker: str,
    min_market_cap: float = 5e9,
    min_avg_volume: float = 300_000,
    require_mc_vol: bool = True,
) -> tuple[bool, str, dict | None]:
    """
    Fetch ticker info from yfinance. Apply filters: US listed common stock; optionally avg volume (require_mc_vol=True).
    When require_mc_vol=False (e.g. TV-LIST): only filter NOT_EQUITY / NOT_US.
    Returns (passed: bool, reject_reason: str, info_dict or None). info_dict: company_name, avg_volume.
    When filter fails but we have info, returns partial info_dict so caller doesn't need a second API call.
    Uses a 24h on-disk cache. (fast_info was spiked: dict(fi) exposes exchange/quoteType/volume,
    but short display names are inferior to .info for manual scans, so the hot path stays .info.)
    """
    hit = _load_info_cache(ticker)
    if hit is not None:
        return hit

    try:
        t = yf.Ticker(ticker)
        i = t.info
        out = _apply_yahoo_info_filters(i, ticker, t, min_market_cap, min_avg_volume, require_mc_vol)
        if out[2] is not None:
            _save_info_cache(ticker, out[0], out[1], out[2])
        return out
    except Exception:
        return False, "INFO_ERROR", None


def fetch_fallback_company_name(ticker: str) -> str:
    """Fetch company name from yfinance when get_ticker_info_and_filter returns None (INFO_ERROR)."""
    try:
        i = yf.Ticker(ticker).info
        return i.get("longName") or i.get("shortName") or ticker
    except Exception:
        return ticker


def _extract_annual_eps_map(financials: pd.DataFrame | None) -> dict[int, float]:
    """
    Build {year: eps} from a yfinance annual financials DataFrame.
    Accepts diluted/basic EPS row naming variants and keeps latest value per year.
    """
    if not isinstance(financials, pd.DataFrame) or financials.empty:
        return {}

    eps_row = None
    for candidate in ("Diluted EPS", "Basic EPS", "DilutedEPS", "BasicEPS"):
        if candidate in financials.index:
            eps_row = financials.loc[candidate]
            break
    if eps_row is None:
        return {}

    out: dict[int, float] = {}
    for col, raw_val in eps_row.items():
        try:
            year = pd.Timestamp(col).year
            eps_val = float(raw_val)
        except Exception:
            continue
        if not math.isfinite(eps_val):
            continue
        out[year] = eps_val
    return out


def get_annual_eps_history_5y(ticker: str) -> dict[int, float] | None:
    """
    Return latest up-to-5 annual EPS points as {year: eps} for one ticker.
    Returns None when no usable annual EPS rows are available.
    """
    try:
        t = yf.Ticker(ticker)
        eps_map = _extract_annual_eps_map(getattr(t, "financials", None))
        if not eps_map:
            eps_map = _extract_annual_eps_map(getattr(t, "income_stmt", None))
        if not eps_map:
            return None

        years_desc = sorted(eps_map.keys(), reverse=True)[:5]
        return {y: eps_map[y] for y in sorted(years_desc)}
    except Exception:
        return None
