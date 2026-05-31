"""
Ticker list I/O and Yahoo Finance info/filtering. No UI or Streamlit.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import pandas as pd
import yfinance as yf

_INFO_CACHE_TTL_SEC = 86400.0  # 24h disk cache for Yahoo metadata
_NAME_RETRY_DELAY_SEC = 0.25
_info_cache_lock = threading.Lock()

# List files (same folder as this package); format: EXCHANGE:SYMBOL comma-separated
TV_LIST_BIG_CAP = "TV-LIST-BIG_CAP_10B.txt"


def read_list_file(filename: str) -> tuple[list[str], dict[str, str], str | None]:
    """
    Read tickers from a list file (EXCHANGE:SYMBOL per entry, comma-separated).
    No cache so you can update the file and next START scan uses the new list.
    Returns (yahoo_tickers, tv_symbol_by_yahoo, error_message). error_message is None on success.
    tv_symbol_by_yahoo maps Yahoo ticker (e.g. BRK-B) -> TradingView symbol (e.g. NYSE:BRK.B).
    """
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return [], {}, None
        out: list[str] = []
        tv_map: dict[str, str] = {}
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                ex, raw_sym = part.split(":", 1)
                ex = ex.strip().upper()
                raw_sym = raw_sym.strip()
                yahoo_sym = raw_sym.replace(".", "-").upper()
                tv_sym = f"{ex}:{raw_sym.upper()}"
            else:
                yahoo_sym = part.replace(".", "-").upper()
                tv_sym = yahoo_sym
            out.append(yahoo_sym)
            tv_map[yahoo_sym] = tv_sym
        return out, tv_map, None
    except FileNotFoundError:
        return [], {}, f"List file not found: {path}. Add {filename} or choose another source."
    except Exception as e:
        return [], {}, f"Could not read list file {filename}: {e}"


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


def _is_symbol_only_name(name: str | None, ticker: str) -> bool:
    if not name or not str(name).strip():
        return True
    return str(name).strip().upper() == ticker.strip().upper()


def _save_info_cache(ticker: str, passed: bool, reason: str, info_dict: dict) -> None:
    if _is_symbol_only_name(info_dict.get("company_name"), ticker):
        return
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


def _company_name_from_info(i: dict, ticker: str) -> str:
    long_name = (i.get("longName") or "").strip()
    if long_name:
        return long_name
    short_name = (i.get("shortName") or "").strip()
    if short_name and not _is_symbol_only_name(short_name, ticker):
        return short_name
    return ticker


def get_cached_company_name(ticker: str) -> str | None:
    """Company name from 24h disk cache only; no network."""
    hit = _load_info_cache(ticker)
    if hit is None:
        return None
    _, _, info_dict = hit
    if not isinstance(info_dict, dict):
        return None
    cached = info_dict.get("company_name")
    if cached and not _is_symbol_only_name(str(cached), ticker):
        return str(cached).strip()
    return None


def _company_name_from_fast_info(fi, ticker: str) -> str | None:
    try:
        if hasattr(fi, "get"):
            data = fi
        elif hasattr(fi, "__iter__") and not isinstance(fi, (str, bytes)):
            data = dict(fi)
        else:
            data = {}
    except Exception:
        data = {}
    for key in ("longName", "shortName"):
        val = data.get(key) if isinstance(data, dict) else None
        if val is None and hasattr(fi, key):
            val = getattr(fi, key, None)
        if val is not None:
            name = str(val).strip()
            if name and not _is_symbol_only_name(name, ticker):
                return name
    return None


def resolve_company_name_fast(ticker: str, *, retries: int = 1) -> str:
    """
    Resolve display name: disk cache -> fast_info -> full .info (same quality as resolve_company_name).
    """
    cached = get_cached_company_name(ticker)
    if cached:
        return cached
    try:
        fi = yf.Ticker(ticker).fast_info
        name = _company_name_from_fast_info(fi, ticker)
        if name:
            return name
    except Exception:
        pass
    return resolve_company_name(ticker, retries=retries)


def build_name_cache(
    tickers: list[str],
    *,
    rate_limit_per_sec: float = 12.0,
    max_workers: int = 8,
    is_cancelled: Callable[[], bool] | None = None,
    on_one_done: Callable[[], None] | None = None,
) -> dict[str, str]:
    """
    Build ticker -> company name for all symbols. Disk hits are instant; misses use fast_info/.info.
    """
    result: dict[str, str] = {}
    pending: list[str] = []
    for t in tickers:
        cached = get_cached_company_name(t)
        if cached:
            result[t] = cached
            if on_one_done:
                on_one_done()
        else:
            pending.append(t)

    if not pending:
        return result

    lock = threading.Lock()
    last_ts = [0.0]

    def _fetch_one(ticker: str) -> tuple[str, str]:
        if is_cancelled and is_cancelled():
            return ticker, ticker
        with lock:
            now = time.monotonic()
            wait = (1.0 / rate_limit_per_sec) - (now - last_ts[0])
            if wait > 0:
                time.sleep(wait)
            last_ts[0] = time.monotonic()
        return ticker, resolve_company_name_fast(ticker)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for ticker, name in pool.map(_fetch_one, pending):
            result[ticker] = name
            if on_one_done:
                on_one_done()
    return result


def resolve_company_name(ticker: str, *, retries: int = 2) -> str:
    """
    Resolve display company name from Yahoo metadata.
    Prefers longName; shortName only when it differs from the ticker symbol.
    Retries .info on failure; does not treat symbol-only cached names as final.
    """
    try:
        _, _, info_dict = get_ticker_info_and_filter(
            ticker, min_market_cap=5e9, min_avg_volume=300_000, require_mc_vol=False
        )
        if isinstance(info_dict, dict):
            cached = info_dict.get("company_name")
            if cached and not _is_symbol_only_name(str(cached), ticker):
                return str(cached).strip()
    except Exception:
        pass

    last = ticker
    for attempt in range(retries + 1):
        try:
            i = yf.Ticker(ticker).info
            name = _company_name_from_info(i, ticker)
            if not _is_symbol_only_name(name, ticker):
                return name
            last = name
        except Exception:
            pass
        if attempt < retries:
            time.sleep(_NAME_RETRY_DELAY_SEC)
    return last


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
    company_name = _company_name_from_info(i, ticker)
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
    return resolve_company_name(ticker)


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


def _round_mcap(val: float) -> str:
    if val >= 1e12:
        return f"{round(val / 1e12, 2)}T"
    if val >= 1e9:
        return f"{round(val / 1e9, 2)}B"
    return f"{round(val / 1e6, 2)}M"


def _days_to_earnings(ticker_obj: yf.Ticker) -> str:
    try:
        cal = ticker_obj.calendar
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date") or cal.get("Earnings Date High")
            if ed is not None:
                if isinstance(ed, (list, tuple)) and ed:
                    ed = ed[0]
                ts = pd.Timestamp(ed)
                days = (ts.normalize() - pd.Timestamp.now().normalize()).days
                if days < 0:
                    return "Today"
                return f"{days}d"
        dates = ticker_obj.get_earnings_dates(limit=4)
        if isinstance(dates, pd.DataFrame) and not dates.empty:
            future = dates.index[dates.index >= pd.Timestamp.now().normalize()]
            if len(future) > 0:
                days = (future[0].normalize() - pd.Timestamp.now().normalize()).days
                return "Today" if days <= 0 else f"{days}d"
    except Exception:
        pass
    return "N/A"


def get_chart_fundamentals(
    ticker: str,
    *,
    close: float | None = None,
    prev_daily_close: float | None = None,
) -> dict:
    """
    Yahoo-backed fundamentals for chart watermark (PE, cap, earnings, description).
    Degrades gracefully to N/A when data is missing.
    """
    out: dict = {
        "company_name": ticker,
        "description": ticker,
        "pe_str": "N/A",
        "mcap_str": "N/A",
        "earn_str": "N/A",
        "daily_chg_str": "+0.00%",
        "market_cap": None,
        "pe": None,
    }
    try:
        t = yf.Ticker(ticker)
        i = t.info or {}
        name = _company_name_from_info(i, ticker)
        out["company_name"] = name
        out["description"] = (i.get("longBusinessSummary") or name)[:120]
        if len(out["description"]) > 120:
            out["description"] = out["description"][:117] + "..."

        mcap = i.get("marketCap")
        if mcap is None:
            shares = i.get("sharesOutstanding") or i.get("impliedSharesOutstanding")
            px = close if close is not None else i.get("regularMarketPrice") or i.get("currentPrice")
            if shares and px:
                mcap = float(shares) * float(px)
        if mcap is not None and mcap > 0:
            out["market_cap"] = float(mcap)
            out["mcap_str"] = _round_mcap(float(mcap))

        pe_ttm = i.get("trailingPE")
        te, _ = _eps_from_yf_info(i)
        px = close if close is not None else i.get("regularMarketPrice")
        pe_final = None
        if pe_ttm is not None:
            try:
                pe_final = float(pe_ttm)
            except (TypeError, ValueError):
                pass
        elif te is not None and te != 0 and px is not None:
            pe_final = float(px) / float(te)
        if pe_final is not None and math.isfinite(pe_final):
            out["pe"] = pe_final
            out["pe_str"] = f"{pe_final:.2f}"

        out["earn_str"] = _days_to_earnings(t)

        if close is not None and prev_daily_close is not None and prev_daily_close != 0:
            chg = (close - prev_daily_close) / prev_daily_close * 100
            sign = "+" if chg >= 0 else ""
            out["daily_chg_str"] = f"{sign}{chg:.2f}%"
    except Exception:
        pass
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
