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

# Streamlit Community Cloud: default user cache dirs are often not writable;
# yfinance can raise on import / first use if cache init fails.
def _is_streamlit_cloud() -> bool:
    try:
        from scan_memory import is_streamlit_cloud

        return bool(is_streamlit_cloud())
    except Exception:
        # Hard Cloud signals only (STREAMLIT_SERVER_PORT is also set by local streamlit).
        return bool(
            os.path.isdir("/mount/src")
            or os.environ.get("HOME", "").startswith("/home/appuser")
        )


def _writable_cache_root() -> str:
    if _is_streamlit_cloud():
        root = "/tmp/vova-screener-cache"
    else:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
    os.makedirs(root, exist_ok=True)
    return root


if _is_streamlit_cloud():
    _tmp_cache = "/tmp"
    os.makedirs(_tmp_cache, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", _tmp_cache)
    os.environ.setdefault("YF_CACHE_DIR", os.path.join(_tmp_cache, "yfinance"))
    os.makedirs(os.environ["YF_CACHE_DIR"], exist_ok=True)
    try:
        import appdirs as _appdirs

        _appdirs.user_cache_dir = lambda *args, **kwargs: os.environ["YF_CACHE_DIR"]  # type: ignore[method-assign]
    except Exception:
        pass

import pandas as pd
import yfinance as yf

_INFO_CACHE_TTL_SEC = 86400.0  # 24h disk cache for Yahoo metadata
_NAME_RETRY_DELAY_SEC = 0.25
_info_cache_lock = threading.Lock()

# List files (same folder as this package); format: EXCHANGE:SYMBOL|Company Name (one per line)
TV_LIST_ETF = "TV-LIST-ETF.txt"
TV_LIST_STOCK_TICKERS = "STOCK-TICKERS.txt"

# TradingView exchange prefix -> Yahoo Finance suffix (Canadian listings)
_TV_TO_YAHOO_SUFFIX: dict[str, str] = {
    "TSX": ".TO",
    "TSXV": ".V",
    "NEO": ".NE",
    "CSE": ".CN",
}
_YAHOO_CANADIAN_SUFFIXES = frozenset(_TV_TO_YAHOO_SUFFIX.values())

# Major US + Canada exchanges (no OTC / pink sheets)
MAJOR_US_EQUITY_EXCHANGES = {
    "NMS", "NYQ", "ASE", "BTS", "BAT", "NGM", "NYS", "PCX",
    "NASDAQ", "NYSE", "AMEX", "BATS", "ARCA",
    "NASDAQGS", "NASDAQCM", "NASDAQGM",
}
CANADIAN_EQUITY_EXCHANGES = {
    "TOR", "TSX", "VAN", "TSXV", "NEO", "CNQ", "CSE",
}
MAJOR_US_CA_EQUITY_EXCHANGES = MAJOR_US_EQUITY_EXCHANGES | CANADIAN_EQUITY_EXCHANGES

OTC_YAHOO_EXCHANGES = frozenset(
    {"OTC", "OTN", "PNK", "OQB", "OQX", "GREY", "CMS", "OOTC", "OTCBB"}
)

_NON_EQUITY_NAME_MARKERS = (
    " ETF",
    " ETN",
    " FUND",
    " TRUST UNITS",
    " TRUST UNIT",
    " PREFERRED",
    " WARRANT",
    " WARRANTS",
    " UNIT",
    " UNITS",
    " DEBENTURE",
    " NOTES DUE",
    " SUBORDINATED",
    " RIGHTS",
    " RIGHT ",
    " DEPOSITARY",
    " DEPOSITORY",
    " AMERICAN DEPOSITARY",
    " ADS ",
    " ADR ",
)


def _tv_to_yahoo_symbol(ex: str, raw_sym: str) -> str:
    """Map TV-style EXCHANGE:SYMBOL to Yahoo ticker (e.g. TSX:SHOP -> SHOP.TO)."""
    ex = ex.strip().upper()
    sym = raw_sym.strip()
    upper = sym.upper()
    if ex in _TV_TO_YAHOO_SUFFIX:
        suffix = _TV_TO_YAHOO_SUFFIX[ex]
        if upper.endswith(suffix.upper()):
            return _normalize_yahoo_ticker(upper)
        if any(upper.endswith(s.upper()) for s in _YAHOO_CANADIAN_SUFFIXES if s != suffix):
            return _normalize_yahoo_ticker(upper)
        base = upper.replace(".", "-") if "." in upper and not upper.endswith(suffix.upper()) else upper
        if base.endswith(suffix.upper().replace(".", "")):
            return _normalize_yahoo_ticker(base)
        return _normalize_yahoo_ticker(f"{base.split('.')[0]}{suffix}")
    return _normalize_yahoo_ticker(sym)


def _normalize_yahoo_ticker(sym: str) -> str:
    """
    Yahoo US share classes use '-' (BRK-B); Canadian listings keep exchange suffix
    with a dot (.TO / .V / .NE / .CN). Never convert those dots to dashes.
    """
    s = str(sym or "").strip().upper().replace("/", "-")
    if not s:
        return s

    # Restore dash-suffixed Canadian tickers produced by older normalize (SHOP-TO -> SHOP.TO)
    for suffix in sorted(_YAHOO_CANADIAN_SUFFIXES, key=len, reverse=True):
        dash_suf = suffix.replace(".", "-").upper()  # -TO, -V, ...
        suf_u = suffix.upper()
        if s.endswith(dash_suf):
            base = s[: -len(dash_suf)].replace(".", "-")
            return f"{base}{suf_u}"
        if s.endswith(suf_u):
            base = s[: -len(suf_u)].replace(".", "-")
            return f"{base}{suf_u}"

    return s.replace(".", "-")

def is_otc_yahoo_exchange(exchange: str | None) -> bool:
    ex = str(exchange or "").strip().upper()
    if not ex:
        return False
    if ex in OTC_YAHOO_EXCHANGES:
        return True
    return "OTC" in ex or ex in ("PNK", "GREY")


def is_major_us_ca_exchange(exchange: str | None) -> bool:
    ex = str(exchange or "").strip().upper()
    return bool(ex) and ex in MAJOR_US_CA_EQUITY_EXCHANGES


def name_suggests_non_common(name: str | None) -> bool:
    n = f" {str(name or '').upper()} "
    return any(marker in n for marker in _NON_EQUITY_NAME_MARKERS)


def tv_part_to_yahoo(tv_part: str) -> str | None:
    """Convert 'NASDAQ:AAPL' or 'TSX:SHOP' to Yahoo symbol."""
    parsed = _parse_list_entry(f"{tv_part.strip()}|")
    return parsed[0] if parsed else None


def _list_file_path(filename: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, filename)


def _parse_list_entry(part: str) -> tuple[str, str, str | None] | None:
    """Parse one entry -> (yahoo_sym, tv_sym, company_name or None). Returns None if empty."""
    part = part.strip()
    if not part or part.startswith("#"):
        return None
    company_name: str | None = None
    if "|" in part:
        tv_part, name = part.split("|", 1)
        name = name.strip()
        if name:
            company_name = name
    else:
        tv_part = part
    tv_part = tv_part.strip()
    if not tv_part:
        return None
    if ":" in tv_part:
        ex, raw_sym = tv_part.split(":", 1)
        ex = ex.strip().upper()
        raw_sym = raw_sym.strip()
        yahoo_sym = _tv_to_yahoo_symbol(ex, raw_sym)
        tv_sym = f"{ex}:{raw_sym.upper()}"
    else:
        yahoo_sym = _normalize_yahoo_ticker(tv_part)
        tv_sym = yahoo_sym
    return yahoo_sym, tv_sym, company_name


def _iter_list_parts(raw: str) -> list[str]:
    """Split file content into entries (newline or legacy comma-separated)."""
    if "\n" in raw:
        return raw.splitlines()
    return raw.split(",")


def read_list_file(filename: str) -> tuple[list[str], dict[str, str], dict[str, str], str | None]:
    """
    Read tickers from a list file.
    Format: EXCHANGE:SYMBOL|Company Name (one per line), or legacy comma-separated EXCHANGE:SYMBOL.
    No cache so you can update the file and next START scan uses the new list.
    Returns (yahoo_tickers, tv_symbol_by_yahoo, company_name_by_yahoo, error_message).
    error_message is None on success.
    """
    path = _list_file_path(filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return [], {}, {}, None
        out: list[str] = []
        tv_map: dict[str, str] = {}
        name_map: dict[str, str] = {}
        for part in _iter_list_parts(raw):
            parsed = _parse_list_entry(part)
            if parsed is None:
                continue
            yahoo_sym, tv_sym, company_name = parsed
            out.append(yahoo_sym)
            tv_map[yahoo_sym] = tv_sym
            if company_name:
                name_map[yahoo_sym] = company_name
        return out, tv_map, name_map, None
    except FileNotFoundError:
        return [], {}, {}, f"List file not found: {path}. Add {filename} or choose another source."
    except Exception as e:
        return [], {}, {}, f"Could not read list file {filename}: {e}"


def write_list_file(
    filename: str,
    entries: list[tuple[str, str, str]],
) -> None:
    """
    Write list file: one line per entry as EXCHANGE:SYMBOL|Company Name.
    entries: (tv_symbol e.g. NYSE:AAPL, yahoo_ticker, company_name).
    """
    path = _list_file_path(filename)
    lines: list[str] = []
    for tv_sym, _yahoo, company_name in entries:
        name = (company_name or "").strip()
        if name:
            lines.append(f"{tv_sym}|{name}")
        else:
            lines.append(tv_sym)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def has_complete_embedded_names(tickers: list[str], name_map: dict[str, str]) -> bool:
    """True when every ticker has a non-empty display name that is not just the symbol."""
    if not tickers or not name_map:
        return False
    for t in tickers:
        name = str(name_map.get(t, "") or "").strip()
        if not name or _is_symbol_only_name(name, t):
            return False
    return True
US_EQUITY_EXCHANGES = {
    "NMS", "NYQ", "ASE", "BTS", "BAT", "NGM", "NYS", "PCX", "OTC", "OTN",
    "NASDAQ", "NYSE", "AMEX", "BATS", "ARCA",
    "NASDAQGS", "NASDAQCM", "NASDAQGM",  # yfinance often returns e.g. "NasdaqGS"
}


def _info_cache_base_dir() -> str:
    d = os.path.join(_writable_cache_root(), "yf_info")
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


def _float_field(i: dict, key: str) -> float | None:
    val = i.get(key)
    if val is None:
        return None
    try:
        out = float(val)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _info_dict_extras(i: dict) -> dict:
    """PE/MC fields persisted in disk cache for watermark fallbacks."""
    out: dict = {}
    for key in (
        "marketCap",
        "trailingPE",
        "forwardPE",
        "sharesOutstanding",
        "impliedSharesOutstanding",
    ):
        val = _float_field(i, key)
        if val is not None:
            out[key] = val
    return out


def _partial_info_dict(
    i: dict,
    *,
    company_name: str,
    avg_vol,
    trailing_eps: float | None,
    forward_eps: float | None,
) -> dict:
    out = {
        "company_name": company_name,
        "avg_volume": avg_vol,
        "trailingEps": trailing_eps,
        "forwardEps": forward_eps,
    }
    out.update(_info_dict_extras(i))
    return out


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
        return False, "NOT_EQUITY", _partial_info_dict(
            i,
            company_name=company_name,
            avg_vol=avg_vol,
            trailing_eps=trailing_eps,
            forward_eps=forward_eps,
        )
    if exchange and exchange not in US_EQUITY_EXCHANGES:
        return False, "NOT_US", _partial_info_dict(
            i,
            company_name=company_name,
            avg_vol=avg_vol,
            trailing_eps=trailing_eps,
            forward_eps=forward_eps,
        )
    if require_mc_vol and (avg_vol is None or (min_avg_volume and avg_vol < min_avg_volume)):
        return False, "LOW_VOL", _partial_info_dict(
            i,
            company_name=company_name,
            avg_vol=avg_vol,
            trailing_eps=trailing_eps,
            forward_eps=forward_eps,
        )

    return True, "", _partial_info_dict(
        i,
        company_name=company_name,
        avg_vol=avg_vol,
        trailing_eps=trailing_eps,
        forward_eps=forward_eps,
    )


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


def _round_mcap(val: float) -> str:
    if val >= 1e12:
        return f"{round(val / 1e12, 2)}T"
    if val >= 1e9:
        return f"{round(val / 1e9, 2)}B"
    return f"{round(val / 1e6, 2)}M"


def _cached_info_dict(ticker: str) -> dict:
    hit = _load_info_cache(ticker)
    if hit is None:
        return {}
    _, _, info_dict = hit
    return dict(info_dict) if isinstance(info_dict, dict) else {}


def _set_if_missing(target: dict, key: str, value) -> None:
    if value is None:
        return
    if target.get(key) is None:
        target[key] = value


def _fast_info_as_dict(fi) -> dict:
    try:
        if hasattr(fi, "get"):
            raw = fi
        elif hasattr(fi, "__iter__") and not isinstance(fi, (str, bytes)):
            raw = dict(fi)
        else:
            raw = {}
    except Exception:
        return {}

    out: dict = {}
    mcap = raw.get("market_cap") or raw.get("marketCap")
    if mcap is None and hasattr(fi, "market_cap"):
        mcap = getattr(fi, "market_cap", None)
    val = _float_field({"marketCap": mcap}, "marketCap") if mcap is not None else None
    if val is not None:
        out["marketCap"] = val

    price = raw.get("last_price") or raw.get("lastPrice") or raw.get("regularMarketPrice")
    if price is None and hasattr(fi, "last_price"):
        price = getattr(fi, "last_price", None)
    val = _float_field({"regularMarketPrice": price}, "regularMarketPrice") if price is not None else None
    if val is not None:
        out["regularMarketPrice"] = val

    shares = (
        raw.get("shares")
        or raw.get("shares_outstanding")
        or raw.get("sharesOutstanding")
    )
    if shares is None and hasattr(fi, "shares"):
        shares = getattr(fi, "shares", None)
    val = _float_field({"sharesOutstanding": shares}, "sharesOutstanding") if shares is not None else None
    if val is not None:
        out["sharesOutstanding"] = val
    return out


_RESOLVE_INFO_TTL_SEC = 3600.0
_resolve_info_cache: dict[str, tuple[float, dict]] = {}


def _resolve_fundamentals_info(ticker: str) -> tuple[dict, yf.Ticker | None]:
    """Merge disk cache, yfinance .info, and fast_info for watermark PE/MC."""
    now = time.time()
    merged: dict = {}
    cached = _resolve_info_cache.get(ticker)
    if cached and now - cached[0] <= _RESOLVE_INFO_TTL_SEC:
        merged = dict(cached[1])
        # Only short-circuit when we already have a business summary for chart "About".
        if merged.get("longBusinessSummary"):
            try:
                return merged, yf.Ticker(ticker)
            except Exception:
                return merged, None

    if not merged:
        disk = _cached_info_dict(ticker)
        merged.update({k: v for k, v in disk.items() if v is not None})

    ticker_obj: yf.Ticker | None = None
    info: dict = {}
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info or {}
    except Exception:
        info = {}

    if info:
        _set_if_missing(merged, "company_name", _company_name_from_info(info, ticker))
        summary = info.get("longBusinessSummary")
        if summary:
            _set_if_missing(merged, "longBusinessSummary", summary)
        te, fe = _eps_from_yf_info(info)
        _set_if_missing(merged, "trailingEps", te)
        _set_if_missing(merged, "forwardEps", fe)
        for key in (
            "marketCap",
            "trailingPE",
            "forwardPE",
            "sharesOutstanding",
            "impliedSharesOutstanding",
            "regularMarketPrice",
            "currentPrice",
        ):
            _set_if_missing(merged, key, _float_field(info, key))

    if ticker_obj is not None:
        try:
            fi_map = _fast_info_as_dict(ticker_obj.fast_info)
        except Exception:
            fi_map = {}
        _set_if_missing(merged, "marketCap", fi_map.get("marketCap"))
        _set_if_missing(merged, "sharesOutstanding", fi_map.get("sharesOutstanding"))
        _set_if_missing(merged, "regularMarketPrice", fi_map.get("regularMarketPrice"))

    _resolve_info_cache[ticker] = (now, dict(merged))
    return merged, ticker_obj


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


def _strip_company_name_prefix(company_name: str, description: str) -> str:
    name = str(company_name or "").strip()
    desc = str(description or "").strip()
    if not name or not desc:
        return desc
    if desc.lower().startswith(name.lower()):
        rest = desc[len(name) :].lstrip(" ,.;:-")
        return rest if rest else desc
    return desc


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
        merged, ticker_obj = _resolve_fundamentals_info(ticker)

        name = merged.get("company_name") or ticker
        out["company_name"] = str(name)
        desc = merged.get("longBusinessSummary") or name or ticker
        desc_str = str(desc)
        out["description"] = _strip_company_name_prefix(str(name), desc_str)

        px = close
        if px is None:
            px = merged.get("regularMarketPrice") or merged.get("currentPrice")
        if px is not None:
            try:
                px = float(px)
            except (TypeError, ValueError):
                px = None

        mcap = merged.get("marketCap")
        if mcap is None:
            shares = merged.get("sharesOutstanding") or merged.get("impliedSharesOutstanding")
            if shares and px is not None:
                mcap = float(shares) * float(px)
        if mcap is not None:
            try:
                mcap = float(mcap)
            except (TypeError, ValueError):
                mcap = None
        if mcap is not None and mcap > 0 and math.isfinite(mcap):
            out["market_cap"] = mcap
            out["mcap_str"] = _round_mcap(mcap)

        pe_ttm = merged.get("trailingPE")
        te = merged.get("trailingEps")
        if te is None:
            te, _ = _eps_from_yf_info(merged)
        pe_final = None
        if pe_ttm is not None:
            try:
                pe_final = float(pe_ttm)
            except (TypeError, ValueError):
                pe_final = None
        elif te is not None and te != 0 and px is not None:
            pe_final = float(px) / float(te)
        if pe_final is not None and math.isfinite(pe_final):
            out["pe"] = pe_final
            out["pe_str"] = f"{pe_final:.2f}"

        if ticker_obj is not None:
            out["earn_str"] = _days_to_earnings(ticker_obj)

        if close is not None and prev_daily_close is not None and prev_daily_close != 0:
            chg = (close - prev_daily_close) / prev_daily_close * 100
            sign = "+" if chg >= 0 else ""
            out["daily_chg_str"] = f"{sign}{chg:.2f}%"
    except Exception:
        pass
    return out

