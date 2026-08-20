#!/usr/bin/env python3
"""FMP EPS filter + PE/PEG valuation helpers. Premium cap: 750 HTTP calls/min (we use 12/s)."""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ticker_data import TV_LIST_STOCK_TICKERS, read_list_file, write_list_file

BASE = "https://financialmodelingprep.com/stable"
CACHE_PATH = ROOT / ".cache" / "fmp_eps.json"
VALUATION_CACHE_PATH = ROOT / ".cache" / "fmp_valuation.json"
RETRY_REASONS = frozenset({"FMP_HTTP", "FMP_RATE_LIMIT", "FMP_ERROR"})
# Bump when liquidity/ATR gates change so old EPS-only PASS rows are rechecked.
QUALITY_GATE = "liq-atr-v1"
MIN_DAILY_ATR_PCT = 1.0
ATR_LEN = 14
MIN_ATR_BARS = 15
US_TV_EXCHANGES = frozenset({"NASDAQ", "NYSE", "AMEX"})
GROWTH_PE_FLOOR = 15.0
GRAHAM_GROWTH_MAX = 5.0
LYNCH_GROWTH_MIN = 15.0
FAIR_VALUE_PE = 15.0
GROWTH_LOOKBACK_YEARS = 5
KNOWN_ADR_RATIO = {"XYF": 6, "TSM": 5, "BABA": 8, "BIDU": 8, "JD": 2, "HDB": 3, "PDD": 4, "NTES": 5}
FALLBACK_FOREIGN_PER_USD = {
    "USD": 1.0,
    "CNY": 7.2,
    "RMB": 7.2,
    "CNH": 7.2,
    "TWD": 32.0,
    "HKD": 7.8,
    "JPY": 150.0,
    "KRW": 1350.0,
    "INR": 84.0,
    "ARS": 1100.0,
    "BRL": 5.5,
    "EUR": 0.92,
    "GBP": 0.79,
    "CAD": 1.37,
    "AUD": 1.55,
}
COMMON_ADR = (2, 3, 4, 5, 6, 8, 10, 20, 25, 40)
FMP_HTTP_RATE_PER_SEC = 12.0
FMP_SCAN_WORKERS = 8
_last_http_at = 0.0
_http_lock = threading.Lock()


@dataclass(frozen=True)
class LiquidityGates:
    min_price: float
    min_vol_avg: float
    min_dollar_adv: float
    min_mkt_cap: float


# Recommended quality floors (listing currency for price/ADV; FMP mktCap is USD).
US_LIQUIDITY = LiquidityGates(
    min_price=5.0,
    min_vol_avg=200_000.0,
    min_dollar_adv=1_000_000.0,
    min_mkt_cap=300_000_000.0,
)
CA_LIQUIDITY = LiquidityGates(
    min_price=2.0,
    min_vol_avg=50_000.0,
    min_dollar_adv=300_000.0,
    min_mkt_cap=100_000_000.0,
)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
        f.flush()
    tmp.replace(path)


def _throttle_fmp_http(rate_per_sec: float = FMP_HTTP_RATE_PER_SEC) -> None:
    """Thread-safe global pacer so concurrent workers still respect the FMP cap."""
    global _last_http_at
    interval = 1.0 / max(rate_per_sec, 0.1)
    with _http_lock:
        now = time.monotonic()
        wait = interval - (now - _last_http_at)
        if wait > 0:
            time.sleep(wait)
        _last_http_at = time.monotonic()


def yahoo_to_fmp(yahoo: str) -> str:
    s = (yahoo or "").strip().upper()
    if not s:
        return s
    if s.endswith((".TO", ".V", ".NE", ".CN")):
        return s
    if len(s) >= 3 and s[-2] == "-" and s[-1].isalpha():
        return f"{s[:-2]}.{s[-1]}"
    return s.replace("-", ".")


def fmp_symbol_candidates(yahoo: str) -> list[str]:
    """
    Symbol forms to try, in order. FMP serves US class shares under the dash
    form (BRK-B); the dotted form answers HTTP 402, so keep it only as fallback.
    """
    raw = (yahoo or "").strip().upper()
    forms = [raw] if raw else []
    mapped = yahoo_to_fmp(yahoo)
    if mapped and mapped not in forms:
        forms.append(mapped)
    return forms


def _num(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        n = float(v)
    except (TypeError, ValueError):
        return None
    return n if math.isfinite(n) else None


def _str(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def load_fmp_api_key() -> str:
    api_key = (os.environ.get("FMP_API_KEY") or "").strip()
    if api_key:
        return api_key
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if t.startswith("FMP_API_KEY="):
                return t.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _fmp_get(path: str, api_key: str, params: dict[str, str]) -> object:
    q = urllib.parse.urlencode({"apikey": api_key, **params})
    url = f"{BASE}{path}?{q}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    last_err: Exception | None = None
    for attempt in range(5):
        _throttle_fmp_http()
        try:
            with urllib.request.urlopen(req, timeout=60) as res:
                data = json.loads(res.read().decode("utf-8"))
            if isinstance(data, dict) and data.get("Error Message"):
                raise RuntimeError(str(data.get("Error Message")))
            return data
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            # Transient network/read timeout: retry with backoff instead of
            # dropping a whole screener exchange or a ticker on the first hiccup.
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
            continue
    if last_err:
        raise last_err
    raise RuntimeError(f"FMP request failed for {path}")


def _year_of(row: dict) -> int | None:
    y = _num(row.get("calendarYear"))
    if y is not None:
        return int(y)
    date = _str(row.get("date")) or ""
    if len(date) >= 4 and date[:4].isdigit():
        return int(date[:4])
    return None


def cagr_pct(first: float, last: float, years: float) -> float | None:
    if years <= 0 or first <= 0 or last <= 0:
        return None
    return ((last / first) ** (1.0 / years) - 1.0) * 100.0


def trailing_eps_cagr_detail(
    points: list[tuple[int, float]],
    lookback_years: int = GROWTH_LOOKBACK_YEARS,
) -> tuple[float | None, int | None]:
    with_m = sorted([(y, e) for y, e in points if e > 0], key=lambda p: p[0])
    if len(with_m) < 2:
        return None, None
    last_y, last_e = with_m[-1]
    target_year = last_y - lookback_years
    first_y, first_e = with_m[0]
    for y, e in with_m:
        if y <= target_year:
            first_y, first_e = y, e
    span = last_y - first_y
    if span < 1:
        return None, None
    return cagr_pct(first_e, last_e, span), span


def trailing_eps_cagr(
    points: list[tuple[int, float]],
    lookback_years: int = GROWTH_LOOKBACK_YEARS,
) -> float | None:
    growth, _span = trailing_eps_cagr_detail(points, lookback_years)
    return growth


def fair_value_rule(
    growth_pct: float | None,
    span_years: int | None = None,
) -> str | None:
    """FAST Graphs bands: gdf / gdf_pe_g / pe_g. Short span (<2y) never Lynch."""
    if span_years is not None and span_years < 2:
        return "gdf_pe_g"
    if growth_pct is None or not math.isfinite(growth_pct):
        return None
    if growth_pct < GRAHAM_GROWTH_MAX:
        return "gdf"
    if growth_pct < LYNCH_GROWTH_MIN:
        return "gdf_pe_g"
    return "pe_g"


def lynch_peg(pe: float | None, growth_pct: float | None) -> float | None:
    if pe is None or growth_pct is None or growth_pct == 0:
        return None
    if not math.isfinite(pe) or not math.isfinite(growth_pct):
        return None
    return pe / growth_pct


def is_undervalued(
    pe: float | None,
    growth_pct: float | None,
    span_years: int | None = None,
) -> bool:
    if pe is None or pe <= 0:
        return False
    if pe < 0.15 or pe > 200:
        return False
    rule = fair_value_rule(growth_pct, span_years)
    if rule in ("gdf", "gdf_pe_g", "pe15"):
        return pe < FAIR_VALUE_PE
    if rule in ("pe_g", "lynch_peg"):
        if growth_pct is None or not math.isfinite(growth_pct):
            return False
        peg = lynch_peg(pe, growth_pct)
        return peg is not None and peg < 1.0
    return False


def _norm_ccy(code: object) -> str | None:
    s = _str(code)
    if not s:
        return None
    u = s.upper()
    if u in {"RMB", "CNH", "CNY"}:
        return "CNY"
    return u


def _rel_close(a: float, b: float, tol: float = 0.15) -> bool:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom <= tol


def _fx_to_listing(reported: str | None, listing: str | None) -> float:
    frm = _norm_ccy(reported) or "USD"
    to = _norm_ccy(listing) or "USD"
    if frm == to:
        return 1.0
    from_per = FALLBACK_FOREIGN_PER_USD.get(frm, 1.0)
    to_per = FALLBACK_FOREIGN_PER_USD.get(to, 1.0)
    if from_per <= 0 or to_per <= 0:
        return 1.0
    return to_per / from_per


def _infer_adr(ticker: str, net_income: float | None, fmp_eps: float | None, shares: float | None) -> int:
    known = KNOWN_ADR_RATIO.get(ticker.split(".")[0].upper(), 1)
    if net_income is None or fmp_eps is None or shares is None or shares <= 0 or fmp_eps == 0:
        return known
    ordinary = net_income / shares
    candidates = [known, *[r for r in COMMON_ADR if r != known]] if known > 1 else list(COMMON_ADR)
    for ratio in candidates:
        if _rel_close(fmp_eps, ordinary * ratio) or _rel_close(fmp_eps, ordinary * ratio * ratio):
            return ratio
    return known


def _share_scale(net_income: float | None, fmp_eps: float | None, shares: float | None, adr: int) -> str:
    if net_income is None or fmp_eps is None or shares is None or shares <= 0 or fmp_eps == 0:
        return "unknown"
    ordinary = net_income / shares
    if _rel_close(fmp_eps, ordinary):
        return "ordinary" if adr > 1 else "ads"
    if adr > 1:
        ads = ordinary * adr
        if _rel_close(fmp_eps, ads):
            return "ads"
        if _rel_close(fmp_eps, ads * adr) or _rel_close(fmp_eps, ordinary * adr * adr):
            return "double_adr"
    return "unknown"


def _per_share_factor(share_scale: str, adr: int) -> float:
    if share_scale == "ordinary":
        return float(adr)
    if share_scale == "double_adr":
        return 1.0 / adr
    return 1.0


def _scale_eps_row(
    ticker: str,
    *,
    fmp_eps: float | None,
    net_income: float | None,
    shares: float | None,
    reported: str | None,
    listing: str | None,
    price: float | None,
) -> tuple[float | None, float | None, bool]:
    """Return (eps_listing_per_ads, pe, reliable)."""
    fx = _fx_to_listing(reported, listing)
    adr = _infer_adr(ticker, net_income, fmp_eps, shares)
    scale = _share_scale(net_income, fmp_eps, shares, adr)
    if scale == "unknown" and adr > 1 and fmp_eps is not None and price is not None and price > 0:
        fx_eps = fmp_eps * fx
        pe_ads = price / fx_eps if fx_eps else None
        pe_double = price / (fx_eps / adr) if fx_eps else None
        if pe_double is not None and 0.15 <= pe_double <= 200 and not (pe_ads is not None and 0.15 <= pe_ads <= 200):
            scale = "double_adr"
    factor = _per_share_factor(scale, adr)
    scaled = fmp_eps * fx * factor if fmp_eps is not None else None
    from_ni = None
    if net_income is not None and shares is not None and shares > 0:
        ads_shares = shares / adr if adr > 1 else shares
        if ads_shares > 0:
            from_ni = (net_income * fx) / ads_shares
    pe_scaled = price / scaled if price and scaled and scaled > 0 else None
    pe_ni = price / from_ni if price and from_ni and from_ni > 0 else None
    eps = scaled
    if from_ni is not None and pe_ni is not None and 0.15 <= pe_ni <= 200:
        if pe_scaled is None or not (0.15 <= pe_scaled <= 200) or (
            scaled is not None and not _rel_close(from_ni, scaled, 0.3)
        ):
            eps = from_ni
    pe = price / eps if price and eps and eps > 0 else None
    reliable = pe is not None and 0.15 <= pe <= 200
    return eps, pe, reliable


def fetch_eps_pe_growth(
    fmp_symbol: str,
    api_key: str,
) -> tuple[bool, str, float | None, float | None, float | None, int | None]:
    try:
        ttm_raw = _fmp_get("/key-metrics-ttm", api_key, {"symbol": fmp_symbol})
        ttm_rows = ttm_raw if isinstance(ttm_raw, list) else []
        ttm = ttm_rows[0] if ttm_rows else {}
        inc_raw = _fmp_get(
            "/income-statement",
            api_key,
            {"symbol": fmp_symbol, "period": "annual", "limit": "8"},
        )
        inc_rows = inc_raw if isinstance(inc_raw, list) else []
        inc0 = inc_rows[0] if inc_rows and isinstance(inc_rows[0], dict) else {}
        eps = _num(ttm.get("netIncomePerShareTTM")) or _num(ttm.get("epsTTM"))
        if eps is None:
            eps = _num(inc0.get("epsdiluted")) or _num(inc0.get("eps"))
        pe = _num(ttm.get("peRatioTTM")) or _num(ttm.get("priceToEarningsRatioTTM"))
        if pe is None:
            try:
                rat_raw = _fmp_get("/ratios-ttm", api_key, {"symbol": fmp_symbol})
                rat_rows = rat_raw if isinstance(rat_raw, list) else []
                rat = rat_rows[0] if rat_rows else {}
                pe = _num(rat.get("priceToEarningsRatioTTM")) or _num(rat.get("peRatioTTM"))
            except Exception:
                pe = None
        reported = _norm_ccy(inc0.get("reportedCurrency"))
        net_income = _num(inc0.get("netIncome"))
        shares = _num(inc0.get("weightedAverageShsOutDil")) or _num(inc0.get("weightedAverageShsOut"))
        need_scale = bool(
            (reported and reported != "USD")
            or fmp_symbol.split(".")[0].upper() in KNOWN_ADR_RATIO
            or (pe is not None and (pe < 0.15 or pe > 200))
        )
        listing = "USD"
        price = None
        if need_scale:
            try:
                profile_raw = _fmp_get("/profile", api_key, {"symbol": fmp_symbol})
                profile_rows = profile_raw if isinstance(profile_raw, list) else []
                profile = profile_rows[0] if profile_rows and isinstance(profile_rows[0], dict) else {}
                listing = _norm_ccy(profile.get("currency")) or "USD"
                price = _num(profile.get("price"))
            except Exception:
                listing = "USD"
            scaled, pe2, _reliable = _scale_eps_row(
                fmp_symbol,
                fmp_eps=eps,
                net_income=net_income,
                shares=shares,
                reported=reported,
                listing=listing,
                price=price,
            )
            if scaled is not None:
                eps = scaled
            if pe2 is not None:
                pe = pe2
        points: list[tuple[int, float]] = []
        for row in inc_rows:
            if not isinstance(row, dict):
                continue
            year = _year_of(row)
            e = _num(row.get("epsdiluted")) or _num(row.get("eps"))
            if year is None or e is None:
                continue
            points.append((year, e))
        growth, span = trailing_eps_cagr_detail(points)
        if eps is None:
            return False, "NO_TRAILING_EPS", None, pe, growth, span
        if eps <= 0:
            return False, "NON_POSITIVE_EPS", eps, pe, growth, span
        return True, "PASS", eps, pe, growth, span
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            return False, "FMP_RATE_LIMIT", None, None, None, None
        return False, "FMP_HTTP", None, None, None, None
    except Exception:
        return False, "FMP_ERROR", None, None, None, None


def tv_exchange(tv: str) -> str:
    return (tv.split(":", 1)[0] if ":" in tv else tv).upper()


def gates_for_tv(
    tv: str,
    *,
    us: LiquidityGates = US_LIQUIDITY,
    ca: LiquidityGates = CA_LIQUIDITY,
) -> LiquidityGates:
    return us if tv_exchange(tv) in US_TV_EXCHANGES else ca


def fetch_profile(symbol: str, api_key: str) -> dict:
    raw = _fmp_get("/profile", api_key, {"symbol": symbol})
    rows = raw if isinstance(raw, list) else []
    row = rows[0] if rows and isinstance(rows[0], dict) else {}
    return row if isinstance(row, dict) else {}


def profile_liquidity_reason(profile: dict, gates: LiquidityGates) -> str | None:
    if profile.get("isEtf") or profile.get("isFund"):
        return "NOT_EQUITY"
    if profile.get("isActivelyTrading") is False:
        return "NOT_TRADING"
    price = _num(profile.get("price"))
    vol = (
        _num(profile.get("volAvg"))
        or _num(profile.get("averageVolume"))
        or _num(profile.get("avgVolume"))
    )
    mcap = _num(profile.get("mktCap")) or _num(profile.get("marketCap"))
    if price is None or price < gates.min_price:
        return "LOW_PRICE"
    if vol is None or vol < gates.min_vol_avg:
        return "LOW_VOL"
    if (vol * price) < gates.min_dollar_adv:
        return "LOW_DOLLAR_VOL"
    if mcap is None or mcap < gates.min_mkt_cap:
        return "LOW_MCAP"
    return None


def _as_dict_rows(raw: object) -> list[dict]:
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]
    if isinstance(raw, dict):
        for key in ("historical", "data", "technicalIndicators"):
            val = raw.get(key)
            if isinstance(val, list):
                return [r for r in val if isinstance(r, dict)]
    return []


def _last_atr_value(rows: list[dict]) -> float | None:
    if not rows:
        return None
    for row in (*rows[:8], *reversed(rows[-8:])):
        v = _num(row.get("atr")) or _num(row.get("ATR"))
        if v is not None:
            return v
    return None


def _ohlc_chronological(rows: list[dict]) -> tuple[list[float], list[float], list[float]]:
    dated: list[tuple[str, float, float, float]] = []
    for row in rows:
        high = _num(row.get("high"))
        low = _num(row.get("low"))
        close = _num(row.get("close")) or _num(row.get("price"))
        if high is None or low is None or close is None:
            continue
        dated.append((_str(row.get("date")) or "", high, low, close))
    dated.sort(key=lambda r: r[0])
    return (
        [r[1] for r in dated],
        [r[2] for r in dated],
        [r[3] for r in dated],
    )


def wilder_atr_last(highs: list[float], lows: list[float], closes: list[float], length: int = ATR_LEN) -> float | None:
    n = len(closes)
    if n < MIN_ATR_BARS or n != len(highs) or n != len(lows):
        return None
    atr = highs[0] - lows[0]
    alpha = 1.0 / length
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        atr = alpha * tr + (1.0 - alpha) * atr
    if not math.isfinite(atr) or atr <= 0:
        return None
    return atr


def fetch_daily_atr(symbol: str, api_key: str) -> tuple[float | None, float | None]:
    """Return (atr, last_close) from FMP. ATR endpoint first, EOD OHLC fallback."""
    try:
        raw = _fmp_get(
            "/technical-indicators/atr",
            api_key,
            {"symbol": symbol, "periodLength": str(ATR_LEN), "timeframe": "1day"},
        )
        rows = _as_dict_rows(raw)
        atr = _last_atr_value(rows)
        close = None
        if rows:
            close = _num(rows[0].get("close")) or _num(rows[-1].get("close"))
        if atr is not None and atr > 0:
            return atr, close
    except Exception:
        pass
    start = (date.today() - timedelta(days=180)).isoformat()
    raw = _fmp_get("/historical-price-eod/full", api_key, {"symbol": symbol, "from": start})
    highs, lows, closes = _ohlc_chronological(_as_dict_rows(raw))
    atr = wilder_atr_last(highs, lows, closes)
    last_close = closes[-1] if closes else None
    return atr, last_close


def daily_atr_pct_reason(
    atr: float | None,
    price: float | None,
    *,
    min_atr_pct: float = MIN_DAILY_ATR_PCT,
) -> tuple[str | None, float | None]:
    if atr is None or atr <= 0 or price is None or price <= 0:
        return "NO_ATR", None
    atr_pct = (atr / price) * 100.0
    if not math.isfinite(atr_pct):
        return "NO_ATR", None
    if atr_pct <= min_atr_pct:
        return "LOW_ATR", atr_pct
    return None, atr_pct


def fetch_profile_and_liquidity(
    symbol: str,
    api_key: str,
    gates: LiquidityGates,
) -> tuple[str | None, dict]:
    profile = fetch_profile(symbol, api_key)
    if not profile:
        return "FMP_ERROR", {}
    return profile_liquidity_reason(profile, gates), profile


def check_daily_atr_pct(
    symbol: str,
    api_key: str,
    profile: dict,
    *,
    min_atr_pct: float = MIN_DAILY_ATR_PCT,
) -> tuple[str | None, float | None]:
    atr, last_close = fetch_daily_atr(symbol, api_key)
    price = _num(profile.get("price")) or last_close
    return daily_atr_pct_reason(atr, price, min_atr_pct=min_atr_pct)


def check_positive_eps_fmp(yahoo: str, api_key: str) -> tuple[bool, str, float | None]:
    symbol = yahoo_to_fmp(yahoo)
    try:
        profile_raw = _fmp_get("/profile", api_key, {"symbol": symbol})
        rows = profile_raw if isinstance(profile_raw, list) else []
        profile = rows[0] if rows else {}
        if profile.get("isEtf") or profile.get("isFund"):
            return False, "NOT_EQUITY", None
        if profile.get("isActivelyTrading") is False:
            return False, "NOT_TRADING", None
        ttm_raw = _fmp_get("/key-metrics-ttm", api_key, {"symbol": symbol})
        ttm_rows = ttm_raw if isinstance(ttm_raw, list) else []
        ttm = ttm_rows[0] if ttm_rows else {}
        eps = (
            _num(ttm.get("netIncomePerShareTTM"))
            or _num(ttm.get("epsTTM"))
            or _num(profile.get("eps"))
        )
        if eps is None:
            inc_raw = _fmp_get(
                "/income-statement",
                api_key,
                {"symbol": symbol, "period": "annual", "limit": "1"},
            )
            inc_rows = inc_raw if isinstance(inc_raw, list) else []
            inc = inc_rows[0] if inc_rows else {}
            eps = _num(inc.get("epsdiluted")) or _num(inc.get("eps"))
        if eps is None:
            return False, "NO_TRAILING_EPS", None
        if eps <= 0:
            return False, "NON_POSITIVE_EPS", eps
        return True, "PASS", eps
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            return False, "FMP_RATE_LIMIT", None
        return False, "FMP_HTTP", None
    except Exception:
        return False, "FMP_ERROR", None


def filter_positive_eps_fmp(
    entries: list[tuple[str, str, str]],
    *,
    api_key: str,
    cache_path: Path = CACHE_PATH,
    resume: bool = True,
    retry_errors: bool = False,
    rate_per_sec: float = 4.0,
) -> tuple[list[tuple[str, str, str]], Counter]:
    cache = _load_json(cache_path) if resume else {}
    checked: dict = cache.setdefault("checked", {})
    if retry_errors:
        for key in list(checked):
            if checked[key].get("reason") in RETRY_REASONS:
                del checked[key]
    passed: list[tuple[str, str, str]] = []
    rejects: Counter = Counter()
    pending: list[tuple[str, str, str]] = []
    for tv, yahoo, name in entries:
        prev = checked.get(yahoo)
        if prev:
            if prev.get("reason") == "PASS":
                passed.append((tv, yahoo, name))
            else:
                rejects[prev.get("reason", "FMP_ERROR")] += 1
        else:
            pending.append((tv, yahoo, name))
    print(f"FMP EPS pending: {len(pending)} (cached: {len(entries) - len(pending)})")
    _ = rate_per_sec
    for i, (tv, yahoo, name) in enumerate(pending, start=1):
        ok, reason, eps = check_positive_eps_fmp(yahoo, api_key)
        if reason == "FMP_RATE_LIMIT":
            time.sleep(4.0)
            ok, reason, eps = check_positive_eps_fmp(yahoo, api_key)
        checked[yahoo] = {
            "reason": reason,
            "tv_part": tv,
            "name": name,
            "trailingEps": eps,
            "fmp": yahoo_to_fmp(yahoo),
        }
        if ok:
            passed.append((tv, yahoo, name))
        else:
            rejects[reason] += 1
        if i % 100 == 0 or i == len(pending):
            _save_json(cache_path, cache)
            print(f"  {i}/{len(pending)}  passed={len(passed)}  last={yahoo} {reason}")
    _save_json(cache_path, cache)
    return passed, rejects


def _valuation_row(
    tv: str,
    yahoo: str,
    name: str,
    eps: float | None,
    pe: float | None,
    growth: float | None,
    reason: str,
    span_years: int | None = None,
) -> dict:
    exchange = tv.split(":", 1)[0] if ":" in tv else ""
    rule = fair_value_rule(growth, span_years) if reason == "PASS" else None
    peg = lynch_peg(pe, growth)
    passed = bool(reason == "PASS" and is_undervalued(pe, growth, span_years))
    return {
        "yahoo": yahoo,
        "tv": tv,
        "name": name,
        "exchange": exchange,
        "epsTtm": eps,
        "peTtm": pe,
        "growth5y": growth,
        "growthSpanYears": span_years,
        "rule": rule or "",
        "pegLynch": peg,
        "pass": passed,
        "epsReason": reason,
    }


def _cached_quality_ok(prev: dict | None) -> bool:
    return bool(prev) and prev.get("qualityGate") == QUALITY_GATE


def scan_eps_and_valuation(
    entries: list[tuple[str, str, str]],
    *,
    api_key: str,
    eps_cache_path: Path = CACHE_PATH,
    val_cache_path: Path = VALUATION_CACHE_PATH,
    resume: bool = True,
    retry_errors: bool = False,
    rate_per_sec: float = 4.0,
    workers: int = FMP_SCAN_WORKERS,
    us_gates: LiquidityGates = US_LIQUIDITY,
    ca_gates: LiquidityGates = CA_LIQUIDITY,
    min_atr_pct: float = MIN_DAILY_ATR_PCT,
) -> tuple[list[tuple[str, str, str]], Counter, list[dict]]:
    eps_cache = _load_json(eps_cache_path) if resume else {}
    val_cache = _load_json(val_cache_path) if resume else {}
    eps_checked: dict = eps_cache.setdefault("checked", {})
    val_checked: dict = val_cache.setdefault("checked", {})
    if retry_errors:
        for key in list(eps_checked):
            if eps_checked[key].get("reason") in RETRY_REASONS:
                del eps_checked[key]
                val_checked.pop(key, None)
        for key in list(val_checked):
            if val_checked[key].get("reason") in RETRY_REASONS:
                del val_checked[key]
    passed: list[tuple[str, str, str]] = []
    rejects: Counter = Counter()
    rows: list[dict] = []
    pending: list[tuple[str, str, str]] = []
    reuse_eps: dict[str, tuple[float | None, float | None, float | None, int | None]] = {}
    for tv, yahoo, name in entries:
        prev_eps = eps_checked.get(yahoo)
        prev_val = val_checked.get(yahoo)
        have_val = bool(prev_val) and prev_val.get("reason") not in RETRY_REASONS
        gated = _cached_quality_ok(prev_eps)
        if prev_eps and have_val and gated:
            reason = prev_eps.get("reason", "FMP_ERROR")
            eps = _num(prev_eps.get("trailingEps"))
            pe = _num(prev_val.get("peTtm"))
            growth = _num(prev_val.get("growth5y"))
            span = prev_val.get("growthSpanYears")
            span_i = int(span) if isinstance(span, (int, float)) and span == int(span) else None
            if reason == "PASS":
                passed.append((tv, yahoo, name))
                rows.append(_valuation_row(tv, yahoo, name, eps, pe, growth, reason, span_i))
            else:
                rejects[reason] += 1
        elif (
            prev_eps
            and have_val
            and not gated
            and prev_eps.get("reason") not in ("PASS", *RETRY_REASONS)
        ):
            # Old EPS rejects still stand; no need to re-hit FMP for liquidity.
            rejects[prev_eps.get("reason", "FMP_ERROR")] += 1
        else:
            pending.append((tv, yahoo, name))
            if prev_eps and prev_val and prev_eps.get("reason") == "PASS":
                span = prev_val.get("growthSpanYears")
                span_i = int(span) if isinstance(span, (int, float)) and span == int(span) else None
                reuse_eps[yahoo] = (
                    _num(prev_eps.get("trailingEps")),
                    _num(prev_val.get("peTtm")),
                    _num(prev_val.get("growth5y")),
                    span_i,
                )
    workers = max(1, int(workers))
    print(
        f"FMP EPS+liquidity+ATR pending: {len(pending)} "
        f"(cached: {len(entries) - len(pending)}, workers={workers})"
    )
    _ = rate_per_sec

    def _persist() -> None:
        try:
            _save_json(val_cache_path, val_cache)
            _save_json(eps_cache_path, eps_cache)
        except OSError as exc:
            print(f"  cache save failed ({exc}); continuing in memory", file=sys.stderr)

    def _scan_one(
        item: tuple[str, str, str],
    ) -> tuple[
        tuple[str, str, str],
        str,
        str,
        float | None,
        float | None,
        float | None,
        float | None,
        int | None,
    ]:
        tv, yahoo, name = item
        gates = gates_for_tv(tv, us=us_gates, ca=ca_gates)
        symbol = yahoo_to_fmp(yahoo)
        reason = "FMP_ERROR"
        eps = pe = growth = atr_pct = None
        span: int | None = None
        for candidate in fmp_symbol_candidates(yahoo):
            symbol = candidate
            try:
                liq_reason, profile = fetch_profile_and_liquidity(symbol, api_key, gates)
            except urllib.error.HTTPError as exc:
                reason = "FMP_RATE_LIMIT" if exc.code == 429 else "FMP_HTTP"
                if reason == "FMP_RATE_LIMIT":
                    time.sleep(4.0)
                    continue
                continue
            except Exception:
                reason = "FMP_ERROR"
                continue
            if liq_reason:
                reason = liq_reason
                if liq_reason in ("FMP_ERROR", "FMP_HTTP", "FMP_RATE_LIMIT"):
                    continue
                break
            cached = reuse_eps.get(yahoo)
            if cached is not None:
                eps, pe, growth, span = cached
                reason = "PASS"
            else:
                _ok, reason, eps, pe, growth, span = fetch_eps_pe_growth(symbol, api_key)
                if reason == "FMP_RATE_LIMIT":
                    time.sleep(4.0)
                    _ok, reason, eps, pe, growth, span = fetch_eps_pe_growth(symbol, api_key)
                if reason in ("FMP_HTTP", "FMP_ERROR"):
                    continue
                if reason != "PASS":
                    break
            try:
                atr_reason, atr_pct = check_daily_atr_pct(
                    symbol, api_key, profile, min_atr_pct=min_atr_pct
                )
            except urllib.error.HTTPError as exc:
                reason = "FMP_RATE_LIMIT" if exc.code == 429 else "FMP_HTTP"
                if reason == "FMP_RATE_LIMIT":
                    time.sleep(4.0)
                    continue
                continue
            except Exception:
                reason = "NO_ATR"
                atr_pct = None
                break
            if atr_reason:
                reason = atr_reason
            break
        return item, symbol, reason, eps, pe, growth, atr_pct, span

    results: dict[
        str, tuple[str, str, float | None, float | None, float | None, float | None, int | None]
    ] = {}
    pending_by_yahoo = {yahoo: (tv, yahoo, name) for tv, yahoo, name in pending}
    done = 0

    def _write_cache(
        item: tuple[str, str, str],
        res: tuple[str, str, float | None, float | None, float | None, float | None, int | None],
    ) -> None:
        tv_i, yahoo_i, name_i = item
        symbol_i, reason_i, eps_i, pe_i, growth_i, atr_i, span_i = res
        rec = {
            "reason": reason_i,
            "tv_part": tv_i,
            "name": name_i,
            "trailingEps": eps_i,
            "fmp": symbol_i,
            "qualityGate": QUALITY_GATE,
            "atrPct": atr_i,
        }
        eps_checked[yahoo_i] = rec
        val_checked[yahoo_i] = {
            "reason": reason_i,
            "tv_part": tv_i,
            "name": name_i,
            "epsTtm": eps_i,
            "peTtm": pe_i,
            "growth5y": growth_i,
            "growthSpanYears": span_i,
            "fmp": symbol_i,
            "qualityGate": QUALITY_GATE,
            "atrPct": atr_i,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_scan_one, item): item for item in pending}
        for fut in as_completed(futures):
            item, symbol, reason, eps, pe, growth, atr_pct, span = fut.result()
            _tv, yahoo, _name = item
            results[yahoo] = (symbol, reason, eps, pe, growth, atr_pct, span)
            done += 1
            if done % 100 == 0 or done == len(pending):
                for y, res in results.items():
                    _write_cache(pending_by_yahoo[y], res)
                _persist()
                print(f"  {done}/{len(pending)}  last={yahoo} {reason}")

    for tv, yahoo, name in pending:
        res = results.get(yahoo)
        if res is None:
            continue
        symbol, reason, eps, pe, growth, atr_pct, span = res
        _write_cache((tv, yahoo, name), res)
        if reason == "PASS":
            passed.append((tv, yahoo, name))
            rows.append(_valuation_row(tv, yahoo, name, eps, pe, growth, reason, span))
        else:
            rejects[reason] += 1
    _persist()
    return passed, rejects, rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter STOCK-TICKERS.txt via FMP TTM EPS > 0")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    api_key = load_fmp_api_key()
    if not api_key:
        print("FMP_API_KEY is not set.", file=sys.stderr)
        return 1
    tickers, tv_map, name_map, err = read_list_file(TV_LIST_STOCK_TICKERS)
    if err:
        print(err, file=sys.stderr)
        return 1
    entries = [(tv_map.get(y) or y, y, name_map.get(y) or "") for y in tickers]
    if args.limit > 0:
        entries = entries[: args.limit]
    passed, rejects = filter_positive_eps_fmp(
        entries,
        api_key=api_key,
        resume=not args.no_resume,
        retry_errors=args.retry_errors,
    )
    print()
    print(f"In {len(entries)} → EPS>0 {len(passed)}")
    for reason, n in sorted(rejects.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {n}")
    if not args.write:
        print(f"Dry-run: would write {len(passed)} lines -> {TV_LIST_STOCK_TICKERS}")
        return 0
    write_list_file(TV_LIST_STOCK_TICKERS, passed)
    print(f"Wrote {len(passed)} lines -> {TV_LIST_STOCK_TICKERS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
