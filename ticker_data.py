"""
Ticker list I/O and Yahoo Finance info/filtering. No UI or Streamlit.
"""
import os
import pandas as pd
import yfinance as yf

# List files (same folder as this package); format: EXCHANGE:SYMBOL comma-separated
TV_LIST_SMALL_CAP = "TV-LIST-SMALL_CAP_2B-10B.txt"
TV_LIST_BIG_CAP = "TV-LIST-BIG_CAP_10B.txt"
TV_LIST_ETF = "TV-LIST-ETF.txt"


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
    """
    try:
        t = yf.Ticker(ticker)
        i = t.info
        quote_type = i.get("quoteType") or ""
        exchange = (i.get("exchange") or "").upper()
        avg_vol = i.get("averageVolume")
        if avg_vol is None and require_mc_vol:
            hist = t.history(period="1mo")
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
    except Exception:
        return False, "INFO_ERROR", None


def fetch_fallback_company_name(ticker: str) -> str:
    """Fetch company name from yfinance when get_ticker_info_and_filter returns None (INFO_ERROR)."""
    try:
        i = yf.Ticker(ticker).info
        return i.get("longName") or i.get("shortName") or ticker
    except Exception:
        return ticker
