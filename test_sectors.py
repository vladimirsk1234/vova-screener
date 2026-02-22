"""
Test script for sector-based grouping of valid ticker results.
Run before changing headless_scanner.py to verify:
  1) yfinance returns sector (and industry) for stocks
  2) Fallback for missing sector (e.g. ETFs) works
  3) Grouping results by sector produces correct structure

Usage: python test_sectors.py
"""
import pandas as pd
import yfinance as yf


def get_sector_from_yfinance(ticker: str) -> tuple:
    """Fetch sector and industry from yfinance (mirrors planned info_dict fields)."""
    try:
        t = yf.Ticker(ticker)
        i = t.info
        sector = i.get("sector")
        industry = i.get("industry")
        if sector is not None and isinstance(sector, str) and not sector.strip():
            sector = None
        if industry is not None and isinstance(industry, str) and not industry.strip():
            industry = None
        return sector, industry
    except Exception:
        return None, None


def build_mock_result_row(symbol: str, company_name: str, sector_from_info, **kwargs) -> dict:
    """Build one result row as in headless_scanner (with Sector key)."""
    sector_display = sector_from_info if sector_from_info else "-"
    return {
        "Symbol": f"https://www.tradingview.com/chart/?symbol={symbol}",
        "Company Name": company_name,
        "Sector": sector_display,
        "TP": kwargs.get("TP", 100.0),
        "SL": kwargs.get("SL", 95.0),
        "RR": kwargs.get("RR", 1.5),
        "MC (B/M)": kwargs.get("MC", 10.0),
        "PE": kwargs.get("PE", 25.0),
        "Position Size (shares)": kwargs.get("Position Size (shares)", 10),
        "Position Value ($)": kwargs.get("Position Value ($)", 1000.0),
        "New": 1,
        "Valid": 1,
        "Strong": 0,
    }


def get_sector_display_order(series: pd.Series) -> list:
    """Order sectors for display: alphabetical, with '—' last (same as planned logic)."""
    uniq = series.dropna().unique().tolist()
    other = "-"  # fallback for missing sector
    if other in uniq:
        uniq.remove(other)
    uniq.sort(key=str.lower)
    if other in series.values:
        uniq.append(other)
    return uniq


def group_results_by_sector(table_rows: list) -> dict:
    """Group list of result dicts by Sector key. Returns {sector_name: list of rows}."""
    df = pd.DataFrame(table_rows)
    if df.empty or "Sector" not in df.columns:
        return {}
    order = get_sector_display_order(df["Sector"])
    out = {}
    for sector in order:
        subset = df[df["Sector"] == sector]
        if not subset.empty:
            out[sector] = subset.to_dict("records")
    return out


def run_tests() -> bool:
    all_ok = True

    # --- Test 1: yfinance returns sector for known stocks ---
    print("Test 1: Fetch sector from yfinance for AAPL, MSFT, JPM, JNJ")
    tickers = ["AAPL", "MSFT", "JPM", "JNJ"]
    sectors_fetched = {}
    for t in tickers:
        sector, industry = get_sector_from_yfinance(t)
        sectors_fetched[t] = (sector, industry)
        print(f"  {t}: sector={sector!r}, industry={industry!r}")
    if not any(s[0] for s in sectors_fetched.values()):
        print("  FAIL: No sector returned for any ticker (yfinance may be down or format changed).")
        all_ok = False
    else:
        print("  PASS: At least one sector returned.\n")

    # --- Test 2: ETF / missing sector → "—" ---
    print("Test 2: ETF or missing sector (SPY) may have no sector -> display as '-'")
    sector_spy, _ = get_sector_from_yfinance("SPY")
    row_etf = build_mock_result_row("SPY", "SPDR S&P 500", sector_spy)
    assert row_etf["Sector"] == (sector_spy if sector_spy else "-"), "Sector fallback should be '-'"
    print(f"  SPY sector from yfinance: {sector_spy!r} -> display: {row_etf['Sector']!r}")
    print("  PASS: Fallback to '-' works.\n")

    # --- Test 3: Build mock table_rows and group by sector ---
    print("Test 3: Build mock results and group by sector")
    table_rows = []
    for t in tickers:
        sector, _ = sectors_fetched[t]
        name = {"AAPL": "Apple", "MSFT": "Microsoft", "JPM": "JPMorgan", "JNJ": "Johnson & Johnson"}.get(t, t)
        table_rows.append(build_mock_result_row(t, name, sector))
    table_rows.append(build_mock_result_row("SPY", "SPDR S&P 500", sector_spy))

    grouped = group_results_by_sector(table_rows)
    print(f"  Sectors found: {list(grouped.keys())}")
    for sec, rows in grouped.items():
        print(f"    {sec}: {len(rows)} ticker(s) — {[r['Company Name'] for r in rows]}")
    if len(grouped) < 1:
        print("  FAIL: No sectors in grouped result.")
        all_ok = False
    else:
        total_rows = sum(len(r) for r in grouped.values())
        if total_rows != len(table_rows):
            print(f"  FAIL: Row count mismatch (grouped={total_rows}, original={len(table_rows)}).")
            all_ok = False
        else:
            print("  PASS: Grouping preserves row count and splits by sector.\n")

    # --- Test 4: Sector order has "—" last ---
    print("Test 4: Display order: alphabetical sectors, '-' last")
    order = get_sector_display_order(pd.Series([r["Sector"] for r in table_rows]))
    print(f"  Order: {order}")
    if "-" in order and order[-1] != "-":
        print("  FAIL: '-' should be last in order.")
        all_ok = False
    else:
        print("  PASS: Order is correct.\n")

    return all_ok


if __name__ == "__main__":
    print("=" * 60)
    print("Sector feature test (no changes to headless_scanner.py)")
    print("=" * 60)
    ok = run_tests()
    print("=" * 60)
    print("OVERALL: PASS" if ok else "OVERALL: FAIL")
    print("=" * 60)
    exit(0 if ok else 1)
