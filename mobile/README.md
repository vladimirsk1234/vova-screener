# Sequence Vova — Expo Go app

Native Expo Go client for Sequence Vova. Streamlit web screener in the repo root stays **unchanged** for result comparison.

## Run on iPhone (Expo Go)

1. Install **Expo Go** from the App Store.
2. From this folder:

```bash
cd mobile
npm start
```

3. Scan the QR code with the Camera / Expo Go app (same Wi‑Fi as your PC).

EAS project id: `57d9162f-b2bf-4549-906c-cfb4a63a3f77`

## What it does

- Scan: Stocks / ETF / Manual, Daily/Weekly/Monthly, BUY TO OPEN / SELL TO CLOSE
- Yahoo Finance OHLC on device (same chart API family as yfinance)
- Results + candle chart + TradingView deep links
- SQLite on iPhone: scan history, trade journal, Update open trades (TP/SL), monthly P&L + CSV share

## Compare with Streamlit

1. Run web: `RUN_SCREENER.bat` (or Streamlit Cloud).
2. Run Expo with the same Manual tickers / TF / risk / min RR.
3. Compare Valid/New/Strong, TP/SL, SELL P&L.

## Parity fixture (optional)

From repo root, with Python env:

```bash
python scripts/export_parity_fixture.py
```

Then in `mobile/`:

```bash
npx tsx scripts/check_parity.ts
```
