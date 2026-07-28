# Sequence Vova — Expo Go app

Native Expo Go client for Sequence Vova. Streamlit web screener in the repo root stays **unchanged** for result comparison.

## Run on iPhone (Expo Go)

1. Install **Expo Go** from the App Store.
2. From this folder:

```bash
cd mobile
npm start
```

3. In Expo Go, tap **Enter URL manually** and type `exp://<your-pc-ip>:8081`
   (this PC: `exp://192.168.1.113:8081`). The iPhone must be on the same Wi‑Fi/LAN as the PC.

EAS project id: `57d9162f-b2bf-4549-906c-cfb4a63a3f77`

## Troubleshooting "Request timed out" in Expo Go

`npm start` runs `expo start --offline` on purpose. Without `--offline`, the CLI stops at an
interactive prompt (`Log in` / `Proceed anonymously`, see expo.fyi/unverified-app-expo-go)
because `app.json` carries an EAS `projectId` while no Expo account is logged in. While that
prompt is waiting, the dev server never finishes serving and Expo Go reports a timeout.

Checks, in order:

1. **Is Metro up?** In a PC browser open `http://192.168.1.113:8081/status` — it must print
   `packager-status:running`.
2. **Can the phone reach the PC?** Open the same URL in iPhone Safari. Timeout here means the
   phone is on a different subnet / has client isolation (common on corporate Wi‑Fi).
   Fallback: enable iPhone **Personal Hotspot** and join it from the PC, then restart with
   `REACT_NATIVE_PACKAGER_HOSTNAME=<pc-ip-on-hotspot>`.
3. **Wrong LAN IP picked?** This PC has several adapters; only `192.168.1.113` (Ethernet 2)
   is routable. Force it:

```bash
$env:REACT_NATIVE_PACKAGER_HOSTNAME="192.168.1.113"; npx expo start --offline
```

4. **QR code instead of manual URL:** log in once with `npx expo login`, then `npm run start:lan`.
5. **`npm run start:tunnel` does not work on this network** — ngrok's endpoint fails TLS here
   (corporate TLS inspection), so `expo start --tunnel` reports
   `ngrok tunnel took too long to connect`. Use LAN or hotspot instead.

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
