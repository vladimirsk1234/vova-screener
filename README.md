# Sequence Vova — Streamlit (production) + React/Mongo candidate

Two apps live in this repo during migration:

| App | Role | Status |
|-----|------|--------|
| Streamlit (`streamlit_app.py`, `headless_scanner.py`) | Production screener on Streamlit Community Cloud | Unchanged, still the product |
| React + NestJS + MongoDB (`apps/`, `packages/`) | Mobile-first candidate, runs locally today | Working: real scans, storage, charts, journal |

Streamlit stays live until the Railway cutover checklist in [docs/architecture/migration.md](docs/architecture/migration.md) is complete.

## Streamlit (unchanged)

```bash
streamlit run headless_scanner.py
```

## React stack locally

```bash
npm install
npm run dev
```

- Web: http://localhost:5173 (phone on same Wi-Fi: `http://<PC-LAN-IP>:5173`)
- API: http://localhost:3001/api
- MongoDB: started automatically as a persistent single-node replica set in `.data/mongo`
  (no Docker or MongoDB install needed). Set `MONGO_URI` to use your own instance.

Only the web port needs to be reachable from the phone — the browser talks to `/api`,
which Vite proxies to the API.

### What works locally

- Full scans over the ticker universe imported from `STOCK-TICKERS.txt` / `TV-LIST-ETF.txt`
  (2308 stocks + 759 ETFs) with live progress over SSE, cancel support
- Bars cached in MongoDB (`barSeries`), so repeat scans skip Yahoo
- Buy and sell signals persisted per run, plus rejected symbols with reason breakdown
- "New since last run" delta per scan
- Chart screen: candles, critical level, TP/SL lines, multi-timeframe status
- Trade journal with mark-to-market, TP/SL auto-close check, monthly P&L report
- Scan settings persisted as a preset

### Useful commands

```bash
npm run dev         # api + web together
npm run dev:api     # NestJS only
npm run dev:web     # Vite only
npm run parity      # TS engine vs Python golden fixture
npm run typecheck   # engine + api + web
```

## Docs

See [docs/architecture/README.md](docs/architecture/README.md).
