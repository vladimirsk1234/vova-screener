# Sequence Vova — Streamlit (production) + React/Mongo candidate

Two apps live in this repo during migration:

| App | Role | Status |
|-----|------|--------|
| Streamlit (`streamlit_app.py`, `headless_scanner.py`) | Production screener on Streamlit Community Cloud | Still the product; optional Nest/Mongo via `VOVA_API_URL` |
| React + NestJS + MongoDB (`apps/`, `packages/`) | Mobile-first candidate on this PC | Working: real scans, storage, charts, journal |

Streamlit stays live until the Railway cutover checklist in [docs/architecture/migration.md](docs/architecture/migration.md) is complete.
**Alternative to Railway:** keep this PC always on — see [home-server.md](docs/architecture/home-server.md).

## Streamlit

```bash
streamlit run headless_scanner.py
```

Optional: point scans at NestJS + Mongo (home server or tunnel):

```toml
# .streamlit/secrets.toml  (see .streamlit/secrets.toml.example)
VOVA_API_URL = "http://127.0.0.1:3001/api"
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

## Home PC server + phone from anywhere (no Railway)

```bat
RUN_HOME_SERVER.bat
RUN_TUNNEL.bat
```

1. Leave the PC on; disable Sleep.
2. `powershell -File scripts\home-server\install-autostart.ps1` — start API/web at logon.
3. `RUN_TUNNEL.bat` — Cloudflare Quick Tunnel; open the `https://….trycloudflare.com` URL on the phone with **mobile data**.
4. Verify: `npm run home-server:verify`

Full guide: [docs/architecture/home-server.md](docs/architecture/home-server.md).

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
npm run dev                 # api + web together
npm run dev:api             # NestJS only
npm run dev:web             # Vite only
npm run home-server         # background home-server start (Windows)
npm run home-server:stop
npm run home-server:verify  # health + smoke MANUAL scan
npm run tunnel              # Cloudflare Quick Tunnel to :5173
npm run parity              # TS engine vs Python golden fixture
npm run typecheck           # engine + api + web
```

## Docs

See [docs/architecture/README.md](docs/architecture/README.md).
