# Sequence Vova — Streamlit (production) + React/Mongo candidate

Two apps live in this repo during migration:

| App | Role | Status |
|-----|------|--------|
| Streamlit (`streamlit_app.py`, `headless_scanner.py`) | Production screener on Streamlit Community Cloud | Still the product; optional Nest/Mongo via `VOVA_API_URL` |
| React + NestJS + MongoDB (`apps/`, `packages/`) | Mobile-first candidate on this PC | Working: background scans, tracked signals, charts, history |

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

- Stocks and ETF are scanned in the background — hourly through the session plus one right after
  each period closes — so **Results** always shows the latest scan without pressing anything
- Results is Stocks / ETF / Manual → D / W / M → New / Valid / Closed, sortable by RR, P&L, mark
  or ticker; Valid and Closed carry P&L
- Scans never filter on RR (MIN RR is "any"), so RR is a sort key rather than a gate — every list
  in the app, Results, Manual and History alike, can be ordered by it
- A signal reaches Valid only by surviving a period close, so one that appears and disappears
  inside a single period never gets a P&L and never lands in History
- Signals close on their stop, their target, or a Sequence Vova sell-to-close on a bullish break —
  and, failing all three, when the scan stops calling the symbol a buy. A symbol Yahoo could not
  deliver is left open, so a data outage never closes a position
- Any signal opens a chart where it can be marked Interested / Not interested; the mark shows in
  the lists and sorts on every tab
- History: win rate, net P&L, avg R, avg RR at entry, avg hold and an equity curve over closed
  signals, for D / W / M / All
- Manual scan for ad-hoc tickers with live SSE progress and a rejected-reason breakdown
- Bars cached in MongoDB (`barSeries`), so repeat scans skip Yahoo
- One setting: Max risk per signal. It is the single source of position size across scans, lists
  and charts, and changing it re-sizes every open signal immediately
- Trades from the old journal are imported on first boot, so History still covers everything closed
  before the app started tracking signals on its own

Background scanning can be tuned with `VOVA_SESSION_SCAN_CRON` (default `5 10-15 * * 1-5`, i.e.
10:05 to 15:05), `VOVA_DAILY_CLOSE_CRON`, `VOVA_WEEKLY_CLOSE_CRON` and `VOVA_MONTHLY_CLOSE_CRON`
(all America/New_York), or switched off entirely with `VOVA_BACKGROUND_SCANS=off`.

An hourly pass re-downloads every symbol on all three timeframes, roughly 12k Yahoo requests in
about a minute. If Yahoo starts throttling, bars fall back to the cached series rather than
failing; watch the `cached/total` figure in the scheduler log and widen the cron if it climbs.

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
npm run smoke:tracker       # signal lifecycle end-to-end, no Yahoo needed
npm run smoke:legacy        # import of the old trade journal
npm run test:e2e            # Playwright (Pixel 7 + desktop)
```

## Docs

See [docs/architecture/README.md](docs/architecture/README.md).
