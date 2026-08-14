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

Put `FMP_API_KEY=` in a repo-root `.env` (Premium; Starter has no Canada / short history).
Without it, scans and charts still work; the Fundamentals page and History EPS tagging do not.
Rebuild the profitable-stock list with `python scripts/fundamentals_fmp.py` (add `--write` to overwrite `STOCK-TICKERS.txt`).

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

- Stocks and ETF are scanned in the background — one hourly pass covering Daily, Weekly and Monthly
  together (09:05–17:05 ET, Mon–Fri); post-close ticks are themselves the period-close scans — so
  **Results** always shows the latest scan without pressing anything
- Results is Stocks / ETF / Manual → D / W / M → New / Valid / Closed, sortable by RR, P&L, mark
  or ticker; Valid and Closed carry P&L
- New is the signals that appeared on the current bar of that timeframe, Valid the ones that appeared
  on an earlier bar and still hold, and each Valid card says how many bars it has been running — a
  symbol the scanner meets for the first time when it has already been running for four bars opens
  straight into Valid instead of sitting next to today's breakouts
- Scans never filter on RR (MIN RR is "any"), so RR is a sort key rather than a gate — every list
  in the app, Results, Manual and History alike, can be ordered by it. RR plays no part in New vs
  Valid either: the age of a signal is measured with the RR requirement off, so the chart badge and
  the tabs always agree
- Only a period-close scan confirms or closes a signal, so one that appears and disappears inside a
  single period never gets a realized P&L and never lands in History
- Closed is the Streamlit SELL TO CLOSE list. A trade ends on the sell-to-close break and on
  nothing else: a stop taken out or a target reached changes what it is worth, not whether it is
  on, and neither does the buy setup lapsing — that just takes the position off screen. A symbol
  Yahoo could not deliver is left exactly as it was, so a data outage never closes a position
- A position is the close scan's trade, replayed from the bars, so it does not have to have been
  opened here to be closed here. Most of any Closed list is symbols the app never reported as a
  buy — a break puts the sequence down, so a symbol closing today is a reject in the buy scan —
  and each one is written down entry and exit together. A trade the app is already carrying is
  priced from the bar the replay entered it on, not the day the app first met the symbol
- Any signal opens a chart where it can be marked Interested / Not interested; the mark shows in
  the lists and sorts on every tab
- History: win rate, net P&L, avg R, avg RR at entry, avg hold and an equity curve over closed
  signals, for D / W / M / All
- Manual scan for ad-hoc tickers with live SSE progress and a rejected-reason breakdown
- Bars cached in MongoDB (`barSeries`), so repeat scans skip Yahoo
- Fundamentals (Fast Graphs–style): FMP Premium via `FMP_API_KEY` in a repo-root `.env`. Yahoo stays the EOD/TA source. Chart button **Fundamentals** opens Summary / Forecasting / Performance / Profile.
- One setting: Max risk per signal. It is the single source of position size across scans, lists
  and charts, and changing it re-sizes every open signal immediately
- Trades from the old journal are imported on first boot, so History still covers everything closed
  before the app started tracking signals on its own

Background scanning can be tuned with `VOVA_SESSION_SCAN_CRON` (default `5 9-17 * * 1-5`, i.e.
09:05 to 17:05 America/New_York), or switched off entirely with `VOVA_BACKGROUND_SCANS=off`.
There are no separate close crons: `periodClose` is decided from the clock when each run starts,
so the 16:05 / 17:05 ticks after the cash close confirm and close tracked signals.

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
npm run smoke:age           # New / Valid split by bar age, D / W / M
npm run smoke:legacy        # import of the old trade journal
npm run smoke:normalize     # one ticker format, one record per trade
npm run smoke:close-live    # real scan vs the Closed tab (needs Yahoo)
npm run test:e2e            # Playwright (Pixel 7 + desktop)
```

## Docs

See [docs/architecture/README.md](docs/architecture/README.md).
