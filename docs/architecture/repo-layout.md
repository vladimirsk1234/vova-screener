# Repo layout

```
apps/
  web/                 React 19 + Vite mobile-first SPA (sole UI)
  api/                 NestJS REST + SSE + in-process scan runner
packages/
  engine/              @vova/engine — pure TS strategy + evaluation (from former mobile/src/engine)
docs/
  architecture/        this design set (includes home-server.md)
scripts/
  home-server/         Windows always-on PC + Cloudflare Tunnel helpers
cloudflared/
  config.example.yml   named-tunnel ingress template
.data/mongo            local MongoDB dbPath (gitignored)
# Python / Streamlit stay at repo ROOT until cutover (not early legacy/)
streamlit_app.py
headless_scanner.py
sequence_vova.py
requirements.txt
STOCK-TICKERS.txt
TV-LIST-ETF.txt
```

## Inside `apps/api`

```
src/db/                Mongoose schemas, local Mongo bootstrap
src/market/            Yahoo client + barSeries cache
src/universe/          ticker file import, universe resolution
src/scans/             runner, progress bus (SSE), controller, background scheduler
src/tracking/          signal lifecycle + Results and History reads
src/instruments/       chart payloads, multi-TF status
src/settings/          the single Max risk setting
src/presets/           persisted chart params
src/dev/               smoke scripts (Mongo, signal tracker)
```

## Planned (later phases)

```
apps/worker/           BullMQ processors (runner body moves here unchanged)
packages/contracts/    Zod schemas shared by api + web
packages/charts/       Lightweight Charts wrappers
packages/ui/           shared tokens
```

## Dependency rules

- `@vova/engine` — zero I/O, zero framework imports (Yahoo client lives in `apps/api`)
- `apps/web` talks to the API over `/api` only; it imports `@vova/engine` for types/pure helpers
- Python Streamlit must not import TS packages; parity goes through fixtures only
- Do not break `streamlit_app.py` / `requirements.txt` while Cloud is production

## Workspaces

npm workspaces (`package.json` root). Node ≥ 20. `npm run dev` starts API + web together.
