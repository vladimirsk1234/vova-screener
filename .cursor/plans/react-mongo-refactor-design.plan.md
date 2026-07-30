---
name: react-mongo-refactor-design
overview: 'Phased refactor into a TypeScript monorepo: pure-TS engine, NestJS + MongoDB + BullMQ, and a single React 19 mobile-first web client (no Expo/standalone app). Streamlit Community Cloud stays the active product until Railway cutover is complete; Python remains the parity oracle. Budget ≤$250/year on Railway.'
todos:
  - id: docs-scaffold
    content: 'docs/architecture scaffold + C4; single client apps/web; no Expo; Streamlit Cloud = prod until cutover'
    status: completed
  - id: adrs
    content: 'ADRs: TS engine, Mongo barSeries, BullMQ, NestJS, Lightweight Charts, Vite SPA, Railway, Yahoo, mobile-first web-only (no Expo), parallel Streamlit until cutover'
    status: completed
  - id: data-model
    content: 'data-model.md: collections, binary barSeries, indexes/TTLs; journal/trades for web via Mongo (no mobile SQLite product)'
    status: completed
  - id: hosting-cost
    content: 'hosting-and-cost.md: Railway four-service + Cloudflare Pages; Streamlit Cloud parallel until cutover; no Expo/EAS costs'
    status: completed
  - id: api-contract
    content: 'api.md: REST + SSE contracts'
    status: completed
  - id: engine-parity
    content: 'engine-parity.md: salvage mobile/src/engine → packages/engine then delete mobile/; Python oracle + golden fixtures'
    status: completed
  - id: frontend-design
    content: 'frontend.md: mobile-first web IA (bottom nav, cards, touch), desktop enhancement, Playwright phone viewports, chart primitives'
    status: completed
  - id: repo-layout
    content: 'repo-layout.md: no apps/mobile; Python/Streamlit at root until cutover (not early legacy/)'
    status: completed
  - id: migration-plan
    content: 'migration.md: 6 phases — salvage+delete Expo, parallel Streamlit, Railway candidate, cutover then retire Streamlit'
    status: completed
  - id: phase1-scaffold
    content: 'Phase 1 code: npm monorepo, @vova/engine, apps/web mobile-first shell running locally'
    status: completed
  - id: phase2-engine
    content: 'Phase 2: scan evaluation + tickers + TV symbols + series codec ported; EMA/BB/MACD chart overlays still open'
    status: completed
  - id: phase3-api
    content: 'Phase 3: NestJS + Mongo + SSE + universe import + bar cache + journal + reports (local, no Docker; BullMQ deferred to Phase 5)'
    status: completed
  - id: phase4-web
    content: 'Phase 4: scan form with live progress, results cards, rejected breakdown, chart overlays, journal, monthly P&L (Playwright gates open)'
    status: completed
  - id: phase5-railway
    content: 'Phase 5: Railway candidate deploy; Streamlit stays on'
    status: pending
  - id: phase6-cutover
    content: 'Phase 6: cutover after exit criteria; retire Streamlit'
    status: pending
isProject: false
---
## Current state (what we are replacing)

- [headless_scanner.py](headless_scanner.py) (1578 lines) is simultaneously the Streamlit UI, the scan orchestrator, the thread pool manager and the progress renderer. `run_scan()` is the only clean seam.
- [sequence_vova.py](sequence_vova.py) (1388 lines) holds all hand-rolled math: ATR/EMA/SMA/MACD/DMI-ADX/Bollinger/Elder plus the sequence state machine and HH/LH/DT/HL/LL/DB structure labelling. No TA-Lib.
- [chart_preview.py](chart_preview.py) builds a Plotly candlestick with critical-level step lines, extension lines, structure markers, fib levels and a multi-timeframe watermark from [watermark_status.py](watermark_status.py).
- State lives entirely in `st.session_state`; nothing persists. Ticker universes are flat files (`EXCHANGE:SYMBOL|Company Name`) in [STOCK-TICKERS.txt](STOCK-TICKERS.txt) / [TV-LIST-ETF.txt](TV-LIST-ETF.txt). There is no authentication.
- A scan of ~2300 symbols runs synchronously inside one Streamlit rerun, hammering Yahoo Finance with 4 download threads + 16 TA threads, with a low-memory streaming mode bolted on ([scan_memory.py](scan_memory.py)).
- **Reusable asset (then delete):** [mobile/](mobile) has a TypeScript engine port ([mobile/src/engine/sequenceVova.ts](mobile/src/engine/sequenceVova.ts), [dataUtils.ts](mobile/src/engine/dataUtils.ts)), Yahoo client, scan helper, SQLite journal schema, and parity harness ([mobile/scripts/check_parity.ts](mobile/scripts/check_parity.ts)). **Expo / standalone app is out of scope** — salvage engine (+ useful mapping/parity code) into `packages/engine`, then **delete `mobile/`**. Journal/trades ship only via Mongo + web UI. UX patterns (bottom tabs, cards, chips) inform the mobile-first web client only as reference.

The refactor promotes the salvaged TS engine into a shared package and builds NestJS + a single React web client around it. Streamlit stays live on Community Cloud until Railway cutover.

## Target architecture

```mermaid
flowchart TD
    subgraph clients [Single client]
        Web["apps/web - React 19 mobile-first Vite SPA"]
    end
    subgraph api [apps/api - NestJS]
        REST[REST controllers]
        SSE[SSE progress stream]
        APP[Application services]
        PORTS[Ports: MarketData, Universe, Repos]
    end
    subgraph worker [apps/worker - separate Railway service]
        SCAN[ScanJobProcessor]
        SYNC[BarSyncProcessor]
        CRON[Repeatable pre-market scan]
    end
    Q[("Redis - BullMQ queue + pub/sub")]
    ENG["packages/engine - pure TS strategy"]
    DB[("MongoDB - self-hosted on Railway")]
    YF[Yahoo Finance]

    Web --> REST
    Web --> SSE
    REST --> APP
    APP --> PORTS
    PORTS --> DB
    APP -->|enqueue| Q
    Q --> SCAN
    Q --> SYNC
    Q --> CRON
    SCAN --> ENG
    SCAN -->|results| DB
    SYNC --> YF
    SYNC --> DB
    SCAN -.progress.-> Q
    Q -.-> SSE
```

The decisive structural change: **scans stop being request-scoped**. A scan becomes a persisted `ScanRun` document processed by a separate worker service, with progress published over Redis and relayed to the browser via SSE. This removes the low-memory hacks, makes runs resumable and cancellable, and gives run history for free.

The second decisive change: **MongoDB becomes the market-data cache**. A `BarSync` job keeps bar series current incrementally; scans read bars from Mongo instead of calling Yahoo ~2300 times per run. This is what makes a large scan fast and removes the rate-limit fragility that today forces retries and backoff in [headless_scanner.py](headless_scanner.py).

The third change, which only a paid worker makes possible: **scans become scheduled**. A repeatable BullMQ job runs the full universe after the US close so results are waiting before the open, instead of the operator pressing START and watching a progress bar. This is the largest single product improvement over the current app and its marginal compute cost is a few cents a month.

The API and worker must stay **separate services**. The scan is a CPU-bound loop; running it in the API process would block Node's event loop and freeze SSE and HTTP for the duration.

## Monorepo layout

pnpm workspaces + Turborepo, TypeScript strict everywhere.

```
apps/
  api/       NestJS 11 REST + SSE
  worker/    Node process running BullMQ processors, shares api's domain modules
  web/       React 19 + Vite 7 SPA — sole UI, mobile-first
packages/
  engine/    @vova/engine    pure functions (salvaged from mobile/src/engine), zero I/O
  contracts/ @vova/contracts Zod schemas -> DTOs + OpenAPI + client types
  charts/    @vova/charts    Lightweight Charts primitives + React chart components
  ui/        @vova/ui        Tailwind v4 + shadcn/ui design system (mobile-first tokens)
# Python/Streamlit stay at repo root as active Cloud product until cutover;
# only then optional move to legacy/ as permanent parity oracle.
# mobile/ is deleted after engine salvage — no Expo app.
```

## Is TypeScript the right engine language? (honest assessment)

Recommendation: **yes, but for specific reasons, not generic ones.**

Arguments for:

- A parity-tested TS port of the state machine already exists ([mobile/src/engine/sequenceVova.ts](mobile/src/engine/sequenceVova.ts), 632 lines) with a fixture-based harness ([mobile/scripts/check_parity.ts](mobile/scripts/check_parity.ts)). Salvaging it into `@vova/engine` avoids a from-scratch rewrite; without a standalone app there is no on-device scan path, so the engine runs only in the worker (and in Vitest).
- The workload is a **scalar loop over bars per symbol**, not matrix algebra. `pinePython` in [sequence_vova.py](sequence_vova.py) is a bar-by-bar loop that the code already had to accelerate with `numba` precisely because pandas cannot vectorise it. That kind of loop over typed arrays is exactly what V8 JITs well, so Python's numeric advantage largely evaporates here.
- One language across web, mobile, server and the shared Zod contracts.

Honest arguments against, and the mitigations:

- **Porting risk is the real cost.** Roughly 700 lines of chart-side math in [sequence_vova.py](sequence_vova.py) (MACD, DMI/ADX, Bollinger, Elder envelope/impulse, peak/trough labelling, fib, extension lines) plus the pandas resampling in [data_utils.py](data_utils.py) are not yet ported. Every ported line is a chance to silently break TradingView parity. Mitigation: keep Streamlit/Python active until Railway cutover, then retain Python as the permanent parity oracle; make the golden-fixture suite a blocking CI gate.
- **Python wins if the roadmap turns quantitative.** Backtesting sweeps, walk-forward optimisation or ML feature work have no real TS equivalent to pandas/vectorbt/statsmodels. Mitigation: this is additive, not a reversal — a Python analytics sidecar can read bar series straight out of MongoDB later without touching the web stack.

## Is MongoDB the right database? (honest assessment)

Recommendation: **yes, with one important correction to the bar storage model.**

Where Mongo is genuinely the better fit:

- The yfinance `.info` payloads cached in [ticker_data.py](ticker_data.py) have a wide, unstable, provider-defined shape. Storing them as documents avoids either a migration per field or a JSON column pretending to be relational.
- BUY rows and SELL rows are different shapes (`TP/SL/RR/Strong` versus `Entry/Exit/P&L`), and a future second strategy will add a third. A `signals` collection with a discriminated payload models that directly.
- The engine never queries individual bars. It always wants *the whole series* for one symbol and interval. So the natural storage unit is the series, which a document store holds better than a row table.

**Correction to an earlier draft of this design:** it originally proposed a MongoDB *time-series collection* with one document per bar. Storing **one document per `(symbol, interval)` holding the series as compact binary column arrays** is better. For ~3,150 instruments across daily/weekly/monthly, per-bar documents mean roughly 3.6M documents and a mandatory `_id` index of comparable size to the payload; the column-array model holds the same data in tens of megabytes and turns the scan hot path into one document read per symbol instead of ~500 row reads. With a paid database the storage argument is secondary, but the read-amplification argument stands on its own: it is the difference between ~3,000 and ~1.5M document reads per full-universe scan. The accepted trade-off is that individual bars are not server-queryable and appending a bar means decode-append-encode; neither matters, because the engine always consumes whole series and `BarSync` touches each series once a day.

Where Mongo is merely adequate: `trades` and monthly P&L are relational-shaped work. That is fine — the volume is tiny and `$group` handles the reporting — but it is not a point in Mongo's favour.

Mongo must be configured as a **single-node replica set**, not a standalone `mongod`, both locally and on Railway. Multi-document transactions and change streams both require it, and the default container templates usually do not enable it.

The credible alternative, to be written up as an ADR rather than dismissed: **Postgres with JSONB** would handle the fundamentals blobs adequately and give real constraints on `trades`. It costs the same to self-host on Railway, so this is purely a data-shape judgement rather than a cost one. Mongo still wins on the heterogeneous signal payloads and schema-free provider data, but the margin is narrower than a free-tier framing would suggest.

## Stack choices and rationale

- **Backend: NestJS + Mongoose.** NestJS's module/DI model is what makes the ports-and-adapters layering below enforceable rather than aspirational, and it gives first-class SSE and validation pipes. Mongoose over Prisma for better access to aggregation pipelines and BSON binary handling.
- **Web: React 19 + Vite + TanStack Router/Query/Table — mobile-first, sole client.** Phone layout is the default (bottom tabs, result cards, full-width CTAs, touch ≥44px). TanStack Table with virtualization is a **desktop (≥1024px) enhancement**, not the primary results UI. Query owns server state; Zustand holds ephemeral UI state. Forms via react-hook-form + Zod from `@vova/contracts`. Optional web `manifest` / Add to Home Screen only — **no Expo, no offline PWA scans**.
- **Design system: Tailwind v4 + shadcn/ui (Radix), mobile-first tokens.** Accessible primitives; terminal-dark palette (`#050505` app, `#1e222d`/`#2a2e39` surfaces, `#2962ff` accent, `#089981`/`#f23645` candles) from [ui_styles.py](ui_styles.py). Default breakpoint = phone; `md`/`lg` enhance.
- **Price charts: TradingView Lightweight Charts v5** wrapped in `@vova/charts`. It is the right fidelity/performance match for candles and already matches the app's visual language. The Plotly-specific tricks map onto v5 features: the `shape="hv"` critical level becomes a stepped line series split by `seq_state`; HH/LH/DT/HL/LL/DB become series markers; TP/SL/fib become price lines; extension lines become a custom series primitive. Weekend gaps come free, so the Plotly `rangebreaks` hack disappears. Licence note: Apache-2.0 and free, but its NOTICE requires the TradingView attribution to stay visible on the chart — a condition to honour, not a cost. `react-plotly.js` stays available as a fallback if any overlay proves impractical.
- **Analytics charts: Recharts** for equity curve, monthly P&L and win-rate panels — a separate concern from price charts, so a separate lighter library is correct.
- **Jobs: BullMQ on a self-hosted Redis service.** Gives retries with backoff, lease/stall recovery, concurrency limits, a built-in rate limiter to replace the hand-rolled 12 req/s token bucket in [headless_scanner.py](headless_scanner.py), repeatable cron jobs for the pre-market scan and nightly `BarSync`, and pub/sub for progress without polling. On Railway a Redis container idles at tens of megabytes, so this costs well under a dollar a month — cheap enough that hand-rolling lease and retry semantics on top of Mongo would be false economy. Queued state is also mirrored into `scanRuns`, so losing Redis loses scheduling, never history.
- **Auth: better-auth or Lucia, self-hosted against the `users` collection.** Deliberately not Clerk/Auth0 — their free tiers add an external dependency and MAU ceiling for what is a single-operator tool.
- **Quality: Vitest, Playwright (Pixel 7 + iPhone 14 required + desktop), Biome, pino.** OpenTelemetry wiring is designed for but left disabled by default.

## Design patterns, applied where they earn their place

- **Hexagonal / ports-and-adapters** in `apps/api`: `MarketDataPort` (adapters: `YahooMarketDataAdapter`, `MongoBarCacheAdapter`, `FixtureAdapter` for tests), `UniversePort` (adapters: `MongoUniverseAdapter`, `LegacyTextFileAdapter` for the one-time import), `ScanRunRepository`, `TradeRepository`. Domain and engine never import Mongoose or `fetch`.
- **Strategy + Registry** for scanners. The code already carries a `scanner_id` field pinned to `"sequence_vova"`; formalise it as a `ScannerStrategy` interface (`describeParams`, `evaluateLast`, `evaluateHistory`) resolved from a registry, so a second strategy is an added file rather than an edited `if`.
- **Repository** for every collection, returning domain objects, never Mongoose documents.
- **Builder** for chart overlay composition, replacing the 800-line procedural `build_sequence_vova_figure`: each overlay (candles, structure markers, critical level, fib, BB, EMA, TP/SL, watermark) is an independently testable `OverlayContributor`.
- **Result / Either** for scan outcomes. The Python code signals rejection through `explain_invalid_buy()` reason strings; make that a discriminated union (`{ status: 'rejected', reason: RejectReason }`) so reject codes are exhaustive and typed instead of stringly.
- **Observer** for progress: worker publishes `ScanProgressEvent`s, API relays via SSE, client subscribes. Replaces Streamlit's rerun-driven progress bars.
- **Command/handler split (CQRS-lite)**: `StartScanCommand` enqueues and returns a run id immediately; reads go through query services backed by aggregation pipelines.
- **Adapter** for symbol mapping (Yahoo suffixes `.TO`/`.V`/`.NE`/`.CN` to TradingView `EXCHANGE:SYMBOL`), lifted from [tradingview_embed.py](tradingview_embed.py) and [mobile/src/tickers/lists.ts](mobile/src/tickers/lists.ts).

## MongoDB data model

Collections, with the indexes that matter:

- `users` — email, hashed password, roles. Unique index on email. (New capability; the app currently has no auth at all.)
- `instruments` — replaces the two text files. `{ yahooTicker, tvSymbol, exchange, companyName, assetType, universes: string[], active }`. Unique on `yahooTicker`; index on `universes`.
- `barSeries` — **one document per `(yahooTicker, interval)`**, not per bar. Shape: `{ yahooTicker, interval, firstDate, lastDate, barCount, dates: BinData, open: BinData, high: BinData, low: BinData, close: BinData, volume: BinData, updatedAt }`, where each `BinData` is a packed typed array (Float32 for prices, Int32 day-offsets for dates) optionally deflated. Unique compound index on `{ yahooTicker, interval }`. This is the cache that decouples scans from Yahoo, and the single-document-read shape is what keeps a full-universe scan fast. `@vova/engine` gets `encodeSeries`/`decodeSeries` helpers so the packing format lives in one tested place.
- `instrumentFundamentals` — market cap, PE, earnings date, description; **TTL index** on `fetchedAt` (24h), replacing the `.cache/yf_info/*.json` file cache.
- `scanRuns` — params snapshot, status (`queued|running|completed|cancelled|failed`), `asOf`, counters, timings, `ownerId`. Index on `{ ownerId, createdAt: -1 }`. Direct successor to the SQLite `scan_runs` table.
- `signals` — one document per emitted row, embedding the typed BUY or SELL payload plus a bar snapshot for chart rendering. Index on `{ runId, symbol }`.
- `trades` — successor to the (former) mobile SQLite `trades` table (entry/tp/sl/shares/status/exit/pnl); journal, history and monthly P&L live on web only. Index on `{ ownerId, status, symbol }`.
- `scanPresets` / `chartPresets` — persisted `IndicatorParams` from [indicator_params.py](indicator_params.py), so chart settings survive sessions instead of dying with `st.session_state.chart_params`.

Every schema is defined once as a Zod schema in `@vova/contracts` and derived into both the Mongoose schema and the API DTOs, so drift is impossible.

## API surface (sketch)

- `POST /api/scans` — validate params, enqueue, return `{ runId }`.
- `GET /api/scans/:id` / `GET /api/scans` — run detail and paginated history.
- `GET /api/scans/:id/events` — SSE progress stream (phase, percent, counters).
- `POST /api/scans/:id/cancel` — sets a cancellation flag the worker polls, mirroring today's `is_cancelled` callback.
- `GET /api/scans/:id/signals` — server-side paginated, sorted, filtered rows for the grid.
- `GET /api/instruments/:ticker/chart?tf=` — bars plus computed overlay series.
- `GET /api/instruments/:ticker/status` — the multi-timeframe watermark payload from [watermark_status.py](watermark_status.py).
- `GET/POST /api/trades`, `POST /api/trades/:id/close`, `GET /api/reports/monthly`.
- `GET/PUT /api/presets/{scan,chart}`.

## Budget: Railway, target under $250/year

No framework or database licence is paid for. Every library in the stack is permissively licensed: React, Vite, TanStack, Tailwind, shadcn/ui + Radix, Recharts, NestJS, Mongoose, Zod, BullMQ, Vitest, Playwright, Biome, pino, react-hook-form, better-auth/Lucia, and Lightweight Charts (Apache-2.0, attribution must stay visible on the chart). MongoDB and Redis are self-hosted from their own community images, so the spend is compute, not licence. No Expo/EAS.

**Railway Hobby, $5/month with the first $5 of usage included.** Metered rates are $10/GB-month of RAM, $20/vCPU-month of CPU, $0.15/GB-month of volume and $0.05/GB of egress, all billed per second on *actual* consumption — so idle services are genuinely cheap and the CPU bursts a scan produces are measured in cents. Hobby allows 6 replicas and caps volumes at 5 GB per service, which is far above what the `barSeries` model needs.

Planned footprint, four Railway services in one project on the private network:

- `api` — NestJS, idles near 200 MB with negligible CPU, roughly $2.50/month.
- `worker` — idles near 120 MB; a three-minute full-universe scan at 1.5 vCPU costs a fraction of a cent, so even a daily scheduled scan stays under $0.10/month. Roughly $1.50/month.
- `mongo` — community image configured as a single-node replica set, WiredTiger cache capped explicitly so it settles near 400 MB, plus a small volume. Roughly $5/month.
- `redis` — BullMQ broker, tens of megabytes with a small AOF volume. Under $0.50/month.

Web assets go to **Cloudflare Pages** rather than a fifth Railway service: static hosting is free there and it saves both the compute and the egress. Backups are a scheduled `mongodump` to Cloudflare R2, whose free allowance covers a dump measured in tens of megabytes; this is the deliberate mitigation for self-hosting rather than using a managed cluster.

That lands around **$9-10/month, roughly $115/year**, plus about $12/year for a domain. Against a $250 ceiling that leaves close to half the budget unspent, which is the point: the headroom is the contingency, not a target to fill.

Two cost decisions worth stating explicitly:

- **Self-hosted Mongo on Railway over Atlas Flex.** Flex starts at $8/month and scales by operations per second toward a $30 cap, so a heavy scan writing thousands of signals can push it above the base tier. Self-hosting on Railway is cheaper, has no operations-per-second tiering, sits on the same private network as the API and worker (so inter-service traffic is not billed as egress), and has no storage ceiling to design around. The thing Flex would buy is managed backups, which the R2 dump replaces for a workload where losing a day of bar cache is a re-sync, not a loss.
- **Stay on Hobby, not Pro.** Pro is $20/month, which alone is $240/year and would consume the entire budget for team features this project does not need.

Deliberately *not* spending the budget on market data. Tiingo at about $10/month or EODHD at about $20/month would eat most or all of the remaining headroom, and neither cleanly replaces Yahoo here: the universe includes TSX/TSXV/NEO/CSE listings that cheap US-only feeds do not cover, and parity was established against Yahoo's unadjusted series. The real fix for Yahoo's fragility is the `barSeries` cache, which turns per-scan fetching into one incremental sync per symbol per day and costs nothing. `MarketDataPort` keeps the swap to a paid provider a single adapter if Yahoo ever blocks in earnest, and the unspent headroom is what would fund it.

What the budget buys that free hosting could not: an always-on worker with no cold start and no runner queue latency, scheduled pre-market scans as the primary workflow instead of a manual START button, a database with no storage ceiling or operations tiering, and a real queue with retries and backoff instead of hand-rolled lease logic. Optional web push for new signals can ride along at no marginal infra cost.

A local Docker Compose stack (`api`, `worker`, `web`, `mongo` as a single-node replica set, `redis`) remains part of the design as the development environment and as a zero-cost fallback, mirroring the Railway topology exactly.

## Web app screens (mobile-first)

Sole client replaces Streamlit after cutover. Phone-canonical IA: bottom tabs **Scan | History | Trades | P&L** (+ Settings). Scan stack: config → progress → **result cards** → chart detail; Rejected as drawer/screen. Desktop (≥1024): virtualized table + optional split chart pane as enhancement. Chart settings = persisted presets; reject reasons = typed diagnostics drawer.

## Migration phases (6 — no Expo)

1. **Scaffold + parity gate.** Monorepo; salvage `mobile/src/engine` (+ parity/yahoo/ticker helpers) → `packages/engine`; **delete `mobile/`**. Extend [scripts/export_parity_fixture.py](scripts/export_parity_fixture.py) + Vitest parity CI. **Streamlit Cloud stays the product**; Python stays at repo root (not early `legacy/`).
2. **Engine completion.** Port remaining chart math from [sequence_vova.py](sequence_vova.py): MACD, DMI/ADX, Bollinger, Elder, SMA major, peak/trough, fib, extension lines, multi-TF watermark.
3. **API + persistence.** NestJS, Mongoose, BarSync, BullMQ + pre-market job, SSE. Local Compose first.
4. **Web client (mobile-first).** Design system, phone layouts/cards/bottom nav first, then desktop table/split; Playwright phone + desktop gates; chart package; journal/reports.
5. **Deploy candidate to Railway.** Four services + Cloudflare Pages + R2 dumps; measure billing ~1 week. **Do not turn off Streamlit Cloud** — operators keep using Streamlit; React is validation.
6. **Cutover.** Side-by-side parity + exit criteria (parity, Playwright, ≥3 scheduled pre-market days, operator sign-off) → retire Streamlit Cloud; optional Python → `legacy/` as permanent oracle. Rollback until then = keep using Streamlit.

## Deliverable of this task

Since the request is for a design, the output is a reviewable document set under `docs/`, not application code:

- `docs/architecture/README.md` — overview, C4-style context/container diagrams, module boundaries.
- `docs/architecture/adr/` — one ADR per decision: TypeScript engine; MongoDB + `barSeries`; self-hosted Mongo on Railway; BullMQ + Redis; Lightweight Charts; NestJS + Mongoose; pnpm/Turborepo; Vite SPA over Next.js; Railway topology; keep Yahoo; **mobile-first web-only (no Expo)**; **parallel Streamlit until Railway cutover**.
- `docs/architecture/data-model.md` — collections, Zod/JSON Schema, indexes, TTLs, binary series encoding; mapping from text files; trades/journal for web (SQLite journal schema as migration reference only).
- `docs/architecture/api.md` — endpoint contracts and the SSE event schema.
- `docs/architecture/engine-parity.md` — indicator/state-machine inventory, salvage-from-mobile gap analysis, golden-fixture strategy.
- `docs/architecture/frontend.md` — mobile-first routes/IA, card-vs-table breakpoints, touch rules, Playwright phone matrix, chart primitives.
- `docs/architecture/hosting-and-cost.md` — Railway topology + Streamlit Cloud parallel until cutover; no Expo/EAS.
- `docs/architecture/migration.md` — 6 phases, delete `mobile/`, parallel-run, cutover exit criteria, Streamlit retirement only after Railway.
- `docs/architecture/repo-layout.md` — no `apps/mobile`; Python at root until cutover.

## Remaining close calls, documented rather than silently picked

- **Vite SPA vs Next.js 15** — Vite SPA: sole client is static on Cloudflare Pages; API is separate; no SEO behind auth; Next SSR would add a billed service.
- **Self-hosted Mongo vs Atlas Flex** — recommending self-hosted on Railway for cost, private networking and no operations-per-second tiering, with the `mongodump`-to-R2 job as the price of giving up managed backups. Reversible quickly if operating it becomes a nuisance, since the connection string is the only thing that changes.
- **Merging `api` and `worker` into one service** — rejected, and worth recording why: it would save around $1.50/month, but the scan is a CPU-bound loop that would block Node's event loop and freeze SSE and HTTP while running. Not a saving worth making.

## Cost guardrails to build in from the start

Because the budget is a real ceiling rather than a guess, the design includes the means to defend it: a Railway usage alert set below the monthly target, an explicit WiredTiger cache cap on the Mongo container so memory does not drift upward with the host, TTL indexes on `instrumentFundamentals` and a retention policy on old `scanRuns`/`signals` so storage does not grow without bound, and a hard cap on scan concurrency so a runaway loop cannot bill CPU indefinitely. All four are cheap to add up front and awkward to retrofit.
