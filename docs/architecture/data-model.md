# Data model (MongoDB)

Single-node replica set. Locally it is started from the `mongodb-memory-server` binary with a
persistent `dbPath` (`.data/mongo`) so no Docker or MongoDB install is required; on Railway the
same code connects through `MONGO_URI`. Replica set (not standalone) so transactions and change
streams stay available.

## Collections (implemented)

### `instruments`
Successor to `STOCK-TICKERS.txt` / `TV-LIST-ETF.txt`, imported on first boot.
`{ yahooTicker, tvSymbol, exchange, companyName, assetType, universes[], active }`.
Unique `yahooTicker`; index `universes`.

### `barSeries`
One document per `(yahooTicker, interval)` with packed binary columns:
`{ yahooTicker, interval, firstDate, lastDate, barCount, dates, open, high, low, close, volume, updatedAt }`.
Unique `{ yahooTicker, interval }`. Hot path: one document read per symbol per scan.

Encoding: `dates` as Int32 day offsets, prices and volume as **Float64**. Float64 (not Float32)
because the engine must reproduce Python results bit-for-bit enough to pass the parity fixture;
rounding cached prices would let cache hits and fresh downloads disagree.

Freshness: `barsMaxAgeHours` (default 12) decides cache vs refetch; `forceRefresh` bypasses it.
A failed Yahoo fetch falls back to stale cache rather than dropping the symbol.

### `scanRuns`
`{ params, status, asOf, counters, reasonCounts, timings, newSymbols[], summary, cancelRequested, error, startedAt, finishedAt }`.
`status`: `queued|running|completed|cancelled|failed`. Index `{ createdAt: -1 }`.

- `counters`: `total`, `downloaded`, `evaluated`, `signals`, `rejected`, `skipped`, `fromCache`
- `reasonCounts`: reject/skip reason histogram (why a scan produced few signals)
- `newSymbols`: symbols absent from the previous completed run with the same source/tf/direction
- `summary`: sell-scan aggregate (win rate, net P&L, invested, avg RR)

### `signals`
One document per BUY/SELL row: `{ runId, kind, symbol, yahooTicker, companyName, isNew, isStrong, rr, payload }`.
Index `{ runId, symbol }`. Charts read bars from `barSeries`, so no bar snapshot is duplicated here.

### `scanRejections`
`{ runId, symbol, reason, createdAt }` with a 30-day TTL — audit data, not history.

### `trades`
Journal: `{ symbol, yahooTicker, companyName, tf, openedAt, asOf, entry, tp, sl, rrAtEntry, shares, riskUsd, status, exitPrice, exitDate, exitReason, pnlUsd, pnlR, runId }`.
Index `{ status, symbol }`. Unrealized P&L is computed on read from cached bars.

### `presets`
`{ key, data }` for `scan` and `chart` params (successor to Streamlit `session_state`).

## Deferred

- `users` / auth — added with the Railway deploy phase
- `instrumentFundamentals` (Yahoo `.info` TTL cache) — the current scan path never calls `.info`;
  the `LOW_VOL` filter is derived from cached bar volume instead
- Full overlay series (EMA/BB/MACD…) — recomputed from `barSeries` by the engine
- Multi-TF watermark — compute-on-read via `GET /instruments/:ticker/status`

## Retention

- `scanRejections`: TTL 30 days
- `instrumentFundamentals` (when added): TTL 24h
- Old `scanRuns` / `signals`: retention policy (e.g. 90 days) before the Railway volume grows
