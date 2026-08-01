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
- `newSymbols`: symbols absent from the previous completed run with the same source and timeframe

### `signals`
One document per buy signal: `{ runId, kind, symbol, yahooTicker, companyName, isNew, isStrong, rr, payload }`.
Index `{ runId, symbol }`. Charts read bars from `barSeries`, so no bar snapshot is duplicated here.

### `scanRejections`
`{ runId, symbol, reason, createdAt }` with a 30-day TTL — audit data, not history.

### `trackedSignals`
The single source for Results and History. Written only by `SignalTrackerService` after a
background scan finishes, so both screens are indexed reads with no per-request maths.

`{ yahooTicker, symbol, tvSymbol, companyName, universe, tf, status, provisional,
openedPeriodKey, openedAsOf, entry, tp, sl, rrAtEntry, shares, riskUsd,
lastSeenPeriodKey, lastSeenAsOf, lastPrice, lastRr, isStrong, unrealizedUsd, unrealizedR, unrealizedPct,
closedPeriodKey, exitDate, exitPrice, exitReason, pnlUsd, pnlR, pnlPct, holdPeriods,
interest, interestRank, interestAt, runId }`.

Lifecycle:

- Every completed Stocks/ETF scan refreshes `lastPrice`, `lastRr` and the unrealized numbers, and
  opens a `provisional` record for a symbol it has not seen before. That is what makes a signal
  appearing mid-session visible in NEW straight away.
- Only a scan that already had its period closed when it started (`run.periodClose`) confirms or
  closes anything: provisional records are either confirmed or deleted, and confirmed records are
  closed with a realized P&L. Intra-period noise therefore never reaches History.
- `exitReason` is one of `SL`, `TP`, `sell_to_close` or `signal_lost` (plus `manual` on imported
  journal rows), checked in that order on the
  first bar after `openedAsOf`. The stop wins over the target on a bar that spans both, because the
  path within a bar is unknowable; `sell_to_close` is the Sequence Vova bullish break, exiting at
  that bar's close. `signal_lost` covers a confirmed signal the scan evaluated and no longer calls a
  buy: the run must hold a rejection for that symbol with a reason other than `NO_DATA` or
  `INSUFFICIENT_DATA`. A symbol the scan never reached, or could not price, is left active — so
  neither a Yahoo outage nor a scan over part of the universe can close positions it never judged.
- A signal reaches VALID by surviving a period close, never by the clock: a provisional record
  whose close scan never ran stays in NEW until some close scan confirms or drops it. Only
  confirmed records can be closed, so a signal that comes and goes inside one period leaves no
  trace, and `totals.active` in History counts confirmed positions only.
- `interest` is set from the chart screen and survives NEW → VALID → CLOSED. `interestRank`
  (2 / 1 / 0) exists only so Mongo can sort marked signals first.

The pre-tracking journal (`trades`) is imported into this collection on boot by
`LegacyTradesMigration`: closed rows arrive as closed signals with the P&L they were recorded
with, open and marked rows as active ones, and `dismissed` rows are left behind. Each source row
is stamped with `migratedAt` / `migratedAs` rather than deleted, so the journal stays as a backup
and a repeat run only picks up what is left. Two things the journal never had are filled in:
`universe`, recovered from `instruments.universes`, and the exit reason `manual`, which exists in
the enum only because the old app let you close a trade by hand.

Indexes: partial-unique `{ yahooTicker, tf, universe }` while `status: 'active'`, plus
`{ universe, tf, status, openedPeriodKey }`, `{ universe, tf, status, closedPeriodKey }`,
`{ universe, tf, status, lastRr }`, `{ universe, tf, status, interestRank }` and
`{ status, tf, closedPeriodKey }` — one index per bucket-and-sort combination the UI offers.

### `presets`
`{ key, data }` for chart params (successor to Streamlit `session_state`) and the `app` key
holding `{ maxRiskUsd }` behind `GET/PUT /api/settings`.

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
