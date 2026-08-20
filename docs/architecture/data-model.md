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

**One ticker, three strings, one of them for the screen.** `yahooTicker` fetches the bars and keys
every record; `tvSymbol` is the TradingView form (`NASDAQ:LMAT`) and is only ever put in a link or a
deep-link URL; `symbol` is that form without its exchange prefix (`LMAT`) and is the only one any
tab prints — Results, History, Rejected, Manual and the chart header alike. `shortSymbol` in the
engine derives it, `companyName` comes from the same line of the list file, and both are applied on
the way in and again on the way out (`toResultRow`), so a record written by an older build cannot
put the same position on screen under two names.

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
`{ params, status, asOf, newestAsOf, counters, reasonCounts, timings, newSymbols[], summary, cancelRequested, error, startedAt, finishedAt }`.
`status`: `queued|running|completed|cancelled|failed`. Index `{ createdAt: -1 }`.

- `counters`: `total`, `downloaded`, `evaluated`, `signals`, `closes`, `rejected`, `skipped`,
  `fromCache`. `closes` is the sell-to-close breaks a buy pass found alongside its signals
- `asOf` is the *oldest* newest-bar in the run — the honest answer to how stale it could be.
  `newestAsOf` is the newest bar of the period most of the universe is in, which is the period the
  screens read the run as being about; see the CLOSED bucket below for why neither extreme works
- `reasonCounts`: reject/skip reason histogram (why a scan produced few signals)
- `newSymbols`: symbols absent from the previous completed run with the same source/tf/direction
- `summary`: sell-scan aggregate (win rate, net P&L, invested, avg RR)

### `signals`
One document per BUY/SELL row: `{ runId, kind, symbol, yahooTicker, companyName, isNew, isStrong, rr, payload }`.
Index `{ runId, symbol }`. Charts read bars from `barSeries`, so no bar snapshot is duplicated here.

A buy run holds both kinds: `kind: 'buy'` is what the scan screen shows, and `kind: 'sell'` is the
close scan run over the same bars in the same pass — the two never name the same symbol, because a
break puts the sequence down and so makes the symbol a reject for the buy scan. The tracker reads
the sell rows; the scan screen filters them out.

### `scanRejections`
`{ runId, symbol, yahooTicker, reason, createdAt }` with a 30-day TTL — audit data, not history.
`symbol` is the display form the Rejected tab prints; `yahooTicker` is what the tracker matches a
position against when it asks which symbols a run could not evaluate.

### `trackedSignals`
The single source for Results and History. Written by `SignalTrackerService` after a
background scan finishes, and by `HistoryRebuildService` when Settings asks for a full ledger
backfill, so both screens are indexed reads with no per-request maths.

`{ yahooTicker, symbol, tvSymbol, companyName, universe, tf, status, provisional, provisionalClose,
signalValid, imported, backfilled, openedPeriodKey, openedAsOf, entry, tp, sl, rrAtEntry, shares, riskUsd,
lastSeenPeriodKey, lastSeenAsOf, lastPrice, lastRr, barsSinceValid, validSinceAsOf, isStrong,
unrealizedUsd, unrealizedR, unrealizedPct,
closedPeriodKey, exitDate, exitPrice, exitReason, pnlUsd, pnlR, pnlPct, holdPeriods,
interest, interestRank, interestAt, runId }`.

**A tracked position is the Streamlit close scan's trade.** That scan replays a symbol's whole
history — take the long on the bar a buy signal appears, give it up on the bar the sequence closes
back through its critical level — from the bars alone, so the trade a symbol is in does not depend
on when this app started watching it. `runCloseLedger` is that replay, and everything below follows
from taking it as the definition rather than tracking only what the app happened to open.

Lifecycle:

- Every completed Stocks/ETF scan refreshes `lastPrice`, `lastRr`, `barsSinceValid` and the
  unrealized numbers, and opens a `provisional` record for a symbol it has not seen before. That is
  what makes a signal appearing mid-session visible straight away. Settings Min RR on NEW/VALID
  reads that live `lastRr` (CLOSED keeps `rrAtEntry`).
- **A break ends a trade whether or not this app recorded its start.** A symbol closing today is
  not a buy today — the break puts the sequence down — so it is a reject in the buy scan and would
  never be heard of again. Every buy pass therefore also runs the close scan over the same bars and
  records what it finds as `kind: 'sell'` signals; a break on a symbol with no record of its own is
  written down complete, entry and exit together. Those adopted trades are most of any close list.
  A close already written down (same symbol, same exit bar) is skipped, so the hourly cadence does
  not stack a copy of each trade an hour.
- **History rebuild** (`POST /history/rebuild`) runs `runCloseLedger` over the `barSeries` cache for
  every Stocks/ETF symbol and every timeframe, and inserts every *closed* ledger trade that is not
  already recorded. A trade is already recorded when any record of that symbol and timeframe shares
  the bar it started on (`openedAsOf`) or the bar it ended on (`exitDate`) — every record and not
  only the closed ones, because a position the app is still carrying, open or breaking on the bar in
  progress, is the same trade the replay is about to find. Rows are marked `backfilled: true`. Open
  ledger tails and `imported` journal rows are left alone. Depth is the Yahoo window already cached:
  Daily ~2y, Weekly/Monthly ~10y (`intervalAndPeriod`).
- **A record's entry follows the replay, not the day the app first met the symbol.** A position met
  four months into its run is priced from the bar it actually started on, and its `entry`, `sl`,
  `tp`, `rrAtEntry` and `openedAsOf` are re-aligned to the replay on every scan. `imported` journal
  trades are exempt: those entries are the user's own record of what they paid.
- Only a scan that already had its period closed when it started (`run.periodClose`) confirms or
  deletes a provisional record, so a signal that comes and goes inside one period never reaches
  History.
- **A trade ends on the sell-to-close break and on nothing else** — the first bar after `openedAsOf`
  whose close falls back through the critical level of the sequence, exiting at that bar's close.
  This is the exit of the Streamlit close scan (`run_sequence_vova_close_scan`), and `exitReason`
  on anything this app closes is always `sell_to_close`. `TP`, `SL` and `signal_lost` remain in the
  enum for the imported journal and for records written before this was the only rule; boot re-opens
  the ones an older build closed
  ([reopen-non-break-exits.service.ts](../../apps/api/src/migrations/reopen-non-break-exits.service.ts)).
- TP and SL are entry-time numbers. SL sizes the position and both state what the setup was worth
  when it was taken; price passing through either changes what the trade is worth, not whether it is
  still on.
- A position the scan stops reporting keeps running: `signalValid` goes false, which drops it off
  NEW and VALID until a scan finds the setup again. The flag is written only by a scan that could
  evaluate the symbol — a `NO_DATA` / `INSUFFICIENT_DATA` reject says nothing about the setup — so
  a Yahoo outage, or simply no scan having run yet, leaves a record showing exactly where it was.
  That is also why an imported journal trade is on screen from the moment it is imported.
  **Live revalidation:** reading NEW or VALID (list or summary badges) also runs `signalAge` over
  the bar cache for every active non-imported row and rewrites `signalValid` / `barsSinceValid` /
  `validSinceAsOf`. A forming bar that kills Seq/Struct between hourly scans therefore leaves the
  screen as soon as Results is opened, and a setup that recovers in the same period with age `0`
  returns to NEW without waiting for the next cron tick. Missing cache is left alone, same as an
  unevaluated scan. The live chart's `syncTrackedAge` does the same bidirectional write when a
  symbol is opened.
- `provisionalClose` is a break on the bar still in progress, which is the only bar that can take
  one back. The record stays `active` and carries the exit it would realize, so CLOSED shows the
  trade for the current period while History waits for the bar to finish; the period-close scan
  either turns it into a real close or clears the exit fields when the bar recovers. A break on any
  earlier bar is settled and is realized by whichever scan finds it, close scan or not — which is
  what a catch-up after a missed close, or after the re-opening migration, mostly finds.
- A symbol whose trade closes and whose setup is a buy again on a later bar opens its next record
  in the same pass, so it does not disappear for a scan while the old record is cleared away.
- `closedPeriodKey` is the calendar slot of the exit bar, not of the scan that found it. They agree
  whenever a scan runs on time; when one is missed for a week, or a re-opened trade turns out to
  have broken long ago, the trade is still filed under the period it actually ended in.
- `barsSinceValid` is the bar the engine says the signal appeared on, counted in bars of `tf` back
  from the latest one: `0` is NEW, anything higher is VALID, and that is the whole rule. It is the
  signal that ages, not the record — a symbol the scanner meets for the first time may already have
  been running for four bars, and it belongs in VALID from the moment it is opened. `validSinceAsOf`
  is the date of that bar, so a card can say how long a trade has been running.
- The age always comes from `signalAge` in the engine, which evaluates with the RR requirement off.
  RR decides which signals a scan reports and how the lists sort, never whether a signal is new: with
  a minimum RR in place the valid flag flips as the ratio drifts across the threshold mid-trade, and
  the age would count bars since the last flip. That is why the chart screen, whose settings default
  to `min_rr: 1.5`, still reports the same age as the tabs.
- Records written before this field existed are filled in on boot by
  [signal-age-backfill.service.ts](../../apps/api/src/migrations/signal-age-backfill.service.ts),
  off the bar cache and counted up to the record's own `lastSeenAsOf` bar. Without it a timeframe
  whose next scan is days away (Weekly, Monthly) would show every active signal as VALID and nothing
  as NEW. A record with no cached bars, or whose structure has since broken, keeps no age and stays
  in VALID — a hand-imported journal trade is not a new signal.
- Both live buckets require `signalValid` not to be false. NEW additionally requires
  `lastSeenPeriodKey` to be the current period, because `barsSinceValid` is only true as of the scan
  that wrote it: a record last priced days ago was new on that bar, not on this one. VALID is the
  exact complement, so the two tab counts always add up and a record nobody has priced this period
  lands in VALID rather than nowhere.
- CLOSED is everything with `closedPeriodKey` on the period the scan reports on, realized or
  `provisionalClose`. That period comes from the bars rather than the clock because
  `closedPeriodKey` does: over a weekend a Monthly scan already runs under the next month while the
  newest bar it can see is the last one of this month. It is the period *most of the universe* is
  in (`run.newestAsOf`), not any one symbol's newest bar: the oldest would be one halted ticker
  away from putting the screen days behind the market, and the plain newest one off-grid bar away
  from putting it a period ahead where nothing has closed yet — and Yahoo does hand out the odd
  series stamped a day off the grid.
- `provisional` no longer decides which tab a signal shows in; it still decides what a period-close
  scan does with the record. Only confirmed records can be closed, so a signal that comes and goes
  inside one period leaves no trace, and `totals.active` in History counts confirmed positions only.
- `interest` is set from the chart screen and survives NEW → VALID → CLOSED. `interestRank`
  (2 / 1 / 0) exists only so Mongo can sort marked signals first.

The pre-tracking journal (`trades`) is imported into this collection on boot by
`LegacyTradesMigration`: closed rows arrive as closed signals with the P&L they were recorded
with, open and marked rows as active ones, and `dismissed` rows are left behind. Each source row
is stamped with `migratedAt` / `migratedAs` rather than deleted, so the journal stays as a backup
and a repeat run only picks up what is left. Three things the journal never had are filled in from
`instruments`: `universe`, the ticker strings (`symbol` / `tvSymbol`) and `companyName`; plus the
exit reason `manual`, which exists in the enum only because the old app let you close a trade by
hand. A journal row is skipped when the app already carries that position, or already has a record
of a trade opened in the same period — a second copy of one trade is worse than a missing line.

**One trade is one record**, matched on the symbol, the timeframe, the universe and the bar the
trade started on or ended on — the bars rather than the periods, because a Monthly trade can close
and another open inside one month. The partial-unique index only guards `status: 'active'`, so
copies used to be possible where one side was closed; `NormalizeSymbols`
([normalize-symbols.service.ts](../../apps/api/src/migrations/normalize-symbols.service.ts)) puts
every record on the one ticker format on boot and drops the extra copies, logging each one because
they carry realized P&L. Of two copies it keeps the scan record over a journal one, then a realized
close over a break still settling on the bar in progress, then the one written most recently.

Indexes: partial-unique `{ yahooTicker, tf, universe }` while `status: 'active'`, plus
`{ universe, tf, status, barsSinceValid }`, `{ universe, tf, status, openedPeriodKey }`,
`{ universe, tf, status, closedPeriodKey }`, `{ universe, tf, status, lastRr }`,
`{ universe, tf, status, interestRank }`, `{ status, tf, closedPeriodKey }` and
`{ universe, tf, closedPeriodKey }` — one index per bucket-and-sort combination the UI offers.

### `presets`
`{ key, data }` for chart params (successor to Streamlit `session_state`) and the `app` key
holding `{ maxRiskUsd, minRr, fundamentalsFilter }` behind `GET/PUT /api/settings`.

### `instrumentFundamentals`
FMP snapshot for the Fundamentals page, Results/History cards, Value screener, and the Settings
valuation filter. One document per `yahooTicker`:

`{ yahooTicker, payload, fairValue, premiumPct, growthRatePct, blendedPe, ltDebtToCapitalTTM,
   epsFairValue, fcfFairValue, dcfFairValue, epsPremiumPct, fcfPremiumPct, dcfPremiumPct,
   stars, bestPremiumPct, interest, scaleVersion, valuationReliable, fetchedAt, updatedAt }`.

`payload` is the assembled `FundamentalsPayload` (metric `eps`; other metrics recompute from
`annual` without FMP). Card / Value fields are denormalized for list/filter reads.
`stars` is how many of EPS / FCF / DCF have `premiumPct < 0` (0–3).

Reads for listed tickers (STOCK-TICKERS / ETF) never call FMP — only Mongo. Unknown Manual
tickers may pull FMP once on first Manual scan. Writes:

- weekday EOD cron (`VOVA_FUNDAMENTALS_FULL_CRON`, default Mon–Fri 18:15 ET) — full 13-endpoint pull
  for every active stock + ETF ticker, then recompute stars
- boot catch-up — if coverage is incomplete, or today's EOD slot was already missed after 18:15 ET,
  the same full/missing walk starts immediately (Nest cron does not fire retroactively)
- unknown Manual miss — one `fetchFresh`, then upsert (not added to the Value universe)

No Mongo TTL: a stale document is served rather than hitting FMP. A failed refresh keeps the
previous document (same stale-fallback idea as `barSeries`).

### `fundamentalsRefreshRuns`
Progress of the latest EOD / boot catch-up walk:
`{ kind, trigger, status, startedAt, finishedAt, total, done, ok, skip, fail }`.
Value screener exposes this as `lastRun` / `lastFullAt` so the UI can show coverage and update age.

## Deferred

- `users` / auth — added with the Railway deploy phase
- Full overlay series (EMA/BB/MACD…) — recomputed from `barSeries` by the engine
- Multi-TF watermark — compute-on-read via `GET /instruments/:ticker/status`

## Retention

- `scanRejections`: TTL 30 days
- Old `scanRuns` / `signals`: retention policy (e.g. 90 days) before the Railway volume grows
