# API surface

Base path `/api`. Implemented in `apps/api` (NestJS). Auth is deferred to the Railway phase
(single local user today), so no endpoint is owner-scoped yet.

## Results

Reads of `trackedSignals`, written only by the background scans — nothing here triggers work.
The "current period" is the `periodKey` of the newest completed scan for that universe and
timeframe, not the wall clock, so buckets always line up with the data on screen.

NEW and VALID split on `barsSinceValid`, the bar the engine says the signal appeared on, which is why
a symbol the scanner meets for the first time can open straight into VALID. RR is not part of that
split anywhere: the age is measured with the RR requirement off, so `/instruments/:ticker/chart`
reports the same `barsSinceValid` as the tabs whatever `minRr` it is called with.

| Method | Path | Notes |
|--------|------|-------|
| GET | `/results?universe&tf&bucket&sort&dir&limit&offset` | `bucket` = `new` (became valid on the latest bar of `tf`) / `valid` (became valid earlier and still is, marked to market) / `closed` (sold to close in the current period). `sort` = `rr`, `pnl`, `interest`, `symbol`, available in every bucket; sorting and paging happen in Mongo. `rr` reads `rrAtEntry` in CLOSED and `lastRr` elsewhere, matching the number on the card. Settings Min RR filters the same way: `lastRr` for NEW/VALID, `rrAtEntry` for CLOSED. Settings `fundamentalsFilter` (`all` / `undervalued` / `overvalued`) further restricts rows to tickers whose Mongo `premiumPct` matches; it does not fetch FMP on the read path. Names without a fair value only appear in `all` |
| GET | `/results/summary` | Bucket counts and scan freshness for every universe × timeframe, for the tab badges |
| GET | `/results/lookup?yahooTicker&tf` | The active tracked signal for a symbol, so a chart opened by URL can show and toggle the mark |
| GET | `/results/signal/:id` | One tracked signal whatever its state — how a closed trade from History is opened on the chart |
| PATCH | `/results/:id/interest` | `{ interest: 'interested' \| 'not_interested' \| null }`; the mark survives NEW → VALID → CLOSED |

CLOSED is the Streamlit SELL TO CLOSE list for the period the scan reports on. A trade ends on the
sell-to-close break and on nothing else, and it ends whether or not this app opened it: every buy
pass runs the close scan over the same bars, and a break on a symbol with no record of its own is
written down entry and exit together. While the exit bar is still running the break is provisional
— the row shows in CLOSED with `provisionalClose: true`, carries the exit it would realize, and
reaches History only if the break survives to the final bar. A break on any earlier bar is settled
and realized straight away.

An open position a scan evaluated and stopped reporting is not on screen at all — it is still
running, and comes back the moment a scan finds its setup again. A symbol the scan could not
evaluate is a different thing and keeps showing on its last numbers.

## History

| Method | Path | Notes |
|--------|------|-------|
| GET | `/history?tf=Daily\|Weekly\|Monthly\|All&groupBy=Daily\|Weekly\|Monthly&range=all\|ytd\|1m\|3m\|6m\|1y\|max&sort&dir` | Win rate, net P&L, avg R, avg RR at entry, avg hold, equity curve and exit-reason histogram over realized trades; aggregated in Mongo. `range` filters by `exitDate` (`max` ≡ `all`). `sort` = `period`, `pnl`, `winRate`, `trades`, `rr` (avg RR at entry). Also carries `timeframes`: each of Daily / Weekly / Monthly with its own trades, win rate, net P&L, avg R and equity curve under the same `range`. Settings Min RR and `fundamentalsFilter` apply here the same way they do on Results |
| GET | `/history/trades?tf&groupBy&periodKey&range&sort&dir&limit&offset` | Closed rows, optionally drilled into one period bucket (intersected with `range`). `sort` = `date`, `pnl`, `r`, `rr`, `interest`, `symbol`. Settings Min RR and `fundamentalsFilter` apply the same way they do on `/history` |
| POST | `/history/rebuild` | Start a background rebuild: `runCloseLedger` over every cached symbol in Stocks/ETF × Daily/Weekly/Monthly, insert missing closed trades into `trackedSignals`. Idempotent; does not delete. Returns `{ started }` immediately |
| GET | `/history/rebuild` | Rebuild job status: `idle\|running\|done\|failed`, progress and insert/skip/noBars counts |

Yahoo bar windows bound how far rebuild can go: Daily `2y`, Weekly/Monthly `10y` (see `intervalAndPeriod`). History `range` filters what is already stored; it cannot invent bars beyond that cache.

## Settings

| Method | Path | Notes |
|--------|------|-------|
| GET/PUT | `/settings` | `{ maxRiskUsd, minRr, fundamentalsFilter }` — user-facing settings; scan parameters are fixed in code. `minRr` floors Results NEW/VALID (and History active) on live `lastRr`, and CLOSED / History closed on entry `rrAtEntry`; `0` disables the filter. `fundamentalsFilter` is `all` (default) / `undervalued` / `overvalued` on current fair-value premium and applies to the same Results lists/counts and History trades/stats; scans still write every signal |

One risk for every signal: `maxRiskUsd` divided by the distance to SL is the position size
everywhere — background scans, manual scans, the tracked signals and the chart. A `PUT` re-sizes
every open tracked signal before it answers, so the response is already consistent with the lists
the UI refetches. Closed signals keep the size they were closed at.

## Scans

Stocks and ETF are scanned by `PeriodSchedulerService`: one hourly pass covering Daily, Weekly and
Monthly together (09:05–17:05 ET, Mon–Fri), and on demand from the Settings sheet. Post-close ticks
are themselves the period-close scans — `periodClose` is decided from the clock when the run
starts — so there are no separate close crons. Passes are skipped, not queued, while an earlier
pass is still running.

A run records `periodClose`, decided when the scan **starts**, and only those runs let the tracker
confirm or close signals. Deciding it at finish would misclassify an hourly pass that began before
the bell and ran past it.

| Method | Path | Notes |
|--------|------|-------|
| GET | `/scans/defaults` | Server-side default params |
| POST | `/scans` | Create a manual ticker run, start it out-of-request, return `{ runId, params }` |
| POST | `/scans/run-now` | `{ tf?: 'Daily' \| 'Weekly' \| 'Monthly' \| 'all' }` — rescan Stocks and ETF now, re-downloading every symbol. Answers `{ started, timeframes, reason? }` as soon as the pass is queued, and `started: false` when one is already running. This is the background pass, so what it produces goes to Results and History; it runs even with `VOVA_BACKGROUND_SCANS=off` |
| GET | `/scans?limit=` | Run history, newest first |
| GET | `/scans/:id` | Run detail: status, counters, `reasonCounts`, timings, `newSymbols`, sell summary, `asOf` (scored bar date), `barsOldestAt` (oldest Yahoo pull) |
| GET | `/scans/:id/signals?limit&offset&onlyNew&onlyStrong` | Signal rows + run + `newSymbols` |
| GET | `/scans/:id/rejections?limit=` | Rejected symbols + reason breakdown; each row carries `detail` (`barDate`, `close`, `criticalLevel`, `seqState`, `rr`, `sl`, `tp`, `minRr`) |
| GET | `/scans/:id/events` | SSE progress stream |
| POST | `/scans/:id/cancel` | Cooperative cancel (flag + in-process abort) |
| DELETE | `/scans/history` | Drop every run, signal, rejection and tracked signal |

Scan params: `source` (`Stocks`/`ETF`/`MANUAL SCAN`), `manualTickers`, `tf`, `direction`,
`minRr`, `riskPerTrade`, `noRrReq`, `useLastHlSl`, `newOnly`, `minAvgVolume`, `maxSymbols`,
`barsMaxAgeHours`, `forceRefresh`.

## Instruments / universe

| Method | Path | Notes |
|--------|------|-------|
| GET | `/instruments/:ticker/chart?tf=&asOf=&minRr=&useLastHlSl=&riskPerTrade=&noRrReq=&lenFast=&lenSlow=&lengthMajor=&lookback=&multiplier=&bbLength=&bbMult=` | Bars + full overlays + watermark + pine; numeric params recompute overlays live. `asOf=YYYY-MM-DD` cuts the series at that bar before the engine sees it, which is what makes the chart behind a closed trade a snapshot of the trade rather than a view of today |
| GET | `/instruments/:ticker/status` | Multi-TF watermark from cached bars |
| GET | `/instruments/fundamentals-cards?tickers=` | Slim valuation for Results / History cards. Reads `instrumentFundamentals` in Mongo; FMP only for names that have never been stored |
| GET | `/instruments/:ticker/fundamentals?metric=` | Fast Graphs–style payload. Reads Mongo; FMP only on a first miss. `metric` = `eps` (default) / `revenue` / `fcf` / `ownerEarnings` (recomputed from stored `annual`) |
| GET | `/instruments/:ticker/dcf?revenueGrowthPct=&ebitdaPct=&capitalExpenditurePct=&longTermGrowthRate=&riskFreeRate=&marketRiskPremium=&taxRate=&costOfEquity=&costOfDebt=&operatingCashFlowPct=` | Unlevered Custom DCF from FMP (`/custom-discounted-cash-flow`). Optional rates as decimals (`0.08` = 8%; `8` is also accepted). Omit them for FMP defaults. In-memory cache 1h keyed by ticker+assumptions — not Mongo, not the scheduled refresh. Lynch fair value is unchanged |

| GET | `/universe/summary` | Counts per universe |
| POST | `/universe/import` | Re-import root ticker text files into `instruments` |

## Presets / health

| Method | Path | Notes |
|--------|------|-------|
| GET/PUT | `/presets/:key` | `chart` params (successor to Streamlit `session_state`) |
| GET | `/health` | Mongo readiness |

## SSE event shape

```json
{
  "runId": "…",
  "phase": "queued|resolving|scanning|saving|completed|cancelled|failed",
  "percent": 42,
  "message": "Scanned 900/2308 · 37 signals",
  "counters": { "total": 2308, "evaluated": 900, "signals": 37, "rejected": 863, "fromCache": 900 }
}
```

## Errors

Per-symbol outcomes are data, not HTTP errors: engine reject reasons land in `scanRejections`
and in `reasonCounts`. HTTP 4xx/5xx is reserved for bad requests and infrastructure failures.

BUY reasons follow the order of the Python oracle `sequence_vova.explain_invalid_buy`, so the
first failing gate is reported: `NO_SEQ_UP` (close is below the critical level — the sequence is
not up), `NO_STRUCT_HL`, `NO_STRUCT_HH`, `STRUCT_INVALID`, `NO_REWARD`, `NO_RISK`,
`RR_TOO_LOW:1.02 (min 1.50)`, `NO_VALID_SIGNAL`. Plus `NO_HH_LAST_PEAK` (BUY hard guard),
`NO_CLOSE_SIGNAL` (SELL to close), `NO_DATA`, `INSUFFICIENT_DATA`, `LOW_VOL`.

A `NO_SEQ_UP` symbol can be VALID on a live TradingView chart at the same moment: TradingView
scores the in-progress bar, the scan scores the stored snapshot named by `detail.barDate`.
