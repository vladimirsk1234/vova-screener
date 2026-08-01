# API surface

Base path `/api`. Implemented in `apps/api` (NestJS). Auth is deferred to the Railway phase
(single local user today), so no endpoint is owner-scoped yet.

## Results

Reads of `trackedSignals`, written only by the background scans — nothing here triggers work.
The "current period" is the `periodKey` of the newest completed scan for that universe and
timeframe, not the wall clock, so buckets always line up with the data on screen.

| Method | Path | Notes |
|--------|------|-------|
| GET | `/results?universe&tf&bucket&sort&dir&limit&offset` | `bucket` = `new` (opened in the current period) / `valid` (opened earlier, marked to market) / `closed` (closed in the current period). `sort` = `rr`, `pnl`, `interest`, `symbol`; sorting and paging happen in Mongo |
| GET | `/results/summary` | Bucket counts and scan freshness for every universe × timeframe, for the tab badges |
| GET | `/results/lookup?yahooTicker&tf` | The active tracked signal for a symbol, so a chart opened by URL can show and toggle the mark |
| PATCH | `/results/:id/interest` | `{ interest: 'interested' \| 'not_interested' \| null }`; the mark survives NEW → VALID → CLOSED |

## History

| Method | Path | Notes |
|--------|------|-------|
| GET | `/history?tf=Daily\|Weekly\|Monthly\|All&groupBy=Daily\|Weekly\|Monthly&sort&dir` | Win rate, net P&L, avg R, avg RR at entry, avg hold, equity curve and exit-reason histogram over closed signals; aggregated in Mongo |
| GET | `/history/trades?tf&groupBy&periodKey&sort&dir&limit&offset` | Closed rows, optionally drilled into one period bucket |

## Settings

| Method | Path | Notes |
|--------|------|-------|
| GET/PUT | `/settings` | `{ maxRiskUsd }` — the only user-facing setting; scan parameters are fixed in code |

## Scans

Only manual scans are started from the UI. Stocks and ETF are scanned by
`PeriodSchedulerService`: hourly through the session plus one right after each period closes.
Session passes are skipped, not queued, while an earlier pass is still running.

A run records `periodClose`, decided when the scan **starts**, and only those runs let the tracker
confirm or close signals. Deciding it at finish would misclassify an hourly pass that began before
the bell and ran past it.

| Method | Path | Notes |
|--------|------|-------|
| GET | `/scans/defaults` | Server-side default params |
| POST | `/scans` | Create run, start it out-of-request, return `{ runId, params }` |
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
| GET | `/instruments/:ticker/chart?tf=&minRr=&useLastHlSl=&riskPerTrade=&noRrReq=&lenFast=&lenSlow=&lengthMajor=&lookback=&multiplier=&bbLength=&bbMult=` | Bars + full overlays + watermark + pine; numeric params recompute overlays live |
| GET | `/instruments/:ticker/status` | Multi-TF watermark from cached bars |
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
