# API surface

Base path `/api`. Implemented in `apps/api` (NestJS). Auth is deferred to the Railway phase
(single local user today), so no endpoint is owner-scoped yet.

## Scans

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

## Trades / reports / presets

| Method | Path | Notes |
|--------|------|-------|
| GET | `/trades?status=open\|closed` | Open trades include `currentPrice`, `unrealizedUsd`, `unrealizedR` |
| POST | `/trades` | Create from a signal card |
| POST | `/trades/:id/close` | `{ exitPrice, exitDate?, exitReason? }` |
| POST | `/trades/refresh` | Auto-close open trades whose TP/SL was touched (cached bars) |
| DELETE | `/trades/:id` | Remove journal entry |
| GET | `/reports/monthly` | Monthly buckets, equity curve, totals |
| GET/PUT | `/presets/:key` | `scan` and `chart` params (successor to Streamlit `session_state`) |
| GET | `/health` | Universe + bar cache stats |

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
