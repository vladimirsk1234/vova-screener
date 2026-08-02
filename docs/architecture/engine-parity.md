# Engine parity

## Source of truth

Python [`sequence_vova.py`](../../sequence_vova.py) remains the oracle (and Streamlit production) until cutover, then stays as CI oracle.

## TS package

[`packages/engine`](../../packages/engine) salvaged from former `mobile/src/engine`:

- `runSequenceVovaPine` / `runSequenceVovaCloseScan` / `runCloseLedger` / `runStructureOverlay` /
  `explainInvalidBuy`
- `dataUtils` OHLC helpers
- Fixtures: `packages/engine/fixtures/parity_sample.json`,
  `packages/engine/fixtures/reject_reasons_parity.json`,
  `packages/engine/fixtures/close_ledger_parity.json`
  (regenerate: `python scripts/export_parity_fixture.py`,
  `python scripts/export_reject_reason_fixture.py`,
  `python scripts/export_close_ledger_fixture.py`)
- Harness: `npm run parity`

## The close scan is a replay, and the whole replay is compared

`runSequenceVovaCloseScan` answers what the Streamlit table renders: the trade that gives up on the
last bar. Tracked positions read the rest of the replay too — which trade a symbol is in, and the
bar it was entered on — so `runCloseLedger` returns every trade in the series and
`check_close_ledger_parity.ts` compares them one for one against
`sequence_vova.run_sequence_vova_close_ledger`. One extra or missing trade shifts every entry after
it, which is exactly the failure that would misprice a closed position without changing the
last-bar answer the older fixture checks.

## Live parity: the close list against today's market

Fixtures prove the arithmetic. They cannot prove the app and Streamlit are looking at the same
bars, and that is the half that decides whether the CLOSED tab agrees with the Streamlit table — a
series ending one bar early moves every break by a bar and empties the list. Two harnesses cover
it, both needing network:

```bash
npx tsx packages/engine/scripts/export_live_close_scan.ts --tf Daily --limit 900
PYTHONPATH=. python3 scripts/check_live_close_parity.py reports/live_close_daily.json
```

reports the close list the engine reads off real symbols against the one
`run_sequence_vova_close_scan` reads off the same bars, and separately against the series
`yf.download` + `headless_scanner._prepare_scan_ohlc` produce for the same symbols.

```bash
VOVA_SMOKE_LIMIT=400 npm run smoke:close-live      # also VOVA_SMOKE_TF=Weekly|Monthly
```

runs a real scan through the whole app — scan runner, tracker, Results query — and asserts the
CLOSED tab shows exactly the closes the scan found, priced from the replay entry, and that a second
pass over the same period adds nothing.

## History window is part of parity

`intervalAndPeriod` must match `data_utils.interval_and_period` (Daily 2y, Weekly/Monthly 10y).
The sequence walk is path-dependent: a shorter window can keep a stale confirmed trough/peak and
silently change SL, RR and the reject reason. YMM Monthly with one leading bar dropped moves SL
from 6.66 to 4.12 and RR from 1.51 to 0.80 — that case is pinned in
`reject_reasons_parity.json`.

Yahoo cannot serve full history here: `1mo&range=max` returns an irregular grid, and `period1=0`
drops the in-progress bar the scan needs.

## Reject reasons

`explainInvalidBuy` mirrors the gate order of `sequence_vova.explain_invalid_buy`
(`NO_SEQ_UP` first). The only intentional string difference is the ` (min x.xx)` threshold
suffix on `RR_TOO_LOW`, which the parity harness strips before comparing.

## Live TradingView bar

Parity is defined against the Python oracle on the same bars, not against a live TradingView
chart. TradingView scores the in-progress bar, a scan scores the stored snapshot, so a close
sitting next to the critical level legitimately reads VALID there and `NO_SEQ_UP` here. Rejected
rows carry `detail` (bar date, close, critical level, seq state, SL/TP/RR) so that difference is
visible instead of looking like an engine bug.

## Chart path (ported)

`runSequenceVovaFull` + `computeOverlays` + watermark helpers live in `@vova/engine`:
MACD, DMI/ADX, Bollinger, Elder envelope/impulse, SMA major, peak/trough labels,
fib, extension lines, trade/DWM watermark text. Python remains the oracle for CI
fixtures; `npm run parity` checks pine/close fixtures and full↔pine last-bar consistency.

## Gate

Parity suite must stay green before trusting worker scans in production cutover.
