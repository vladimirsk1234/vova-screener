# Engine parity

## Source of truth

Python [`sequence_vova.py`](../../sequence_vova.py) remains the oracle (and Streamlit production) until cutover, then stays as CI oracle.

## TS package

[`packages/engine`](../../packages/engine) salvaged from former `mobile/src/engine`:

- `runSequenceVovaPine` / `runSequenceVovaCloseScan` / `runStructureOverlay` / `explainInvalidBuy`
- `dataUtils` OHLC helpers
- Fixture: `packages/engine/fixtures/parity_sample.json`
- Harness: `npm run parity`

## Chart path (ported)

`runSequenceVovaFull` + `computeOverlays` + watermark helpers live in `@vova/engine`:
MACD, DMI/ADX, Bollinger, Elder envelope/impulse, SMA major, peak/trough labels,
fib, extension lines, trade/DWM watermark text. Python remains the oracle for CI
fixtures; `npm run parity` checks pine/close fixtures and full↔pine last-bar consistency.

## Gate

Parity suite must stay green before trusting worker scans in production cutover.
