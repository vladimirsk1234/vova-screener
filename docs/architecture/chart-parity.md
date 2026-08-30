# Chart visual parity — acceptance

## Goal

React Lightweight Charts chart matches Streamlit/Plotly Sequence Vova chart for the same
ticker/timeframe/settings, and exposes compact drawing tools.

## Automated gates

| Gate | How |
|------|-----|
| Engine pine/close fixture | `npm run parity -w @vova/engine` |
| Full overlay self-check | same command — full↔pine last-bar + overlays length |
| Web/API typecheck | `npm run typecheck` |
| Playwright chart smoke | `npm run test:e2e -w @vova/web` (Pixel 7 + desktop) |

Playwright cases:

1. Chart page loads host + watermark (or pine card) for a known ticker.
2. Settings sheet opens; toggling Fibonacci/EMA persists after Save preset (`/api/presets/chart`).
3. Drawing toolbar: create trend line (two clicks), undo removes it, redo restores.
4. Timeframe chip Weekly → Monthly remounts chart without crash.
5. Snapshot: default Streamlit theme (`bg #707585`) on desktop 1280×720 and Pixel 7.

## Manual side-by-side gate

Open the same symbol (e.g. a recent scan hit) in:

1. Streamlit Plotly preview
2. React `/chart/:yahooTicker`
3. TradingView deep link

Check:

- Candles align on date/OHLC (Yahoo series)
- Critical level color flips with seq state (green up / red down)
- HH/LH/DT and HL/LL/DB labels on confirmed pivots
- Extension lines continue to the right edge
- Break markers present when structure breaks
- Optional Fib 0.382/0.5/0.618 and EMA/SMA/BB when enabled
- TP/SL dashed lines when enabled
- Watermark D/W/M + trade line text matches Streamlit intent
- Settings Reset restores Streamlit defaults from `indicator_params.py`

Allowed differences: canvas vs SVG fonts, LWC attribution logo, no full TV drawing
toolbar / Pine studies UI.

## Escape hatch

`Open in TradingView` remains for native TV UX.
