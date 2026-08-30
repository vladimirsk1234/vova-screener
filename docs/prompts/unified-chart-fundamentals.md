# Unified chart window: one price chart, metrics swap only

Implementation prompt for merging Technical Analysis and Fundamentals into a single `/chart/:ticker` window. The weekly price chart stays mounted; the TA / Fundamentals toggle changes only the panel under the chart.

Do not mount a second valuation chart. Do not overlay FAST Graphs series on the candlesticks. Full Fundamentals tabs (Forecasting, DCF, Performance, Profile) are out of scope for this pass.

---

## Prompt

Role & Goal
You are an expert React developer specializing in iPhone-first mobile UI/UX and financial charting. Refactor the chart screen so Technical Analysis (TA) and Fundamentals share one window. The price chart must stay in place. The TA / Fundamentals toggle changes only the metrics and buttons under the chart. Do not break existing calculations, data hooks, or the sequence chart engine.

This is a React PWA (SV Screener). Charts use TradingView Lightweight Charts (`lightweight-charts`), not an embedded TradingView widget.

Current architecture (do not invent a different one)
- Card tap already opens `/chart/:ticker` (optional `?trade=<id>`). See `apps/web/src/components/SignalCard.tsx` and `apps/web/src/pages/ManualPage.tsx`.
- TA is `apps/web/src/pages/ChartPage.tsx`. Candlesticks mount via `apps/web/src/components/mountSequenceChart.ts`.
- Fundamentals is a separate route `/fundamentals/:ticker` in `apps/web/src/pages/FundamentalsPage.tsx`. It currently mounts a second FAST Graphs–style chart via `apps/web/src/components/mountValuationChart.ts`. That second chart must NOT appear in the unified window.
- Shell already hides header + bottom nav on `/chart/` and `/fundamentals/` (`isChart` in `apps/web/src/App.tsx`).
- Valuation numbers come from `buildValuationSeries()` in `packages/engine/src/fundamentalsValuation.ts` and `api.fundamentals(ticker, 'eps')`. Reuse these hooks for metrics only. Do not change API contracts or engine formulas.
- Today’s TA bottom grid (Interested, Not Interested, Fundamentals, TradingView) lives in `.chart-actions` in `apps/web/src/styles.css`. Replace the Fundamentals link with the view toggle; leave the rest of TA as it is unless a button must move to make room for the two-button toggle.

Hard rule: one chart, never two
- There is a single chart host: the existing price candlestick (`.chart-stage` / `.chart-host` + `mountSequenceChart`).
- Do NOT mount `mountValuationChart` on this screen.
- Do NOT overlay fair-value / EPS / Normal P/E series on the candlesticks.
- Do NOT remount, destroy, resize-swap, or navigate away when the user taps Fundamentals.
- The chart window stays exactly where it is. Only the panel under the chart changes.

Required UX
1. One window. Card tap still opens `/chart/:ticker`. No modal, no second page for fundamentals.
2. Layout, top to bottom:
   - Existing chart header (back, ticker, settings) — keep.
   - The same price chart. Timeframe is Weekly. In Fundamentals view the series stays Weekly. Do not replace it with an annual valuation chart.
   - Directly under the chart window: the same button row region as today (`.chart-actions`).
   - That row has exactly two view buttons: "TA" (or "Technical Analysis") and "Fundamentals" (or "Fundamental Analysis"). Selected state like existing `Chips` / `.selected`.
   - Under those two buttons (or in that same below-chart region): the view-specific metrics.
3. TA view: keep Technical Analysis as it works now — Weekly / Monthly chips, overlays, watermark, settings, trade snapshot, RR / TP / SL, drawings, crosshair, Interested / Not Interested / TradingView. Do not redesign TA.
4. Fundamentals view: the Weekly price chart does not change. Swap only the below-chart content to fundamental metrics already computed today, at minimum:
   - Fair value
   - Price vs fair / premium
   - Earnings (EPS / how much the company earns)
   - Existing fundamental option buttons under the chart, same chips as now: metric (EPS, Sales/sh, FCF/sh, Owner earn.) and window (5Y / 10Y / MAX)
5. Switching TA ↔ Fundamentals must not reset ticker, trade snapshot, settings, drawings, or unmount the chart host. Keep the Weekly price series on screen.
6. Redirect `/fundamentals/:ticker` → `/chart/:ticker?view=fundamentals` so old links work. Optional `?view=fundamentals` is enough; keep `?trade=`. Update `apps/web/src/lib/tabMemory.ts` if it still treats `/fundamentals/...` as a first-class path.

Out of scope — finalize later
- Do not port the full Fundamentals tabs (Forecasting, DCF, Performance, Profile) in this pass.
- Do not add FAST Graphs forecast bars onto the price chart.
- Do not restyle SignalCard or Settings filters.
- Do not replace Lightweight Charts with a TradingView widget.

State & modularity
- View state: `'ta' | 'fundamentals'`, synced to `?view=`.
- Keep ChartPage as the shell. Extract a small below-chart fundamentals metrics panel that reuses `['fundamentals', ticker]` + `buildValuationSeries` for numbers and chips only.
- Keep TA queries as they are. For the visible series use Weekly when the fundamentals view is active (or keep the already-Weekly chart). Do not fetch a second chart type.
- iPhone: no overflow, ~44px tap targets, `--safe-bottom` on the below-chart actions. The chart keeps flex-grow (`.chart-stage`); metrics sit under it and may scroll if needed. Do not shrink the chart into a second stacked chart layout.

Constraints
- Do not change backend valuation formulas, FMP fetch, or sequence-engine calculations.
- Do not assume missing logic. If something is unclear, ask one concrete question and wait.
- Follow existing React Query + React Router patterns.
- Update `apps/web/e2e/chart-parity.spec.ts`: the old Fundamentals `<Link>` goes away; assert the TA / Fundamentals toggle instead. Keep TradingView + Interested disabled-when-untracked coverage.
- Keep existing English UI labels in English.

Definition of done
- One chart window. The candlestick host never swaps to a valuation chart.
- Two buttons under the chart: TA and Fundamentals.
- TA unchanged aside from that toggle replacing the old Fundamentals navigation link.
- Fundamentals: same Weekly price chart + fair value / earnings / current fundamental chips under the chart.
- Old `/fundamentals/:ticker` redirects into this window.
- No second chart, no FAST Graphs overlay. Everything else later.
