# Frontend (mobile-first web)

Sole client: `apps/web`. No Expo.

## IA (≤640px = canonical)

Bottom tabs: **Results | History**. Settings is a sheet behind the gear in the header.

Results nests three tab rows, all held in the URL (`/results/:universe/:tf/:bucket`) so every
view is linkable and survives a reload:

1. Stocks · ETF · Manual
2. W · M
3. New · Valid · Closed

New holds the signals that appeared on the current bar of the selected timeframe, Valid the ones that
appeared on an earlier bar and still hold — so the two lists never mix a fresh breakout with a trade
that has been running for four bars. That is the only rule; the RR settings on the chart screen do
not move a symbol between the tabs, and the chart badge reads the same age the tabs split on.

Closed holds the trades sold to close on the newest bar's period. A break on the bar still running
shows there with a `CLOSING` badge: it counts as closed for the period on screen and joins History
only once that bar finishes. An open position a scan evaluated and stopped reporting is on no list
at all — it is still running, and comes back the moment a scan finds its setup again. One the scan
could not price is a different thing and stays in Valid on its last numbers.

Manual is the only screen with a Scan button for arbitrary tickers. Stocks and ETF come from
background scans, which Settings can also start on demand: pick a timeframe or All, press Run scan
now, and the sheet reports which universe and timeframe is going until every list has been rebuilt.
Every card links to `/chart/:ticker`, where Interested / Not Interested marks the tracked signal;
the mark shows as a badge in the lists and is a sort key in every bucket. A closed card passes its
id (`?trade=`), and the chart then opens as a snapshot of that trade: the series cut at the bar it
broke on, entry and exit marked, its own TP and SL drawn, and a Snapshot/Live toggle for the chart
as it stands today.

History covers W / M / All over closed trades, groupable by day, week or month, with an equity
curve per timeframe above the period list so Weekly and Monthly can be compared at a glance.

Results on phone = **cards**. TanStack Table only as desktop (≥1024px) enhancement.

## Touch / layout rules

- `viewport-fit=cover`, `100dvh`, safe-area padding
- Tap targets ≥44px; full-width primary CTA
- Chip toggles + switches; no hover-only actions
- Sticky bottom nav with content padding-bottom
- Chart: Lightweight Charts overlays (critical/HHLL/extensions/fib/MA/BB/TP-SL), HTML watermark, settings sheet, drawing toolbar; Open in TradingView
- Chart parity gates: [`chart-parity.md`](./chart-parity.md)

## Tokens

From Streamlit/Expo palette: bg `#050505`, surfaces `#1e222d`/`#2a2e39`, accent `#2962ff`, up `#089981`, down `#f23645`.

## State

- Server state: TanStack Query. `staleTime` 60s, `placeholderData: keepPreviousData` when
  switching tabs, a 5-minute `refetchInterval` on Results, and neighbouring timeframe/bucket
  queries prefetched on arrival
- Lists page at 100 rows through `useInfiniteQuery`; sorting and paging happen on the server
- `SortChips` renders every sort selector, so a key behaves the same on Results, Manual and
  History: clicking the active key flips direction, a fresh key starts from its natural end
  (descending for RR, P&L and dates; ascending for A-Z). Manual sorts client-side — a handful of
  ad-hoc tickers never justifies a round trip
- `ChartPage` is `React.lazy` so Lightweight Charts stays out of the main bundle
- Ephemeral UI: local React state / Zustand later
- Forms: react-hook-form + Zod from `@vova/contracts` later

## Quality gates

Playwright projects: Pixel 7 + iPhone 14 required; desktop optional enhancement. Optional web manifest / Add to Home Screen — no offline service worker.
