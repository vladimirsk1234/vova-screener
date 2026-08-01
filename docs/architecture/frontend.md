# Frontend (mobile-first web)

Sole client: `apps/web`. No Expo.

## IA (≤640px = canonical)

Bottom tabs: **Results | History**. Settings is a sheet behind the gear in the header.

Results nests three tab rows, all held in the URL (`/results/:universe/:tf/:bucket`) so every
view is linkable and survives a reload:

1. Stocks · ETF · Manual
2. D · W · M
3. New · Valid · Closed

Manual is the only screen with a Scan button; Stocks and ETF come from background scans.
Every card links to `/chart/:ticker`, where Interested / Not Interested marks the tracked signal;
the mark shows as a badge in the lists and is a sort key in every bucket.

History covers D / W / M / All over closed signals, groupable by day, week or month.

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
- `ChartPage` is `React.lazy` so Lightweight Charts stays out of the main bundle
- Ephemeral UI: local React state / Zustand later
- Forms: react-hook-form + Zod from `@vova/contracts` later

## Quality gates

Playwright projects: Pixel 7 + iPhone 14 required; desktop optional enhancement. Optional web manifest / Add to Home Screen — no offline service worker.
