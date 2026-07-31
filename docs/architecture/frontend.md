# Frontend (mobile-first web)

Sole client: `apps/web`. No Expo.

## IA (≤640px = canonical)

Bottom tabs: **Scan | History | Trades | P&L** (+ Settings later).

Scan flow: Config → Progress → Results (cards) → Chart detail. Rejected as drawer/screen.

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

- Server state: TanStack Query (when API exists)
- Ephemeral UI: local React state / Zustand later
- Forms: react-hook-form + Zod from `@vova/contracts` later

## Quality gates

Playwright projects: Pixel 7 + iPhone 14 required; desktop optional enhancement. Optional web manifest / Add to Home Screen — no offline service worker.
