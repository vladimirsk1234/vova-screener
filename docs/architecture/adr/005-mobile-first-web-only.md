# ADR 005: Mobile-first web only (no Expo)

## Status
Superseded by the on-device iPhone app in `apps/ios` (`packages/device`). The phone stores bars and signals in SQLite and does not use the Railway web UI.

## Context
Operator wants phone UX without building/maintaining a standalone app.

## Decision
Single client `apps/web`, mobile-first. Salvage engine from Expo `mobile/`, then remove Expo from the product.

## Consequences
No EAS/App Store; phone via browser / Add to Home Screen on free or custom URL.
