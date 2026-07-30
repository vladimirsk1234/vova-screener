# ADR 005: Mobile-first web only (no Expo)

## Status
Accepted

## Context
Operator wants phone UX without building/maintaining a standalone app.

## Decision
Single client `apps/web`, mobile-first. Salvage engine from Expo `mobile/`, then remove Expo from the product.

## Consequences
No EAS/App Store; phone via browser / Add to Home Screen on free or custom URL.
