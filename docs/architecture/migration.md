# Migration

## Parallel-run rule

Streamlit Cloud = production until cutover. React/Railway = candidate. Do not move Python to `legacy/` early. Do not break Cloud entrypoints.

## Phases

1. **Scaffold + parity** — monorepo, salvage engine, delete Expo `mobile/`, docs, local web shell. **Done.**
2. **Engine completion** — scan evaluation, ticker parsing, TradingView symbols, binary series codec ported; remaining Python chart indicators (EMA/BB/MACD overlays) still open. **Mostly done.**
3. **API + persistence** — NestJS + MongoDB + SSE + universe import + bar cache + journal + reports, running locally without Docker. **Done** (BullMQ/Redis swap deferred to Phase 5).
4. **Web client** — scan form with live progress, results cards, rejected breakdown, chart with overlays, journal, monthly P&L. **Done**; Playwright phone gates still open.
5. **Railway candidate** — deploy api/worker/mongo/redis + Pages, move the runner into a BullMQ worker; Streamlit stays on.
6. **Cutover** — parity + Playwright + ≥3 scheduled pre-market days + operator sign-off → retire Streamlit.

## Local-first deviations (intentional, revisited in Phase 5)

| Design target | Local today | Why |
|---------------|-------------|-----|
| Mongo in Docker Compose | embedded persistent replica set in `.data/mongo` | no Docker on the dev machine; identical Mongoose code |
| BullMQ on Redis | in-process runner with cancel + SSE progress | no Redis locally; the runner body is queue-agnostic |
| Yahoo `.info` filters (`NOT_EQUITY`, `NOT_US`, `LOW_VOL`) | `LOW_VOL` from cached bar volume, the rest skipped | ticker lists are pre-filtered; `.info` is slow and rate-limited |

## Cutover exit criteria

1. Full-universe scan parity vs Streamlit on same `asOf`
2. Playwright phone + desktop smoke green
3. Scheduled scan ≥3 trading days
4. Explicit operator confirmation

## Rollback

Until Phase 6 complete: keep using Streamlit Cloud. Candidate stack can be paused without user impact.
