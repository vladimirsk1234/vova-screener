# ADR 007: Railway Hobby + Yahoo

## Status
Accepted

## Context
Budget ≤ $250/year; universe includes non-US listings.

## Decision
Self-host Mongo/Redis/api/worker on Railway Hobby; Pages for web; keep Yahoo via `barSeries` cache rather than paid market data.

## Consequences
~$115/yr estimated; headroom for contingency; Yahoo fragility mitigated by cache + sync job.
