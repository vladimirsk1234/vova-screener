# ADR 002: MongoDB + binary barSeries

## Status
Accepted

## Context
Scans need whole OHLC series per symbol, not bar-level SQL queries. Heterogeneous signal shapes.

## Decision
Self-hosted Mongo; one `barSeries` document per `(symbol, interval)` with packed columns.

## Consequences
~1 read/symbol/scan; bars not individually queryable (acceptable).
