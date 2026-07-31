# ADR 001: TypeScript engine

## Status
Accepted

## Context
Strategy lives in Python today; a partial TS port already existed under `mobile/`.

## Decision
Promote salvaged TS into `@vova/engine` for the worker. Keep Python as parity oracle.

## Consequences
One language on the server hot path; porting risk mitigated by fixture CI.
