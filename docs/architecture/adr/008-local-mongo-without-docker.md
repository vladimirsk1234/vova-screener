# ADR 008 — Local MongoDB without Docker, in-process scan runner

Status: accepted (local development only)

## Context

The design targets MongoDB + Redis/BullMQ in Docker Compose locally and on Railway. The
development machine has neither Docker nor MongoDB installed, and the operator needs the full app
(scan, storage, charts, journal) working on that machine now.

## Decision

Two local-only substitutions, both behind the same interfaces used in production:

1. **MongoDB**: if `MONGO_URI` is unset, the API starts a persistent single-node replica set from
   the `mongodb-memory-server` binary with `dbPath = .data/mongo`. This is a real `mongod`, so
   Mongoose schemas, indexes, TTLs and aggregation behave exactly as on Railway. Setting
   `MONGO_URI` (Railway, Compose, or an installed mongod) bypasses it entirely.
2. **Queue**: scans run in-process, started out-of-request, with an `AbortController` for cancel
   and an RxJS subject fanning progress out over SSE. Progress publishing and cancellation are the
   only couplings, so moving the runner body into a BullMQ processor in Phase 5 does not change the
   scan logic.

## Consequences

- No Docker/MongoDB install required to run the whole stack locally
- First boot downloads a ~600 MB `mongod` binary once (cached under the user profile); on Windows
  the first spawn can fail with `EBUSY` while antivirus scans it — retry once
- Scans die with the API process locally; durability of in-flight runs arrives with BullMQ
- Concurrency is bounded by one process, which is fine for a single operator

## Alternatives rejected

- **Install MongoDB Community locally** — needs admin rights, more setup per machine
- **SQLite/file adapter for local storage** — a second persistence implementation to keep in sync
  with the Mongo one; parity bugs would only show up in production
