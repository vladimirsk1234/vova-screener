# ADR 003: BullMQ + Redis

## Status
Accepted

## Context
Scans must not block HTTP/SSE; need retries, cron, progress pub/sub.

## Decision
BullMQ on self-hosted Redis; separate `worker` service from `api`.

## Consequences
Small idle cost; history mirrored in `scanRuns` so Redis loss ≠ data loss.
