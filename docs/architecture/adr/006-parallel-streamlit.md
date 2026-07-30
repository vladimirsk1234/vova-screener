# ADR 006: Parallel Streamlit until Railway cutover

## Status
Accepted

## Context
Need zero downtime for the working screener during rewrite.

## Decision
Streamlit Community Cloud remains production until explicit cutover exit criteria. React/Railway is candidate only. Python stays at repo root (not early `legacy/`).

## Consequences
Two UIs during migration; parity gates required before retirement.
