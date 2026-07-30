# Hosting and cost

Budget ceiling: **$250/year**.

## Production until cutover

**Streamlit Community Cloud** — free, unchanged. Do not migrate traffic away until Phase 6 exit criteria.

## Candidate stack (Railway Hobby + Cloudflare)

| Service | Role | Rough $/mo |
|---------|------|------------|
| `api` | NestJS | ~2.50 |
| `worker` | BullMQ scans | ~1.50 |
| `mongo` | self-hosted RS | ~5 |
| `redis` | BullMQ | ~0.50 |
| Cloudflare Pages | `apps/web` static | 0 |
| R2 | mongodump backups | ~0 |

≈ **$9–10/mo (~$115/yr)** + optional domain ~$12/yr. Domain is **optional** — phone works on `*.pages.dev`.

## Local (no Railway required)

- `npm run dev:web` on this PC
- Later: Docker Compose for api/worker/mongo/redis (Docker not installed on this machine yet)

## Guardrails

Railway usage alert; WiredTiger cache cap; TTL on fundamentals; retention on old runs/signals; scan concurrency hard cap.
