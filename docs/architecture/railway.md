# Railway (NestJS API candidate)

Streamlit stays on Community Cloud until cutover. The Dockerfile builds the React UI
and runs the Nest API, which serves `/api/*` plus the SPA on `/` so Railpack does not
treat the repo as Python.

## Services

| Service | Role |
|---------|------|
| `api` | Dockerfile at repo root → Nest on `PORT` (API + static web UI) |
| `mongo` | Railway MongoDB plugin / template |

## API service

- **Git branch must be** `cursor/react-mongo-refactor-design-18b3` (or another branch that
  contains `Dockerfile` / Nest). `main` is still Streamlit-era and has **no** Dockerfile —
  Railpack then prints `Detected Python` and fails.
- Builder: prefer **Dockerfile** in Service Settings → Build (also set in [`railway.toml`](../../railway.toml)).
- Fallback if Railpack still runs: [`railpack.json`](../../railpack.json) forces `provider: node`
  and `npm run start -w @vova/api`.
- Start: `npm run start -w @vova/api` (ts-node; engine is TypeScript source)
- UI: Vite build copied into the image; Nest serves `apps/web/dist` at `/`
- Healthcheck: `GET /api/health`
- Public networking: generate a domain on the API service — open `https://<host>/` for the UI

If logs still say `Detected Python` after a push that includes `Dockerfile`:
1. Settings → Source → confirm branch + root `/`
2. Settings → Build → Builder = Dockerfile → Redeploy
3. Or trigger **Redeploy** from the deployment menu (Git auto-deploys sometimes ignore Dockerfile)

### Variables

| Name | Value |
|------|--------|
| `MONGO_URI` | **Required.** Reference the Mongo service, e.g. `${{Mongo.MONGO_URL}}` |
| `PORT` | Set by Railway automatically |

Without `MONGO_URI` the API **exits on boot** in production (`NODE_ENV=production`). Healthcheck
`/api/health` will stay `service unavailable` until Mongo is attached and the API restarts.

## Mongo

1. In the same Railway project: **Add service → Database → MongoDB** (or Mongo template).
2. Prefer a connection string that supports replica-set semantics if the plugin provides one
   (transactions / change streams). If the template is standalone and scans fail on
   transactions, switch to a replica-set capable image.
3. On the API service → Variables → add `MONGO_URI` from the Mongo service variable reference.
4. Redeploy the API after the variable is set.

## Deploy checklist

1. Push Dockerfile + `railway.toml` to the linked GitHub branch.
2. Confirm build uses Docker (not “Detected Python”).
3. Attach Mongo + `MONGO_URI`.
4. Expose the API and open `https://<host>/api/health`.
