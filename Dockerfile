# Build the React UI, then run the Nest API which serves /api + the static SPA.
FROM node:20-bookworm-slim AS web-build

WORKDIR /app

COPY package.json package-lock.json ./
COPY apps/api/package.json apps/api/
COPY apps/web/package.json apps/web/
COPY packages/engine/package.json packages/engine/

RUN npm ci

COPY apps/web apps/web
COPY packages/engine packages/engine

RUN npm run build -w @vova/web

FROM node:20-bookworm-slim

WORKDIR /app

COPY package.json package-lock.json ./
COPY apps/api/package.json apps/api/
COPY apps/web/package.json apps/web/
COPY packages/engine/package.json packages/engine/

RUN npm ci --omit=dev

COPY apps/api apps/api
COPY packages/engine packages/engine
COPY STOCK-TICKERS.txt TV-LIST-ETF.txt ./
COPY --from=web-build /app/apps/web/dist apps/web/dist

ENV NODE_ENV=production
EXPOSE 3001

CMD ["npm", "run", "start", "-w", "@vova/api"]
