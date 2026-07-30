FROM node:20-bookworm-slim

WORKDIR /app

COPY package.json package-lock.json ./
COPY apps/api/package.json apps/api/
COPY apps/web/package.json apps/web/
COPY packages/engine/package.json packages/engine/

RUN npm ci --omit=dev

COPY apps/api apps/api
COPY packages/engine packages/engine

ENV NODE_ENV=production
EXPOSE 3001

CMD ["npm", "run", "start", "-w", "@vova/api"]
