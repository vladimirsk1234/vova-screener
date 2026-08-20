import 'reflect-metadata';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Logger } from '@nestjs/common';
import { NestFactory } from '@nestjs/core';
import type { NestExpressApplication } from '@nestjs/platform-express';
import type { NextFunction, Request, Response } from 'express';
import { AppModule } from './app.module';

/** Vite build output: apps/web/dist (from apps/api/src → ../../web/dist). */
const WEB_DIST = join(__dirname, '..', '..', 'web', 'dist');

/** Load repo-root `.env` so `FMP_API_KEY` / `MONGO_URI` work without exporting them in the shell. */
function loadDotEnv() {
  const candidates = [
    join(process.cwd(), '.env'),
    join(__dirname, '..', '..', '..', '.env'),
    join(__dirname, '..', '..', '.env'),
  ];
  for (const file of candidates) {
    if (!existsSync(file)) continue;
    for (const line of readFileSync(file, 'utf8').split(/\r?\n/)) {
      const t = line.trim();
      if (!t || t.startsWith('#')) continue;
      const eq = t.indexOf('=');
      if (eq <= 0) continue;
      const key = t.slice(0, eq).trim();
      let val = t.slice(eq + 1).trim();
      if (
        (val.startsWith('"') && val.endsWith('"')) ||
        (val.startsWith("'") && val.endsWith("'"))
      ) {
        val = val.slice(1, -1);
      }
      if (process.env[key] == null || process.env[key] === '') process.env[key] = val;
    }
    break;
  }
}

async function bootstrap() {
  loadDotEnv();
  const app = await NestFactory.create<NestExpressApplication>(AppModule, { bufferLogs: false });
  app.setGlobalPrefix('api');
  app.enableCors({ origin: true });

  if (existsSync(WEB_DIST)) {
    app.useStaticAssets(WEB_DIST, { index: false });
    app.use((req: Request, res: Response, next: NextFunction) => {
      if (req.method !== 'GET' && req.method !== 'HEAD') return next();
      if (req.path.startsWith('/api')) return next();
      res.sendFile(join(WEB_DIST, 'index.html'));
    });
  }

  const port = Number(process.env.PORT ?? 3001);
  await app.listen(port, '0.0.0.0');
  new Logger('Bootstrap').log(
    existsSync(WEB_DIST)
      ? `UI + API listening on http://localhost:${port}/`
      : `API listening on http://localhost:${port}/api`,
  );
  new Logger('Bootstrap').log(`Ready — GET /api/health on port ${port}`);
}

void bootstrap();
