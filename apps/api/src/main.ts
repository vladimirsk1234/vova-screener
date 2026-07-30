import 'reflect-metadata';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { Logger } from '@nestjs/common';
import { NestFactory } from '@nestjs/core';
import type { NestExpressApplication } from '@nestjs/platform-express';
import { AppModule } from './app.module';

/** Vite build output: apps/web/dist (from apps/api/src → ../../web/dist). */
const WEB_DIST = join(__dirname, '..', '..', 'web', 'dist');

async function bootstrap() {
  const app = await NestFactory.create<NestExpressApplication>(AppModule, { bufferLogs: false });
  app.setGlobalPrefix('api');
  app.enableCors({ origin: true });

  if (existsSync(WEB_DIST)) {
    app.useStaticAssets(WEB_DIST, { index: false });
    app.use((req, res, next) => {
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
}

void bootstrap();
