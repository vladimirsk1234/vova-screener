import 'reflect-metadata';
import { Logger } from '@nestjs/common';
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule, { bufferLogs: false });
  app.setGlobalPrefix('api');
  app.enableCors({ origin: true });
  const port = Number(process.env.PORT ?? 3001);
  await app.listen(port, '0.0.0.0');
  new Logger('Bootstrap').log(`API listening on http://localhost:${port}/api`);
}

void bootstrap();
