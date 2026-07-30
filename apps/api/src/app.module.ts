import { Controller, Get, Module, ServiceUnavailableException } from '@nestjs/common';
import { InjectConnection } from '@nestjs/mongoose';
import type { Connection } from 'mongoose';
import { DbModule } from './db/db.module';
import { InstrumentsModule } from './instruments/instruments.module';
import { MarketModule } from './market/market.module';
import { PresetsModule } from './presets/presets.module';
import { ReportsModule } from './reports/reports.module';
import { ScansModule } from './scans/scans.module';
import { TradesModule } from './trades/trades.module';
import { UniverseModule } from './universe/universe.module';

@Controller('health')
class HealthController {
  constructor(@InjectConnection() private readonly connection: Connection) {}

  @Get()
  health() {
    const ready = this.connection.readyState === 1;
    if (!ready) throw new ServiceUnavailableException({ ok: false, mongo: 'down' });
    return { ok: true, mongo: 'up' };
  }
}

@Module({
  imports: [
    DbModule,
    MarketModule,
    UniverseModule,
    ScansModule,
    InstrumentsModule,
    TradesModule,
    ReportsModule,
    PresetsModule,
  ],
  controllers: [HealthController],
})
export class AppModule {}
