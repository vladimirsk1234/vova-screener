import { Controller, Get, Module } from '@nestjs/common';
import { DbModule } from './db/db.module';
import { InstrumentsModule } from './instruments/instruments.module';
import { MarketModule } from './market/market.module';
import { PresetsModule } from './presets/presets.module';
import { ReportsModule } from './reports/reports.module';
import { ScansModule } from './scans/scans.module';
import { TradesModule } from './trades/trades.module';
import { UniverseModule } from './universe/universe.module';
import { BarsService } from './market/bars.service';
import { UniverseService } from './universe/universe.service';

@Controller('health')
class HealthController {
  constructor(
    private readonly bars: BarsService,
    private readonly universe: UniverseService,
  ) {}

  @Get()
  async health() {
    return {
      ok: true,
      universe: await this.universe.summary(),
      cache: await this.bars.stats(),
    };
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
