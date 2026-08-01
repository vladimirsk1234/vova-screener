import { Module } from '@nestjs/common';
import { MarketModule } from '../market/market.module';
import { SettingsModule } from '../settings/settings.module';
import { LegacyTradesMigration } from './legacy-trades.service';
import { SignalAgeBackfill } from './signal-age-backfill.service';

@Module({
  imports: [SettingsModule, MarketModule],
  providers: [LegacyTradesMigration, SignalAgeBackfill],
  exports: [LegacyTradesMigration, SignalAgeBackfill],
})
export class MigrationsModule {}
