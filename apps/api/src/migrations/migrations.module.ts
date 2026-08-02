import { Module } from '@nestjs/common';
import { MarketModule } from '../market/market.module';
import { SettingsModule } from '../settings/settings.module';
import { LegacyTradesMigration } from './legacy-trades.service';
import { ReopenNonBreakExits } from './reopen-non-break-exits.service';
import { SignalAgeBackfill } from './signal-age-backfill.service';

@Module({
  imports: [SettingsModule, MarketModule],
  providers: [LegacyTradesMigration, ReopenNonBreakExits, SignalAgeBackfill],
  exports: [LegacyTradesMigration, ReopenNonBreakExits, SignalAgeBackfill],
})
export class MigrationsModule {}
