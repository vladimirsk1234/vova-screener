import { Module } from '@nestjs/common';
import { MarketModule } from '../market/market.module';
import { SettingsModule } from '../settings/settings.module';
import { LegacyTradesMigration } from './legacy-trades.service';
import { NormalizeSymbols } from './normalize-symbols.service';
import { ReopenNonBreakExits } from './reopen-non-break-exits.service';
import { SignalAgeBackfill } from './signal-age-backfill.service';

// NormalizeSymbols is listed after the journal import so it also sees whatever that pass created.
@Module({
  imports: [SettingsModule, MarketModule],
  providers: [LegacyTradesMigration, NormalizeSymbols, ReopenNonBreakExits, SignalAgeBackfill],
  exports: [LegacyTradesMigration, NormalizeSymbols, ReopenNonBreakExits, SignalAgeBackfill],
})
export class MigrationsModule {}
