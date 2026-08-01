import { Module } from '@nestjs/common';
import { SettingsModule } from '../settings/settings.module';
import { LegacyTradesMigration } from './legacy-trades.service';

@Module({
  imports: [SettingsModule],
  providers: [LegacyTradesMigration],
  exports: [LegacyTradesMigration],
})
export class MigrationsModule {}
