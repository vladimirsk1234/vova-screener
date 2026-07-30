import { Module } from '@nestjs/common';
import { MarketModule } from '../market/market.module';
import { UniverseModule } from '../universe/universe.module';
import { ProgressBus } from './progress.bus';
import { ScanRunnerService } from './scan-runner.service';
import { ScansController } from './scans.controller';
import { ScansService } from './scans.service';

@Module({
  imports: [MarketModule, UniverseModule],
  controllers: [ScansController],
  providers: [ScansService, ScanRunnerService, ProgressBus],
  exports: [ScansService],
})
export class ScansModule {}
