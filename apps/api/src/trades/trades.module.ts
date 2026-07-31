import { Body, Controller, Delete, Get, Module, Param, Post, Query } from '@nestjs/common';
import { MarketModule } from '../market/market.module';
import { TradesService, type CreateTradeDto } from './trades.service';

@Controller('trades')
class TradesController {
  constructor(private readonly trades: TradesService) {}

  @Get()
  list(
    @Query('status') status?: 'open' | 'closed' | 'dismissed',
    @Query('tf') tf?: 'Daily' | 'Weekly' | 'Monthly',
  ) {
    return this.trades.list(status, tf);
  }

  @Post()
  create(@Body() body: CreateTradeDto) {
    return this.trades.create({ ...body, source: body.source ?? 'manual' });
  }

  @Post('refresh')
  refresh(@Query('tf') tf?: 'Daily' | 'Weekly' | 'Monthly') {
    return this.trades.refresh(tf ? { tf } : {});
  }

  @Post(':id/close')
  close(
    @Param('id') id: string,
    @Body() body: { exitPrice: number; exitDate?: string; exitReason?: string },
  ) {
    return this.trades.close(id, body);
  }

  @Post(':id/dismiss')
  dismiss(@Param('id') id: string) {
    return this.trades.dismiss(id);
  }

  @Delete(':id')
  remove(@Param('id') id: string) {
    return this.trades.remove(id);
  }
}

@Module({
  imports: [MarketModule],
  controllers: [TradesController],
  providers: [TradesService],
  exports: [TradesService],
})
export class TradesModule {}
