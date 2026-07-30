import { Body, Controller, Delete, Get, Module, Param, Post, Query } from '@nestjs/common';
import { MarketModule } from '../market/market.module';
import { TradesService, type CreateTradeDto } from './trades.service';

@Controller('trades')
class TradesController {
  constructor(private readonly trades: TradesService) {}

  @Get()
  list(@Query('status') status?: 'open' | 'closed') {
    return this.trades.list(status);
  }

  @Post()
  create(@Body() body: CreateTradeDto) {
    return this.trades.create(body);
  }

  @Post('refresh')
  refresh() {
    return this.trades.refresh();
  }

  @Post(':id/close')
  close(
    @Param('id') id: string,
    @Body() body: { exitPrice: number; exitDate?: string; exitReason?: string },
  ) {
    return this.trades.close(id, body);
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
