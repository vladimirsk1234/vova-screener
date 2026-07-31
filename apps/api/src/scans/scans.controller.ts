import { Body, Controller, Delete, Get, Param, Post, Query, Sse } from '@nestjs/common';
import { concat, map, of, type Observable } from 'rxjs';
import type { Timeframe } from '@vova/engine';
import { ProgressBus } from './progress.bus';
import { ScansService } from './scans.service';
import type { ScanParamsApi } from './scan-runner.service';

@Controller('scans')
export class ScansController {
  constructor(
    private readonly scans: ScansService,
    private readonly bus: ProgressBus,
  ) {}

  @Get('defaults')
  defaults() {
    return this.scans.defaults();
  }

  @Delete('history')
  resetHistory() {
    return this.scans.resetHistory();
  }

  @Post()
  start(@Body() body: Partial<ScanParamsApi>) {
    return this.scans.start(body ?? {}, { trigger: 'manual' });
  }

  @Get()
  list(@Query('limit') limit?: string, @Query('tf') tf?: Timeframe) {
    return this.scans.list({
      limit: limit ? Number(limit) : undefined,
      tf,
    });
  }

  @Get(':id')
  get(@Param('id') id: string) {
    return this.scans.get(id);
  }

  @Get(':id/signals')
  signals(
    @Param('id') id: string,
    @Query('limit') limit?: string,
    @Query('offset') offset?: string,
    @Query('onlyNew') onlyNew?: string,
    @Query('onlyStrong') onlyStrong?: string,
  ) {
    return this.scans.listSignals(id, {
      limit: limit ? Number(limit) : undefined,
      offset: offset ? Number(offset) : undefined,
      onlyNew: onlyNew === 'true',
      onlyStrong: onlyStrong === 'true',
    });
  }

  @Get(':id/rejections')
  rejections(@Param('id') id: string, @Query('limit') limit?: string) {
    return this.scans.listRejections(id, limit ? Number(limit) : undefined);
  }

  @Post(':id/cancel')
  cancel(@Param('id') id: string) {
    return this.scans.cancel(id);
  }

  @Sse(':id/events')
  events(@Param('id') id: string): Observable<{ data: string }> {
    const snapshot = this.bus.snapshot(id);
    const initial = snapshot
      ? of(snapshot)
      : of({ runId: id, phase: 'queued' as const, percent: 0, message: 'Queued' });
    return concat(initial, this.bus.stream(id)).pipe(map((e) => ({ data: JSON.stringify(e) })));
  }
}
