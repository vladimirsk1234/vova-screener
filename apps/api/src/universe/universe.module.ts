import { Controller, Get, Module, Post } from '@nestjs/common';
import { UniverseService } from './universe.service';

@Controller('universe')
class UniverseController {
  constructor(private readonly universe: UniverseService) {}

  @Get('summary')
  summary() {
    return this.universe.summary();
  }

  @Post('import')
  import() {
    return this.universe.importFromFiles();
  }
}

@Module({
  controllers: [UniverseController],
  providers: [UniverseService],
  exports: [UniverseService],
})
export class UniverseModule {}
