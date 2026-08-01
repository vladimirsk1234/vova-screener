/** Chart payloads and the multi-timeframe status watermark. */
import { Injectable, NotFoundException } from '@nestjs/common';
import {
  ATR_LEN,
  buildDwmLines,
  buildTradeLine,
  buildWatermarkParts,
  defaultIndicatorParams,
  indicatorParamsFromDict,
  maxBarsForTf,
  runSequenceVovaFull,
  runSequenceVovaPine,
  type IndicatorParams,
  type Timeframe,
} from '@vova/engine';
import { BarsService } from '../market/bars.service';
import { UniverseService } from '../universe/universe.service';

const TFS: Timeframe[] = ['Daily', 'Weekly', 'Monthly'];

function sliceSeries<T>(arr: T[], start: number): T[] {
  return arr.slice(start);
}

function remapIdx(idx: number, start: number): number {
  return idx - start;
}

@Injectable()
export class InstrumentsService {
  constructor(
    private readonly bars: BarsService,
    private readonly universe: UniverseService,
  ) {}

  async chart(
    yahooTicker: string,
    tf: Timeframe,
    opts: {
      minRr?: number;
      useLastHlSl?: boolean;
      riskPerTrade?: number;
      noRrReq?: boolean;
      chartParams?: Partial<IndicatorParams>;
    } = {},
  ) {
    const { bars } = await this.bars.getBars(yahooTicker, tf, { maxAgeHours: 12 });
    if (!bars?.length) throw new NotFoundException(`no bars for ${yahooTicker}`);

    const params = indicatorParamsFromDict({
      ...defaultIndicatorParams(),
      ...(opts.chartParams ?? {}),
      min_rr: opts.minRr ?? opts.chartParams?.min_rr ?? 1.5,
      use_last_hl_sl: opts.useLastHlSl ?? opts.chartParams?.use_last_hl_sl ?? true,
      risk_dollars: opts.riskPerTrade ?? opts.chartParams?.risk_dollars ?? 100,
      no_rr_req: opts.noRrReq ?? opts.chartParams?.no_rr_req ?? false,
      atr_len: ATR_LEN,
    });

    const full = runSequenceVovaFull(bars, { params });
    const pine = runSequenceVovaPine(bars, {
      atr_len: params.atr_len,
      min_rr: params.min_rr,
      use_last_hl_sl: params.use_last_hl_sl,
      risk_dollars: params.risk_dollars,
      no_rr_req: params.no_rr_req,
      direction: 'buy',
    });

    const keep = maxBarsForTf(tf);
    const start = Math.max(0, bars.length - keep);
    const instrument = await this.universe.findOne(yahooTicker);

    const [dailyCached, weeklyCached, monthlyCached] = await Promise.all([
      tf === 'Daily' ? Promise.resolve(bars) : this.bars.getCached(yahooTicker, 'Daily'),
      tf === 'Weekly' ? Promise.resolve(bars) : this.bars.getCached(yahooTicker, 'Weekly'),
      tf === 'Monthly' ? Promise.resolve(bars) : this.bars.getCached(yahooTicker, 'Monthly'),
    ]);

    const dwmLines = full
      ? buildDwmLines({
          chartBars: bars,
          dailyBars: dailyCached,
          weeklyBars: weeklyCached,
          monthlyBars: monthlyCached,
          chartTf: tf,
          params,
        })
      : {};
    const tradeLine = full ? buildTradeLine(full, params, bars.length - 1) : '';
    const watermark = full
      ? buildWatermarkParts({
          fundamentals: {
            company_name: instrument?.companyName ?? yahooTicker,
          },
          full,
          params,
          dwmLines,
          chartTf: tf,
          ticker: instrument?.tvSymbol ?? yahooTicker,
          tradeLine,
        })
      : null;

    const overlay = full
      ? {
          critical: sliceSeries(full.critical_level_series, start),
          seqState: sliceSeries(full.seq_state_series, start),
          bullishBreak: sliceSeries(full.bullish_break, start),
          bearishBreak: sliceSeries(full.bearish_break, start),
          lastPeak: full.last_peak,
          lastTrough: full.last_trough,
          tp: Number.isFinite(full.TP) ? full.TP : null,
          sl: Number.isFinite(full.SL) ? full.SL : null,
          rr: Number.isFinite(full.RR) ? full.RR : null,
          peaks: full.peaks
            .filter((p) => p.idx >= start)
            .map((p) => ({ ...p, idx: remapIdx(p.idx, start) })),
          troughs: full.troughs
            .filter((t) => t.idx >= start)
            .map((t) => ({ ...t, idx: remapIdx(t.idx, start) })),
          extensionLines: full.extension_lines.map((ln) => ({
            kind: ln.kind,
            x0Idx: remapIdx(ln.x0_idx, start),
            y0: ln.y0,
            x1Idx: remapIdx(ln.x1_idx, start),
            y1: ln.y1,
            // Keep absolute slope origin for extend-right even if x0 is off-window.
            rawX0Idx: ln.x0_idx - start,
            rawX1Idx: ln.x1_idx - start,
          })),
          fib: full.fib
            ? {
                high: full.fib.high,
                highIdx: remapIdx(full.fib.high_idx, start),
                low: full.fib.low,
                lowIdx: remapIdx(full.fib.low_idx, start),
                fib382: full.fib.fib_382,
                fib500: full.fib.fib_500,
                fib618: full.fib.fib_618,
              }
            : null,
          overlays: {
            emaFast: sliceSeries(full.overlays.ema_fast, start),
            emaSlow: sliceSeries(full.overlays.ema_slow, start),
            smaMajor: sliceSeries(full.overlays.sma_major, start),
            envUpper: sliceSeries(full.overlays.env_upper, start),
            envLower: sliceSeries(full.overlays.env_lower, start),
            bbBasis: sliceSeries(full.overlays.bb_basis, start),
            bbUpper: sliceSeries(full.overlays.bb_upper, start),
            bbLower: sliceSeries(full.overlays.bb_lower, start),
          },
          impulseColors: sliceSeries(full.impulse_colors, start),
          atrPct: full.ATR_pct,
          adx: full.ADX,
          signalBarIndex:
            full.signal_bar_index != null && full.signal_bar_index >= start
              ? remapIdx(full.signal_bar_index, start)
              : null,
          seqStateFinal: full.seq_state_final,
          criticalLevel: full.critical_level,
        }
      : null;

    return {
      yahooTicker,
      tvSymbol: instrument?.tvSymbol ?? yahooTicker,
      companyName: instrument?.companyName ?? yahooTicker,
      tf,
      bars: bars.slice(start),
      overlay,
      pine: pine
        ? {
            valid: pine.Valid,
            isNew: pine.New,
            strong: pine.Strong,
            // Same number the Results tabs split on, so the badge here cannot disagree with them.
            barsSinceValid: pine.bars_since_valid,
            validSinceAsOf:
              pine.valid_since_index != null
                ? (bars[pine.valid_since_index]?.date ?? null)
                : null,
            tp: Number.isFinite(pine.TP) ? pine.TP : null,
            sl: Number.isFinite(pine.SL) ? pine.SL : null,
            rr: Number.isFinite(pine.RR) ? pine.RR : null,
            close: pine.Close,
            atr: pine.ATR,
            lastPeakWasHh: pine.last_peak_was_hh,
            lastTroughWasHl: pine.last_trough_was_hl,
          }
        : null,
      watermark: watermark
        ? {
            lines: watermark.lines,
            main: watermark.main,
            tradeLine,
            dwmLines,
            description: watermark.description,
          }
        : null,
      params,
    };
  }

  async status(yahooTicker: string) {
    const out: Record<string, unknown> = {};
    for (const tf of TFS) {
      const bars = await this.bars.getCached(yahooTicker, tf);
      if (!bars?.length) {
        out[tf] = null;
        continue;
      }
      const full = runSequenceVovaFull(bars, { atr_len: ATR_LEN });
      const pine = runSequenceVovaPine(bars, { atr_len: ATR_LEN, direction: 'buy' });
      out[tf] = {
        asOf: bars[bars.length - 1].date,
        seqState: full ? full.seq_state_final : null,
        lastPeakWasHh: pine?.last_peak_was_hh ?? null,
        lastTroughWasHl: pine?.last_trough_was_hl ?? null,
        valid: pine?.Valid ?? false,
        barsSinceValid: pine?.bars_since_valid ?? null,
        validSinceAsOf:
          pine?.valid_since_index != null ? (bars[pine.valid_since_index]?.date ?? null) : null,
        rr: pine && Number.isFinite(pine.RR) ? pine.RR : null,
        close: bars[bars.length - 1].close,
      };
    }
    return { yahooTicker, timeframes: out };
  }
}
