/** Chart payloads and the multi-timeframe status watermark. */
import { Injectable, Logger, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import type { Model } from 'mongoose';
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
  shortSymbol,
  signalAge,
  type IndicatorParams,
  type OhlcSeries,
  type Timeframe,
} from '@vova/engine';
import { TRACKED_SIGNAL } from '../db/schemas';
import { BarsService } from '../market/bars.service';
import { UniverseService } from '../universe/universe.service';
import { FundamentalsService, formatDailyChgStr } from './fundamentals.service';

const TFS: Timeframe[] = ['Daily', 'Weekly', 'Monthly'];

/** Same freshness the hourly scan uses for unknown Manual tickers. */
const CHART_BARS_MAX_AGE_HOURS = 0.5;

function sliceSeries<T>(arr: T[], start: number): T[] {
  return arr.slice(start);
}

function remapIdx(idx: number, start: number): number {
  return idx - start;
}

@Injectable()
export class InstrumentsService {
  private readonly log = new Logger(InstrumentsService.name);

  constructor(
    @InjectModel(TRACKED_SIGNAL) private readonly tracked: Model<any>,
    private readonly bars: BarsService,
    private readonly universe: UniverseService,
    private readonly fundamentals: FundamentalsService,
  ) {}

  /**
   * `asOf` turns this into a snapshot: the series is cut at that bar before the engine sees it, so
   * the structure, the critical level and every overlay are what they were on that bar rather than
   * what later bars have since made of them. That is what the chart behind a closed trade needs —
   * a trade closed in March is unreadable against the structure of today.
   */
  async chart(
    yahooTicker: string,
    tf: Timeframe,
    opts: {
      minRr?: number;
      useLastHlSl?: boolean;
      riskPerTrade?: number;
      noRrReq?: boolean;
      asOf?: string;
      chartParams?: Partial<IndicatorParams>;
      /** Skip the TA window trim (80 weekly bars) so Fundamentals can show the full cached series. */
      fullSeries?: boolean;
    } = {},
  ) {
    const listed = await this.universe.isInTrackedUniverse(yahooTicker);
    let series: OhlcSeries | null = null;
    if (listed) {
      // Listed: barSeries from Mongo only — hourly/EOD scan fills it; never Yahoo on chart open.
      series = await this.bars.getCached(yahooTicker, tf);
    } else {
      const live = await this.bars.getBars(yahooTicker, tf, {
        maxAgeHours: CHART_BARS_MAX_AGE_HOURS,
      });
      series = live.bars;
    }
    if (!series?.length) throw new NotFoundException(`no bars for ${yahooTicker}`);
    const asOf = opts.asOf;
    const bars = asOf ? series.filter((bar) => bar.date <= asOf) : series;
    if (!bars.length) throw new NotFoundException(`no bars for ${yahooTicker} up to ${opts.asOf}`);

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
    // The overlays above are drawn with the params the caller asked for, TradingView-style. The
    // signal state below is not: `min_rr` is the only thing these two options change, and RR is a
    // number this app reports rather than a gate it applies. TP, SL and RR come out the same either
    // way, while VALID / NEW / STRONG stay the structural answer the Results tabs are built on.
    const pine = runSequenceVovaPine(bars, {
      atr_len: params.atr_len,
      min_rr: 0,
      use_last_hl_sl: params.use_last_hl_sl,
      risk_dollars: params.risk_dollars,
      no_rr_req: true,
      direction: 'buy',
    });
    const age = signalAge(bars);

    // Live chart only: a closed-trade snapshot must not rewrite today's tracked row. Self-heal
    // closes the gap where Results still shows NEW from the last scan while the chart already
    // reads NO SIGNAL on fresher bars.
    if (!asOf) {
      await this.syncTrackedAge(yahooTicker, tf, bars[bars.length - 1].date, age);
    }

    const keep = opts.fullSeries ? bars.length : maxBarsForTf(tf);
    const start = Math.max(0, bars.length - keep);
    const instrument = await this.universe.findOne(yahooTicker);
    const tvSymbol = instrument?.tvSymbol ?? yahooTicker;
    const symbol = shortSymbol(tvSymbol);

    // Trimmed the same way as the chart series: on a snapshot the D/W/M status has to read as it
    // did on that bar, not as it reads today.
    const upTo = (other: OhlcSeries | null) =>
      !other || !asOf ? other : other.filter((bar) => bar.date <= asOf);
    const lastClose = bars[bars.length - 1]?.close ?? null;
    const [dailyCached, weeklyCached, monthlyCached, chartFund] = await Promise.all([
      tf === 'Daily' ? Promise.resolve(bars) : this.bars.getCached(yahooTicker, 'Daily').then(upTo),
      tf === 'Weekly'
        ? Promise.resolve(bars)
        : this.bars.getCached(yahooTicker, 'Weekly').then(upTo),
      tf === 'Monthly'
        ? Promise.resolve(bars)
        : this.bars.getCached(yahooTicker, 'Monthly').then(upTo),
      this.fundamentals.getChartFundamentals(yahooTicker, { close: lastClose }),
    ]);

    const dailyBars = dailyCached;
    const dailyClose = dailyBars?.length ? dailyBars[dailyBars.length - 1].close : lastClose;
    const prevDailyClose =
      dailyBars && dailyBars.length >= 2 ? dailyBars[dailyBars.length - 2].close : null;

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
            company_name: instrument?.companyName ?? chartFund.company_name ?? yahooTicker,
            pe_str: chartFund.pe_str,
            earn_str: chartFund.earn_str,
            mcap_str: chartFund.mcap_str,
            daily_chg_str: formatDailyChgStr(dailyClose, prevDailyClose),
            description: chartFund.description,
          },
          full,
          params,
          dwmLines,
          chartTf: tf,
          // Same string the cards print, so the chart and the lists never name a symbol differently.
          ticker: symbol,
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
      /** Display form, no exchange prefix; `tvSymbol` is what the TradingView link needs. */
      symbol,
      tvSymbol,
      companyName: instrument?.companyName ?? yahooTicker,
      tf,
      /** The bar the series was cut at, or null when this is the live chart. */
      asOf: asOf ?? null,
      bars: bars.slice(start),
      overlay,
      pine: pine
        ? {
            valid: pine.Valid,
            isNew: pine.New,
            strong: pine.Strong,
            ...age,
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
      // Same rule as the background scans: RR is reported, never required.
      const pine = runSequenceVovaPine(bars, {
        atr_len: ATR_LEN,
        direction: 'buy',
        min_rr: 0,
        no_rr_req: true,
      });
      out[tf] = {
        asOf: bars[bars.length - 1].date,
        seqState: full ? full.seq_state_final : null,
        lastPeakWasHh: pine?.last_peak_was_hh ?? null,
        lastTroughWasHl: pine?.last_trough_was_hl ?? null,
        valid: pine?.Valid ?? false,
        ...signalAge(bars),
        rr: pine && Number.isFinite(pine.RR) ? pine.RR : null,
        close: bars[bars.length - 1].close,
      };
    }
    return { yahooTicker, timeframes: out };
  }

  /**
   * Align the tracked row with the live engine verdict the chart just computed. Results reads
   * stored fields; without this a symbol can sit in NEW for up to an hour after the setup dies.
   *
   * Skips imported journal trades (their entry is the user's), provisional closes (the break is
   * still settling), and any row whose last scan priced a newer bar than this chart series.
   */
  private async syncTrackedAge(
    yahooTicker: string,
    tf: Timeframe,
    barAsOf: string,
    age: { barsSinceValid: number | null; validSinceAsOf: string | null },
  ) {
    try {
      // Bidirectional: hide when structure dies, and put the row back on screen when it recovers
      // so Results does not wait for the next hourly scan after someone opened the chart.
      const set: Record<string, unknown> = {
        barsSinceValid: age.barsSinceValid,
        validSinceAsOf: age.validSinceAsOf,
        signalValid: age.barsSinceValid != null,
      };

      await this.tracked
        .updateOne(
          {
            yahooTicker,
            tf,
            status: 'active',
            imported: { $ne: true },
            provisionalClose: { $ne: true },
            $or: [{ lastSeenAsOf: { $exists: false } }, { lastSeenAsOf: { $lte: barAsOf } }],
          },
          { $set: set },
        )
        .exec();
    } catch (err) {
      this.log.warn(
        `tracked-age sync ${yahooTicker}/${tf} failed: ${(err as Error).message}`,
      );
    }
  }
}
