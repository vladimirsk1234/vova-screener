/**
 * Chart payload in the exact shape `GET /instruments/:ticker/chart` returns, built on the phone.
 * Mirrors `InstrumentsService.chart`; the watermark leaves out the FMP market cap / P/E / earnings.
 */
import { maxBarsForTf } from '../../engine/src/dataUtils.ts';
import { defaultIndicatorParams, indicatorParamsFromDict, type IndicatorParams } from '../../engine/src/indicatorParams.ts';
import { ATR_LEN } from '../../engine/src/evaluate.ts';
import { runSequenceVovaPine } from '../../engine/src/sequenceVova.ts';
import { runSequenceVovaFull } from '../../engine/src/sequenceVovaFull.ts';
import { signalAge } from '../../engine/src/signalAge.ts';
import { shortSymbol } from '../../engine/src/tickers.ts';
import type { OhlcSeries, Timeframe } from '../../engine/src/types.ts';
import { buildDwmLines, buildTradeLine, buildWatermarkParts } from '../../engine/src/watermark.ts';

export type ChartOptions = {
  minRr?: number;
  useLastHlSl?: boolean;
  riskPerTrade?: number;
  noRrReq?: boolean;
  asOf?: string;
  chartParams?: Partial<IndicatorParams>;
  fullSeries?: boolean;
};

export type ChartInput = {
  yahooTicker: string;
  tvSymbol: string;
  companyName: string;
  tf: 'Weekly' | 'Monthly';
  bars: OhlcSeries;
  dailyBars: OhlcSeries | null;
  weeklyBars: OhlcSeries | null;
  monthlyBars: OhlcSeries | null;
};

export function buildChartPayload(input: ChartInput, opts: ChartOptions = {}) {
  const { tf } = input;
  const asOf = opts.asOf;
  const bars = asOf ? input.bars.filter((bar) => bar.date <= asOf) : input.bars;
  if (!bars.length) throw new ChartError(`no bars for ${input.yahooTicker}${asOf ? ` up to ${asOf}` : ''}`);

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
    min_rr: 0,
    use_last_hl_sl: params.use_last_hl_sl,
    risk_dollars: params.risk_dollars,
    no_rr_req: true,
    direction: 'buy',
  });
  const age = signalAge(bars);

  const keep = opts.fullSeries ? bars.length : maxBarsForTf(tf as Timeframe);
  const start = Math.max(0, bars.length - keep);
  const symbol = shortSymbol(input.tvSymbol || input.yahooTicker);

  const upTo = (other: OhlcSeries | null) =>
    !other || !asOf ? other : other.filter((bar) => bar.date <= asOf);
  const dailyCached = upTo(input.dailyBars);
  const weeklyCached = tf === 'Weekly' ? bars : upTo(input.weeklyBars);
  const monthlyCached = tf === 'Monthly' ? bars : upTo(input.monthlyBars);
  const lastClose = bars[bars.length - 1]?.close ?? null;
  const dailyClose = dailyCached?.length ? dailyCached[dailyCached.length - 1].close : lastClose;
  const prevDailyClose =
    dailyCached && dailyCached.length >= 2 ? dailyCached[dailyCached.length - 2].close : null;

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
          company_name: input.companyName || input.yahooTicker,
          daily_chg_str: formatDailyChgStr(dailyClose, prevDailyClose),
        },
        full,
        params,
        dwmLines,
        chartTf: tf,
        ticker: symbol,
        tradeLine,
        omitFundamentals: true,
      })
    : null;

  const slice = <T>(arr: T[]) => arr.slice(start);
  const remap = (idx: number) => idx - start;

  const overlay = full
    ? {
        critical: slice(full.critical_level_series),
        seqState: slice(full.seq_state_series),
        bullishBreak: slice(full.bullish_break),
        bearishBreak: slice(full.bearish_break),
        lastPeak: full.last_peak,
        lastTrough: full.last_trough,
        tp: Number.isFinite(full.TP) ? full.TP : null,
        sl: Number.isFinite(full.SL) ? full.SL : null,
        rr: Number.isFinite(full.RR) ? full.RR : null,
        peaks: full.peaks.filter((p) => p.idx >= start).map((p) => ({ ...p, idx: remap(p.idx) })),
        troughs: full.troughs.filter((t) => t.idx >= start).map((t) => ({ ...t, idx: remap(t.idx) })),
        extensionLines: full.extension_lines.map((ln) => ({
          kind: ln.kind,
          x0Idx: remap(ln.x0_idx),
          y0: ln.y0,
          x1Idx: remap(ln.x1_idx),
          y1: ln.y1,
          rawX0Idx: ln.x0_idx - start,
          rawX1Idx: ln.x1_idx - start,
        })),
        fib: full.fib
          ? {
              high: full.fib.high,
              highIdx: remap(full.fib.high_idx),
              low: full.fib.low,
              lowIdx: remap(full.fib.low_idx),
              fib382: full.fib.fib_382,
              fib500: full.fib.fib_500,
              fib618: full.fib.fib_618,
            }
          : null,
        overlays: {
          emaFast: slice(full.overlays.ema_fast),
          emaSlow: slice(full.overlays.ema_slow),
          smaMajor: slice(full.overlays.sma_major),
          envUpper: slice(full.overlays.env_upper),
          envLower: slice(full.overlays.env_lower),
          bbBasis: slice(full.overlays.bb_basis),
          bbUpper: slice(full.overlays.bb_upper),
          bbLower: slice(full.overlays.bb_lower),
        },
        impulseColors: slice(full.impulse_colors),
        atrPct: full.ATR_pct,
        adx: full.ADX,
        signalBarIndex:
          full.signal_bar_index != null && full.signal_bar_index >= start
            ? remap(full.signal_bar_index)
            : null,
        seqStateFinal: full.seq_state_final,
        criticalLevel: full.critical_level,
      }
    : null;

  return {
    payload: {
      yahooTicker: input.yahooTicker,
      symbol,
      tvSymbol: input.tvSymbol || input.yahooTicker,
      companyName: input.companyName || input.yahooTicker,
      tf,
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
    },
    age,
    lastBarDate: bars[bars.length - 1].date,
  };
}

export class ChartError extends Error {}

export function formatDailyChgStr(
  close: number | null | undefined,
  prev: number | null | undefined,
): string {
  if (close == null || prev == null || !Number.isFinite(close) || !Number.isFinite(prev) || prev === 0) {
    return '';
  }
  const chg = ((close - prev) / prev) * 100;
  if (!Number.isFinite(chg)) return '';
  const sign = chg >= 0 ? '+' : '';
  return `${sign}${chg.toFixed(2)}%`;
}
