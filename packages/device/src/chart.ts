/** Technical chart payload. Company name and daily price change stay; FMP fields do not. */
import {
  buildDwmLines,
  buildTradeLine,
  buildWatermarkParts,
  defaultIndicatorParams,
  indicatorParamsFromDict,
  maxBarsForTf,
  runSequenceVovaFull,
  runSequenceVovaPine,
  signalAge,
  type OhlcBar,
  type OhlcSeries,
} from '../../engine/src/index.ts';
import type { UserTf } from './types';

export type ChartPayload = {
  yahooTicker: string;
  symbol: string;
  companyName: string;
  tf: UserTf;
  asOf: string | null;
  bars: OhlcBar[];
  overlay: {
    critical: (number | null)[];
    seqState: number[];
    bullishBreak: boolean[];
    bearishBreak: boolean[];
    peaks: Array<{ idx: number; price: number; label: string }>;
    troughs: Array<{ idx: number; price: number; label: string }>;
    extensionLines: Array<{ kind: 'high' | 'low'; x0: number; y0: number; x1: number; y1: number }>;
    fib: { high: number; low: number; fib382: number; fib500: number; fib618: number } | null;
    emaFast: (number | null)[];
    emaSlow: (number | null)[];
    bbUpper: (number | null)[];
    bbLower: (number | null)[];
    bbBasis: (number | null)[];
  } | null;
  pine: {
    valid: boolean;
    isNew: boolean;
    strong: boolean;
    barsSinceValid: number | null;
    tp: number | null;
    sl: number | null;
    rr: number | null;
    close: number;
  } | null;
  watermarkLines: string[];
};

export function buildChart(opts: {
  yahooTicker: string;
  symbol: string;
  companyName: string;
  tf: UserTf;
  bars: OhlcSeries;
  asOf?: string | null;
  dailyClose?: number | null;
  prevDailyClose?: number | null;
  weeklyBars?: OhlcSeries | null;
  monthlyBars?: OhlcSeries | null;
}): ChartPayload {
  const asOf = opts.asOf ?? null;
  const series = asOf ? opts.bars.filter((bar) => bar.date <= asOf) : opts.bars;
  const params = indicatorParamsFromDict({
    ...defaultIndicatorParams(),
    min_rr: 0,
    no_rr_req: true,
    use_last_hl_sl: true,
  });
  const full = series.length ? runSequenceVovaFull(series, { params }) : null;
  const pine = series.length
    ? runSequenceVovaPine(series, {
        atr_len: params.atr_len,
        min_rr: 0,
        use_last_hl_sl: true,
        no_rr_req: true,
        direction: 'buy',
      })
    : null;
  const age = series.length ? signalAge(series) : { barsSinceValid: null, validSinceAsOf: null };
  const keep = maxBarsForTf(opts.tf);
  const start = Math.max(0, series.length - keep);
  const bars = series.slice(start);
  const upTo = (other: OhlcSeries | null | undefined) =>
    !other ? null : asOf ? other.filter((bar) => bar.date <= asOf) : other;
  const dwm = full
    ? buildDwmLines({
        chartBars: series,
        weeklyBars: opts.tf === 'Weekly' ? series : upTo(opts.weeklyBars),
        monthlyBars: opts.tf === 'Monthly' ? series : upTo(opts.monthlyBars),
        chartTf: opts.tf,
        params,
      })
    : {};
  const tradeLine = full ? buildTradeLine(full, params, series.length - 1) : '';
  const watermark = full
    ? buildWatermarkParts({
        fundamentals: {
          company_name: opts.companyName,
          daily_chg_str: formatDailyChgStr(opts.dailyClose, opts.prevDailyClose),
        },
        full,
        params,
        dwmLines: dwm,
        chartTf: opts.tf,
        ticker: opts.symbol,
        tradeLine,
        omitFundamentals: true,
      })
    : null;

  const slice = <T>(arr: T[] | undefined) => (arr ? arr.slice(start) : []);
  return {
    yahooTicker: opts.yahooTicker,
    symbol: opts.symbol,
    companyName: opts.companyName,
    tf: opts.tf,
    asOf,
    bars,
    overlay: full
      ? {
          critical: slice(full.critical_level_series).map(numOrNull),
          seqState: slice(full.seq_state_series),
          bullishBreak: slice(full.bullish_break),
          bearishBreak: slice(full.bearish_break),
          peaks: full.peaks.filter((p) => p.idx >= start).map((p) => ({ ...p, idx: p.idx - start })),
          troughs: full.troughs.filter((p) => p.idx >= start).map((p) => ({ ...p, idx: p.idx - start })),
          extensionLines: full.extension_lines.map((ln) => ({
            kind: ln.kind,
            x0: ln.x0_idx - start,
            y0: ln.y0,
            x1: ln.x1_idx - start,
            y1: ln.y1,
          })),
          fib: full.fib
            ? {
                high: full.fib.high,
                low: full.fib.low,
                fib382: full.fib.fib_382,
                fib500: full.fib.fib_500,
                fib618: full.fib.fib_618,
              }
            : null,
          emaFast: slice(full.overlays.ema_fast).map(numOrNull),
          emaSlow: slice(full.overlays.ema_slow).map(numOrNull),
          bbUpper: slice(full.overlays.bb_upper).map(numOrNull),
          bbLower: slice(full.overlays.bb_lower).map(numOrNull),
          bbBasis: slice(full.overlays.bb_basis).map(numOrNull),
        }
      : null,
    pine: pine
      ? {
          valid: pine.Valid,
          isNew: pine.New,
          strong: pine.Strong,
          barsSinceValid: age.barsSinceValid,
          tp: Number.isFinite(pine.TP) ? pine.TP : null,
          sl: Number.isFinite(pine.SL) ? pine.SL : null,
          rr: Number.isFinite(pine.RR) ? pine.RR : null,
          close: pine.Close,
        }
      : null,
    watermarkLines: watermark?.lines ?? [opts.companyName],
  };
}

function numOrNull(v: number | null): number | null {
  return v != null && Number.isFinite(v) ? v : null;
}

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
