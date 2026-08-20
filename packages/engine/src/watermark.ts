/** Watermark helpers — port of watermark_status.py (trade line + D/W/M). */
import type { IndicatorParams } from './indicatorParams';
import { defaultIndicatorParams } from './indicatorParams';
import {
  runSequenceVovaFull,
  structureSnapshotFromFull,
  type SequenceVovaFullResult,
  type StructureSnapshot,
} from './sequenceVovaFull';
import type { OhlcSeries, Timeframe } from './types';
import { emojiState, seqDisplay, seqStructStatus, structDisplay, type SeqStructStatus } from './seqStruct';

export {
  emojiState,
  seqDisplay,
  seqStructStatus,
  structDisplay,
  type SeqStructStatus,
} from './seqStruct';

export function seqStructFromBars(
  bars: OhlcSeries,
  tf: Timeframe,
  params?: IndicatorParams,
): SeqStructStatus | null {
  const snap = snapshotForBars(bars, params);
  if (!snap) return null;
  return seqStructStatus(snap, tf === 'Daily' ? snap.sma_above ?? null : null);
}

export function snapshotForBars(
  bars: OhlcSeries,
  params: IndicatorParams = defaultIndicatorParams(),
): StructureSnapshot | null {
  if (bars.length < 2) return null;
  const full = runSequenceVovaFull(bars, { params });
  if (!full) return null;
  const snap = structureSnapshotFromFull(full);
  const sma = full.overlays.sma_major[full.overlays.sma_major.length - 1];
  snap.sma_above = sma != null ? full.Close > sma : null;
  snap.sma_major = sma ?? undefined;
  return snap;
}

export type DwmLines = Partial<Record<'daily' | 'weekly' | 'monthly', string>>;

export function buildDwmLines(opts: {
  chartBars: OhlcSeries;
  dailyBars?: OhlcSeries | null;
  weeklyBars?: OhlcSeries | null;
  monthlyBars?: OhlcSeries | null;
  chartTf: Timeframe;
  params?: IndicatorParams;
}): DwmLines {
  const params = opts.params ?? defaultIndicatorParams();
  const lengthMajor = params.length_major;
  const lines: DwmLines = {};

  const dailySnap =
    opts.chartTf === 'Daily'
      ? snapshotForBars(opts.chartBars, params)
      : opts.dailyBars
        ? snapshotForBars(opts.dailyBars, params)
        : null;
  const weeklySnap =
    opts.chartTf === 'Weekly'
      ? snapshotForBars(opts.chartBars, params)
      : opts.weeklyBars
        ? snapshotForBars(opts.weeklyBars, params)
        : null;
  const monthlySnap =
    opts.chartTf === 'Monthly'
      ? snapshotForBars(opts.chartBars, params)
      : opts.monthlyBars
        ? snapshotForBars(opts.monthlyBars, params)
        : null;

  if (dailySnap) {
    const dSeq = seqDisplay(dailySnap, dailySnap.sma_above ?? null);
    const [dStructE, dStructL] = structDisplay(dailySnap, dailySnap.sma_above ?? null);
    const maE = dailySnap.sma_above ? '🟢' : '🔴';
    lines.daily = `D: Seq ${emojiState(dSeq)}   Struct ${dStructE}${dStructL}   SMA ${lengthMajor} ${maE}`;
  }
  if (weeklySnap) {
    const wSeq = seqDisplay(weeklySnap);
    const [wStructE, wStructL] = structDisplay(weeklySnap);
    lines.weekly = `W: Seq ${emojiState(wSeq)}   Struct ${wStructE}${wStructL}`;
  }
  if (monthlySnap) {
    const mSeq = seqDisplay(monthlySnap);
    const [mStructE, mStructL] = structDisplay(monthlySnap);
    lines.monthly = `M: Seq ${emojiState(mSeq)}   Struct ${mStructE}${mStructL}`;
  }
  return lines;
}

export function atrEmoji(val: number, lowT: number, highT: number): string {
  if (val > highT) return '🔴';
  if (val >= lowT) return '🟡';
  return '🟢';
}

export function buildTradeLine(
  full: SequenceVovaFullResult,
  params: IndicatorParams,
  barIndexLast: number,
): string {
  const seqState = full.seq_state_final;
  const lastTroughHl = full.last_trough_was_hl;
  const lastPeakHh = full.last_peak_was_hh;
  const lastPeak = full.last_peak;
  const structInvalid = full.struct_invalid_seq_down;
  const close = full.Close;
  const structOk =
    (lastTroughHl || (lastPeak != null && close > lastPeak && lastTroughHl)) &&
    lastPeakHh &&
    !structInvalid;
  const seqOk = seqState === 1;

  const crit = full.critical_level;
  const atr = full.ATR || 0;
  const lastTrough = full.last_trough;
  let sl = close - atr;
  if (crit != null && crit < close) sl = Math.min(sl, crit);
  if (params.use_last_hl_sl && lastTroughHl && lastTrough != null && lastTrough < close) {
    sl = Math.min(sl, lastTrough);
  }
  const risk = close - sl;
  const reward = lastPeak != null ? lastPeak - close : 0;
  const rr = risk > 0 ? reward / risk : Number.NaN;
  const valid = params.no_rr_req
    ? seqOk && structOk
    : seqOk && structOk && rr >= params.min_rr && risk > 0 && reward > 0;

  const bearishBreakLast = full.bearish_break[full.bearish_break.length - 1] ?? false;
  const sigIdx = full.signal_bar_index;
  const rrLabel = (val: number) => (Number.isFinite(val) ? val.toFixed(2) : 'N/A');

  if (valid && bearishBreakLast) return `🆕 NEW | R/R: ${rrLabel(rr)}`;
  if (valid) {
    const barsSince = sigIdx != null ? barIndexLast - sigIdx : 0;
    return `✅ VALID | R/R: ${rrLabel(rr)} | Bars ${barsSince}`;
  }
  let debug = '';
  if (!seqOk) debug += 'Seq❌ ';
  if (!structOk) debug += 'Struct❌ ';
  if (!params.no_rr_req && !debug && Number.isFinite(rr) && rr < params.min_rr) {
    return `❌ R/R too low: ${rr.toFixed(2)} (need ${params.min_rr.toFixed(2)})`;
  }
  if (!debug) return 'NO SETUP: Risk/Reward Invalid';
  return `NO SETUP: ${debug.trim()}`;
}

export type ChartFundamentals = {
  company_name?: string;
  daily_chg_str?: string;
  mcap_str?: string;
  pe_str?: string;
  earn_str?: string;
  description?: string | null;
};

export function buildWatermarkParts(opts: {
  fundamentals: ChartFundamentals;
  full: SequenceVovaFullResult;
  params: IndicatorParams;
  dwmLines: DwmLines;
  chartTf: Timeframe;
  ticker: string;
  tradeLine: string;
}): { main: string; lines: string[]; description: string | null } {
  const { fundamentals, full, params, dwmLines, chartTf, ticker, tradeLine } = opts;
  const name = String(fundamentals.company_name || ticker);
  const rawDesc = fundamentals.description;
  const description =
    rawDesc && String(rawDesc).trim().toLowerCase() !== name.toLowerCase()
      ? String(rawDesc).trim()
      : null;

  const rows: string[] = [];
  const dChg = fundamentals.daily_chg_str ?? '';
  const mcap = fundamentals.mcap_str ?? 'N/A';
  rows.push(`${ticker} (${chartTf}) | ${dChg} | ${mcap}`);
  rows.push(`PE: ${fundamentals.pe_str ?? 'N/A'} | Earn: ${fundamentals.earn_str ?? 'N/A'}`);

  const atrE = atrEmoji(full.ATR_pct, params.atr_low_thresh, params.atr_high_thresh);
  const adxSuffix = full.Valid ? `   ADX: ${full.ADX.toFixed(2)}` : '';
  rows.push(`ATR: ${full.ATR.toFixed(2)} (${full.ATR_pct.toFixed(2)}%) ${atrE}${adxSuffix}`);

  if (dwmLines.daily) rows.push(dwmLines.daily);
  if (dwmLines.weekly) rows.push(dwmLines.weekly);
  if (dwmLines.monthly) rows.push(dwmLines.monthly);
  rows.push(tradeLine);

  return { main: `${name}\n${rows.join('\n')}`, lines: [name, ...rows], description };
}
