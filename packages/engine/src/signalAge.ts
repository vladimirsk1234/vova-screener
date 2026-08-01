/**
 * How old a buy signal is, in bars of the series it is measured on.
 *
 * This is the only number that splits the NEW and VALID lists, so it must mean the same thing on
 * every screen: `0` is a signal that appeared on the last bar, `1` and up is a signal that has been
 * running for that many bars, `null` is no signal at all.
 *
 * RR is deliberately switched off. RR decides which signals a scan is willing to report and how the
 * lists are sorted, never how old a signal is — with a minimum RR in place the valid flag flips on
 * and off as the ratio drifts across the threshold mid-trade, and the age would then count bars
 * since the last flip instead of bars since the signal appeared. With RR off the valid flag is
 * purely structural (sequence up plus intact HH/HL structure), which is what "new signal" means.
 */
import { runSequenceVovaPine } from './sequenceVova';
import type { OhlcSeries } from './types';

export const ATR_LEN_AGE = 14;

export type SignalAge = {
  /** Bars of the series since the signal appeared: 0 on the bar it appeared on. */
  barsSinceValid: number | null;
  /** Date of the bar the signal appeared on. */
  validSinceAsOf: string | null;
};

export function signalAge(bars: OhlcSeries): SignalAge {
  const out = runSequenceVovaPine(bars, {
    atr_len: ATR_LEN_AGE,
    min_rr: 0,
    no_rr_req: true,
    use_last_hl_sl: true,
    direction: 'buy',
  });
  const idx = out?.valid_since_index ?? null;
  return {
    barsSinceValid: out?.bars_since_valid ?? null,
    validSinceAsOf: idx != null ? (bars[idx]?.date ?? null) : null,
  };
}
