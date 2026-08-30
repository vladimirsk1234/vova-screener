/** User-facing timeframes and query parsing. Daily is not a product timeframe. */
import type { Timeframe } from '@vova/engine';

/** Scan / Results / History strategy timeframes. Engine may still read Daily bars internally. */
export const TIMEFRAMES = ['Weekly', 'Monthly'] as const satisfies readonly Timeframe[];

export type UserTimeframe = (typeof TIMEFRAMES)[number];
export type HistoryTf = UserTimeframe | 'All';
/**
 * Exit-date grouping axis — not a scan timeframe. `Day` buckets by calendar exit date.
 * `Daily` is accepted as a legacy alias of `Day`.
 */
export const HISTORY_GROUP_BYS = ['Day', 'Weekly', 'Monthly'] as const;
export type HistoryGroupBy = (typeof HISTORY_GROUP_BYS)[number];

export function isUserTimeframe(value: unknown): value is UserTimeframe {
  return TIMEFRAMES.includes(value as UserTimeframe);
}

/** Unknown values (including Daily) become Weekly — never Daily. */
export function parseTf(value?: string): UserTimeframe {
  return isUserTimeframe(value) ? value : 'Weekly';
}

/** Missing / All / unknown (including Daily) become All. */
export function parseHistoryTf(value?: string): HistoryTf {
  if (value === 'All' || value == null || value === '') return 'All';
  return isUserTimeframe(value) ? value : 'All';
}

/** Calendar-day grouping. `Daily` is the old chip name, not the Daily timeframe. */
export function parseHistoryGroupBy(value?: string): HistoryGroupBy {
  if (value === 'Weekly' || value === 'Monthly') return value;
  if (value === 'Day' || value === 'Daily') return 'Day';
  return 'Day';
}

/**
 * Restrict a Mongo match so History/Results All is Weekly+Monthly only.
 * An explicit engine Timeframe (including Daily in internal tests) still filters to that tf.
 */
export function withUserTf(
  match: Record<string, unknown>,
  tf: Timeframe | HistoryTf,
): Record<string, unknown> {
  if (tf === 'All') return { ...match, tf: { $in: [...TIMEFRAMES] } };
  return { ...match, tf };
}
