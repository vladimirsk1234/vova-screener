/**
 * Mongo filters for Settings UV/OV on the per-trade entry snapshot.
 * As-of premium maths live in `@vova/engine` (`resolvePremiumPctAtEntry`).
 */
import type { FundamentalsFilter } from '../settings/settings.module';

/** Same UV/OV rule the History / CLOSED Mongo filter applies. */
export function matchesEntryPremium(
  premiumPctAtEntry: number | null | undefined,
  filter: FundamentalsFilter,
): boolean {
  if (filter === 'all') return true;
  if (premiumPctAtEntry == null || !Number.isFinite(premiumPctAtEntry)) return false;
  return filter === 'undervalued' ? premiumPctAtEntry < 0 : premiumPctAtEntry > 0;
}

function premiumClause(filter: Exclude<FundamentalsFilter, 'all'>): Record<string, unknown> {
  return { premiumPctAtEntry: filter === 'undervalued' ? { $lt: 0 } : { $gt: 0 } };
}

/**
 * History + Results CLOSED: filter on the per-trade snapshot.
 * `$and` so an existing `$or` on the base match is not overwritten.
 */
export function withEntryPremiumFilter(
  match: Record<string, unknown>,
  filter: FundamentalsFilter,
): Record<string, unknown> {
  if (filter === 'all') return match;
  return { $and: [match, premiumClause(filter)] };
}

/**
 * Results NEW/VALID: stamped trades use `premiumPctAtEntry`; unstamped rows
 * (`$exists: false` only — explicit null is "unknown") fall back to today's
 * live ticker set until a stamp lands.
 */
export function withLiveOrEntryPremiumFilter(
  match: Record<string, unknown>,
  filter: FundamentalsFilter,
  liveTickers: string[] | null,
): Record<string, unknown> {
  if (filter === 'all') return match;
  const stamped = premiumClause(filter);
  if (!liveTickers) return { $and: [match, stamped] };
  return {
    $and: [
      match,
      {
        $or: [
          stamped,
          {
            $and: [{ premiumPctAtEntry: { $exists: false } }, { yahooTicker: { $in: liveTickers } }],
          },
        ],
      },
    ],
  };
}
