/** Position size and P&L — same rounding as the server tracker. */
import type { UserTf } from './types';

export function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

export function finiteOrNull(n: number): number | null {
  return Number.isFinite(n) ? round2(n) : null;
}

export function sharesFromRisk(
  entry: number,
  sl: number | null | undefined,
  riskUsd: number,
): number {
  if (sl == null || !Number.isFinite(sl) || !Number.isFinite(entry)) return 0;
  const risk = entry - sl;
  if (risk <= 0 || riskUsd <= 0) return 0;
  return Math.max(0, Math.round(riskUsd / risk));
}

export type Pnl = { usd: number | null; r: number | null; pct: number | null };

export function computePnl(
  entry: number,
  sl: number | null | undefined,
  shares: number,
  price: number | null | undefined,
): Pnl {
  if (price == null || !Number.isFinite(price) || !Number.isFinite(entry)) {
    return { usd: null, r: null, pct: null };
  }
  const usd = round2((price - entry) * (shares || 0));
  const risk = sl != null && Number.isFinite(sl) ? entry - sl : 0;
  const invested = entry * (shares || 0);
  return {
    usd,
    r: risk > 0 ? round2((price - entry) / risk) : null,
    pct: invested > 0 ? round2((usd / invested) * 100) : null,
  };
}

export function holdPeriods(tf: UserTf, from?: string | null, to?: string | null): number | null {
  if (!from || !to) return null;
  const ms = Date.parse(`${to}T12:00:00Z`) - Date.parse(`${from}T12:00:00Z`);
  if (!Number.isFinite(ms) || ms < 0) return null;
  const days = ms / 86_400_000;
  if (tf === 'Weekly') return round2(days / 7);
  return round2(days / 30.4375);
}

export function holdUnitLabel(tf: UserTf | 'All'): string {
  if (tf === 'Weekly') return 'weeks';
  if (tf === 'Monthly') return 'months';
  return 'periods';
}

export function interestRank(interest: 'interested' | 'not_interested' | null): number {
  if (interest === 'interested') return 2;
  if (interest === 'not_interested') return 0;
  return 1;
}
