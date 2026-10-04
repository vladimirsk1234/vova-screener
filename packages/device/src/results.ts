/** Results buckets. Same split as the server, without the FMP valuation filter or UV sort. */
import { barPeriodKey } from '../../../apps/api/src/scans/period.ts';
import type { Bucket, ResultSort, ScanMeta, SortDir, TrackedSignal, Universe, UserTf } from './types';

export type ResultsQuery = {
  signals: TrackedSignal[];
  universe: Universe;
  tf: UserTf;
  bucket: Bucket;
  sort: ResultSort;
  dir: SortDir;
  minRr: number;
  scan: ScanMeta;
  limit?: number;
  offset?: number;
};

export function listResults(q: ResultsQuery): { total: number; rows: TrackedSignal[] } {
  const rows = q.signals.filter((row) => inBucket(row, q));
  rows.sort((a, b) => compare(a, b, q.bucket, q.sort, q.dir));
  const offset = Math.max(q.offset ?? 0, 0);
  const limit = Math.min(Math.max(q.limit ?? 100, 1), 500);
  return { total: rows.length, rows: rows.slice(offset, offset + limit) };
}

export function bucketCounts(
  signals: TrackedSignal[],
  universe: Universe,
  tf: UserTf,
  minRr: number,
  scan: ScanMeta,
): { new: number; valid: number; closed: number } {
  const count = (bucket: Bucket) =>
    signals.filter((row) => inBucket(row, { signals, universe, tf, bucket, sort: 'rr', dir: 'desc', minRr, scan })).length;
  return { new: count('new'), valid: count('valid'), closed: count('closed') };
}

function inBucket(row: TrackedSignal, q: ResultsQuery): boolean {
  if (row.universe !== q.universe || row.tf !== q.tf) return false;
  if (q.bucket === 'closed') {
    const period = q.scan.newestAsOf ? barPeriodKey(q.tf, q.scan.newestAsOf) : q.scan.periodKey;
    if (row.closedPeriodKey !== period) return false;
    if (!(row.status === 'closed' || row.provisionalClose)) return false;
    if (q.minRr > 0 && (row.rrAtEntry == null || row.rrAtEntry < q.minRr)) return false;
    return true;
  }
  if (row.status !== 'active' || row.provisionalClose || row.signalValid === false) return false;
  if (q.minRr > 0 && (row.lastRr == null || row.lastRr < q.minRr)) return false;
  const isNew = row.barsSinceValid === 0 && row.lastSeenPeriodKey === q.scan.periodKey;
  return q.bucket === 'new' ? isNew : !isNew;
}

function compare(a: TrackedSignal, b: TrackedSignal, bucket: Bucket, sort: ResultSort, dir: SortDir): number {
  const sign = dir === 'asc' ? 1 : -1;
  const num = (n: number | null) => (n == null || !Number.isFinite(n) ? null : n);
  if (sort === 'symbol') return a.symbol.localeCompare(b.symbol) * sign;
  if (sort === 'interest') {
    const d = (a.interestRank - b.interestRank) * sign;
    return d || a.symbol.localeCompare(b.symbol);
  }
  if (sort === 'pnl') {
    const av = num(bucket === 'new' ? null : a.pnlUsd ?? a.unrealizedUsd);
    const bv = num(bucket === 'new' ? null : b.pnlUsd ?? b.unrealizedUsd);
    return cmpNullsLast(av, bv, sign) || a.symbol.localeCompare(b.symbol);
  }
  const av = num(bucket === 'closed' ? a.rrAtEntry : a.lastRr);
  const bv = num(bucket === 'closed' ? b.rrAtEntry : b.lastRr);
  return cmpNullsLast(av, bv, sign) || a.symbol.localeCompare(b.symbol);
}

/** Descending sorts push missing numbers to the end, matching the server's Mongo order. */
function cmpNullsLast(a: number | null, b: number | null, sign: number): number {
  if (a == null && b == null) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  return (a - b) * sign;
}
