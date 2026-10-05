/** Results buckets and order. Mirrors `ResultsService` without the FMP valuation filter or UV sort. */
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
  const rows = q.signals.filter((row) => inBucket(row, q.universe, q.tf, q.bucket, q.minRr, q.scan));
  rows.sort(resultsComparator(q.bucket, q.sort, q.dir));
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
  const counts = { new: 0, valid: 0, closed: 0 };
  for (const row of signals) {
    if (inBucket(row, universe, tf, 'new', minRr, scan)) counts.new += 1;
    else if (inBucket(row, universe, tf, 'valid', minRr, scan)) counts.valid += 1;
    else if (inBucket(row, universe, tf, 'closed', minRr, scan)) counts.closed += 1;
  }
  return counts;
}

export function inBucket(
  row: TrackedSignal,
  universe: Universe,
  tf: UserTf,
  bucket: Bucket,
  minRr: number,
  scan: ScanMeta,
): boolean {
  if (row.universe !== universe || row.tf !== tf) return false;
  if (bucket === 'closed') {
    const period = scan.newestAsOf ? barPeriodKey(tf, scan.newestAsOf) : scan.periodKey;
    if (row.closedPeriodKey !== period) return false;
    if (!(row.status === 'closed' || row.provisionalClose)) return false;
    if (minRr > 0 && !(row.rrAtEntry != null && row.rrAtEntry >= minRr)) return false;
    return true;
  }
  if (row.status !== 'active' || row.provisionalClose || row.signalValid === false) return false;
  if (minRr > 0 && !(row.lastRr != null && row.lastRr >= minRr)) return false;
  const isNew = row.barsSinceValid === 0 && row.lastSeenPeriodKey === scan.periodKey;
  return bucket === 'new' ? isNew : !isNew;
}

/** Mongo order: a missing value sorts below every number, then ties fall back to symbol A-Z. */
export function mongoCompare(a: number | string | null | undefined, b: number | string | null | undefined): number {
  const an = a == null || (typeof a === 'number' && !Number.isFinite(a));
  const bn = b == null || (typeof b === 'number' && !Number.isFinite(b));
  if (an && bn) return 0;
  if (an) return -1;
  if (bn) return 1;
  if (typeof a === 'string' && typeof b === 'string') return a < b ? -1 : a > b ? 1 : 0;
  return (a as number) - (b as number);
}

function resultsComparator(bucket: Bucket, sort: ResultSort, dir: SortDir) {
  const order = dir === 'asc' ? 1 : -1;
  const key = (row: TrackedSignal): number | string | null => {
    if (sort === 'rr') return bucket === 'closed' ? row.rrAtEntry : row.lastRr;
    if (sort === 'pnl') return bucket === 'closed' ? row.pnlUsd : row.unrealizedUsd;
    if (sort === 'interest') return row.interestRank;
    return row.symbol;
  };
  return (a: TrackedSignal, b: TrackedSignal) => {
    const primary = mongoCompare(key(a), key(b)) * order;
    if (primary || sort === 'symbol') return primary;
    return mongoCompare(a.symbol, b.symbol);
  };
}
