import { useEffect, useMemo } from 'react';
import { Navigate, useParams, useSearchParams } from 'react-router-dom';
import { keepPreviousData, useInfiniteQuery, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  BUCKETS,
  TIMEFRAMES,
  UNIVERSES,
  api,
  type Bucket,
  type ResultSort,
  type SortDir,
  type Timeframe,
  type Universe,
} from '../lib/api';
import { formatAge, TF_SHORT } from '../lib/format';
import { SegmentedTabs } from '../components/SegmentedTabs';
import { SignalCard } from '../components/SignalCard';
import { SortChips, type SortOption } from '../components/SortChips';

const PAGE_SIZE = 100;

const BUCKET_LABEL: Record<Bucket, string> = {
  new: 'New',
  valid: 'Valid',
  closed: 'Closed',
};

const SORTS: SortOption<ResultSort>[] = [
  { value: 'rr', label: 'RR' },
  { value: 'pnl', label: 'P&L' },
  { value: 'interest', label: 'Marked' },
  { value: 'symbol', label: 'A-Z', from: 'asc' },
];

const BUCKET_HINT: Record<Bucket, string> = {
  new: 'Signals that appeared in the current period.',
  valid: 'Signals from earlier periods that are still valid, marked to market.',
  closed: 'Signals closed in the current period.',
};

function isUniverse(value: string | undefined): value is Universe {
  return UNIVERSES.includes(value as Universe);
}

function isTimeframe(value: string | undefined): value is Timeframe {
  return TIMEFRAMES.includes(value as Timeframe);
}

function isBucket(value: string | undefined): value is Bucket {
  return BUCKETS.includes(value as Bucket);
}

export function ResultsPage() {
  const params = useParams();
  const [search, setSearch] = useSearchParams();
  const queryClient = useQueryClient();

  const universe = isUniverse(params.universe) ? params.universe : null;
  const tf = isTimeframe(params.tf) ? params.tf : 'Daily';
  const bucket = isBucket(params.bucket) ? params.bucket : 'new';
  const sort = (search.get('sort') as ResultSort | null) ?? (bucket === 'closed' ? 'pnl' : 'rr');
  const dir = (search.get('dir') as SortDir | null) ?? 'desc';

  const summary = useQuery({
    queryKey: ['results-summary'],
    queryFn: api.resultsSummary,
    staleTime: 60_000,
    refetchInterval: 5 * 60_000,
  });

  const page = useInfiniteQuery({
    queryKey: ['results', universe, tf, bucket, sort, dir],
    enabled: Boolean(universe),
    initialPageParam: 0,
    queryFn: ({ pageParam }) =>
      api.results({
        universe: universe as Universe,
        tf,
        bucket,
        sort,
        dir,
        limit: PAGE_SIZE,
        offset: pageParam,
      }),
    getNextPageParam: (last, all) => {
      const loaded = all.reduce((sum, p) => sum + p.rows.length, 0);
      return loaded < last.total ? loaded : undefined;
    },
    placeholderData: keepPreviousData,
    staleTime: 60_000,
    refetchInterval: 5 * 60_000,
  });

  const rows = useMemo(() => page.data?.pages.flatMap((p) => p.rows) ?? [], [page.data]);
  const first = page.data?.pages[0];
  const counts = universe ? summary.data?.[universe]?.[tf]?.counts : undefined;
  const scan = first?.scan ?? (universe ? summary.data?.[universe]?.[tf]?.scan : undefined);

  // Switching timeframe or bucket is the most common gesture here, so warm the neighbours.
  useEffect(() => {
    if (!universe || page.isLoading) return;
    const neighbours: Array<{ tf: Timeframe; bucket: Bucket }> = [
      ...TIMEFRAMES.filter((t) => t !== tf).map((t) => ({ tf: t, bucket })),
      ...BUCKETS.filter((b) => b !== bucket).map((b) => ({ tf, bucket: b })),
    ];
    for (const next of neighbours) {
      void queryClient.prefetchInfiniteQuery({
        queryKey: ['results', universe, next.tf, next.bucket, sort, dir],
        initialPageParam: 0,
        queryFn: () =>
          api.results({
            universe,
            tf: next.tf,
            bucket: next.bucket,
            sort,
            dir,
            limit: PAGE_SIZE,
            offset: 0,
          }),
        staleTime: 60_000,
      });
    }
  }, [universe, tf, bucket, sort, dir, page.isLoading, queryClient]);

  if (!universe) return <Navigate to="/results/Stocks/Daily/new" replace />;

  const scanAge = formatAge(scan?.finishedAt);

  return (
    <div>
      <section className="results-head">
        <SegmentedTabs
          label="Universe"
          segments={[
            ...UNIVERSES.map((u) => ({
              value: u,
              to: `/results/${u}/${tf}/${bucket}`,
              label: u,
            })),
            { value: 'manual' as const, to: '/results/manual', label: 'Manual' },
          ]}
        />

        <SegmentedTabs
          label="Timeframe"
          size="sm"
          segments={TIMEFRAMES.map((t) => ({
            value: t,
            to: `/results/${universe}/${t}/${bucket}`,
            label: TF_SHORT[t],
          }))}
        />

        <SegmentedTabs
          label="Bucket"
          size="sm"
          segments={BUCKETS.map((b) => ({
            value: b,
            to: `/results/${universe}/${tf}/${b}`,
            label: BUCKET_LABEL[b],
            badge: counts?.[b],
          }))}
        />

        <div className="results-meta">
          <span className="muted small">
            {scan?.running ? 'Scanning now' : scanAge ? `Scanned ${scanAge}` : 'No scan yet'}
            {scan?.asOf ? ` · bar ${scan.asOf}` : ''}
          </span>
          <SortChips
            options={SORTS}
            value={sort}
            dir={dir}
            onChange={(next, nextDir) => setSearch({ sort: next, dir: nextDir }, { replace: true })}
          />
        </div>
      </section>

      {page.isLoading ? <p className="empty">Loading…</p> : null}
      {page.error ? <p className="error">{(page.error as Error).message}</p> : null}

      {!page.isLoading && rows.length === 0 ? (
        <p className="empty">
          Nothing here. {BUCKET_HINT[bucket]}
          {scan?.running ? ' A scan is running right now.' : ''}
        </p>
      ) : null}

      {rows.map((row) => (
        <SignalCard key={row.id} row={row} bucket={bucket} />
      ))}

      {page.hasNextPage ? (
        <button
          type="button"
          className="btn btn-accent"
          disabled={page.isFetchingNextPage}
          onClick={() => void page.fetchNextPage()}
        >
          {page.isFetchingNextPage
            ? 'Loading…'
            : `Load more (${rows.length} of ${first?.total ?? 0})`}
        </button>
      ) : null}
    </div>
  );
}
