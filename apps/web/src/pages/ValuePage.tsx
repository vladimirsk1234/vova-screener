import { useSearchParams } from 'react-router-dom';
import { keepPreviousData, useInfiniteQuery } from '@tanstack/react-query';
import {
  api,
  type SortDir,
  type ValueScreenerSort,
} from '../lib/api';
import { Chips } from '../components/Chips';
import { SortChips, type SortOption } from '../components/SortChips';
import { UniverseTabs } from '../components/UniverseTabs';
import { ValueCard } from '../components/ValueCard';
import { useRestoreChartScroll } from '../lib/useRestoreChartScroll';

const PAGE_SIZE = 100;

const STAR_FILTERS = ['undervalued', '4', '3', '2', '1', '0'] as const;
type StarChip = (typeof STAR_FILTERS)[number];

const STAR_LABEL: Record<StarChip, string> = {
  undervalued: 'Undervalued',
  '4': '4/4',
  '3': '3/4',
  '2': '2/4',
  '1': '1/4',
  '0': '0/4',
};

const SORTS: SortOption<ValueScreenerSort>[] = [
  { value: 'stars', label: 'Stars' },
  { value: 'eps', label: 'EPS %', from: 'asc' },
  { value: 'fcf', label: 'FCF %', from: 'asc' },
  { value: 'dcf', label: 'DCF %', from: 'asc' },
  { value: 'interest', label: 'Marked', from: 'desc' },
  { value: 'symbol', label: 'A-Z', from: 'asc' },
];

function isStarFilter(value: string | null): value is StarChip {
  return STAR_FILTERS.includes(value as StarChip);
}

function isSort(value: string | null): value is ValueScreenerSort {
  return SORTS.some((s) => s.value === value);
}

function formatEodEt(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  const parts = Object.fromEntries(
    new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/New_York',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      hourCycle: 'h23',
    })
      .formatToParts(d)
      .map((p) => [p.type, p.value]),
  );
  return `${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute} ET`;
}

function statusLine(first: {
  total: number;
  counts: { undervalued: number };
  coverage?: { universe: number; complete: number };
  lastFullAt?: string | null;
  lastRun?: {
    status: string;
    done: number;
    total: number;
    startedAt: string | null;
    finishedAt: string | null;
    ok: number;
    fail: number;
  } | null;
  stars: StarChip;
}): string {
  const cov = first.coverage;
  const scored = cov?.complete ?? first.counts.undervalued;
  const universe = cov?.universe;
  const run = first.lastRun;
  const bits: string[] = [];

  if (first.stars === 'undervalued') {
    bits.push(`${first.total} undervalued`);
  } else {
    bits.push(`${first.total} at ${first.stars}/4`);
  }

  if (universe != null) {
    bits.push(`${scored} scored`);
    bits.push(`${universe} STOCK-TICKERS`);
  }

  if (run?.status === 'running') {
    const started = formatEodEt(run.startedAt);
    bits.push(
      `Updating ${run.done}/${run.total}${started ? ` · started ${started}` : ''}`,
    );
  } else {
    const eod = formatEodEt(first.lastFullAt ?? run?.finishedAt);
    if (eod) {
      const fail = run && run.fail > 0 ? ` · ${run.fail} fail` : '';
      bits.push(`EOD ${eod}${fail}`);
    }
  }

  return bits.join(' · ');
}

export function ValuePage() {
  const [search, setSearch] = useSearchParams();
  const rawStars = search.get('stars');
  const rawSort = search.get('sort');
  const stars: StarChip = isStarFilter(rawStars) ? rawStars : 'undervalued';
  const sort: ValueScreenerSort = isSort(rawSort) ? rawSort : 'stars';
  const dir = (search.get('dir') as SortDir | null) ?? 'desc';

  const page = useInfiniteQuery({
    queryKey: ['value-screener', stars, sort, dir],
    initialPageParam: 0,
    queryFn: ({ pageParam }) =>
      api.fundamentalsScreener({
        stars,
        sort,
        dir,
        limit: PAGE_SIZE,
        offset: pageParam,
      }),
    getNextPageParam: (last, pages) => {
      const loaded = pages.reduce((n, p) => n + p.rows.length, 0);
      return loaded < last.total ? loaded : undefined;
    },
    placeholderData: keepPreviousData,
    staleTime: 60_000,
    refetchInterval: (q) =>
      q.state.data?.pages[0]?.lastRun?.status === 'running' ? 5_000 : false,
  });

  const rows = page.data?.pages.flatMap((p) => p.rows) ?? [];
  const first = page.data?.pages[0];
  const counts = first?.counts;

  useRestoreChartScroll(!page.isLoading);

  const setFilter = (next: StarChip) => {
    const params: Record<string, string> = { sort, dir };
    if (next !== 'undervalued') params.stars = next;
    setSearch(params, { replace: true });
  };

  return (
    <div>
      <section className="results-head">
        <UniverseTabs />
        <Chips
          label="Stars"
          value={stars}
          options={STAR_FILTERS}
          format={(v) => {
            const count =
              v === 'undervalued'
                ? counts?.undervalued
                : counts?.[Number(v) as 0 | 1 | 2 | 3 | 4];
            const label = STAR_LABEL[v];
            return count != null ? `${label} ${count}` : label;
          }}
          onChange={setFilter}
        />
        <div className="results-meta">
          <span className="muted small">
            {first
              ? statusLine({ ...first, stars })
              : 'Ranking Stocks by EPS / FCF / DCF undervaluation and LT D/C'}
          </span>
          <SortChips
            options={SORTS}
            value={sort}
            dir={dir}
            onChange={(next, nextDir) => {
              const params: Record<string, string> = { sort: next, dir: nextDir };
              if (stars !== 'undervalued') params.stars = stars;
              setSearch(params, { replace: true });
            }}
          />
        </div>
      </section>

      {page.isLoading ? <p className="empty">Loading…</p> : null}
      {page.error ? <p className="error">{(page.error as Error).message}</p> : null}

      {!page.isLoading && rows.length === 0 ? (
        <p className="empty">
          {first?.lastRun?.status === 'running'
            ? `Updating fundamentals ${first.lastRun.done}/${first.lastRun.total}. Names appear here as the EOD job scores them.`
            : 'No names match this filter yet. The weekday EOD job scores every STOCK-TICKERS name into Mongo; open this tab again after the update finishes.'}
        </p>
      ) : null}

      {rows.map((row) => (
        <ValueCard key={row.yahooTicker} row={row} />
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
