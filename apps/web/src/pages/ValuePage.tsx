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

const PAGE_SIZE = 100;

const STAR_FILTERS = ['undervalued', '3', '2', '1'] as const;
type StarChip = (typeof STAR_FILTERS)[number];

const STAR_LABEL: Record<StarChip, string> = {
  undervalued: 'All',
  '3': '3/3',
  '2': '2/3',
  '1': '1/3',
};

const SORTS: SortOption<ValueScreenerSort>[] = [
  { value: 'stars', label: 'Stars' },
  { value: 'eps', label: 'EPS %', from: 'asc' },
  { value: 'fcf', label: 'FCF %', from: 'asc' },
  { value: 'dcf', label: 'DCF %', from: 'asc' },
  { value: 'symbol', label: 'A-Z', from: 'asc' },
];

function isStarFilter(value: string | null): value is StarChip {
  return STAR_FILTERS.includes(value as StarChip);
}

function isSort(value: string | null): value is ValueScreenerSort {
  return SORTS.some((s) => s.value === value);
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
  });

  const rows = page.data?.pages.flatMap((p) => p.rows) ?? [];
  const first = page.data?.pages[0];
  const counts = first?.counts;

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
              v === 'undervalued' ? counts?.undervalued : counts?.[Number(v) as 1 | 2 | 3];
            const label = STAR_LABEL[v];
            return count != null ? `${label} ${count}` : label;
          }}
          onChange={setFilter}
        />
        <div className="results-meta">
          <span className="muted small">
            {first
              ? `${first.total} names from STOCK-TICKERS · ${stars === 'undervalued' ? '1/3 and up' : `${stars}/3`}`
              : 'Ranking Stocks by EPS / FCF / DCF undervaluation'}
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
          No names match this filter yet. Fundamentals refresh into Mongo daily; a Stocks scan fills
          Seq/Struct on the cards.
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
