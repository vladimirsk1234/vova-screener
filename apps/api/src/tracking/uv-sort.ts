/**
 * Results / History UV sort. Premiums live on instrumentFundamentals, not the trade,
 * so the bucket is loaded first and then joined by yahooTicker.
 */

export type UvPremia = {
  bestPremiumPct?: number | null;
  epsPremiumPct?: number | null;
};

export type UvRow = {
  yahooTicker: string;
  symbol: string;
};

function finite(n: number | null | undefined): number | null {
  return n != null && Number.isFinite(n) ? n : null;
}

function lookup<T>(ticker: string, cards: Record<string, T | undefined>): T | undefined {
  return cards[ticker] ?? cards[ticker.toUpperCase()];
}

/** Sort key: most undervalued = lowest premium. Falls back to EPS when best is missing. */
export function uvSortKey(card: UvPremia | undefined): number | null {
  const best = finite(card?.bestPremiumPct);
  if (best != null) return best;
  return finite(card?.epsPremiumPct);
}

function compareNullable(av: number | null, bv: number | null, sign: number): number {
  if (av == null && bv == null) return 0;
  if (av == null) return 1;
  if (bv == null) return -1;
  if (av === bv) return 0;
  return av < bv ? -sign : sign;
}

export function compareUvRows<T extends UvRow>(
  a: T,
  b: T,
  cards: Record<string, UvPremia | undefined>,
  dir: 'asc' | 'desc',
): number {
  const sign = dir === 'asc' ? 1 : -1;
  const aCard = lookup(a.yahooTicker, cards);
  const bCard = lookup(b.yahooTicker, cards);
  const byBest = compareNullable(uvSortKey(aCard), uvSortKey(bCard), sign);
  if (byBest !== 0) return byBest;
  const byEps = compareNullable(finite(aCard?.epsPremiumPct), finite(bCard?.epsPremiumPct), sign);
  if (byEps !== 0) return byEps;
  return a.symbol.localeCompare(b.symbol);
}

export function sortByUndervaluation<T extends UvRow>(
  rows: T[],
  cards: Record<string, UvPremia | undefined>,
  dir: 'asc' | 'desc',
): T[] {
  return [...rows].sort((a, b) => compareUvRows(a, b, cards, dir));
}
