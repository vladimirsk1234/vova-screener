/** Peak concurrent capital: the cash pool needed to take every selected signal. */
function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

export type CapitalTrade = {
  id: string;
  openedAsOf: string | null;
  exitDate: string | null;
  positionValue: number;
};

export type PeakCapitalStats = {
  peakCapitalUsd: number;
  peakCapitalAsOf: string | null;
  peakConcurrentPositions: number;
  openCapitalUsd: number;
};

type CapitalEvent = {
  date: string;
  kind: 'open' | 'close';
  value: number;
  openedAsOf: string;
};

/**
 * Sweep-line over open/close dates. Same-day: a position that was already open releases
 * first, so that cash can fund a new trade; a same-day round-trip still needs capital
 * (its open is applied before its own close).
 *
 * `rangeFrom` clips already-open trades onto the start of the History lookback so a YTD
 * peak is the peak *during* YTD, not during the earlier life of a long holder.
 * Still-open trades (no `exitDate`, or a close after `rangeEnd`) stay in the curve to
 * the end of the window and become `openCapitalUsd`.
 */
export function computePeakCapital(
  trades: CapitalTrade[],
  opts: { rangeFrom?: string | null; rangeEnd?: string | null } = {},
): PeakCapitalStats {
  const rangeFrom = normalizeDate(opts.rangeFrom);
  const rangeEnd = normalizeDate(opts.rangeEnd);
  const unique = dedupeTrades(trades);

  const events: CapitalEvent[] = [];
  let current = 0;
  let positions = 0;

  for (const trade of unique) {
    const value = Number(trade.positionValue);
    if (!Number.isFinite(value) || value <= 0) continue;
    const openedAsOf = normalizeDate(trade.openedAsOf);
    if (!openedAsOf) continue;
    const exitDate = normalizeDate(trade.exitDate);

    if (rangeFrom && exitDate && exitDate < rangeFrom) continue;
    if (rangeEnd && openedAsOf > rangeEnd) continue;

    const alreadyOpenAtStart = Boolean(rangeFrom && openedAsOf < rangeFrom);
    if (alreadyOpenAtStart) {
      current += value;
      positions += 1;
    } else {
      events.push({ date: openedAsOf, kind: 'open', value, openedAsOf });
    }

    if (exitDate && (!rangeEnd || exitDate <= rangeEnd)) {
      events.push({ date: exitDate, kind: 'close', value, openedAsOf });
    }
  }

  events.sort((a, b) => {
    if (a.date !== b.date) return a.date < b.date ? -1 : 1;
    return eventRank(a) - eventRank(b);
  });

  let peak = current;
  let peakAsOf: string | null = current > 0 ? rangeFrom : null;
  let peakPositions = positions;

  for (const event of events) {
    if (event.kind === 'close') {
      current -= event.value;
      positions -= 1;
    } else {
      current += event.value;
      positions += 1;
    }
    if (current > peak + 1e-9) {
      peak = current;
      peakAsOf = event.date;
      peakPositions = positions;
    }
  }

  return {
    peakCapitalUsd: round2(Math.max(0, peak)),
    peakCapitalAsOf: peak > 0 ? peakAsOf : null,
    peakConcurrentPositions: Math.max(0, peakPositions),
    openCapitalUsd: round2(Math.max(0, current)),
  };
}

/** Realized closed P&L ÷ peak concurrent capital, as a percent. */
export function roiOnPeakPct(pnlUsd: number, peakCapitalUsd: number): number | null {
  if (!(peakCapitalUsd > 0)) return null;
  return round2((pnlUsd / peakCapitalUsd) * 100);
}

function dedupeTrades(trades: CapitalTrade[]): CapitalTrade[] {
  const seen = new Set<string>();
  const unique: CapitalTrade[] = [];
  for (const trade of trades) {
    if (!trade.id || seen.has(trade.id)) continue;
    seen.add(trade.id);
    unique.push(trade);
  }
  return unique;
}

/** Prior-day close → open → same-day close, so capital recycles without dropping a same-day round-trip. */
function eventRank(event: CapitalEvent): number {
  if (event.kind === 'close' && event.openedAsOf < event.date) return 0;
  if (event.kind === 'open') return 1;
  return 2;
}

function normalizeDate(value: string | null | undefined): string | null {
  if (!value) return null;
  const day = String(value).slice(0, 10);
  return /^\d{4}-\d{2}-\d{2}$/.test(day) ? day : null;
}
