/** History EPS-at-entry stamps. Nest-free so unit tests can import under strip-types. */

export type HistoryEpsEnrichResult = {
  configured: boolean;
  scanned: number;
  updated: number;
  skipped: number;
  errors: number;
  remaining: number;
};

export type EpsHit = { eps: number | null; date: string | null };

/**
 * Explicit null = looked up or failed (same spirit as premium-at-entry).
 * Never invent an EPS number.
 */
export const EPS_UNKNOWN_STAMP = {
  epsAtEntry: null,
  epsPositiveAtEntry: null,
  epsAtEntryAsOf: null,
} as const;

export function epsStampFromHit(hit: EpsHit): {
  epsAtEntry: number | null;
  epsPositiveAtEntry: boolean | null;
  epsAtEntryAsOf: string | null;
} {
  if (hit.eps == null) return { ...EPS_UNKNOWN_STAMP };
  return {
    epsAtEntry: hit.eps,
    epsPositiveAtEntry: hit.eps > 0,
    epsAtEntryAsOf: hit.date,
  };
}

export function enrichRemaining(remainingBefore: number, written: number): number {
  return Math.max(0, remainingBefore - written);
}

/**
 * `epsAsOf` uses up to 2 FMP calls per ticker (annual then quarterly income).
 * 200ms between tickers → ≤600 calls/min if both fire and HTTP is instant,
 * leaving headroom under a 750/min key when a batch is 50–100.
 */
export const FMP_EPS_ENRICH_TICKER_GAP_MS = 200;

export const PENDING_EPS = {
  openedAsOf: { $type: 'string', $ne: '' },
  epsPositiveAtEntry: { $exists: false },
};

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export type HistoryEpsTracked = {
  find(filter: unknown): {
    select(fields: string): {
      limit(n: number): {
        lean<T>(): { exec(): Promise<T> };
      };
    };
  };
  countDocuments(filter: unknown): { exec(): Promise<number> };
  updateOne(filter: unknown, update: unknown): Promise<unknown>;
};

export type HistoryEpsFmp = {
  resolveFmpSymbol(ticker: string): Promise<string>;
  epsAsOf(symbol: string, asOf: string): Promise<EpsHit>;
};

export type HistoryEpsLog = {
  warn(message: string): void;
};

export async function enrichHistoryEps(
  tracked: HistoryEpsTracked,
  fmp: HistoryEpsFmp,
  opts?: { limit?: number; tickerGapMs?: number; log?: HistoryEpsLog },
): Promise<HistoryEpsEnrichResult> {
  const cap = Math.min(Math.max(opts?.limit ?? 40, 1), 200);
  const gapMs = opts?.tickerGapMs ?? FMP_EPS_ENRICH_TICKER_GAP_MS;
  const pending = await tracked
    .find(PENDING_EPS)
    .select('_id yahooTicker openedAsOf')
    .limit(cap)
    .lean<Array<{ _id: unknown; yahooTicker: string; openedAsOf: string }>>()
    .exec();

  const remainingBefore = await tracked.countDocuments(PENDING_EPS).exec();

  let updated = 0;
  let skipped = 0;
  let errors = 0;
  const epsCache = new Map<string, EpsHit>();

  for (let i = 0; i < pending.length; i += 1) {
    if (i > 0 && gapMs > 0) await sleep(gapMs);
    const doc = pending[i];
    const asOf = doc.openedAsOf;
    const ticker = String(doc.yahooTicker || '').trim();
    if (!ticker || !/^\d{4}-\d{2}-\d{2}$/.test(asOf)) {
      await tracked.updateOne({ _id: doc._id }, { $set: { ...EPS_UNKNOWN_STAMP } });
      updated += 1;
      skipped += 1;
      continue;
    }
    const cacheKey = `${ticker.toUpperCase()}|${asOf}`;
    try {
      let hit = epsCache.get(cacheKey);
      if (!hit) {
        const fmpSymbol = await fmp.resolveFmpSymbol(ticker);
        hit = await fmp.epsAsOf(fmpSymbol, asOf);
        epsCache.set(cacheKey, hit);
      }
      await tracked.updateOne({ _id: doc._id }, { $set: epsStampFromHit(hit) });
      updated += 1;
    } catch (err) {
      errors += 1;
      opts?.log?.warn(`EPS-at-entry failed for ${ticker} @ ${asOf}: ${(err as Error).message}`);
      try {
        await tracked.updateOne({ _id: doc._id }, { $set: { ...EPS_UNKNOWN_STAMP } });
        updated += 1;
      } catch (writeErr) {
        opts?.log?.warn(
          `EPS-at-entry null stamp failed for ${ticker}: ${(writeErr as Error).message}`,
        );
      }
    }
  }

  return {
    configured: true,
    scanned: pending.length,
    updated,
    skipped,
    errors,
    remaining: enrichRemaining(remainingBefore, updated),
  };
}
