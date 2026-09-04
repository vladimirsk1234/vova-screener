import type { HistoryEpsEnrichResult } from './api';

/**
 * Keep posting enrich-eps until remaining is 0.
 * Backs off between batches so a long History tag does not hammer the API.
 */
export async function runHistoryEpsEnrichLoop(
  enrich: (limit: number) => Promise<HistoryEpsEnrichResult>,
  opts?: {
    limit?: number;
    sleep?: (ms: number) => Promise<void>;
    onBatch?: (batch: HistoryEpsEnrichResult, totals: HistoryEpsEnrichResult) => void;
  },
): Promise<HistoryEpsEnrichResult> {
  const limit = opts?.limit ?? 80;
  const wait = opts?.sleep ?? ((ms) => new Promise<void>((resolve) => setTimeout(resolve, ms)));
  let delayMs = 400;
  let tagged = 0;
  let scanned = 0;
  let skipped = 0;
  let errors = 0;
  let last: HistoryEpsEnrichResult | undefined;

  for (;;) {
    last = await enrich(limit);
    tagged += last.updated;
    scanned += last.scanned;
    skipped += last.skipped;
    errors += last.errors;
    const totals: HistoryEpsEnrichResult = {
      configured: last.configured,
      scanned,
      updated: tagged,
      skipped,
      errors,
      remaining: last.remaining,
    };
    opts?.onBatch?.(last, totals);
    if (last.remaining <= 0) return totals;
    if (last.scanned === 0 || last.updated === 0) return totals;
    await wait(delayMs);
    delayMs = Math.min(Math.round(delayMs * 1.5), 4000);
  }
}
