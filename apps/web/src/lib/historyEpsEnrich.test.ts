import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { runHistoryEpsEnrichLoop } from './historyEpsEnrich.ts';
import type { HistoryEpsEnrichResult } from './api.ts';

function batch(
  partial: Partial<HistoryEpsEnrichResult> & Pick<HistoryEpsEnrichResult, 'updated' | 'remaining'>,
): HistoryEpsEnrichResult {
  return {
    configured: true,
    scanned: partial.scanned ?? partial.updated,
    skipped: 0,
    errors: 0,
    ...partial,
  };
}

describe('runHistoryEpsEnrichLoop', () => {
  it('loops until remaining is 0 and sums tagged docs', async () => {
    const sleeps: number[] = [];
    const calls: number[] = [];
    const leftovers = [12, 4, 0];
    const result = await runHistoryEpsEnrichLoop(
      async (limit) => {
        calls.push(limit);
        const remaining = leftovers.shift() ?? 0;
        return batch({ updated: 8, remaining, scanned: 8 });
      },
      { sleep: async (ms) => { sleeps.push(ms); } },
    );
    assert.deepEqual(calls, [80, 80, 80]);
    assert.equal(result.updated, 24);
    assert.equal(result.remaining, 0);
    assert.deepEqual(sleeps, [400, 600]);
  });

  it('stops if a batch writes nothing so a stuck queue cannot spin', async () => {
    let n = 0;
    const result = await runHistoryEpsEnrichLoop(
      async () => {
        n += 1;
        return batch({ updated: 0, remaining: 9, scanned: 0 });
      },
      { sleep: async () => { throw new Error('should not backoff'); } },
    );
    assert.equal(n, 1);
    assert.equal(result.remaining, 9);
    assert.equal(result.updated, 0);
  });
});
