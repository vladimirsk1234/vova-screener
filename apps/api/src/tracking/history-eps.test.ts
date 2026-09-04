import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  EPS_UNKNOWN_STAMP,
  enrichRemaining,
  epsStampFromHit,
  HistoryEpsService,
} from './history-eps.service.ts';

describe('epsStampFromHit', () => {
  it('does not invent an EPS number when FMP has no report', () => {
    assert.deepEqual(epsStampFromHit({ eps: null, date: null }), { ...EPS_UNKNOWN_STAMP });
    assert.equal(typeof EPS_UNKNOWN_STAMP.epsAtEntry, 'object');
    assert.equal(EPS_UNKNOWN_STAMP.epsAtEntry, null);
  });

  it('keeps the reported EPS and only derives the sign', () => {
    assert.deepEqual(epsStampFromHit({ eps: 2.15, date: '2024-06-30' }), {
      epsAtEntry: 2.15,
      epsPositiveAtEntry: true,
      epsAtEntryAsOf: '2024-06-30',
    });
    assert.deepEqual(epsStampFromHit({ eps: -0.4, date: '2023-12-31' }), {
      epsAtEntry: -0.4,
      epsPositiveAtEntry: false,
      epsAtEntryAsOf: '2023-12-31',
    });
  });
});

describe('enrichRemaining', () => {
  it('drops remaining by the number of docs written (including null stamps)', () => {
    assert.equal(enrichRemaining(10, 3), 7);
    assert.equal(enrichRemaining(2, 2), 0);
    assert.equal(enrichRemaining(1, 4), 0);
  });
});

function trackedMock(docs: Array<{ _id: string; yahooTicker: string; openedAsOf: string }>) {
  const updates: Array<{ filter: { _id: unknown }; update: { $set: Record<string, unknown> } }> = [];
  const pending = {
    openedAsOf: { $type: 'string', $ne: '' },
    epsPositiveAtEntry: { $exists: false },
  };
  return {
    updates,
    find(filter: unknown) {
      assert.deepEqual(filter, pending);
      return {
        select() {
          return {
            limit() {
              return {
                lean() {
                  return { exec: async () => docs };
                },
              };
            },
          };
        },
      };
    },
    countDocuments(filter: unknown) {
      assert.deepEqual(filter, pending);
      return { exec: async () => docs.length + 5 };
    },
    async updateOne(filter: { _id: unknown }, update: { $set: Record<string, unknown> }) {
      updates.push({ filter, update });
    },
  };
}

describe('HistoryEpsService.enrich', () => {
  it('writes null on FMP error so remaining decreases and no EPS is invented', async () => {
    const tracked = trackedMock([
      { _id: 'a', yahooTicker: 'AAA', openedAsOf: '2024-01-15' },
      { _id: 'b', yahooTicker: 'BBB', openedAsOf: '2024-02-01' },
    ]);
    const fmp = {
      configured: () => true,
      resolveFmpSymbol: async (ticker: string) => ticker,
      epsAsOf: async () => {
        throw new Error('FMP request failed (429) for /income-statement');
      },
    };
    const svc = new HistoryEpsService(tracked as never, fmp as never);
    const result = await svc.enrich(80, { tickerGapMs: 0 });

    assert.equal(result.errors, 2);
    assert.equal(result.updated, 2);
    assert.equal(result.skipped, 0);
    assert.equal(result.remaining, 5);
    assert.equal(tracked.updates.length, 2);
    for (const row of tracked.updates) {
      assert.deepEqual(row.update.$set, { ...EPS_UNKNOWN_STAMP });
      assert.equal(row.update.$set.epsAtEntry, null);
      assert.equal(typeof row.update.$set.epsAtEntry === 'number', false);
    }
  });

  it('writes reported EPS on success and never substitutes another number', async () => {
    const tracked = trackedMock([{ _id: 'a', yahooTicker: 'AAPL', openedAsOf: '2024-03-01' }]);
    const fmp = {
      configured: () => true,
      resolveFmpSymbol: async () => 'AAPL',
      epsAsOf: async () => ({ eps: 6.42, date: '2023-09-30' }),
    };
    const svc = new HistoryEpsService(tracked as never, fmp as never);
    const result = await svc.enrich(40, { tickerGapMs: 0 });

    assert.equal(result.errors, 0);
    assert.equal(result.updated, 1);
    assert.equal(result.remaining, 5);
    assert.deepEqual(tracked.updates[0]?.update.$set, {
      epsAtEntry: 6.42,
      epsPositiveAtEntry: true,
      epsAtEntryAsOf: '2023-09-30',
    });
  });

  it('stamps invalid ticker/asOf as unknown so they leave the pending queue', async () => {
    const tracked = trackedMock([{ _id: 'bad', yahooTicker: '', openedAsOf: 'not-a-date' }]);
    const fmp = {
      configured: () => true,
      resolveFmpSymbol: async () => {
        throw new Error('should not fetch');
      },
      epsAsOf: async () => {
        throw new Error('should not fetch');
      },
    };
    const svc = new HistoryEpsService(tracked as never, fmp as never);
    const result = await svc.enrich(10, { tickerGapMs: 0 });

    assert.equal(result.updated, 1);
    assert.equal(result.skipped, 1);
    assert.equal(result.errors, 0);
    assert.equal(result.remaining, 5);
    assert.deepEqual(tracked.updates[0]?.update.$set, { ...EPS_UNKNOWN_STAMP });
  });
});
