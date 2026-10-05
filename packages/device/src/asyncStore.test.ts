/**
 * expo-sqlite on the phone answers every call asynchronously over the native bridge, so two
 * callers interleave statements on the one connection. node:sqlite (used elsewhere in the tests)
 * finishes each call before the next starts and hides that. This store adapter yields between
 * statements the way the device does.
 */
import assert from 'node:assert/strict';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { describe, it } from 'node:test';
import { createCachedStore } from './cachedStore';
import { createSqlStore, migrate, type SqlAsync } from './store';
import type { TrackedSignal } from './types';

const tick = () => new Promise<void>((resolve) => setImmediate(resolve));

function deviceLikeSql(file: string): SqlAsync {
  const db = new DatabaseSync(file);
  return {
    async exec(sql) {
      await tick();
      db.exec(sql);
    },
    async run(sql, params = []) {
      await tick();
      db.prepare(sql).run(...params);
    },
    async all<T>(sql: string, params: readonly (string | number | null)[] = []) {
      await tick();
      return db.prepare(sql).all(...params) as T[];
    },
    async get<T>(sql: string, params: readonly (string | number | null)[] = []) {
      await tick();
      return (db.prepare(sql).get(...params) as T | undefined) ?? null;
    },
  };
}

function signal(i: number): TrackedSignal {
  return {
    id: `s${i}`,
    yahooTicker: `T${i}`,
    symbol: `T${i}`,
    tvSymbol: `T${i}`,
    companyName: `T${i}`,
    universe: 'Stocks',
    tf: 'Weekly',
    status: 'active',
    provisional: false,
    provisionalClose: false,
    signalValid: true,
    imported: false,
    openedPeriodKey: '2026-W40',
    openedAsOf: '2026-09-28',
    openedAt: '2026-09-28T12:00:00.000Z',
    entry: 10,
    tp: 12,
    sl: 9,
    rrAtEntry: 2,
    shares: 100,
    riskUsd: 100,
    lastSeenPeriodKey: '2026-W40',
    lastSeenAsOf: '2026-09-28',
    lastPrice: 10,
    lastRr: 2,
    barsSinceValid: 1,
    validSinceAsOf: '2026-09-21',
    isStrong: false,
    unrealizedUsd: 0,
    unrealizedR: 0,
    unrealizedPct: 0,
    closedPeriodKey: null,
    closedAt: null,
    exitDate: null,
    exitPrice: null,
    exitReason: null,
    pnlUsd: null,
    pnlR: null,
    pnlPct: null,
    holdPeriods: null,
    interest: null,
    interestRank: 1,
  };
}

describe('SQLite store with device-like async calls', () => {
  it('keeps a chart Save that lands while a scan is rewriting signals, across a cold start', async () => {
    const file = path.join(mkdtempSync(path.join(tmpdir(), 'vova-async-')), 'sequence-vova.db');
    const sql = deviceLikeSql(file);
    await migrate(sql);
    const store = createCachedStore(createSqlStore(sql));

    const many = Array.from({ length: 300 }, (_, i) => signal(i));
    const preset = { bg_color: '#707585', show_fib: true, len_fast: 9, len_slow: 21, length_major: 100 };
    // A scan pass saving its signals, a resize from Settings, and the user pressing Save preset.
    const results = await Promise.allSettled([
      store.saveSignals(many),
      store.saveSignals(many.slice(0, 50)),
      store.putPreset('chart', preset),
    ]);
    assert.deepEqual(
      results.map((r) => r.status),
      ['fulfilled', 'fulfilled', 'fulfilled'],
      JSON.stringify(results.map((r) => (r.status === 'rejected' ? String(r.reason) : 'ok'))),
    );

    // Cold start: a new connection to the same file, no in-memory cache.
    const reopened = createSqlStore(deviceLikeSql(file));
    assert.deepEqual(await reopened.getPreset('chart'), preset);
    assert.equal((await reopened.listSignals()).length, 50);
  });

  it('reports Save only once the preset is committed, not while a scan transaction is still open', async () => {
    const file = path.join(mkdtempSync(path.join(tmpdir(), 'vova-commit-')), 'sequence-vova.db');
    const sql = deviceLikeSql(file);
    await migrate(sql);
    const store = createCachedStore(createSqlStore(sql));
    const preset = { show_fib: true, len_fast: 9 };

    // A scan pass starts writing its signals; the user presses Save while that is in flight.
    const scanWrite = store.saveSignals(Array.from({ length: 400 }, (_, i) => signal(i)));
    await tick();
    await store.putPreset('chart', preset);

    // "Saved" is on screen now. If iOS kills the app at this moment, the next launch reads the
    // file from a fresh connection: the preset must already be there.
    const afterKill = createSqlStore(deviceLikeSql(file));
    assert.deepEqual(await afterKill.getPreset('chart'), preset);
    await scanWrite;
  });

  it('does not report a Save as done when the disk write failed', async () => {
    const file = path.join(mkdtempSync(path.join(tmpdir(), 'vova-fail-')), 'sequence-vova.db');
    const sql = deviceLikeSql(file);
    await migrate(sql);
    const disk = createSqlStore(sql);
    const failing = { ...disk, putPreset: async () => Promise.reject(new Error('disk full')) };
    const store = createCachedStore(failing);
    await assert.rejects(store.putPreset('chart', { show_fib: true }));
    assert.equal(await store.getPreset('chart'), null);
  });
});
