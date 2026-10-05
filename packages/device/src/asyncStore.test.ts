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

  it('replaces the signal set and still finishes while another read is waiting', async () => {
    const file = path.join(mkdtempSync(path.join(tmpdir(), 'vova-replace-')), 'sequence-vova.db');
    const sql = deviceLikeSql(file);
    await migrate(sql);
    const store = createCachedStore(createSqlStore(sql));
    await store.saveSignals([signal(1), signal(2), signal(3)]);
    const rewritten = [signal(2), signal(4)];
    const write = store.saveSignals(rewritten);
    const read = store.listSignals();
    const [, listed] = await Promise.all([write, read]);
    const during = listed.map((row) => row.id).sort().join();
    // A read that overlaps the rewrite sees the previous set or the committed one, never an empty table.
    assert.ok(during === 's1,s2,s3' || during === 's2,s4', during);
    const cold = createSqlStore(deviceLikeSql(file));
    assert.deepEqual((await cold.listSignals()).map((row) => row.id).sort(), ['s2', 's4']);
  });

  it('does not wipe signals when a rewrite is interrupted before the new rows are stored', async () => {
    const file = path.join(mkdtempSync(path.join(tmpdir(), 'vova-wipe-')), 'sequence-vova.db');
    const real = deviceLikeSql(file);
    await migrate(real);
    await createSqlStore(real).saveSignals([signal(1), signal(2)]);

    // expo-sqlite has committed a statement even when the JS transaction later fails. Model that:
    // BEGIN/COMMIT are ignored, and a full-table DELETE before any insert aborts the rewrite.
    let inserted = false;
    const autocommit: SqlAsync = {
      async exec(sql) {
        if (/^\s*BEGIN/i.test(sql) || /^\s*COMMIT/i.test(sql) || /^\s*ROLLBACK/i.test(sql)) return;
        await real.exec(sql);
      },
      async run(sql, params) {
        const wipe = /^\s*DELETE\s+FROM\s+signals\s*$/i.test(sql);
        if (wipe && !inserted) {
          await real.run(sql, params);
          throw new Error('wiped before insert');
        }
        if (/^\s*INSERT/i.test(sql)) inserted = true;
        await real.run(sql, params);
      },
      all: (sql, params) => real.all(sql, params),
      get: (sql, params) => real.get(sql, params),
    };
    const store = createCachedStore(createSqlStore(autocommit));
    await store.saveSignals([signal(9)]);
    const ids = (await createSqlStore(deviceLikeSql(file)).listSignals()).map((row) => row.id).sort();
    assert.deepEqual(ids, ['s9']);
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
