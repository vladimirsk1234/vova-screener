import assert from 'node:assert/strict';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { describe, it } from 'node:test';
import vm from 'node:vm';
import { createDeviceApi } from './api';
import { bridgeReplyScript, sqlFromExpo, type ExpoSqliteLike } from './bridge';
import { createCachedStore } from './cachedStore';
import { parseUniverse } from './scan';
import { createSqlStore, migrate } from './store';
import type { TrackedSignal } from './types';

/** expo-sqlite's async API shape over a real SQLite file, yielding between calls like the native bridge. */
function fakeExpoDb(file: string): ExpoSqliteLike {
  const db = new DatabaseSync(file);
  const tick = () => new Promise<void>((r) => setImmediate(r));
  return {
    async execAsync(source) {
      await tick();
      db.exec(source);
    },
    async runAsync(source, ...params) {
      await tick();
      return db.prepare(source).run(...params);
    },
    async getAllAsync<T>(source: string, ...params: Array<string | number | null>) {
      await tick();
      return db.prepare(source).all(...params) as T[];
    },
    async getFirstAsync<T>(source: string, ...params: Array<string | number | null>) {
      await tick();
      return (db.prepare(source).get(...params) as T | undefined) ?? null;
    },
  };
}

/** What App.tsx does at launch: open, migrate, cache, API. */
async function launchApp(file: string) {
  const sql = sqlFromExpo(fakeExpoDb(file));
  await migrate(sql);
  return createDeviceApi({
    store: createCachedStore(createSqlStore(sql)),
    universe: parseUniverse('NASDAQ:AAA|Alpha', 'AMEX:EEE|ETF'),
  });
}

/** The web side: deviceBridge.ts posts this JSON; App.tsx injects the reply script back. */
async function webRequest(api: Awaited<ReturnType<typeof launchApp>>, id: number, method: string, apiPath: string, body?: unknown) {
  const raw = JSON.stringify({ id, method, path: apiPath, body: body === undefined ? null : JSON.stringify(body) });
  const script = await bridgeReplyScript(api, raw);
  assert.ok(script);
  let reply: { id: number; status: number; body: string } | null = null;
  const context = vm.createContext({
    window: { __vovaReply: (rid: number, status: number, text: string) => (reply = { id: rid, status, body: text }) },
  });
  vm.runInContext(script, context);
  assert.ok(reply, 'reply delivered');
  const got = reply as { id: number; status: number; body: string };
  assert.equal(got.id, id);
  return { status: got.status, data: JSON.parse(got.body) };
}

function openPosition(i: number): TrackedSignal {
  return {
    id: `p${i}`,
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

describe('chart preset migration to the gray default', () => {
  it('turns the old black default background gray once and keeps the rest of the preset', async () => {
    const file = path.join(mkdtempSync(path.join(tmpdir(), 'vova-migrate-')), 'sequence-vova.db');
    const first = await launchApp(file);
    await webRequest(first, 1, 'PUT', '/presets/chart', { bg_color: '#000000', wm_text_color: '#000000', show_fib: true, len_fast: 9 });

    const relaunch = await launchApp(file);
    await relaunch.migratePresets();
    const migrated = await webRequest(relaunch, 1, 'GET', '/presets/chart');
    assert.deepEqual(migrated.data, { bg_color: '#707585', wm_text_color: '#000000', show_fib: true, len_fast: 9 });

    // A black chosen after the migration is the user's choice and stays.
    await webRequest(relaunch, 2, 'PUT', '/presets/chart', { ...migrated.data, bg_color: '#000000' });
    const again = await launchApp(file);
    await again.migratePresets();
    assert.equal((await webRequest(again, 1, 'GET', '/presets/chart')).data.bg_color, '#000000');
  });
});

describe('chart Save through the iPhone bridge and expo-sqlite adapter', () => {
  it('persists visibility switches, MA lengths and colours across a cold start', async () => {
    const file = path.join(mkdtempSync(path.join(tmpdir(), 'vova-bridge-')), 'sequence-vova.db');
    const api = await launchApp(file);

    // Seed open positions so a Max-risk change rewrites the signals table in a transaction.
    const seedSql = sqlFromExpo(fakeExpoDb(file));
    await createSqlStore(seedSql).saveSignals(Array.from({ length: 200 }, (_, i) => openPosition(i)));
    const relaunchedForSeed = await launchApp(file);

    const saved = {
      bg_color: '#707585',
      wm_text_color: '#000000',
      show_fib: true,
      show_short_ema: true,
      show_center_ema: false,
      show_sma_major: true,
      show_bb: true,
      show_tp_sl: true,
      len_fast: 9,
      len_slow: 21,
      length_major: 100,
    };
    const [riskChange, save] = await Promise.all([
      webRequest(relaunchedForSeed, 1, 'PUT', '/settings', { maxRiskUsd: 250 }),
      webRequest(relaunchedForSeed, 2, 'PUT', '/presets/chart', saved),
    ]);
    assert.equal(riskChange.status, 200, JSON.stringify(riskChange.data));
    assert.equal(save.status, 200, JSON.stringify(save.data));
    void api;

    // Cold start / app update: new process, same file, empty caches.
    const coldStart = await launchApp(file);
    const read = await webRequest(coldStart, 1, 'GET', '/presets/chart');
    assert.equal(read.status, 200);
    assert.deepEqual(read.data, saved);
    const settings = await webRequest(coldStart, 2, 'GET', '/settings');
    assert.equal(settings.data.maxRiskUsd, 250);
  });
});
