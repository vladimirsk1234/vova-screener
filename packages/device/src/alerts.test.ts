import assert from 'node:assert/strict';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { describe, it } from 'node:test';
import {
  closeReminders,
  DEFAULT_ALERT_PREFS,
  nextMonthlyCloses,
  nextWeeklyCloses,
  nyWallClockToDate,
  scanAlerts,
} from './alerts';
import { ALERT_PREFS_KEY, createDeviceApi, type Notifier } from './api';
import { createCachedStore } from './cachedStore';
import { openNodeStore } from './nodeStore.test-helper';
import { parseUniverse } from './scan';
import type { Reminder } from './alerts';
import type { ScanMeta, TrackedSignal } from './types';

const scan: ScanMeta = {
  periodKey: '2026-W40',
  asOf: '2026-09-28',
  newestAsOf: '2026-09-28',
  finishedAt: null,
  running: false,
  status: 'completed',
};

function row(partial: Partial<TrackedSignal> & Pick<TrackedSignal, 'id' | 'yahooTicker'>): TrackedSignal {
  return {
    symbol: partial.yahooTicker,
    tvSymbol: partial.yahooTicker,
    companyName: partial.yahooTicker,
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
    barsSinceValid: 3,
    validSinceAsOf: '2026-09-07',
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
    ...partial,
  };
}

describe('scan alerts', () => {
  const before = [
    row({ id: 'old-new', yahooTicker: 'OLD', barsSinceValid: 0 }),
    row({ id: 'run', yahooTicker: 'RUN' }),
    row({ id: 'brk', yahooTicker: 'BRK' }),
  ];
  const after = [
    row({ id: 'old-new', yahooTicker: 'OLD', barsSinceValid: 0 }),
    row({ id: 'fresh', yahooTicker: 'FRESH', barsSinceValid: 0, interest: 'interested' }),
    row({ id: 'run', yahooTicker: 'RUN', status: 'closed', closedPeriodKey: '2026-W40', exitDate: '2026-09-28', exitPrice: 11 }),
    row({ id: 'brk', yahooTicker: 'BRK', provisionalClose: true }),
    row({ id: 'adopted', yahooTicker: 'ADOPT', status: 'closed', exitDate: '2026-09-28', exitPrice: 8 }),
  ];
  const all = {
    ...DEFAULT_ALERT_PREFS,
    newSignals: true,
    sellToClose: true,
    closing: true,
  };

  it('reports only what changed in NEW, CLOSED and CLOSING', () => {
    const alerts = scanAlerts({ before, after, universe: 'Stocks', tf: 'Weekly', scan, prefs: all });
    assert.deepEqual(
      alerts.map((a) => [a.kind, a.symbols]),
      [
        ['new', ['FRESH']],
        ['closed', ['ADOPT', 'RUN']],
        ['closing', ['BRK']],
      ],
    );
    assert.equal(alerts[0].title, 'Stocks W · 1 new signal');
  });

  it('respects the kind, scope and Interested filters', () => {
    assert.deepEqual(scanAlerts({ before, after, universe: 'Stocks', tf: 'Weekly', scan, prefs: DEFAULT_ALERT_PREFS }), []);
    assert.deepEqual(scanAlerts({ before, after, universe: 'Stocks', tf: 'Weekly', scan, prefs: { ...all, stocks: false } }), []);
    assert.deepEqual(scanAlerts({ before, after, universe: 'Stocks', tf: 'Weekly', scan, prefs: { ...all, weekly: false } }), []);
    const interested = scanAlerts({ before, after, universe: 'Stocks', tf: 'Weekly', scan, prefs: { ...all, onlyInterested: true } });
    assert.deepEqual(interested.map((a) => a.symbols), [['FRESH']]);
  });
});

describe('close reminders', () => {
  it('converts New York wall-clock times across DST', () => {
    assert.equal(nyWallClockToDate(2026, 7, 10, 16, 15).toISOString(), '2026-07-10T20:15:00.000Z');
    assert.equal(nyWallClockToDate(2026, 12, 11, 16, 15).toISOString(), '2026-12-11T21:15:00.000Z');
  });

  it('schedules Friday 16:15 ET and the last weekday of the month', () => {
    const now = new Date('2026-10-05T01:00:00Z');
    assert.deepEqual(
      nextWeeklyCloses(now, 2).map((d) => d.toISOString()),
      ['2026-10-09T20:15:00.000Z', '2026-10-16T20:15:00.000Z'],
    );
    // October 2026 ends on a Saturday, so the monthly close is Friday the 30th.
    assert.equal(nextMonthlyCloses(now, 1)[0].toISOString(), '2026-10-30T20:15:00.000Z');
    const reminders = closeReminders({ ...DEFAULT_ALERT_PREFS, weeklyCloseReminder: true, monthlyCloseReminder: true }, now);
    assert.equal(reminders.length, 6 + 3);
    assert.ok(reminders.every((r) => r.id.startsWith('reminder-') && r.at > now));
    assert.deepEqual(closeReminders(DEFAULT_ALERT_PREFS, now), []);
  });
});

function fakeNotifier(initial: 'granted' | 'undetermined' | 'denied') {
  let state: 'granted' | 'undetermined' | 'denied' = initial;
  const sent: Array<[string, string]> = [];
  let reminders: Reminder[] = [];
  let requests = 0;
  const notifier: Notifier = {
    permission: async () => state,
    request: async () => {
      requests += 1;
      state = 'granted';
      return state;
    },
    notify: async (title, body) => {
      sent.push([title, body]);
    },
    scheduleReminders: async (next) => {
      reminders = next;
    },
  };
  return { notifier, sent, get reminders() { return reminders; }, get requests() { return requests; } };
}

describe('Save survives a cold start (same SQLite file, new process)', () => {
  it('writes alert choices and the chart preset on Save and reads them back after reopening', async () => {
    const file = path.join(mkdtempSync(path.join(tmpdir(), 'vova-save-')), 'sequence-vova.db');
    const universe = parseUniverse('NASDAQ:AAA|Alpha', 'AMEX:EEE|ETF');
    const fake = fakeNotifier('undetermined');
    const api = createDeviceApi({
      store: createCachedStore(await openNodeStore(file)),
      universe,
      notifier: fake.notifier,
      now: () => new Date('2026-10-05T01:00:00Z'),
    });
    const handle = async (method: string, url: string, body?: unknown) =>
      JSON.parse((await api.handle(method, url, body === undefined ? null : JSON.stringify(body))).body);

    const untouched = await handle('GET', '/device/alerts');
    assert.equal(untouched.prefs.newSignals, false);
    assert.equal(fake.requests, 0);

    const saved = await handle('PUT', '/device/alerts', {
      ...untouched.prefs,
      newSignals: true,
      weeklyCloseReminder: true,
      etf: false,
    });
    assert.equal(saved.permission, 'granted');
    assert.equal(fake.requests, 1);
    assert.equal(fake.reminders.length, 6);

    const chartPreset = { bg_color: '#000000', wm_text_color: '#000000', show_fib: true, show_bb: true };
    await handle('PUT', '/presets/chart', chartPreset);

    // A fresh process on the same file: what an app update or cold start sees.
    const disk = await openNodeStore(file);
    assert.deepEqual(await disk.getPreset('chart'), chartPreset);
    const alertsOnDisk = (await disk.getPreset(ALERT_PREFS_KEY)) as Record<string, boolean>;
    assert.equal(alertsOnDisk.newSignals, true);
    assert.equal(alertsOnDisk.etf, false);

    const relaunched = createDeviceApi({
      store: createCachedStore(disk),
      universe,
      notifier: fake.notifier,
      now: () => new Date('2026-10-05T01:00:00Z'),
    });
    const read = JSON.parse((await relaunched.handle('GET', '/device/alerts')).body);
    assert.equal(read.prefs.newSignals, true);
    assert.equal(read.prefs.weeklyCloseReminder, true);
    const chart = JSON.parse((await relaunched.handle('GET', '/presets/chart')).body);
    assert.deepEqual(chart, chartPreset);
    await relaunched.restoreAlerts();
    assert.equal(fake.reminders.length, 6);
  });

  it('never notifies without permission', async () => {
    const file = path.join(mkdtempSync(path.join(tmpdir(), 'vova-deny-')), 'sequence-vova.db');
    const fake = fakeNotifier('denied');
    const api = createDeviceApi({
      store: createCachedStore(await openNodeStore(file)),
      universe: parseUniverse('NASDAQ:AAA|Alpha', ''),
      notifier: fake.notifier,
    });
    const res = JSON.parse(
      (await api.handle('PUT', '/device/alerts', JSON.stringify({ ...DEFAULT_ALERT_PREFS, weeklyCloseReminder: true }))).body,
    );
    assert.equal(res.permission, 'denied');
    assert.equal(fake.requests, 0);
    assert.deepEqual(fake.reminders, []);
  });
});
