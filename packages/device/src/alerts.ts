/**
 * Alert choices and what a finished scan pass is worth telling the user about. The kinds are the
 * ones the Results screen already distinguishes: NEW, CLOSED (sell to close) and CLOSING, scoped
 * by universe / timeframe and optionally to symbols marked Interested.
 *
 * Scans only run while the app is open, so these fire then. The close reminders are plain iOS
 * calendar notifications: they arrive with the app closed, but they cannot scan anything.
 */
import { barPeriodKey, partsInNy } from '../../../apps/api/src/scans/period.ts';
import { inBucket } from './results';
import type { ScanMeta, TrackedSignal, Universe, UserTf } from './types';

export type AlertPrefs = {
  newSignals: boolean;
  sellToClose: boolean;
  closing: boolean;
  scanFinished: boolean;
  weeklyCloseReminder: boolean;
  monthlyCloseReminder: boolean;
  stocks: boolean;
  etf: boolean;
  weekly: boolean;
  monthly: boolean;
  onlyInterested: boolean;
};

export const DEFAULT_ALERT_PREFS: AlertPrefs = {
  newSignals: false,
  sellToClose: false,
  closing: false,
  scanFinished: false,
  weeklyCloseReminder: false,
  monthlyCloseReminder: false,
  stocks: true,
  etf: true,
  weekly: true,
  monthly: true,
  onlyInterested: false,
};

export function sanitizeAlertPrefs(raw: unknown): AlertPrefs {
  const input = (raw ?? {}) as Partial<Record<keyof AlertPrefs, unknown>>;
  const out = { ...DEFAULT_ALERT_PREFS };
  for (const key of Object.keys(DEFAULT_ALERT_PREFS) as Array<keyof AlertPrefs>) {
    if (typeof input[key] === 'boolean') out[key] = input[key] as boolean;
  }
  return out;
}

export function anyAlertEnabled(prefs: AlertPrefs): boolean {
  return (
    prefs.newSignals ||
    prefs.sellToClose ||
    prefs.closing ||
    prefs.scanFinished ||
    prefs.weeklyCloseReminder ||
    prefs.monthlyCloseReminder
  );
}

export type ScanAlert = { kind: 'new' | 'closed' | 'closing'; title: string; body: string; symbols: string[] };

const TF_SHORT: Record<UserTf, string> = { Weekly: 'W', Monthly: 'M' };

/** What one universe × timeframe pass changed, as the Results tabs would show it. */
export function scanAlerts(opts: {
  before: TrackedSignal[];
  after: TrackedSignal[];
  universe: Universe;
  tf: UserTf;
  scan: ScanMeta;
  prefs: AlertPrefs;
}): ScanAlert[] {
  const { universe, tf, prefs, scan } = opts;
  if (universe === 'Stocks' ? !prefs.stocks : !prefs.etf) return [];
  if (tf === 'Weekly' ? !prefs.weekly : !prefs.monthly) return [];
  const mine = (row: TrackedSignal) => row.universe === universe && row.tf === tf;
  const wanted = (row: TrackedSignal) => !prefs.onlyInterested || row.interest === 'interested';
  const before = opts.before.filter(mine);
  const after = opts.after.filter(mine).filter(wanted);
  const label = `${universe} ${TF_SHORT[tf]}`;
  const out: ScanAlert[] = [];

  if (prefs.newSignals) {
    const wasNew = new Set(before.filter((r) => inBucket(r, universe, tf, 'new', 0, scan)).map((r) => r.yahooTicker));
    const fresh = after.filter((r) => inBucket(r, universe, tf, 'new', 0, scan) && !wasNew.has(r.yahooTicker));
    if (fresh.length) out.push(alert('new', `${label} · ${count(fresh.length, 'new signal')}`, fresh));
  }
  if (prefs.sellToClose) {
    const wasClosed = new Set(before.filter((r) => r.status === 'closed').map((r) => r.id));
    const closed = after.filter((r) => r.status === 'closed' && !wasClosed.has(r.id));
    if (closed.length) out.push(alert('closed', `${label} · ${count(closed.length, 'sell to close')}`, closed));
  }
  if (prefs.closing) {
    const wasClosing = new Set(before.filter((r) => r.provisionalClose).map((r) => r.id));
    const closing = after.filter((r) => r.provisionalClose && !wasClosing.has(r.id));
    if (closing.length) {
      out.push(alert('closing', `${label} · ${count(closing.length, 'closing on the bar in progress', true)}`, closing));
    }
  }
  return out;
}

function alert(kind: ScanAlert['kind'], title: string, rows: TrackedSignal[]): ScanAlert {
  const symbols = [...new Set(rows.map((r) => r.symbol))].sort();
  const shown = symbols.slice(0, 12).join(', ');
  const more = symbols.length > 12 ? ` +${symbols.length - 12} more` : '';
  return { kind, title, body: `${shown}${more}`, symbols };
}

function count(n: number, noun: string, invariant = false): string {
  return `${n} ${noun}${invariant || n === 1 ? '' : 's'}`;
}

/** UTC instant for a wall-clock time in America/New_York. */
export function nyWallClockToDate(year: number, month: number, day: number, hour: number, minute: number): Date {
  const target = Date.UTC(year, month - 1, day, hour, minute);
  let t = target;
  // Twice, so an instant that lands across a DST switch settles on the right offset.
  for (let i = 0; i < 2; i++) t = target - (nyWallAsUtc(t) - t);
  return new Date(t);
}

function nyWallAsUtc(ms: number): number {
  const parts = Object.fromEntries(
    new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/New_York',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      hourCycle: 'h23',
    })
      .formatToParts(new Date(ms))
      .map((p) => [p.type, p.value]),
  );
  return Date.UTC(Number(parts.year), Number(parts.month) - 1, Number(parts.day), Number(parts.hour), Number(parts.minute));
}

const CLOSE_HOUR = 16;
const CLOSE_MINUTE = 15;

/** The next `count` Friday 16:15 ET instants after `now` (weekly bar close, holidays not modelled). */
export function nextWeeklyCloses(now: Date, count: number): Date[] {
  const out: Date[] = [];
  const today = partsInNy(now);
  for (let offset = 0; out.length < count && offset < 7 * (count + 2); offset++) {
    const probe = new Date(Date.UTC(today.year, today.month - 1, today.day + offset, 12));
    if (probe.getUTCDay() !== 5) continue;
    const at = nyWallClockToDate(probe.getUTCFullYear(), probe.getUTCMonth() + 1, probe.getUTCDate(), CLOSE_HOUR, CLOSE_MINUTE);
    if (at > now) out.push(at);
  }
  return out;
}

/** The next `count` last-weekday-of-month 16:15 ET instants after `now` (monthly bar close). */
export function nextMonthlyCloses(now: Date, count: number): Date[] {
  const out: Date[] = [];
  const today = partsInNy(now);
  for (let m = 0; out.length < count && m < count + 2; m++) {
    const lastDay = new Date(Date.UTC(today.year, today.month - 1 + m + 1, 0, 12));
    while (lastDay.getUTCDay() === 0 || lastDay.getUTCDay() === 6) lastDay.setUTCDate(lastDay.getUTCDate() - 1);
    const at = nyWallClockToDate(lastDay.getUTCFullYear(), lastDay.getUTCMonth() + 1, lastDay.getUTCDate(), CLOSE_HOUR, CLOSE_MINUTE);
    if (at > now) out.push(at);
  }
  return out;
}

export type Reminder = { id: string; at: Date; title: string; body: string };

/** Calendar reminders iOS can deliver with the app closed. They remind; they do not scan. */
export function closeReminders(prefs: AlertPrefs, now: Date): Reminder[] {
  const out: Reminder[] = [];
  if (prefs.weeklyCloseReminder) {
    for (const at of nextWeeklyCloses(now, 6)) {
      out.push({
        id: `reminder-weekly-${barPeriodKey('Weekly', at.toISOString().slice(0, 10))}`,
        at,
        title: 'Weekly bar closed',
        body: 'Open SV Screener and run a scan to confirm new signals and sell-to-close breaks.',
      });
    }
  }
  if (prefs.monthlyCloseReminder) {
    for (const at of nextMonthlyCloses(now, 3)) {
      out.push({
        id: `reminder-monthly-${at.toISOString().slice(0, 7)}`,
        at,
        title: 'Monthly bar closed',
        body: 'Open SV Screener and run a scan to confirm new signals and sell-to-close breaks.',
      });
    }
  }
  return out;
}
