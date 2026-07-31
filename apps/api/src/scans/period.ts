/** Calendar period keys in America/New_York for Daily / Weekly / Monthly history slots. */
import type { Timeframe } from '@vova/engine';

export const MARKET_TZ = 'America/New_York';

export type NyDateParts = {
  year: number;
  month: number;
  day: number;
  /** YYYY-MM-DD in America/New_York */
  dateStr: string;
  /** 0=Sun … 6=Sat in America/New_York */
  weekday: number;
};

export function partsInNy(date: Date = new Date()): NyDateParts {
  const fmt = new Intl.DateTimeFormat('en-US', {
    timeZone: MARKET_TZ,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    weekday: 'short',
  });
  const parts = Object.fromEntries(fmt.formatToParts(date).map((p) => [p.type, p.value]));
  const year = Number(parts.year);
  const month = Number(parts.month);
  const day = Number(parts.day);
  const weekdayMap: Record<string, number> = {
    Sun: 0,
    Mon: 1,
    Tue: 2,
    Wed: 3,
    Thu: 4,
    Fri: 5,
    Sat: 6,
  };
  return {
    year,
    month,
    day,
    dateStr: `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`,
    weekday: weekdayMap[parts.weekday ?? 'Mon'] ?? 1,
  };
}

/** ISO week key: YYYY-Www (week starts Monday, ISO-8601). */
export function isoWeekKey(year: number, month: number, day: number): string {
  const utc = new Date(Date.UTC(year, month - 1, day));
  const dayNum = utc.getUTCDay() || 7;
  utc.setUTCDate(utc.getUTCDate() + 4 - dayNum);
  const isoYear = utc.getUTCFullYear();
  const yearStart = new Date(Date.UTC(isoYear, 0, 1));
  const week = Math.ceil(((utc.getTime() - yearStart.getTime()) / 86_400_000 + 1) / 7);
  return `${isoYear}-W${String(week).padStart(2, '0')}`;
}

export function periodKey(tf: Timeframe, date: Date = new Date()): string {
  const { year, month, day, dateStr } = partsInNy(date);
  if (tf === 'Daily') return dateStr;
  if (tf === 'Monthly') return `${year}-${String(month).padStart(2, '0')}`;
  return isoWeekKey(year, month, day);
}

/** True on the last Mon–Fri of the month in America/New_York. */
export function isLastTradingDayOfMonth(date: Date = new Date()): boolean {
  const today = partsInNy(date);
  if (today.weekday === 0 || today.weekday === 6) return false;

  let offset = 1;
  while (offset <= 7) {
    const probe = new Date(date.getTime() + offset * 86_400_000);
    const next = partsInNy(probe);
    if (next.weekday !== 0 && next.weekday !== 6) {
      return next.month !== today.month || next.year !== today.year;
    }
    offset += 1;
  }
  return false;
}
