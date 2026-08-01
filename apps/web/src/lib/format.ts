/** Display helpers shared by Results and History. */
import type { HistoryTf, Timeframe } from './api';

export function money(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—';
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export function signedMoney(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—';
  return `${n >= 0 ? '+' : ''}${money(n)}`;
}

export function num(n: number | null | undefined, digits = 2): string {
  if (n == null || !Number.isFinite(n)) return '—';
  return n.toFixed(digits);
}

export function pct(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—';
  return `${n >= 0 ? '+' : ''}${n.toFixed(2)}%`;
}

/** `YYYY-MM-DD` / `YYYY-Www` / `YYYY-MM` rendered for the selected granularity. */
export function periodLabel(periodKey: string | null | undefined, tf: Timeframe | HistoryTf): string {
  if (!periodKey || periodKey === 'unknown') return periodKey || '—';
  if (tf === 'Monthly') {
    const [y, m] = periodKey.split('-');
    if (!y || !m) return periodKey;
    return new Date(Number(y), Number(m) - 1, 1).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'long',
    });
  }
  if (/^\d{4}-W\d{2}$/.test(periodKey)) return periodKey.replace('-W', ' · week ');
  if (/^\d{4}-\d{2}-\d{2}$/.test(periodKey)) {
    return new Date(`${periodKey}T12:00:00`).toLocaleDateString(undefined, {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
    });
  }
  return periodKey;
}

/**
 * Age of the data behind a screen. A scan evaluates a stored bar snapshot, TradingView draws
 * the live in-progress bar, so the age of the snapshot explains most disagreements.
 */
export function formatAge(iso: string | null | undefined, now = Date.now()): string | null {
  if (!iso) return null;
  const ts = new Date(iso).getTime();
  if (!Number.isFinite(ts)) return null;
  const minutes = Math.max(0, Math.round((now - ts) / 60_000));
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

/**
 * How long a signal has been valid, in bars of its own timeframe — the number the NEW / VALID
 * split is made on, so "1 bar" is the youngest thing VALID can hold.
 */
export function barsLabel(bars: number | null | undefined): string {
  if (bars == null || !Number.isFinite(bars)) return '—';
  if (bars === 0) return 'this bar';
  return `${bars} bar${bars === 1 ? '' : 's'}`;
}

export function holdLabel(tf: HistoryTf): string {
  if (tf === 'Daily') return 'days';
  if (tf === 'Weekly') return 'weeks';
  if (tf === 'Monthly') return 'months';
  return 'periods';
}

export const TF_SHORT: Record<Timeframe, string> = { Daily: 'D', Weekly: 'W', Monthly: 'M' };
