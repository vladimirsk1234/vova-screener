/** SQLite-shaped store. The phone and the Node tests share this schema. */
import type { OhlcBar } from '../../engine/src/types.ts';
import {
  DEFAULT_SETTINGS,
  type AppSettings,
  type BenchmarkCache,
  type CachedBars,
  type ScanMeta,
  type TrackedSignal,
  type Universe,
  type UserTf,
} from './types';

export type SqlAsync = {
  exec(sql: string): Promise<void>;
  run(sql: string, params?: readonly (string | number | null)[]): Promise<void>;
  all<T>(sql: string, params?: readonly (string | number | null)[]): Promise<T[]>;
  get<T>(sql: string, params?: readonly (string | number | null)[]): Promise<T | null>;
};

const SCHEMA = `
CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK (id = 1), json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS bars (
  yahoo_ticker TEXT NOT NULL,
  tf TEXT NOT NULL,
  json TEXT NOT NULL,
  PRIMARY KEY (yahoo_ticker, tf)
);
CREATE TABLE IF NOT EXISTS signals (
  id TEXT PRIMARY KEY,
  universe TEXT NOT NULL,
  tf TEXT NOT NULL,
  json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scan_meta (
  universe TEXT NOT NULL,
  tf TEXT NOT NULL,
  json TEXT NOT NULL,
  PRIMARY KEY (universe, tf)
);
CREATE TABLE IF NOT EXISTS benchmark (id INTEGER PRIMARY KEY CHECK (id = 1), json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS presets (key TEXT PRIMARY KEY, json TEXT NOT NULL);
`;

export async function migrate(db: SqlAsync): Promise<void> {
  await db.exec(SCHEMA);
}

export type BarsTf = UserTf | 'Daily';

export type ScreenerStore = {
  getSettings(): Promise<AppSettings>;
  putSettings(next: AppSettings): Promise<void>;
  getBars(ticker: string, tf: BarsTf): Promise<CachedBars | null>;
  putBars(row: CachedBars): Promise<void>;
  listBarKeys(): Promise<Array<{ yahooTicker: string; tf: BarsTf }>>;
  listSignals(): Promise<TrackedSignal[]>;
  saveSignals(rows: TrackedSignal[]): Promise<void>;
  upsertSignals(rows: TrackedSignal[]): Promise<void>;
  getScanMeta(universe: Universe, tf: UserTf): Promise<ScanMeta | null>;
  putScanMeta(universe: Universe, tf: UserTf, meta: ScanMeta): Promise<void>;
  clearScanMeta(): Promise<void>;
  getBenchmark(): Promise<BenchmarkCache | null>;
  putBenchmark(row: BenchmarkCache): Promise<void>;
  getPreset<T>(key: string): Promise<T | null>;
  putPreset(key: string, data: unknown): Promise<void>;
};

function parse<T>(json: string | null | undefined): T | null {
  if (!json) return null;
  return JSON.parse(json) as T;
}

export function createSqlStore(db: SqlAsync): ScreenerStore {
  return {
    async getSettings() {
      const row = await db.get<{ json: string }>('SELECT json FROM settings WHERE id = 1');
      return { ...DEFAULT_SETTINGS, ...(parse<Partial<AppSettings>>(row?.json) ?? {}) };
    },
    async putSettings(next) {
      await db.run(
        'INSERT INTO settings (id, json) VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET json = excluded.json',
        [JSON.stringify(next)],
      );
    },
    async getBars(ticker, tf) {
      const row = await db.get<{ json: string }>(
        'SELECT json FROM bars WHERE yahoo_ticker = ? AND tf = ?',
        [ticker, tf],
      );
      return parse<CachedBars>(row?.json);
    },
    async putBars(row) {
      await db.run(
        `INSERT INTO bars (yahoo_ticker, tf, json) VALUES (?, ?, ?)
         ON CONFLICT(yahoo_ticker, tf) DO UPDATE SET json = excluded.json`,
        [row.yahooTicker, row.tf, JSON.stringify(row)],
      );
    },
    async listBarKeys() {
      const rows = await db.all<{ yahoo_ticker: string; tf: string }>('SELECT yahoo_ticker, tf FROM bars');
      return rows.map((r) => ({ yahooTicker: r.yahoo_ticker, tf: r.tf as BarsTf }));
    },
    async listSignals() {
      const rows = await db.all<{ json: string }>('SELECT json FROM signals');
      return rows.map((r) => JSON.parse(r.json) as TrackedSignal);
    },
    async saveSignals(rows) {
      await db.exec('BEGIN');
      try {
        await db.run('DELETE FROM signals');
        for (const row of rows) {
          await db.run('INSERT INTO signals (id, universe, tf, json) VALUES (?, ?, ?, ?)', [
            row.id,
            row.universe,
            row.tf,
            JSON.stringify(row),
          ]);
        }
        await db.exec('COMMIT');
      } catch (err) {
        await db.exec('ROLLBACK');
        throw err;
      }
    },
    async upsertSignals(rows) {
      for (const row of rows) {
        await db.run(
          `INSERT INTO signals (id, universe, tf, json) VALUES (?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET universe = excluded.universe, tf = excluded.tf, json = excluded.json`,
          [row.id, row.universe, row.tf, JSON.stringify(row)],
        );
      }
    },
    async getScanMeta(universe, tf) {
      const row = await db.get<{ json: string }>(
        'SELECT json FROM scan_meta WHERE universe = ? AND tf = ?',
        [universe, tf],
      );
      return parse<ScanMeta>(row?.json);
    },
    async putScanMeta(universe, tf, meta) {
      await db.run(
        `INSERT INTO scan_meta (universe, tf, json) VALUES (?, ?, ?)
         ON CONFLICT(universe, tf) DO UPDATE SET json = excluded.json`,
        [universe, tf, JSON.stringify(meta)],
      );
    },
    async clearScanMeta() {
      await db.run('DELETE FROM scan_meta');
    },
    async getBenchmark() {
      const row = await db.get<{ json: string }>('SELECT json FROM benchmark WHERE id = 1');
      return parse<BenchmarkCache>(row?.json);
    },
    async putBenchmark(row) {
      await db.run(
        'INSERT INTO benchmark (id, json) VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET json = excluded.json',
        [JSON.stringify(row)],
      );
    },
    async getPreset<T>(key: string) {
      const row = await db.get<{ json: string }>('SELECT json FROM presets WHERE key = ?', [key]);
      return parse<T>(row?.json);
    },
    async putPreset(key, data) {
      await db.run(
        'INSERT INTO presets (key, json) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET json = excluded.json',
        [key, JSON.stringify(data ?? null)],
      );
    },
  };
}

export type { OhlcBar };
