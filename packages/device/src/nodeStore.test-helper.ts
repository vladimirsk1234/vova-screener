/** Node's built-in sqlite for tests. Never import this from app code — Metro cannot bundle node:sqlite. */
import { DatabaseSync } from 'node:sqlite';
import { createSqlStore, migrate, type ScreenerStore, type SqlAsync } from './store';

/** Fields the iPhone app must never persist or render. */
export const FMP_FIELDS = [
  'epsAtEntry',
  'epsPositiveAtEntry',
  'premiumPctAtEntry',
  'undervaluedAtEntry',
  'fairValue',
  'peTTM',
  'pegTTM',
] as const;

export function assertNoFundamentalFields(value: unknown): void {
  const text = JSON.stringify(value);
  for (const field of FMP_FIELDS) {
    if (text.includes(`"${field}"`)) {
      throw new Error(`FMP field ${field} leaked into device data`);
    }
  }
}

export async function openNodeStore(filename: string): Promise<ScreenerStore> {
  const db = new DatabaseSync(filename);
  const sql: SqlAsync = {
    async exec(sqlText) {
      db.exec(sqlText);
    },
    async run(sqlText, params = []) {
      db.prepare(sqlText).run(...params);
    },
    async all<T>(sqlText: string, params: readonly (string | number | null)[] = []) {
      return db.prepare(sqlText).all(...params) as T[];
    },
    async get<T>(sqlText: string, params: readonly (string | number | null)[] = []) {
      return (db.prepare(sqlText).get(...params) as T | undefined) ?? null;
    },
  };
  await migrate(sql);
  return createSqlStore(sql);
}
