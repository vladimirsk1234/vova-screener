import {
  createSqlStore,
  migrate,
  type ScreenerStore,
  type SqlAsync,
} from '@vova/device';
import * as SQLite from 'expo-sqlite';

/** Local SQLite file. It stays on the phone and is still there after a restart. */
export async function openPhoneStore(): Promise<ScreenerStore> {
  const db = await SQLite.openDatabaseAsync('sequence-vova.db');
  const sql: SqlAsync = {
    exec: (statement) => db.execAsync(statement),
    run: async (statement, params = []) => {
      await db.runAsync(statement, ...(params as SQLite.SQLiteBindValue[]));
    },
    all: (statement, params = []) => db.getAllAsync(statement, ...(params as SQLite.SQLiteBindValue[])),
    get: async (statement, params = []) =>
      (await db.getFirstAsync(statement, ...(params as SQLite.SQLiteBindValue[]))) ?? null,
  };
  await migrate(sql);
  return createSqlStore(sql);
}
