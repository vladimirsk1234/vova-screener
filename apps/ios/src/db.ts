import { createSqlStore, migrate, sqlFromExpo, type ExpoSqliteLike, type ScreenerStore } from '@vova/device';
import * as SQLite from 'expo-sqlite';

/**
 * `Documents/SQLite/sequence-vova.db` (expo-sqlite's default directory on iOS). It stays on the
 * phone across app updates and restarts; only deleting the app removes it.
 */
export async function openPhoneStore(): Promise<ScreenerStore> {
  const db = await SQLite.openDatabaseAsync('sequence-vova.db');
  const sql = sqlFromExpo(db as unknown as ExpoSqliteLike);
  await migrate(sql);
  return createSqlStore(sql);
}
