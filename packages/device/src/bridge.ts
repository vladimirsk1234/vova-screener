/**
 * The native half of the WebView bridge and the expo-sqlite adapter, kept free of React Native
 * imports so tests run the exact code the app runs.
 */
import type { SqlAsync } from './store';

type BindValue = string | number | null;

/** The slice of expo-sqlite's `SQLiteDatabase` the store uses. */
export type ExpoSqliteLike = {
  execAsync(source: string): Promise<void>;
  runAsync(source: string, ...params: BindValue[]): Promise<unknown>;
  getAllAsync<T>(source: string, ...params: BindValue[]): Promise<T[]>;
  getFirstAsync<T>(source: string, ...params: BindValue[]): Promise<T | null>;
};

export function sqlFromExpo(db: ExpoSqliteLike): SqlAsync {
  return {
    exec: (statement) => db.execAsync(statement),
    run: async (statement, params = []) => {
      await db.runAsync(statement, ...params);
    },
    all: <T>(statement: string, params: readonly BindValue[] = []) => db.getAllAsync<T>(statement, ...params),
    get: async <T>(statement: string, params: readonly BindValue[] = []) =>
      (await db.getFirstAsync<T>(statement, ...params)) ?? null,
  };
}

export type BridgeApi = { handle(method: string, path: string, body?: string | null): Promise<{ status: number; body: string }> };

/**
 * One message from the WebView (`window.ReactNativeWebView.postMessage`) to the JavaScript the app
 * injects back. Returns null for anything that is not a bridge request.
 */
export async function bridgeReplyScript(api: BridgeApi, raw: string): Promise<string | null> {
  let msg: { id: number; method: string; path: string; body: string | null };
  try {
    msg = JSON.parse(raw);
  } catch {
    return null;
  }
  if (typeof msg?.id !== 'number' || typeof msg.path !== 'string') return null;
  const res = await api.handle(msg.method, msg.path, msg.body);
  return `window.__vovaReply && window.__vovaReply(${msg.id}, ${res.status}, ${JSON.stringify(res.body)}); true;`;
}
