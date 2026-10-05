/**
 * In-memory layer over the SQLite store. Everything a screen reads is loaded from disk once per
 * app launch, so moving between Results, History and the chart never waits on SQLite or Yahoo.
 * Writes reach disk first and only then the cache: a Save that fails on disk must not look saved
 * for the rest of the session and vanish on the next launch.
 */
import type { BarsTf, ScreenerStore } from './store';
import type { AppSettings, BenchmarkCache, CachedBars, ScanMeta, TrackedSignal, Universe, UserTf } from './types';

const BAR_CACHE_LIMIT = 600;

export function createCachedStore(disk: ScreenerStore): ScreenerStore {
  let settings: AppSettings | null = null;
  let signals: TrackedSignal[] | null = null;
  let benchmark: BenchmarkCache | null | undefined;
  const metas = new Map<string, ScanMeta | null>();
  const presets = new Map<string, unknown>();
  const bars = new Map<string, CachedBars | null>();

  const barKey = (ticker: string, tf: BarsTf) => `${ticker}|${tf}`;
  const remember = (key: string, row: CachedBars | null) => {
    bars.delete(key);
    bars.set(key, row);
    if (bars.size > BAR_CACHE_LIMIT) {
      const oldest = bars.keys().next().value;
      if (oldest !== undefined) bars.delete(oldest);
    }
  };

  return {
    async getSettings() {
      settings ??= await disk.getSettings();
      return settings;
    },
    async putSettings(next) {
      await disk.putSettings(next);
      settings = next;
    },
    async getBars(ticker, tf) {
      const key = barKey(ticker, tf);
      if (bars.has(key)) {
        const hit = bars.get(key) ?? null;
        remember(key, hit);
        return hit;
      }
      const row = await disk.getBars(ticker, tf);
      remember(key, row);
      return row;
    },
    async putBars(row) {
      await disk.putBars(row);
      remember(barKey(row.yahooTicker, row.tf), row);
    },
    listBarKeys: () => disk.listBarKeys(),
    async listSignals() {
      signals ??= await disk.listSignals();
      return signals;
    },
    async saveSignals(rows) {
      await disk.saveSignals(rows);
      signals = rows.slice();
    },
    async upsertSignals(rows) {
      await disk.upsertSignals(rows);
      const current = signals ?? (await disk.listSignals());
      const byId = new Map(rows.map((row) => [row.id, row]));
      const known = new Set(current.map((row) => row.id));
      signals = [...current.map((row) => byId.get(row.id) ?? row), ...rows.filter((row) => !known.has(row.id))];
    },
    async getScanMeta(universe: Universe, tf: UserTf) {
      const key = `${universe}|${tf}`;
      if (!metas.has(key)) metas.set(key, await disk.getScanMeta(universe, tf));
      return metas.get(key) ?? null;
    },
    async putScanMeta(universe, tf, meta) {
      await disk.putScanMeta(universe, tf, meta);
      metas.set(`${universe}|${tf}`, meta);
    },
    async clearScanMeta() {
      await disk.clearScanMeta();
      metas.clear();
    },
    async getBenchmark() {
      if (benchmark === undefined) benchmark = await disk.getBenchmark();
      return benchmark;
    },
    async putBenchmark(row) {
      await disk.putBenchmark(row);
      benchmark = row;
    },
    async getPreset<T>(key: string) {
      if (!presets.has(key)) presets.set(key, await disk.getPreset(key));
      return (presets.get(key) ?? null) as T | null;
    },
    async putPreset(key, data) {
      await disk.putPreset(key, data);
      presets.set(key, data);
    },
  };
}
