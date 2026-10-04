import {
  DEFAULT_SETTINGS,
  parseUniverse,
  resizeActive,
  scanUniverse,
  type AppSettings,
  type ScanProgress,
  type ScreenerStore,
  type Universe,
  type UserTf,
} from '@vova/device';
import type { ParsedEntry } from '@vova/engine';
import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { openPhoneStore } from './db';
import { ETF_TEXT, STOCKS_TEXT } from './universeData';

type ScanState = ScanProgress & { label: string };

type ScreenerContextValue = {
  ready: boolean;
  error: string | null;
  store: ScreenerStore | null;
  settings: AppSettings;
  rev: number;
  universe: Record<Universe, ParsedEntry[]>;
  scan: ScanState | null;
  refresh: () => void;
  saveSettings: (patch: Partial<AppSettings>) => Promise<void>;
  runScan: (tf: UserTf | 'all') => Promise<void>;
  cancelScan: () => void;
};

const Ctx = createContext<ScreenerContextValue | null>(null);

export function useScreener(): ScreenerContextValue {
  const value = useContext(Ctx);
  if (!value) throw new Error('Screener context missing');
  return value;
}

export function ScreenerProvider({ children }: { children: ReactNode }) {
  const [store, setStore] = useState<ScreenerStore | null>(null);
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rev, setRev] = useState(0);
  const [scan, setScan] = useState<ScanState | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const universe = useMemo(() => parseUniverse(STOCKS_TEXT, ETF_TEXT), []);

  useEffect(() => {
    let cancelled = false;
    openPhoneStore()
      .then(async (opened) => {
        if (cancelled) return;
        setStore(opened);
        setSettings(await opened.getSettings());
        setReady(true);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const refresh = () => setRev((n) => n + 1);

  const saveSettings = async (patch: Partial<AppSettings>) => {
    if (!store) return;
    const next = { ...settings, ...patch };
    if (!(next.maxRiskUsd > 0) || !(next.minRr >= 0)) return;
    await store.putSettings(next);
    if (next.maxRiskUsd !== settings.maxRiskUsd) {
      const signals = await store.listSignals();
      await store.saveSignals(resizeActive(signals, next.maxRiskUsd));
    }
    setSettings(next);
    refresh();
  };

  const runScan = async (tf: UserTf | 'all') => {
    if (!store || scan?.phase === 'running') return;
    const controller = new AbortController();
    abortRef.current = controller;
    const frames: UserTf[] = tf === 'all' ? ['Weekly', 'Monthly'] : [tf];
    try {
      for (const frame of frames) {
        for (const name of ['Stocks', 'ETF'] as const) {
          if (controller.signal.aborted) break;
          setScan({
            phase: 'running',
            evaluated: 0,
            total: universe[name].length,
            signals: 0,
            closes: 0,
            rejected: 0,
            message: `Starting ${name} ${frame}`,
            label: `${name} ${frame === 'Weekly' ? 'W' : 'M'}`,
          });
          const result = await scanUniverse({
            store,
            entries: universe[name],
            universe: name,
            tf: frame,
            settings,
            signal: controller.signal,
            concurrency: 4,
            onProgress: (progress) => setScan({ ...progress, label: `${name} ${frame === 'Weekly' ? 'W' : 'M'}` }),
          });
          if (result.phase === 'failed') {
            setScan({ ...result, label: `${name} ${frame}` });
            return;
          }
        }
      }
      refresh();
    } finally {
      abortRef.current = null;
    }
  };

  const cancelScan = () => abortRef.current?.abort();

  const value: ScreenerContextValue = {
    ready,
    error,
    store,
    settings,
    rev,
    universe,
    scan,
    refresh,
    saveSettings,
    runScan,
    cancelScan,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
