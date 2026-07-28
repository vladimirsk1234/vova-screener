import React, { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react';
import type {
  OhlcCacheEntry,
  RejectedRow,
  ResultRow,
  ScanParams,
  ScanProgress,
} from '../types';
import { defaultChartParams, type IndicatorParams } from '../indicatorParams';
import { defaultScanParams, runScan } from '../scan/runScan';
import { journalNewBuySignals, saveScanRun } from '../db/journal';

type ScanContextValue = {
  params: ScanParams;
  setParams: React.Dispatch<React.SetStateAction<ScanParams>>;
  chartParams: IndicatorParams;
  setChartParams: React.Dispatch<React.SetStateAction<IndicatorParams>>;
  results: ResultRow[];
  rejected: RejectedRow[];
  asOf: string | null;
  ohlcCache: Record<string, OhlcCacheEntry>;
  progress: ScanProgress;
  scanning: boolean;
  lastRunId: number | null;
  startScan: () => Promise<void>;
  stopScan: () => void;
  selectedSymbol: string | null;
  setSelectedSymbol: (s: string | null) => void;
};

const ScanContext = createContext<ScanContextValue | null>(null);

export function ScanProvider({ children }: { children: React.ReactNode }) {
  const [params, setParams] = useState<ScanParams>(defaultScanParams);
  const [chartParams, setChartParams] = useState<IndicatorParams>(defaultChartParams);
  const [results, setResults] = useState<ResultRow[]>([]);
  const [rejected, setRejected] = useState<RejectedRow[]>([]);
  const [asOf, setAsOf] = useState<string | null>(null);
  const [ohlcCache, setOhlcCache] = useState<Record<string, OhlcCacheEntry>>({});
  const [progress, setProgress] = useState<ScanProgress>({
    phase: 'idle',
    downloadPct: 0,
    processPct: 0,
    message: '',
  });
  const [scanning, setScanning] = useState(false);
  const [lastRunId, setLastRunId] = useState<number | null>(null);
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const stopScan = useCallback(() => {
    abortRef.current?.abort();
    setScanning(false);
    setProgress((p) => ({ ...p, phase: 'cancelled', message: 'Stopped' }));
  }, []);

  const startScan = useCallback(async () => {
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setScanning(true);
    setResults([]);
    setRejected([]);
    setSelectedSymbol(null);
    setProgress({
      phase: 'download',
      downloadPct: 0,
      processPct: 0,
      message: 'Starting…',
    });
    try {
      const outcome = await runScan(params, {
        signal: ac.signal,
        onProgress: (p) => {
          setProgress((prev) => ({
            phase: p.phase,
            downloadPct: p.phase === 'download' ? p.pct : prev.downloadPct,
            processPct: p.phase === 'process' ? p.pct : prev.processPct,
            message: p.message,
          }));
        },
      });
      if (outcome.cancelled) {
        setProgress((p) => ({ ...p, phase: 'cancelled', message: 'Cancelled' }));
        return;
      }
      setResults(outcome.rows);
      setRejected(outcome.rejected);
      setAsOf(outcome.asOf);
      setOhlcCache(outcome.ohlcCache);
      const runId = await saveScanRun(params, outcome.rows, outcome.asOf);
      setLastRunId(runId);
      await journalNewBuySignals(outcome.rows, params, outcome.asOf);
      setProgress({
        phase: 'done',
        downloadPct: 100,
        processPct: 100,
        message: `Done — ${outcome.rows.length} rows`,
      });
    } catch (e) {
      setProgress({
        phase: 'idle',
        downloadPct: 0,
        processPct: 0,
        message: e instanceof Error ? e.message : String(e),
      });
    } finally {
      setScanning(false);
    }
  }, [params]);

  const value = useMemo(
    () => ({
      params,
      setParams,
      chartParams,
      setChartParams,
      results,
      rejected,
      asOf,
      ohlcCache,
      progress,
      scanning,
      lastRunId,
      startScan,
      stopScan,
      selectedSymbol,
      setSelectedSymbol,
    }),
    [
      params,
      chartParams,
      results,
      rejected,
      asOf,
      ohlcCache,
      progress,
      scanning,
      lastRunId,
      startScan,
      stopScan,
      selectedSymbol,
    ],
  );

  return <ScanContext.Provider value={value}>{children}</ScanContext.Provider>;
}

export function useScan() {
  const ctx = useContext(ScanContext);
  if (!ctx) throw new Error('useScan outside provider');
  return ctx;
}
