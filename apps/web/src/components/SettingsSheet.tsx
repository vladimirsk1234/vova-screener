import { useEffect, useMemo, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  TIMEFRAMES,
  UNIVERSES,
  api,
  type ResultsSummary,
  type Timeframe,
  type Universe,
} from '../lib/api';
import { formatAge, TF_SHORT } from '../lib/format';
import { Chips } from './Chips';

/** Every list the tracked universes feed, invalidated together when a scan changes them. */
const SCAN_FED = ['results', 'results-summary', 'history', 'history-trades', 'tracked-signal'];

type ScanTf = Timeframe | 'all';
const SCAN_TFS: readonly ScanTf[] = ['all', ...TIMEFRAMES];

/** The universe and timeframe a scan is on right now, or null when nothing is running. */
function scanning(data: ResultsSummary | undefined): { universe: Universe; tf: Timeframe } | null {
  if (!data) return null;
  for (const universe of UNIVERSES) {
    for (const tf of TIMEFRAMES) {
      if (data[universe]?.[tf]?.scan.running) return { universe, tf };
    }
  }
  return null;
}

/** Newest finish across every universe and timeframe — how fresh the screens are at best. */
function lastFinished(data: ResultsSummary | undefined): string | null {
  let newest: string | null = null;
  for (const universe of UNIVERSES) {
    for (const tf of TIMEFRAMES) {
      const at = data?.[universe]?.[tf]?.scan.finishedAt;
      if (at && (!newest || at > newest)) newest = at;
    }
  }
  return newest;
}

/**
 * The whole settings surface: one risk number, rescanning, and maintenance. Scan parameters are
 * fixed in the backend (buy signals, no RR floor), so there is nothing else to tune here.
 */
export function SettingsSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
  const queryClient = useQueryClient();
  const settings = useQuery({ queryKey: ['settings'], queryFn: api.settings, enabled: open });
  const [draft, setDraft] = useState('');
  const [scanTf, setScanTf] = useState<ScanTf>('all');
  // Queued, but the first run document may not exist yet — without this the button would flick
  // back to idle in the seconds between the request returning and the scan showing up.
  const [awaiting, setAwaiting] = useState(false);

  useEffect(() => {
    if (settings.data) setDraft(String(settings.data.maxRiskUsd));
  }, [settings.data]);

  // Shares its key with the Results header, so opening this sheet is also what speeds that up
  // while a scan is on: react-query polls at the shortest interval any observer asks for.
  const summary = useQuery({
    queryKey: ['results-summary'],
    queryFn: api.resultsSummary,
    enabled: open,
    refetchInterval: (query) => (awaiting || scanning(query.state.data) ? 5_000 : false),
  });
  const busy = scanning(summary.data);
  const wasBusy = useRef(false);

  // A finished pass has rewritten every list, and this poll is the only thing that knows.
  useEffect(() => {
    if (busy) {
      wasBusy.current = true;
      return;
    }
    if (!wasBusy.current) return;
    wasBusy.current = false;
    setAwaiting(false);
    for (const key of SCAN_FED) void queryClient.invalidateQueries({ queryKey: [key] });
  }, [busy, queryClient]);

  // Nothing ever started: background scanning is off, or the pass failed before writing a run.
  useEffect(() => {
    if (!awaiting) return;
    const timer = setTimeout(() => setAwaiting(false), 120_000);
    return () => clearTimeout(timer);
  }, [awaiting]);

  const rescan = useMutation({
    mutationFn: () => api.runScanNow(scanTf),
    onSuccess: (result) => {
      if (result.started) setAwaiting(true);
      void queryClient.invalidateQueries({ queryKey: ['results-summary'] });
    },
  });

  const running = awaiting || Boolean(busy);
  const scanAge = useMemo(() => formatAge(lastFinished(summary.data)), [summary.data]);

  const save = useMutation({
    mutationFn: (maxRiskUsd: number) => api.saveSettings({ maxRiskUsd }),
    onSuccess: (next) => {
      queryClient.setQueryData(['settings'], next);
      // The server re-sizes every open position before answering, so everything showing a share
      // count or an unrealized number is now stale.
      for (const key of ['results', 'results-summary', 'history', 'history-trades', 'tracked-signal', 'chart']) {
        void queryClient.invalidateQueries({ queryKey: [key] });
      }
    },
  });

  const reset = useMutation({
    mutationFn: api.resetHistory,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['results'] });
      void queryClient.invalidateQueries({ queryKey: ['results-summary'] });
      void queryClient.invalidateQueries({ queryKey: ['history'] });
      void queryClient.invalidateQueries({ queryKey: ['history-trades'] });
    },
  });

  if (!open) return null;

  const commit = () => {
    const next = Number(draft);
    if (!Number.isFinite(next) || next <= 0) {
      setDraft(String(settings.data?.maxRiskUsd ?? ''));
      return;
    }
    if (next === settings.data?.maxRiskUsd) return;
    save.mutate(next);
  };

  return (
    <>
      <div className="sheet-backdrop" onClick={onClose} aria-hidden />
      <div className="chart-settings-sheet" role="dialog" aria-label="Settings">
      <div className="chart-settings-head">
        <strong>Settings</strong>
        <button type="button" className="btn-sm ghost" onClick={onClose}>
          Close
        </button>
      </div>

      <div className="field">
        <label htmlFor="max-risk">Max risk per signal ($)</label>
        <input
          id="max-risk"
          type="number"
          inputMode="decimal"
          min={1}
          step={1}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
          }}
        />
      </div>
      <p className="muted small">
        One risk for every signal: position size is this divided by the distance to SL. Changing it
        re-sizes all open signals right away. Closed ones keep the size they were closed at.
      </p>

      <div className="field">
        <Chips
          label="Rescan"
          value={scanTf}
          options={SCAN_TFS}
          disabled={running || rescan.isPending}
          onChange={setScanTf}
          format={(v) => (v === 'all' ? 'All' : TF_SHORT[v])}
        />
      </div>
      <div className="chart-settings-actions">
        <button
          type="button"
          className="btn-sm"
          disabled={running || rescan.isPending}
          onClick={() => rescan.mutate()}
        >
          {running || rescan.isPending ? 'Scanning…' : 'Run scan now'}
        </button>
      </div>
      <p className="muted small">
        {busy
          ? `Scanning ${busy.universe} ${TF_SHORT[busy.tf]}. Stocks and ETF go one timeframe at a time, and the lists fill in as each finishes.`
          : running
            ? 'Starting. Stocks and ETF go one timeframe at a time, and the lists fill in as each finishes.'
            : 'Re-downloads every symbol in Stocks and ETF and rebuilds the lists from it. Scans otherwise run on their own, hourly through the session and once each period closes.'}
        {scanAge ? ` Last finished ${scanAge}.` : ' No scan has finished yet.'}
      </p>
      {rescan.data && !rescan.data.started ? (
        <p className="muted small">{rescan.data.reason ?? 'A scan is already running.'}</p>
      ) : null}
      {rescan.error ? <p className="error">{(rescan.error as Error).message}</p> : null}

      <div className="chart-settings-actions">
        <button
          type="button"
          className="btn-sm danger"
          disabled={reset.isPending}
          onClick={() => {
            if (!window.confirm('Delete every scan run and tracked signal? This cannot be undone.')) {
              return;
            }
            reset.mutate();
          }}
        >
          {reset.isPending ? 'Resetting…' : 'Reset all history'}
        </button>
      </div>
      {reset.data ? (
        <p className="muted small">
          Deleted {reset.data.deletedRuns} runs and {reset.data.deletedSignals} tracked signals.
        </p>
      ) : null}
        {save.error ? <p className="error">{(save.error as Error).message}</p> : null}
      </div>
    </>
  );
}
