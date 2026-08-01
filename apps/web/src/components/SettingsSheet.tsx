import { useEffect, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '../lib/api';

/**
 * The whole settings surface: one risk number plus maintenance. Scan parameters are fixed in
 * the backend (buy signals, no RR floor), so there is nothing else to tune here.
 */
export function SettingsSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
  const queryClient = useQueryClient();
  const settings = useQuery({ queryKey: ['settings'], queryFn: api.settings, enabled: open });
  const [draft, setDraft] = useState('');

  useEffect(() => {
    if (settings.data) setDraft(String(settings.data.maxRiskUsd));
  }, [settings.data]);

  const save = useMutation({
    mutationFn: (maxRiskUsd: number) => api.saveSettings({ maxRiskUsd }),
    onSuccess: (next) => {
      queryClient.setQueryData(['settings'], next);
      void queryClient.invalidateQueries({ queryKey: ['results'] });
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
        Position size for every tracked signal is derived from this and the distance to SL. Changing
        it re-sizes active signals on the next background scan.
      </p>

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
