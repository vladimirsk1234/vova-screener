import { useEffect, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, type AlertPrefs } from '../lib/api';
import { Switch } from './Chips';

const PERMISSION_TEXT: Record<string, string> = {
  granted: 'Notifications are allowed.',
  denied: 'Notifications are off for SV Screener in iOS Settings — turn them on there to receive alerts.',
  undetermined: 'iOS will ask for permission when you save with an alert on.',
  unavailable: 'Notifications are not available here.',
};

/** iPhone app only: which local notifications to raise, written to the phone by Save. */
export function AlertSettings({ open }: { open: boolean }) {
  const queryClient = useQueryClient();
  const state = useQuery({ queryKey: ['device-alerts'], queryFn: api.deviceAlerts, enabled: open });
  const [draft, setDraft] = useState<AlertPrefs | null>(null);

  useEffect(() => {
    if (state.data && !draft) setDraft(state.data.prefs);
  }, [state.data, draft]);

  const save = useMutation({
    mutationFn: (prefs: AlertPrefs) => api.saveDeviceAlerts(prefs),
    onSuccess: (next) => {
      queryClient.setQueryData(['device-alerts'], next);
      setDraft(next.prefs);
    },
  });

  if (!draft) return null;
  const set = (key: keyof AlertPrefs) => (v: boolean) => setDraft({ ...draft, [key]: v });
  const dirty = JSON.stringify(draft) !== JSON.stringify(state.data?.prefs);
  const permission = save.data?.permission ?? state.data?.permission ?? 'undetermined';

  return (
    <>
      <span className="field-label">Alerts</span>
      <Switch label="New signals" checked={draft.newSignals} onChange={set('newSignals')} />
      <Switch label="Sell to close" checked={draft.sellToClose} onChange={set('sellToClose')} />
      <Switch label="Closing on the bar in progress" checked={draft.closing} onChange={set('closing')} />
      <Switch label="Scan finished" checked={draft.scanFinished} onChange={set('scanFinished')} />
      <Switch
        label="Weekly close reminder (Fri 16:15 ET)"
        checked={draft.weeklyCloseReminder}
        onChange={set('weeklyCloseReminder')}
      />
      <Switch
        label="Monthly close reminder (last weekday 16:15 ET)"
        checked={draft.monthlyCloseReminder}
        onChange={set('monthlyCloseReminder')}
      />
      <span className="field-label">Alert on</span>
      <Switch label="Stocks" checked={draft.stocks} onChange={set('stocks')} />
      <Switch label="ETF" checked={draft.etf} onChange={set('etf')} />
      <Switch label="Weekly" checked={draft.weekly} onChange={set('weekly')} />
      <Switch label="Monthly" checked={draft.monthly} onChange={set('monthly')} />
      <Switch label="Only symbols marked Interested" checked={draft.onlyInterested} onChange={set('onlyInterested')} />
      <div className="chart-settings-actions">
        <button
          type="button"
          className="btn-sm btn-accent"
          disabled={save.isPending || (!dirty && !save.isError)}
          onClick={() => save.mutate(draft)}
        >
          {save.isPending ? 'Saving…' : !dirty && save.isSuccess ? 'Saved' : 'Save alerts'}
        </button>
      </div>
      <p className="muted small">
        {PERMISSION_TEXT[permission]} New, sell-to-close, closing and scan-finished alerts come from a
        scan, and scans only run while the app is open — iOS pauses a closed app, so nothing is
        scanned or alerted then. The close reminders are scheduled with iOS and arrive even with the
        app closed; they remind you to open the app and scan, they do not scan on their own.
      </p>
      {save.error ? <p className="error">{(save.error as Error).message}</p> : null}
    </>
  );
}
