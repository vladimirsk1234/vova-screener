import { useEffect, useState } from 'react';
import type { ScanProgressEvent } from './api';

const TERMINAL = ['completed', 'cancelled', 'failed'];

/** Subscribes to the API's SSE progress stream for one run. */
export function useScanProgress(runId: string | null, generation = 0) {
  const [event, setEvent] = useState<ScanProgressEvent | null>(null);

  useEffect(() => {
    if (!runId) {
      setEvent(null);
      return;
    }
    setEvent(null);
    const source = new EventSource(`/api/scans/${runId}/events`);
    source.onmessage = (msg) => {
      try {
        const parsed = JSON.parse(msg.data) as ScanProgressEvent;
        setEvent(parsed);
        if (TERMINAL.includes(parsed.phase)) source.close();
      } catch {
        /* ignore malformed frame */
      }
    };
    source.onerror = () => source.close();
    return () => source.close();
  }, [runId, generation]);

  return event;
}
