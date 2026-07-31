/** Progress fan-out: worker publishes, SSE relays (Observer pattern per design). */
import { Injectable } from '@nestjs/common';
import { Subject, type Observable } from 'rxjs';

export type ScanProgressEvent = {
  runId: string;
  phase: 'queued' | 'resolving' | 'scanning' | 'saving' | 'completed' | 'cancelled' | 'failed';
  percent: number;
  message: string;
  counters?: Record<string, number>;
};

@Injectable()
export class ProgressBus {
  private readonly streams = new Map<string, Subject<ScanProgressEvent>>();
  private readonly last = new Map<string, ScanProgressEvent>();

  private subject(runId: string) {
    let s = this.streams.get(runId);
    if (!s) {
      s = new Subject<ScanProgressEvent>();
      this.streams.set(runId, s);
    }
    return s;
  }

  publish(event: ScanProgressEvent) {
    this.last.set(event.runId, event);
    this.subject(event.runId).next(event);
    if (['completed', 'cancelled', 'failed'].includes(event.phase)) {
      setTimeout(() => {
        this.streams.get(event.runId)?.complete();
        this.streams.delete(event.runId);
      }, 1_000);
    }
  }

  snapshot(runId: string) {
    return this.last.get(runId);
  }

  stream(runId: string): Observable<ScanProgressEvent> {
    return this.subject(runId).asObservable();
  }
}
