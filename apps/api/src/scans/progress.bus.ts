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

const TERMINAL: ScanProgressEvent['phase'][] = ['completed', 'cancelled', 'failed'];

@Injectable()
export class ProgressBus {
  private readonly streams = new Map<string, Subject<ScanProgressEvent>>();
  private readonly last = new Map<string, ScanProgressEvent>();
  private readonly closeTimers = new Map<string, ReturnType<typeof setTimeout>>();

  private subject(runId: string) {
    let s = this.streams.get(runId);
    if (!s) {
      s = new Subject<ScanProgressEvent>();
      this.streams.set(runId, s);
    }
    return s;
  }

  private cancelClose(runId: string) {
    const timer = this.closeTimers.get(runId);
    if (!timer) return;
    clearTimeout(timer);
    this.closeTimers.delete(runId);
  }

  private closeStream(runId: string) {
    this.streams.get(runId)?.complete();
    this.streams.delete(runId);
  }

  /** Drop snapshot and live subscribers so a reused runId can start a fresh pass. */
  reset(runId: string) {
    this.cancelClose(runId);
    this.last.delete(runId);
    this.closeStream(runId);
  }

  publish(event: ScanProgressEvent) {
    this.cancelClose(event.runId);
    this.last.set(event.runId, event);
    this.subject(event.runId).next(event);
    if (TERMINAL.includes(event.phase)) {
      const timer = setTimeout(() => {
        this.closeTimers.delete(event.runId);
        this.closeStream(event.runId);
      }, 1_000);
      this.closeTimers.set(event.runId, timer);
    }
  }

  snapshot(runId: string) {
    return this.last.get(runId);
  }

  stream(runId: string): Observable<ScanProgressEvent> {
    return this.subject(runId).asObservable();
  }
}
