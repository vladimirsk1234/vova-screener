import type { StructureSnapshot } from './sequenceVovaFull';

export function emojiState(state: number): string {
  if (state === 1) return '🟢';
  if (state === -1) return '🔴';
  return '🟡';
}

export function seqDisplay(snap: StructureSnapshot, smaMajorAbove: boolean | null = null): number {
  const crit = snap.critical_level;
  const close = snap.close;
  const seq = snap.seq_state;
  if (crit != null && close != null) {
    if (close > crit) return 1;
    if (close < crit) return -1;
  }
  if (crit == null && smaMajorAbove != null) return smaMajorAbove ? 1 : -1;
  return seq;
}

export function structDisplay(
  snap: StructureSnapshot,
  smaAbove: boolean | null = null,
): [string, string] {
  const invalid = snap.struct_invalid;
  const troughHl = snap.last_trough_was_hl;
  const peakHh = snap.last_peak_was_hh;
  const lastPeak = snap.last_peak;
  const close = snap.close;
  const seq = snap.seq_state;
  const lastLh = snap.last_lh;
  const seqHigh = snap.seq_high;

  if (close != null && lastPeak != null) {
    const hasHl = troughHl && !invalid;
    const newHighAboveLh =
      seq === 1 && lastLh != null && seqHigh != null && seqHigh > lastLh;
    const green = hasHl && (peakHh || newHighAboveLh);
    const yellow = hasHl && !green;
    if (green) return ['🟢', ' (HL+HH)'];
    if (yellow) return ['🟡', ' (HL)'];
    return ['🔴', ''];
  }
  if (smaAbove) return ['🟡', ''];
  return ['🔴', ''];
}

/** Compact Seq/Struct for result cards — same rules as the chart watermark. */
export type SeqStructStatus = {
  seq: number;
  seqEmoji: string;
  structEmoji: string;
  /** `(HL+HH)`, `(HL)`, or empty. */
  structLabel: string;
};

export function seqStructStatus(
  snap: StructureSnapshot,
  smaMajorAbove: boolean | null = null,
): SeqStructStatus {
  const seq = seqDisplay(snap, smaMajorAbove);
  const [structEmoji, structLabel] = structDisplay(snap, smaMajorAbove);
  return { seq, seqEmoji: emojiState(seq), structEmoji, structLabel };
}
