import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import type { StructureSnapshot } from './sequenceVovaFull.ts';
import { seqStructStatus } from './seqStruct.ts';

function snap(over: Partial<StructureSnapshot>): StructureSnapshot {
  return {
    seq_state: 1,
    critical_level: 10,
    close: 12,
    last_peak_was_hh: true,
    last_trough_was_hl: true,
    last_peak: 14,
    last_trough: 8,
    last_lh: 9,
    seq_high: 13,
    struct_invalid: false,
    ...over,
  };
}

describe('seqStructStatus', () => {
  it('marks HL+HH green when trough is HL and peak is HH', () => {
    const got = seqStructStatus(snap({}), true);
    assert.equal(got.seq, 1);
    assert.equal(got.seqEmoji, '🟢');
    assert.equal(got.structEmoji, '🟢');
    assert.equal(got.structLabel, ' (HL+HH)');
  });

  it('marks HL yellow without HH', () => {
    const got = seqStructStatus(
      snap({ last_peak_was_hh: false, last_lh: 20, seq_high: 15 }),
    );
    assert.equal(got.structEmoji, '🟡');
    assert.equal(got.structLabel, ' (HL)');
  });

  it('marks seq red when close is below critical', () => {
    const got = seqStructStatus(snap({ close: 8, critical_level: 10 }));
    assert.equal(got.seq, -1);
    assert.equal(got.seqEmoji, '🔴');
  });
});
