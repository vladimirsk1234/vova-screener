import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { fundamentalsCatchUpKind, refreshProgressPct } from './fundamentals-catchup.ts';

describe('fundamentalsCatchUpKind', () => {
  const ready = {
    fmpConfigured: true,
    busy: false,
    universe: 100,
    complete: 100,
    pastEodSlot: true,
    todayFullDone: false,
  };

  it('waits when the universe list is still empty', () => {
    assert.equal(fundamentalsCatchUpKind({ ...ready, universe: 0, complete: 0 }), null);
  });

  it('fills missing names even before the EOD slot', () => {
    assert.equal(
      fundamentalsCatchUpKind({ ...ready, complete: 40, pastEodSlot: false }),
      'missing',
    );
  });

  it('runs a full pull after the weekday EOD slot when today is not done', () => {
    assert.equal(fundamentalsCatchUpKind(ready), 'full');
  });

  it('skips when today already has a post-close full pull and coverage is complete', () => {
    assert.equal(fundamentalsCatchUpKind({ ...ready, todayFullDone: true }), null);
  });

  it('does not start a second job while one is running', () => {
    assert.equal(fundamentalsCatchUpKind({ ...ready, busy: true, complete: 10 }), null);
  });
});

describe('refreshProgressPct', () => {
  it('uses the running job when present', () => {
    assert.equal(
      refreshProgressPct({
        run: { status: 'running', done: 25, total: 100 },
        coverage: { complete: 1, universe: 100 },
      }),
      25,
    );
  });

  it('falls back to coverage when idle', () => {
    assert.equal(
      refreshProgressPct({
        run: { status: 'completed', done: 100, total: 100 },
        coverage: { complete: 40, universe: 80 },
      }),
      50,
    );
  });
});
