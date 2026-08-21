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
    completedPassToday: false,
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

  it('does not restart missing after a pass already finished today', () => {
    assert.equal(
      fundamentalsCatchUpKind({
        ...ready,
        complete: 40,
        pastEodSlot: false,
        completedPassToday: true,
      }),
      null,
    );
  });

  it('still runs evening full after a morning missing pass', () => {
    assert.equal(
      fundamentalsCatchUpKind({
        ...ready,
        complete: 40,
        completedPassToday: true,
        todayFullDone: false,
        pastEodSlot: true,
      }),
      'full',
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
  it('uses universe coverage so a remaining-batch job does not look stuck at 2%', () => {
    assert.equal(
      refreshProgressPct({
        run: { status: 'running', done: 20, total: 827 },
        coverage: { complete: 2025, universe: 2095 },
      }),
      97,
    );
  });

  it('falls back to the running job when coverage is empty', () => {
    assert.equal(
      refreshProgressPct({
        run: { status: 'running', done: 25, total: 100 },
        coverage: { complete: 0, universe: 0 },
      }),
      25,
    );
  });
});
