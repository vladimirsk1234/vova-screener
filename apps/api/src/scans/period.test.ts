import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  FUNDAMENTALS_EOD_MINUTES,
  isFullRunAfterTodaysClose,
  isPastFundamentalsEodSlot,
  nyTimeMinutes,
  partsInNy,
} from './period.ts';

/** Build a Date that is `minutes` after NY midnight on the given NY calendar date. */
function atNyMinutes(dateStr: string, minutes: number): Date {
  const hour = Math.floor(minutes / 60);
  const minute = minutes % 60;
  // Fixed offset America/New_York without DST gymnastics for unit tests: use noon UTC
  // probes via Intl round-trip from a known US Eastern instant.
  const probe = new Date(`${dateStr}T12:00:00.000Z`);
  const noonNy = nyTimeMinutes(probe);
  const deltaMin = minutes - noonNy;
  return new Date(probe.getTime() + deltaMin * 60_000);
}

describe('isPastFundamentalsEodSlot', () => {
  it('is false on weekends', () => {
    // 2026-08-22 is Saturday
    const sat = atNyMinutes('2026-08-22', FUNDAMENTALS_EOD_MINUTES + 30);
    assert.equal(partsInNy(sat).weekday, 6);
    assert.equal(isPastFundamentalsEodSlot(sat), false);
  });

  it('is false before 18:15 ET on a weekday', () => {
    // 2026-08-20 is Thursday
    const before = atNyMinutes('2026-08-20', FUNDAMENTALS_EOD_MINUTES - 1);
    assert.equal(partsInNy(before).weekday, 4);
    assert.equal(isPastFundamentalsEodSlot(before), false);
  });

  it('is true at/after 18:15 ET on a weekday', () => {
    const at = atNyMinutes('2026-08-20', FUNDAMENTALS_EOD_MINUTES);
    assert.equal(isPastFundamentalsEodSlot(at), true);
    const after = atNyMinutes('2026-08-20', FUNDAMENTALS_EOD_MINUTES + 45);
    assert.equal(isPastFundamentalsEodSlot(after), true);
  });
});

describe('isFullRunAfterTodaysClose', () => {
  it('is false when last run is missing', () => {
    assert.equal(isFullRunAfterTodaysClose(null, atNyMinutes('2026-08-20', 20 * 60)), false);
  });

  it('is false when last run is before today 16:00 ET', () => {
    const now = atNyMinutes('2026-08-20', 20 * 60);
    const morning = atNyMinutes('2026-08-20', 10 * 60);
    assert.equal(isFullRunAfterTodaysClose(morning, now), false);
  });

  it('is true when last run is after today 16:00 ET', () => {
    const now = atNyMinutes('2026-08-20', 20 * 60);
    const eod = atNyMinutes('2026-08-20', 18 * 60 + 30);
    assert.equal(isFullRunAfterTodaysClose(eod, now), true);
  });

  it('is false when last run was yesterday', () => {
    const now = atNyMinutes('2026-08-20', 20 * 60);
    const yesterday = atNyMinutes('2026-08-19', 19 * 60);
    assert.equal(isFullRunAfterTodaysClose(yesterday, now), false);
  });
});
