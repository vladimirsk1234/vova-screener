import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { computePeakCapital, roiOnPeakPct, type CapitalTrade } from './peak-capital.ts';

function trade(
  id: string,
  openedAsOf: string | null,
  exitDate: string | null,
  positionValue: number,
): CapitalTrade {
  return { id, openedAsOf, exitDate, positionValue };
}

describe('computePeakCapital', () => {
  it('recycles same-day capital: a close funds a new open, so they do not stack', () => {
    const peak = computePeakCapital([
      trade('a', '2026-01-01', '2026-01-10', 100_000),
      trade('b', '2026-01-10', '2026-01-20', 80_000),
    ]);
    assert.equal(peak.peakCapitalUsd, 100_000);
    assert.equal(peak.peakCapitalAsOf, '2026-01-01');
    assert.equal(peak.peakConcurrentPositions, 1);
    assert.equal(peak.openCapitalUsd, 0);
  });

  it('sums overlapping positions', () => {
    const peak = computePeakCapital([
      trade('a', '2026-01-01', '2026-01-20', 100_000),
      trade('b', '2026-01-10', '2026-01-30', 50_000),
    ]);
    assert.equal(peak.peakCapitalUsd, 150_000);
    assert.equal(peak.peakCapitalAsOf, '2026-01-10');
    assert.equal(peak.peakConcurrentPositions, 2);
    assert.equal(peak.openCapitalUsd, 0);
  });

  it('keeps still-open trades in the curve until the window ends', () => {
    const peak = computePeakCapital([
      trade('open', '2026-01-01', null, 100_000),
      trade('closed', '2026-01-05', '2026-01-10', 40_000),
    ]);
    assert.equal(peak.peakCapitalUsd, 140_000);
    assert.equal(peak.peakCapitalAsOf, '2026-01-05');
    assert.equal(peak.peakConcurrentPositions, 2);
    assert.equal(peak.openCapitalUsd, 100_000);
  });

  it('clips already-open trades onto rangeFrom so the peak is inside the lookback', () => {
    const peak = computePeakCapital(
      [
        trade('long', '2025-06-01', '2026-03-01', 100_000),
        trade('ytd', '2026-02-01', '2026-04-01', 50_000),
        trade('before', '2025-01-01', '2025-12-15', 400_000),
      ],
      { rangeFrom: '2026-01-01', rangeEnd: '2026-09-04' },
    );
    assert.equal(peak.peakCapitalUsd, 150_000);
    assert.equal(peak.peakCapitalAsOf, '2026-02-01');
    assert.equal(peak.peakConcurrentPositions, 2);
    assert.equal(peak.openCapitalUsd, 0);
  });

  it('releases a pre-range holder on rangeFrom before booking that day\'s new opens', () => {
    const peak = computePeakCapital(
      [
        trade('old', '2025-12-01', '2026-01-01', 100_000),
        trade('fresh', '2026-01-01', '2026-02-01', 80_000),
      ],
      { rangeFrom: '2026-01-01' },
    );
    assert.equal(peak.peakCapitalUsd, 100_000);
    assert.equal(peak.peakCapitalAsOf, '2026-01-01');
    assert.equal(peak.peakConcurrentPositions, 1);
    assert.equal(peak.openCapitalUsd, 0);
  });

  it('counts a same-day open-and-close toward the peak', () => {
    const peak = computePeakCapital([trade('round', '2026-01-10', '2026-01-10', 25_000)]);
    assert.equal(peak.peakCapitalUsd, 25_000);
    assert.equal(peak.peakCapitalAsOf, '2026-01-10');
    assert.equal(peak.peakConcurrentPositions, 1);
    assert.equal(peak.openCapitalUsd, 0);
  });

  it('dedupes by trade id', () => {
    const peak = computePeakCapital([
      trade('a', '2026-01-01', '2026-01-10', 100_000),
      trade('a', '2026-01-01', '2026-01-10', 100_000),
    ]);
    assert.equal(peak.peakCapitalUsd, 100_000);
    assert.equal(peak.peakConcurrentPositions, 1);
  });

  it('ignores zero-size and undated rows', () => {
    const peak = computePeakCapital([
      trade('zero', '2026-01-01', null, 0),
      trade('nodate', null, null, 50_000),
    ]);
    assert.deepEqual(peak, {
      peakCapitalUsd: 0,
      peakCapitalAsOf: null,
      peakConcurrentPositions: 0,
      openCapitalUsd: 0,
    });
  });

  it('does not stack Weekly and Monthly — caller chooses the trade set', () => {
    const weekly = [
      trade('w1', '2026-01-01', '2026-02-01', 200_000),
      trade('w2', '2026-01-15', '2026-03-01', 100_000),
    ];
    const monthly = [trade('m1', '2026-01-01', '2026-06-01', 300_000)];
    assert.equal(computePeakCapital(weekly).peakCapitalUsd, 300_000);
    assert.equal(computePeakCapital(monthly).peakCapitalUsd, 300_000);
    assert.equal(computePeakCapital([...weekly, ...monthly]).peakCapitalUsd, 600_000);
  });
});

describe('roiOnPeakPct', () => {
  it('is closed P&L divided by peak, as a percent', () => {
    assert.equal(roiOnPeakPct(70_000, 700_000), 10);
    assert.equal(roiOnPeakPct(60_000, 300_000), 20);
  });

  it('is null when there is no capital pool', () => {
    assert.equal(roiOnPeakPct(100, 0), null);
    assert.equal(roiOnPeakPct(100, -1), null);
  });
});
