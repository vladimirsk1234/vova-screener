import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  computePeakCapital,
  roiOnAvgPct,
  roiOnPeakPct,
  type CapitalTrade,
} from './peak-capital.ts';

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
      avgCapitalUsd: 0,
      windowFrom: null,
      windowTo: null,
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

describe('avgCapitalUsd', () => {
  it('forward-fills between event days so idle stretches pull the mean down', () => {
    const peak = computePeakCapital(
      [
        trade('a', '2026-01-01', '2026-01-05', 100_000),
        trade('b', '2026-01-20', '2026-01-25', 100_000),
      ],
      { rangeFrom: '2026-01-01', rangeEnd: '2026-01-31' },
    );
    // 9 days at 100k (Jan 1–4 and 20–24), 22 idle/close days at 0.
    assert.equal(peak.avgCapitalUsd, 29_032.26);
    assert.equal(peak.windowFrom, '2026-01-01');
    assert.equal(peak.windowTo, '2026-01-31');
    assert.equal(peak.peakCapitalUsd, 100_000);
  });

  it('includes already-open capital from rangeFrom and zeros after the last close', () => {
    const peak = computePeakCapital(
      [
        trade('long', '2025-06-01', '2026-03-01', 100_000),
        trade('ytd', '2026-02-01', '2026-04-01', 50_000),
        trade('before', '2025-01-01', '2025-12-15', 400_000),
      ],
      { rangeFrom: '2026-01-01', rangeEnd: '2026-09-04' },
    );
    // Jan 31d@100k + Feb 28d@150k + Mar 31d@50k + 157 idle days @0 = 8_850_000 / 247.
    assert.equal(peak.avgCapitalUsd, 35_829.96);
    assert.equal(peak.peakCapitalUsd, 150_000);
    assert.equal(peak.openCapitalUsd, 0);
  });

  it('keeps still-open size through rangeEnd after the last event', () => {
    const peak = computePeakCapital(
      [
        trade('open', '2026-01-01', null, 100_000),
        trade('closed', '2026-01-05', '2026-01-10', 40_000),
      ],
      { rangeFrom: '2026-01-01', rangeEnd: '2026-01-20' },
    );
    // Jan 1–4 @100k, 5–9 @140k, 10–20 @100k → (4*100 + 5*140 + 11*100) / 20.
    assert.equal(peak.avgCapitalUsd, 110_000);
    assert.equal(peak.openCapitalUsd, 100_000);
  });

  it('is the event-window mean when History range is all (no rangeFrom)', () => {
    const peak = computePeakCapital([
      trade('a', '2026-01-01', '2026-01-10', 100_000),
      trade('b', '2026-01-10', '2026-01-20', 80_000),
    ]);
    // Jan 1–9 @100k, 10–19 @80k, 20 @0 → 1_700_000 / 20.
    assert.equal(peak.avgCapitalUsd, 85_000);
    assert.equal(peak.windowFrom, '2026-01-01');
    assert.equal(peak.windowTo, '2026-01-20');
  });

  it('counts a same-day round-trip as zero EOD capital that day', () => {
    const peak = computePeakCapital([trade('round', '2026-01-10', '2026-01-10', 25_000)]);
    assert.equal(peak.peakCapitalUsd, 25_000);
    assert.equal(peak.avgCapitalUsd, 0);
  });
});

describe('roiOnPeakPct / roiOnAvgPct', () => {
  it('is closed P&L divided by the pool, as a percent', () => {
    assert.equal(roiOnPeakPct(70_000, 700_000), 10);
    assert.equal(roiOnPeakPct(60_000, 300_000), 20);
    assert.equal(roiOnAvgPct(70_000, 350_000), 20);
  });

  it('is null when there is no capital pool', () => {
    assert.equal(roiOnPeakPct(100, 0), null);
    assert.equal(roiOnPeakPct(100, -1), null);
    assert.equal(roiOnAvgPct(100, 0), null);
  });
});
