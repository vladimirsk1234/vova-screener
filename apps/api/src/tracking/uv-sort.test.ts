import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { sortByUndervaluation } from './uv-sort.ts';

function row(symbol: string, yahooTicker = symbol) {
  return { symbol, yahooTicker };
}

describe('sortByUndervaluation', () => {
  it('orders −30, −5, +10, null with nulls last (asc = most undervalued first)', () => {
    const rows = [row('NULL'), row('PLUS'), row('DEEP'), row('MILD')];
    const cards = {
      DEEP: { bestPremiumPct: -30, epsPremiumPct: -30 },
      MILD: { bestPremiumPct: -5, epsPremiumPct: -5 },
      PLUS: { bestPremiumPct: 10, epsPremiumPct: 10 },
    };
    const asc = sortByUndervaluation(rows, cards, 'asc').map((r) => r.symbol);
    assert.deepEqual(asc, ['DEEP', 'MILD', 'PLUS', 'NULL']);
    const desc = sortByUndervaluation(rows, cards, 'desc').map((r) => r.symbol);
    assert.deepEqual(desc, ['PLUS', 'MILD', 'DEEP', 'NULL']);
  });

  it('breaks a bestPremiumPct tie with epsPremiumPct then symbol', () => {
    const rows = [row('BBB'), row('AAA'), row('CCC')];
    const cards = {
      AAA: { bestPremiumPct: -10, epsPremiumPct: 4 },
      BBB: { bestPremiumPct: -10, epsPremiumPct: -2 },
      CCC: { bestPremiumPct: -10, epsPremiumPct: 4 },
    };
    assert.deepEqual(
      sortByUndervaluation(rows, cards, 'asc').map((r) => r.symbol),
      ['BBB', 'AAA', 'CCC'],
    );
  });

  it('falls back to epsPremiumPct when bestPremiumPct is missing', () => {
    const rows = [row('A'), row('B')];
    const cards = {
      A: { epsPremiumPct: -8 },
      B: { epsPremiumPct: -20 },
    };
    assert.deepEqual(
      sortByUndervaluation(rows, cards, 'asc').map((r) => r.symbol),
      ['B', 'A'],
    );
  });
});
