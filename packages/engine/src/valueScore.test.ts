import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  bestValuePremium,
  compareValueRows,
  interestRankOf,
  rowMatchesStarsFilter,
  scoreValueStars,
  VALUE_INTEREST_RANK,
} from './valueScore.ts';

describe('scoreValueStars', () => {
  it('counts FCF-only as 1/3, not 2', () => {
    const got = scoreValueStars({
      epsPremiumPct: 12,
      fcfPremiumPct: -8,
      dcfPremiumPct: 4,
    });
    assert.equal(got.stars, 1);
    assert.equal(got.epsUndervalued, false);
    assert.equal(got.fcfUndervalued, true);
    assert.equal(got.dcfUndervalued, false);
  });

  it('counts EPS-only as 1/3', () => {
    assert.equal(
      scoreValueStars({ epsPremiumPct: -3, fcfPremiumPct: 1, dcfPremiumPct: 0 }).stars,
      1,
    );
  });

  it('counts EPS+DCF as 2/3', () => {
    const got = scoreValueStars({
      epsPremiumPct: -10,
      fcfPremiumPct: 2,
      dcfPremiumPct: -1,
    });
    assert.equal(got.stars, 2);
    assert.equal(got.epsUndervalued, true);
    assert.equal(got.dcfUndervalued, true);
  });

  it('counts all three as 3/3', () => {
    assert.equal(
      scoreValueStars({ epsPremiumPct: -1, fcfPremiumPct: -2, dcfPremiumPct: -3 }).stars,
      3,
    );
  });

  it('does not count null or NaN as a star', () => {
    assert.equal(
      scoreValueStars({ epsPremiumPct: null, fcfPremiumPct: Number.NaN, dcfPremiumPct: undefined as unknown as null }).stars,
      0,
    );
    assert.equal(
      scoreValueStars({ epsPremiumPct: -4, fcfPremiumPct: null, dcfPremiumPct: null }).stars,
      1,
    );
  });

  it('treats exactly 0 premium as not undervalued', () => {
    assert.equal(
      scoreValueStars({ epsPremiumPct: 0, fcfPremiumPct: 0, dcfPremiumPct: 0 }).stars,
      0,
    );
  });
});

describe('bestValuePremium', () => {
  it('returns the most negative finite premium', () => {
    assert.equal(
      bestValuePremium({ epsPremiumPct: -5, fcfPremiumPct: -20, dcfPremiumPct: 3 }),
      -20,
    );
  });

  it('ignores nulls', () => {
    assert.equal(
      bestValuePremium({ epsPremiumPct: null, fcfPremiumPct: -1, dcfPremiumPct: null }),
      -1,
    );
  });
});

describe('rowMatchesStarsFilter', () => {
  it('undervalued is 1/3 and up; all includes 0/3; 0/1/2/3 are exact', () => {
    assert.equal(rowMatchesStarsFilter(0, 'undervalued'), false);
    assert.equal(rowMatchesStarsFilter(1, 'undervalued'), true);
    assert.equal(rowMatchesStarsFilter(0, 'all'), true);
    assert.equal(rowMatchesStarsFilter(0, '0'), true);
    assert.equal(rowMatchesStarsFilter(1, '0'), false);
    assert.equal(rowMatchesStarsFilter(2, '2'), true);
    assert.equal(rowMatchesStarsFilter(3, '2'), false);
  });
});

describe('compareValueRows', () => {
  const row = (
    symbol: string,
    stars: number,
    premia: {
      eps?: number | null;
      fcf?: number | null;
      dcf?: number | null;
      best?: number | null;
      rank?: number;
    },
  ) => ({
    symbol,
    stars,
    epsPremiumPct: premia.eps ?? null,
    fcfPremiumPct: premia.fcf ?? null,
    dcfPremiumPct: premia.dcf ?? null,
    bestPremiumPct: premia.best ?? null,
    interestRank: premia.rank ?? VALUE_INTEREST_RANK.none,
  });

  it('sorts 3/3 before 1/3 when stars desc', () => {
    const a = row('AAA', 1, { best: -50 });
    const b = row('BBB', 3, { best: -1 });
    const rows = [a, b].sort((x, y) => compareValueRows(x, y, 'stars', 'desc'));
    assert.equal(rows[0].symbol, 'BBB');
  });

  it('breaks equal stars with the more negative premium', () => {
    const a = row('AAA', 2, { best: -5 });
    const b = row('BBB', 2, { best: -30 });
    const rows = [a, b].sort((x, y) => compareValueRows(x, y, 'stars', 'desc'));
    assert.equal(rows[0].symbol, 'BBB');
  });

  it('filters undervalued then paginates 3/3 first', () => {
    const rows = [
      row('ZERO', 0, { best: -40 }),
      row('ONE', 1, { best: -10 }),
      row('TWO', 2, { best: -8 }),
      row('THREE', 3, { best: -2 }),
    ];
    const filtered = rows.filter((r) => rowMatchesStarsFilter(r.stars, 'undervalued'));
    filtered.sort((a, b) => compareValueRows(a, b, 'stars', 'desc'));
    assert.deepEqual(
      filtered.map((r) => r.symbol),
      ['THREE', 'TWO', 'ONE'],
    );
    assert.equal(filtered.slice(0, 2)[0].symbol, 'THREE');
    assert.equal(filtered.slice(0, 2).length, 2);
  });

  it('sorts interested above unmarked above not_interested when Marked desc', () => {
    const rows = [
      row('NO', 3, { rank: VALUE_INTEREST_RANK.not_interested }),
      row('YES', 1, { rank: VALUE_INTEREST_RANK.interested }),
      row('PLAIN', 2, { rank: VALUE_INTEREST_RANK.none }),
    ].sort((a, b) => compareValueRows(a, b, 'interest', 'desc'));
    assert.deepEqual(
      rows.map((r) => r.symbol),
      ['YES', 'PLAIN', 'NO'],
    );
  });

  it('treats a missing interestRank as unmarked', () => {
    const a = row('YES', 1, { rank: VALUE_INTEREST_RANK.interested });
    const b = { ...row('PLAIN', 3, {}), interestRank: undefined };
    const rows = [b, a].sort((x, y) => compareValueRows(x, y, 'interest', 'desc'));
    assert.equal(rows[0].symbol, 'YES');
    assert.equal(rows[1].symbol, 'PLAIN');
  });
});

describe('interestRankOf', () => {
  it('treats a cleared mark as unmarked (1)', () => {
    assert.equal(interestRankOf(null), 1);
    assert.equal(interestRankOf('interested'), 2);
    assert.equal(interestRankOf('not_interested'), 0);
  });
});
