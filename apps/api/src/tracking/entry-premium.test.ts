import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  matchesEntryPremium,
  withEntryPremiumFilter,
  withLiveOrEntryPremiumFilter,
} from './entry-premium.ts';

describe('matchesEntryPremium', () => {
  it('filters UV / OV on the entry snapshot and keeps nulls only in all', () => {
    assert.equal(matchesEntryPremium(-12, 'undervalued'), true);
    assert.equal(matchesEntryPremium(8, 'undervalued'), false);
    assert.equal(matchesEntryPremium(0, 'undervalued'), false);
    assert.equal(matchesEntryPremium(null, 'undervalued'), false);
    assert.equal(matchesEntryPremium(undefined, 'undervalued'), false);

    assert.equal(matchesEntryPremium(8, 'overvalued'), true);
    assert.equal(matchesEntryPremium(-12, 'overvalued'), false);
    assert.equal(matchesEntryPremium(0, 'overvalued'), false);
    assert.equal(matchesEntryPremium(null, 'overvalued'), false);

    assert.equal(matchesEntryPremium(-12, 'all'), true);
    assert.equal(matchesEntryPremium(8, 'all'), true);
    assert.equal(matchesEntryPremium(null, 'all'), true);
    assert.equal(matchesEntryPremium(undefined, 'all'), true);
    assert.equal(matchesEntryPremium(0, 'all'), true);
  });
});

describe('withEntryPremiumFilter', () => {
  const base = { status: 'closed', universe: 'Stocks' };

  it('leaves the match alone when the filter is all', () => {
    assert.deepEqual(withEntryPremiumFilter(base, 'all'), base);
  });

  it('ANDs premiumPctAtEntry < 0 for undervalued (nulls excluded)', () => {
    assert.deepEqual(withEntryPremiumFilter(base, 'undervalued'), {
      $and: [base, { premiumPctAtEntry: { $lt: 0 } }],
    });
  });

  it('ANDs premiumPctAtEntry > 0 for overvalued (nulls excluded)', () => {
    assert.deepEqual(withEntryPremiumFilter(base, 'overvalued'), {
      $and: [base, { premiumPctAtEntry: { $gt: 0 } }],
    });
  });
});

describe('withLiveOrEntryPremiumFilter', () => {
  const base = { status: 'active', universe: 'Stocks' };

  it('falls back to live tickers only for rows that are not yet stamped', () => {
    const got = withLiveOrEntryPremiumFilter(base, 'undervalued', ['AAA', 'BBB']);
    assert.deepEqual(got, {
      $and: [
        base,
        {
          $or: [
            { premiumPctAtEntry: { $lt: 0 } },
            {
              $and: [
                { premiumPctAtEntry: { $exists: false } },
                { yahooTicker: { $in: ['AAA', 'BBB'] } },
              ],
            },
          ],
        },
      ],
    });
  });

  it('does not treat explicit null as unstamped (no live fallback)', () => {
    const json = JSON.stringify(withLiveOrEntryPremiumFilter(base, 'overvalued', ['ZZZ']));
    assert.equal(json.includes('"premiumPctAtEntry":null'), false);
    assert.equal(json.includes('$exists'), true);
  });
});
