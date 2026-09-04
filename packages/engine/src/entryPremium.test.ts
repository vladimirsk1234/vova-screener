import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  resolvePremiumPctAtEntry,
  stampFromResolved,
  stampIfResolvable,
  type EntryPremiumPayload,
} from './entryPremium.ts';

function annualYear(
  year: number,
  opts: { operatingEps: number; price?: number | null; date?: string },
) {
  return {
    date: opts.date ?? `${year}-12-31`,
    year,
    price: opts.price ?? null,
    eps: opts.operatingEps,
    operatingEps: opts.operatingEps,
    revenuePerShare: null,
    fcfPerShare: null,
    ownerEarningsPerShare: null,
    pe: null,
    revenue: null,
    netIncome: null,
    operatingCashFlow: null,
    freeCashFlow: null,
  };
}

/** Flat Op. EPS → ~0% growth → 15× (GDF…P/E=G). FV ≈ 2 × 15 = 30. */
function flatPayload(extra?: EntryPremiumPayload): EntryPremiumPayload {
  return {
    scale: { reliable: true },
    annual: [2019, 2020, 2021, 2022, 2023].map((y) =>
      annualYear(y, { operatingEps: 2, price: 30 }),
    ),
    ...extra,
  };
}

describe('resolvePremiumPctAtEntry', () => {
  it('computes card premium from as-of annuals and the entry price', () => {
    const cheap = resolvePremiumPctAtEntry({
      asOf: '2024-03-15',
      entryPrice: 8,
      payload: flatPayload(),
    });
    assert.ok(cheap.premiumPct != null && cheap.premiumPct < 0, `cheap=${cheap.premiumPct}`);
    assert.equal(cheap.asOf, '2024-03-15');

    const rich = resolvePremiumPctAtEntry({
      asOf: '2024-03-15',
      entryPrice: 80,
      payload: flatPayload(),
    });
    assert.ok(rich.premiumPct != null && rich.premiumPct > 0, `rich=${rich.premiumPct}`);
  });

  it('returns null when payload or as-of annuals are missing (no invented value)', () => {
    assert.deepEqual(
      resolvePremiumPctAtEntry({ asOf: '2024-01-01', entryPrice: 10, payload: null }),
      { premiumPct: null, asOf: null },
    );
    assert.deepEqual(
      resolvePremiumPctAtEntry({
        asOf: '2010-01-01',
        entryPrice: 10,
        payload: flatPayload(),
      }),
      { premiumPct: null, asOf: null },
    );
    assert.deepEqual(
      resolvePremiumPctAtEntry({
        asOf: '2024-01-01',
        entryPrice: 10,
        payload: { ...flatPayload(), scale: { reliable: false } },
      }),
      { premiumPct: null, asOf: null },
    );
  });

  it('ignores fiscal years after openedAsOf', () => {
    const withFuture = flatPayload({
      annual: [
        ...(flatPayload().annual ?? []),
        annualYear(2025, { operatingEps: 80, price: 10 }),
      ],
    });
    const asOf = resolvePremiumPctAtEntry({
      asOf: '2024-03-15',
      entryPrice: 20,
      payload: withFuture,
    });
    const baseline = resolvePremiumPctAtEntry({
      asOf: '2024-03-15',
      entryPrice: 20,
      payload: flatPayload(),
    });
    assert.equal(asOf.premiumPct, baseline.premiumPct);
  });
});

describe('stampIfResolvable / backfill no-op', () => {
  it('is a no-op when stored fundamentals are missing', () => {
    assert.equal(stampIfResolvable('2024-03-15', 20, null), null);
    assert.equal(stampIfResolvable('2024-03-15', 20, {}), null);
    assert.equal(stampIfResolvable('2024-03-15', 20, { annual: [] }), null);
  });

  it('writes nulls (not a number) when a payload exists but valuation is impossible', () => {
    const stamp = stampIfResolvable('2024-03-15', 20, {
      scale: { reliable: false },
      annual: [annualYear(2023, { operatingEps: 2, price: 30 })],
    });
    assert.deepEqual(stamp, {
      premiumPctAtEntry: null,
      undervaluedAtEntry: null,
      premiumPctAtEntryAsOf: null,
    });
  });

  it('maps a resolved premium onto the trade fields', () => {
    assert.deepEqual(stampFromResolved({ premiumPct: -7.5, asOf: '2024-03-15' }), {
      premiumPctAtEntry: -7.5,
      undervaluedAtEntry: true,
      premiumPctAtEntryAsOf: '2024-03-15',
    });
    assert.deepEqual(stampFromResolved({ premiumPct: 3, asOf: '2024-03-15' }), {
      premiumPctAtEntry: 3,
      undervaluedAtEntry: false,
      premiumPctAtEntryAsOf: '2024-03-15',
    });
  });
});
