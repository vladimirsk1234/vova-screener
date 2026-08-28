import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { cardFundamentalsFromStoredDoc, cardPremiaFromStored } from './card-premia.ts';

describe('cardPremiaFromStored', () => {
  it('maps EPS / FCF / DCF premia and keeps default premiumPct', () => {
    const card = cardPremiaFromStored({
      premiumPct: 5,
      epsPremiumPct: -30,
      fcfPremiumPct: -5,
      dcfPremiumPct: 10,
      bestPremiumPct: -30,
    });
    assert.equal(card.premiumPct, 5);
    assert.equal(card.epsPremiumPct, -30);
    assert.equal(card.fcfPremiumPct, -5);
    assert.equal(card.dcfPremiumPct, 10);
    assert.equal(card.bestPremiumPct, -30);
  });

  it('falls back to premiumPct for EPS and recomputes best when star fields are missing', () => {
    const card = cardPremiaFromStored({
      premiumPct: -12,
      fcfPremiumPct: -40,
      dcfPremiumPct: 8,
    });
    assert.equal(card.premiumPct, -12);
    assert.equal(card.epsPremiumPct, -12);
    assert.equal(card.fcfPremiumPct, -40);
    assert.equal(card.dcfPremiumPct, 8);
    assert.equal(card.bestPremiumPct, -40);
  });

  it('returns null premia when nothing is stored', () => {
    const card = cardPremiaFromStored({});
    assert.equal(card.premiumPct, null);
    assert.equal(card.epsPremiumPct, null);
    assert.equal(card.fcfPremiumPct, null);
    assert.equal(card.dcfPremiumPct, null);
    assert.equal(card.bestPremiumPct, null);
  });
});

describe('cardFundamentalsFromStoredDoc', () => {
  it('includes the three premia next to the existing card fields', () => {
    const card = cardFundamentalsFromStoredDoc({
      fairValue: 42,
      premiumPct: 1.5,
      growthRatePct: 12,
      blendedPe: 14.8,
      ltDebtToCapitalTTM: 0.2,
      epsPremiumPct: -7,
      fcfPremiumPct: 3,
      dcfPremiumPct: -11,
    });
    assert.equal(card.fairValue, 42);
    assert.equal(card.growthRatePct, 12);
    assert.equal(card.blendedPe, 14.8);
    assert.equal(card.ltDebtToCapitalTTM, 0.2);
    assert.equal(card.premiumPct, 1.5);
    assert.equal(card.epsPremiumPct, -7);
    assert.equal(card.fcfPremiumPct, 3);
    assert.equal(card.dcfPremiumPct, -11);
    assert.equal(card.bestPremiumPct, -11);
  });
});
