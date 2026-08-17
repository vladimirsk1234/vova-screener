import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  compactAbsPct,
  realizedRrLabel,
  realizedRrRatio,
  signedMultiple,
} from './format.ts';

describe('compactAbsPct', () => {
  it('drops the decimal on whole percents', () => {
    assert.equal(compactAbsPct(50), '50');
    assert.equal(compactAbsPct(-15), '15');
  });

  it('keeps one decimal when the tenth matters', () => {
    assert.equal(compactAbsPct(50.4), '50.4');
    assert.equal(compactAbsPct(-15.2), '15.2');
  });
});

describe('realizedRrLabel', () => {
  it('formats winner / loser percents without signs', () => {
    assert.equal(realizedRrLabel(50, -15), '50 / 15');
  });

  it('is a dash when either side is missing', () => {
    assert.equal(realizedRrLabel(50, null), '—');
    assert.equal(realizedRrLabel(null, -15), '—');
  });
});

describe('realizedRrRatio', () => {
  it('divides winner pct by loser pct', () => {
    assert.equal(realizedRrRatio(50, -15), 3.33);
  });
});

describe('signedMultiple', () => {
  it('shows profit-to-risk as a signed multiple', () => {
    assert.equal(signedMultiple(1.1), '+1.10×');
    assert.equal(signedMultiple(-0.66), '-0.66×');
  });
});
