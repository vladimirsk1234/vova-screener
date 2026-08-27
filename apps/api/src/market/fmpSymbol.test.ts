import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  fmpMappedSymbol,
  fmpSymbolCandidates,
  yahooToFmpSymbol,
} from './fmpSymbol.ts';

describe('fmpSymbolCandidates', () => {
  it('keeps US class shares in dash form first', () => {
    assert.deepEqual(fmpSymbolCandidates('BRK-B'), ['BRK-B', 'BRK.B']);
    assert.deepEqual(fmpSymbolCandidates('BF-B'), ['BF-B', 'BF.B']);
    assert.deepEqual(fmpSymbolCandidates('MOG-A'), ['MOG-A', 'MOG.A']);
  });

  it('leaves Canadian exchange suffixes unchanged', () => {
    assert.deepEqual(fmpSymbolCandidates('SHOP.TO'), ['SHOP.TO']);
    assert.deepEqual(fmpSymbolCandidates('AAV.V'), ['AAV.V']);
  });

  it('maps ordinary tickers without extra candidates', () => {
    assert.deepEqual(fmpSymbolCandidates('AAPL'), ['AAPL']);
  });
});

describe('yahooToFmpSymbol', () => {
  it('returns the preferred first candidate', () => {
    assert.equal(yahooToFmpSymbol('BRK-B'), 'BRK-B');
    assert.equal(yahooToFmpSymbol('SHOP.TO'), 'SHOP.TO');
    assert.equal(yahooToFmpSymbol('AAPL'), 'AAPL');
  });
});

describe('fmpMappedSymbol', () => {
  it('maps class shares to dotted fallback only', () => {
    assert.equal(fmpMappedSymbol('BRK-B'), 'BRK.B');
    assert.equal(fmpMappedSymbol('SHOP.TO'), 'SHOP.TO');
  });
});
