import { describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { readManualSearchHistory, rememberManualSearch, rememberResolvedManualSearch } from './manualSearchHistory.ts';

const store = new Map<string, string>();

beforeEach(() => {
  store.clear();
  (globalThis as { localStorage?: Storage }).localStorage = {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => {
      store.set(key, value);
    },
    removeItem: (key: string) => {
      store.delete(key);
    },
    clear: () => store.clear(),
    key: () => null,
    get length() {
      return store.size;
    },
  } as Storage;
});

describe('rememberManualSearch', () => {
  it('keeps newest first and overwrites the 11th', () => {
    for (let i = 1; i <= 11; i++) rememberManualSearch(`T${i}`);
    assert.deepEqual(readManualSearchHistory(), [
      'T11',
      'T10',
      'T9',
      'T8',
      'T7',
      'T6',
      'T5',
      'T4',
      'T3',
      'T2',
    ]);
  });

  it('moves a repeat to the front without a duplicate', () => {
    rememberManualSearch('AAPL');
    rememberManualSearch('TSLA');
    rememberManualSearch('AAPL');
    assert.deepEqual(readManualSearchHistory(), ['AAPL', 'TSLA']);
  });
});

describe('rememberResolvedManualSearch', () => {
  it('replaces the bare typed ticker with the canonical Yahoo symbol', () => {
    rememberManualSearch('RBY');
    rememberManualSearch('NVDA');
    assert.deepEqual(rememberResolvedManualSearch('RBY', 'RBY.TO'), ['RBY.TO', 'NVDA']);
  });
});
