import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_RESULTS_PATH,
  HISTORY_GROUP_BYS,
  TIMEFRAMES,
  isUserTimeframe,
  normalizeHistoryFilters,
  rewriteLegacyResultsPath,
  type Timeframe,
} from './userTimeframes.ts';

describe('TIMEFRAMES', () => {
  it('is Weekly and Monthly only', () => {
    assert.deepEqual([...TIMEFRAMES], ['Weekly', 'Monthly']);
    assert.equal(isUserTimeframe('Daily'), false);
    assert.equal(isUserTimeframe('Weekly'), true);
    assert.equal(isUserTimeframe('Monthly'), true);
    assert.equal(TIMEFRAMES.includes('Daily' as Timeframe), false);
  });
});

describe('HISTORY_GROUP_BYS', () => {
  it('keeps calendar-day grouping as Day, not Daily', () => {
    assert.deepEqual([...HISTORY_GROUP_BYS], ['Day', 'Weekly', 'Monthly']);
    assert.equal((HISTORY_GROUP_BYS as readonly string[]).includes('Daily'), false);
  });
});

describe('legacy Daily Results URLs', () => {
  it('rewrites Daily to Weekly and leaves other paths alone', () => {
    assert.equal(DEFAULT_RESULTS_PATH, '/results/Stocks/Weekly/new');
    assert.equal(
      rewriteLegacyResultsPath('/results/Stocks/Daily/valid'),
      '/results/Stocks/Weekly/valid',
    );
    assert.equal(
      rewriteLegacyResultsPath('/results/ETF/Weekly/new'),
      '/results/ETF/Weekly/new',
    );
  });
});

describe('normalizeHistoryFilters', () => {
  it('defaults missing or Daily timeframe to All, and Daily groupBy to Day', () => {
    assert.deepEqual(normalizeHistoryFilters({}), {
      universe: 'Stocks',
      tf: 'All',
      groupBy: 'Day',
      range: 'all',
    });
    assert.deepEqual(
      normalizeHistoryFilters({ universe: 'ETF', tf: 'Daily', groupBy: 'Daily', range: '1y' }),
      { universe: 'ETF', tf: 'All', groupBy: 'Day', range: '1y' },
    );
  });
});
