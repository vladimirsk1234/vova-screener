import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  HISTORY_GROUP_BYS,
  TIMEFRAMES,
  isUserTimeframe,
  parseHistoryGroupBy,
  parseHistoryTf,
  parseTf,
  withUserTf,
} from './tf.ts';

describe('TIMEFRAMES', () => {
  it('is Weekly and Monthly only', () => {
    assert.deepEqual([...TIMEFRAMES], ['Weekly', 'Monthly']);
    assert.equal(isUserTimeframe('Daily'), false);
    assert.equal(isUserTimeframe('Weekly'), true);
    assert.equal(isUserTimeframe('Monthly'), true);
  });
});

describe('parseTf', () => {
  it('accepts Weekly and Monthly', () => {
    assert.equal(parseTf('Weekly'), 'Weekly');
    assert.equal(parseTf('Monthly'), 'Monthly');
  });

  it('does not default unknown or Daily to Daily', () => {
    assert.equal(parseTf('Daily'), 'Weekly');
    assert.equal(parseTf('1d'), 'Weekly');
    assert.equal(parseTf(undefined), 'Weekly');
    assert.equal(parseTf(''), 'Weekly');
  });
});

describe('parseHistoryTf', () => {
  it('keeps All and the user timeframes', () => {
    assert.equal(parseHistoryTf('All'), 'All');
    assert.equal(parseHistoryTf('Weekly'), 'Weekly');
    assert.equal(parseHistoryTf('Monthly'), 'Monthly');
  });

  it('treats Daily and unknown as All, not Daily', () => {
    assert.equal(parseHistoryTf('Daily'), 'All');
    assert.equal(parseHistoryTf(undefined), 'All');
    assert.equal(parseHistoryTf('nope'), 'All');
  });
});

describe('parseHistoryGroupBy', () => {
  it('keeps calendar-day grouping under Day, including the Daily alias', () => {
    assert.equal(parseHistoryGroupBy('Day'), 'Day');
    assert.equal(parseHistoryGroupBy('Daily'), 'Day');
    assert.equal(parseHistoryGroupBy(undefined), 'Day');
    assert.deepEqual([...HISTORY_GROUP_BYS], ['Day', 'Weekly', 'Monthly']);
  });

  it('accepts Weekly and Monthly grouping', () => {
    assert.equal(parseHistoryGroupBy('Weekly'), 'Weekly');
    assert.equal(parseHistoryGroupBy('Monthly'), 'Monthly');
  });
});

describe('withUserTf', () => {
  it('restricts All to Weekly and Monthly so Daily rows stay hidden', () => {
    assert.deepEqual(withUserTf({ status: 'closed' }, 'All'), {
      status: 'closed',
      tf: { $in: ['Weekly', 'Monthly'] },
    });
    assert.deepEqual(withUserTf({ status: 'closed' }, 'Weekly'), {
      status: 'closed',
      tf: 'Weekly',
    });
  });
});
