import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { fundamentalsUpdateBanner, refreshPollMs } from './fundamentalsRefresh.ts';

describe('fundamentalsUpdateBanner', () => {
  it('shows Updating N/total while a job is running', () => {
    const banner = fundamentalsUpdateBanner({
      lastRun: { status: 'running', done: 20, total: 827 },
      coverage: { complete: 2025, universe: 2095 },
    });
    assert.equal(banner?.text, 'Updating 20/827 · 2025/2095 scored');
    assert.equal(banner?.pct, 97);
  });

  it('hides when the last run finished even if some names are still missing', () => {
    assert.equal(
      fundamentalsUpdateBanner({
        lastRun: { status: 'completed', done: 827, total: 827 },
        coverage: { complete: 2025, universe: 2095 },
      }),
      null,
    );
  });

  it('hides when coverage is empty and no job is running', () => {
    assert.equal(
      fundamentalsUpdateBanner({
        coverage: { complete: 0, universe: 0 },
      }),
      null,
    );
  });

  it('hides when coverage is complete and no job is running', () => {
    assert.equal(
      fundamentalsUpdateBanner({
        lastRun: { status: 'completed', done: 100, total: 100 },
        coverage: { complete: 100, universe: 100 },
      }),
      null,
    );
  });
});

describe('refreshPollMs', () => {
  it('polls only while a job is running', () => {
    assert.equal(
      refreshPollMs({ lastRun: { status: 'running', done: 1, total: 2 } }),
      3_000,
    );
    assert.equal(refreshPollMs({ coverage: { complete: 1, universe: 10 } }), false);
    assert.equal(refreshPollMs({ coverage: { complete: 0, universe: 0 } }), false);
    assert.equal(refreshPollMs({ coverage: { complete: 10, universe: 10 } }), false);
    assert.equal(refreshPollMs(undefined), false);
  });
});
