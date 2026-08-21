import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { fundamentalsUpdateBanner, refreshPollMs } from './fundamentalsRefresh.ts';

describe('fundamentalsUpdateBanner', () => {
  it('shows Updating N/total while a job is running', () => {
    const banner = fundamentalsUpdateBanner({
      lastRun: { status: 'running', done: 12, total: 80 },
      coverage: { complete: 40, universe: 100 },
    });
    assert.equal(banner?.text, 'Updating 12/80 · 40/100 scored');
    assert.equal(banner?.pct, 15);
  });

  it('shows Starting… when coverage is incomplete but last run is idle', () => {
    const banner = fundamentalsUpdateBanner({
      lastRun: { status: 'completed', done: 80, total: 80 },
      coverage: { complete: 40, universe: 100 },
    });
    assert.equal(banner?.text, 'Starting fundamentals update… · 40/100 scored');
    assert.equal(banner?.pct, 40);
  });

  it('shows Starting… when coverage is still empty', () => {
    const banner = fundamentalsUpdateBanner({
      coverage: { complete: 0, universe: 0 },
    });
    assert.equal(banner?.text, 'Starting fundamentals update…');
    assert.equal(banner?.pct, 0);
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
  it('polls quickly while running or incomplete', () => {
    assert.equal(
      refreshPollMs({ lastRun: { status: 'running', done: 1, total: 2 } }),
      3_000,
    );
    assert.equal(refreshPollMs({ coverage: { complete: 1, universe: 10 } }), 5_000);
    assert.equal(refreshPollMs({ coverage: { complete: 0, universe: 0 } }), 5_000);
    assert.equal(refreshPollMs({ coverage: { complete: 10, universe: 10 } }), false);
    assert.equal(refreshPollMs(undefined), 5_000);
  });
});
