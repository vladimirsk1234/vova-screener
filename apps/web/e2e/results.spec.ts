import { expect, test } from '@playwright/test';

/**
 * Results is the app's front door: three nested tab rows over the latest background scan.
 * These run against a live API — with no data the lists are empty but the chrome still holds.
 */
test.describe('results shell', () => {
  test('lands on Stocks / Daily / New and keeps the tabs in the URL', async ({ page }) => {
    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
    await page.goto('/');
    await expect(page).toHaveURL(/\/results\/Stocks\/Daily\/new$/);

    await expect(page.getByRole('tab', { name: 'Stocks' })).toHaveClass(/active/);
    await expect(page.getByRole('tab', { name: 'D', exact: true })).toHaveClass(/active/);
    await expect(page.getByRole('tab', { name: /^New/ })).toHaveClass(/active/);
  });

  test('switching universe, timeframe and bucket navigates', async ({ page }) => {
    await page.goto('/results/Stocks/Daily/new');

    // Every row builds its links from the current route, so wait for a tab to light up before
    // clicking the next one: the URL changes ahead of the re-render, and a click landing in
    // between would follow a link still pointing at the route we just left.
    const weekly = page.getByRole('tab', { name: 'W', exact: true });
    await weekly.click();
    await expect(page).toHaveURL(/\/results\/Stocks\/Weekly\/new$/);
    await expect(weekly).toHaveClass(/active/);

    const closed = page.getByRole('tab', { name: /^Closed/ });
    await closed.click();
    await expect(page).toHaveURL(/\/results\/Stocks\/Weekly\/closed$/);
    await expect(closed).toHaveClass(/active/);

    await page.getByRole('tab', { name: 'ETF' }).click();
    await expect(page).toHaveURL(/\/results\/ETF\/Weekly\/closed$/);
  });

  test('sorting is reflected in the query string', async ({ page }) => {
    await page.goto('/results/Stocks/Daily/valid');
    await page.getByRole('button', { name: /^P&L/ }).click();
    await expect(page).toHaveURL(/sort=pnl&dir=desc/);
    await page.getByRole('button', { name: /^P&L/ }).click();
    await expect(page).toHaveURL(/sort=pnl&dir=asc/);
    await page.getByRole('button', { name: /^Marked/ }).click();
    await expect(page).toHaveURL(/sort=interest/);
  });

  test('every bucket can be sorted by RR', async ({ page }) => {
    for (const bucket of ['new', 'valid', 'closed']) {
      await page.goto(`/results/Stocks/Daily/${bucket}`);
      const rr = page.getByRole('group', { name: 'Sort' }).getByRole('button', { name: /^RR/ });
      await expect(rr).toBeVisible();
      // NEW and VALID already default to RR, CLOSED to P&L, so assert the toggle rather than a
      // fixed starting direction. The arrow is what says the chip has caught up with the URL; a
      // second click before that reads the old direction and asks for the same sort again.
      const before = await rr.textContent();
      await rr.click();
      await expect(page).toHaveURL(/sort=rr/);
      await expect(rr).not.toHaveText(before ?? '');

      const descending = (await rr.textContent())?.includes('↓');
      await rr.click();
      await expect(page).toHaveURL(new RegExp(`sort=rr&dir=${descending ? 'asc' : 'desc'}$`));
    }
  });

  test('manual is the only place with a scan button', async ({ page }) => {
    await page.goto('/results/Stocks/Daily/new');
    await expect(page.getByRole('button', { name: 'START SCAN' })).toHaveCount(0);

    await page.getByRole('tab', { name: 'Manual' }).click();
    await expect(page).toHaveURL(/\/results\/manual$/);
    await expect(page.getByRole('button', { name: 'START SCAN' })).toBeVisible();
  });

  test('settings sheet holds max risk, min RR, fundamentals, rescan and reset', async ({ page }) => {
    await page.goto('/results/Stocks/Daily/new');
    await page.getByRole('button', { name: 'Settings' }).click();
    const sheet = page.getByRole('dialog', { name: 'Settings' });
    await expect(sheet.getByLabel('Max risk per signal ($)')).toBeVisible();
    await expect(sheet.getByLabel('Min RR')).toBeVisible();
    for (const label of ['Undervalued', 'Overvalued']) {
      await expect(sheet.getByRole('button', { name: label, exact: true })).toBeVisible();
    }
    await expect(sheet.getByRole('button', { name: 'All', exact: true })).toHaveCount(2);
    for (const label of ['D', 'W', 'M']) {
      await expect(sheet.getByRole('button', { name: label, exact: true })).toBeVisible();
    }
    await expect(sheet.getByRole('button', { name: /Run scan now|Scanning/ })).toBeVisible();
    await expect(sheet.getByRole('button', { name: 'Reset all history' })).toBeVisible();
    await sheet.getByRole('button', { name: 'Close' }).click();
    await expect(page.getByRole('dialog', { name: 'Settings' })).toHaveCount(0);
  });

  test('the rescan button asks for the selected timeframe', async ({ page }) => {
    // Stubbed: a real press starts a full pass over Stocks and ETF, which is not something a test
    // run should leave going. What is worth checking is the request and what the sheet does next.
    let asked: string | null = null;
    await page.route('**/api/scans/run-now', async (route) => {
      asked = route.request().postDataJSON()?.tf ?? null;
      await route.fulfill({ json: { started: true, timeframes: ['Weekly'] } });
    });

    await page.goto('/results/Stocks/Daily/new');
    await page.getByRole('button', { name: 'Settings' }).click();
    const sheet = page.getByRole('dialog', { name: 'Settings' });

    const run = sheet.getByRole('button', { name: 'Run scan now' });
    if ((await run.count()) === 0) test.skip(true, 'a scan is already running against this API');
    await sheet.getByRole('button', { name: 'W', exact: true }).click();
    await run.click();

    // The pass is queued rather than awaited, so the button is what says the request landed.
    await expect(sheet.getByRole('button', { name: 'Scanning…' })).toBeVisible();
    expect(asked).toBe('Weekly');
  });

  test('history exposes Stocks/ETF, every timeframe plus All', async ({ page }) => {
    await page.goto('/history');
    for (const label of ['Stocks', 'ETF']) {
      await expect(page.getByRole('button', { name: label, exact: true }).first()).toBeVisible();
    }
    for (const label of ['Daily', 'Weekly', 'Monthly', 'All']) {
      await expect(page.getByRole('button', { name: label, exact: true }).first()).toBeVisible();
    }
    await expect(page.getByText('Win rate')).toBeVisible();
  });

  test('Results status and timeframe survive a trip through History', async ({ page }) => {
    await page.goto('/results/Stocks/Daily/new');

    const weekly = page.getByRole('tab', { name: 'W', exact: true });
    await weekly.click();
    await expect(page).toHaveURL(/\/results\/Stocks\/Weekly\/new$/);

    const valid = page.getByRole('tab', { name: /^Valid/ });
    await valid.click();
    await expect(page).toHaveURL(/\/results\/Stocks\/Weekly\/valid$/);

    await page.getByRole('link', { name: 'History' }).click();
    await expect(page).toHaveURL(/\/history$/);

    // History keeps its own timeframe pick across remounts.
    await page.getByRole('button', { name: 'Weekly', exact: true }).first().click();
    await expect(page.getByRole('button', { name: 'Weekly', exact: true }).first()).toHaveClass(
      /active/,
    );

    await page.getByRole('link', { name: 'Results' }).click();
    await expect(page).toHaveURL(/\/results\/Stocks\/Weekly\/valid$/);
    await expect(page.getByRole('tab', { name: 'W', exact: true })).toHaveClass(/active/);
    await expect(page.getByRole('tab', { name: /^Valid/ })).toHaveClass(/active/);

    await page.getByRole('link', { name: 'History' }).click();
    await expect(page.getByRole('button', { name: 'Weekly', exact: true }).first()).toHaveClass(
      /active/,
    );
  });

  test('a signal card opens the chart', async ({ page }) => {
    // A single scan fills NEW only with the symbols that broke out on the bar it evaluated;
    // everything valid for longer lands in VALID, so try both before giving up.
    let card = page.locator('.signal-card').first();
    for (const bucket of ['new', 'valid']) {
      await page.goto(`/results/Stocks/Daily/${bucket}`);
      await expect(page.getByText('Loading…')).toHaveCount(0);
      card = page.locator('.signal-card').first();
      if ((await card.count()) > 0) break;
    }
    if ((await card.count()) === 0) test.skip(true, 'no tracked signals in this database');
    await expect(card.getByRole('button', { name: 'Interested', exact: true })).toBeVisible();
    await expect(card.getByRole('button', { name: 'Not Interested' })).toBeVisible();
    await card.locator('.signal-card-title').click();
    await expect(page).toHaveURL(/\/chart\//);
    await expect(page.getByRole('button', { name: 'Interested', exact: true })).toBeVisible();
  });

  test('Results tabs survive closing the session and reopening /', async ({ page }) => {
    await page.goto('/results/ETF/Weekly/closed?sort=pnl&dir=desc');
    await expect(page).toHaveURL(/\/results\/ETF\/Weekly\/closed\?sort=pnl&dir=desc/);

    // sessionStorage dies with the tab; localStorage is what a cold PWA reopen keeps.
    await page.evaluate(() => sessionStorage.clear());
    await page.goto('/');
    await expect(page).toHaveURL(/\/results\/ETF\/Weekly\/closed\?sort=pnl&dir=desc/);
    await expect(page.getByRole('tab', { name: 'ETF' })).toHaveClass(/active/);
    await expect(page.getByRole('tab', { name: 'W', exact: true })).toHaveClass(/active/);
    await expect(page.getByRole('tab', { name: /^Closed/ })).toHaveClass(/active/);
  });

  test('History is restored from / after a cold reopen', async ({ page }) => {
    await page.goto('/history');
    await expect(page).toHaveURL(/\/history$/);

    await page.evaluate(() => sessionStorage.clear());
    await page.goto('/');
    await expect(page).toHaveURL(/\/history$/);
    await expect(page.getByText('Win rate')).toBeVisible();
  });

  test('Chart is restored from / after a cold reopen', async ({ page }) => {
    await page.goto('/chart/AAPL');
    await expect(page).toHaveURL(/\/chart\/AAPL/);

    await page.evaluate(() => sessionStorage.clear());
    await page.goto('/');
    await expect(page).toHaveURL(/\/chart\/AAPL/);

    // The restore seeds Results underneath the chart, so Back stays inside the app instead of
    // leaving for whatever the tab was showing before.
    await page.goBack();
    await expect(page).toHaveURL(/\/results\/Stocks\/Daily\/new$/);
  });

  test('Back on a chart opened straight from a URL lands on Results', async ({ page }) => {
    await page.goto('/chart/AAPL');
    await page.getByRole('button', { name: 'Back', exact: true }).click();
    await expect(page).toHaveURL(/\/results\/Stocks\/Daily\/new$/);
  });

  test('Results bottom nav still returns to the last Results tabs after History', async ({
    page,
  }) => {
    await page.goto('/results/Stocks/Monthly/valid?sort=rr&dir=asc');
    await page.getByRole('link', { name: 'History' }).click();
    await expect(page).toHaveURL(/\/history$/);

    await page.evaluate(() => sessionStorage.clear());
    await page.reload();
    await expect(page).toHaveURL(/\/history$/);

    await page.getByRole('link', { name: 'Results' }).click();
    await expect(page).toHaveURL(/\/results\/Stocks\/Monthly\/valid\?sort=rr&dir=asc/);
  });
});
