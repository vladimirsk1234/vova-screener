import { expect, test } from '@playwright/test';

/**
 * Results is the app's front door: three nested tab rows over the latest background scan.
 * These run against a live API — with no data the lists are empty but the chrome still holds.
 */
test.describe('results shell', () => {
  test('lands on Stocks / Daily / New and keeps the tabs in the URL', async ({ page }) => {
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

  test('settings sheet holds max risk and reset', async ({ page }) => {
    await page.goto('/results/Stocks/Daily/new');
    await page.getByRole('button', { name: 'Settings' }).click();
    const sheet = page.getByRole('dialog', { name: 'Settings' });
    await expect(sheet.getByLabel('Max risk per signal ($)')).toBeVisible();
    await expect(sheet.getByRole('button', { name: 'Reset all history' })).toBeVisible();
    await sheet.getByRole('button', { name: 'Close' }).click();
    await expect(page.getByRole('dialog', { name: 'Settings' })).toHaveCount(0);
  });

  test('history exposes every timeframe plus All', async ({ page }) => {
    await page.goto('/history');
    for (const label of ['Daily', 'Weekly', 'Monthly', 'All']) {
      await expect(page.getByRole('button', { name: label, exact: true }).first()).toBeVisible();
    }
    await expect(page.getByText('Win rate')).toBeVisible();
  });

  test('a signal card opens the chart', async ({ page }) => {
    // NEW is the one bucket a freshly scanned database is guaranteed to fill; VALID and CLOSED
    // need a signal to survive into a later period, which no single scan can produce.
    await page.goto('/results/Stocks/Daily/new');
    await expect(page.getByText('Loading…')).toHaveCount(0);
    const card = page.locator('.signal-card').first();
    if ((await card.count()) === 0) test.skip(true, 'no tracked signals in this database');
    await card.click();
    await expect(page).toHaveURL(/\/chart\//);
    await expect(page.getByRole('button', { name: 'Interested', exact: true })).toBeVisible();
  });
});
