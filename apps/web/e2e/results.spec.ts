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

    await page.getByRole('tab', { name: 'W', exact: true }).click();
    await expect(page).toHaveURL(/\/results\/Stocks\/Weekly\/new$/);

    await page.getByRole('tab', { name: /^Closed/ }).click();
    await expect(page).toHaveURL(/\/results\/Stocks\/Weekly\/closed$/);

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
    await page.goto('/results/Stocks/Daily/valid');
    await expect(page.getByText('Loading…')).toHaveCount(0);
    const card = page.locator('.signal-card').first();
    if ((await card.count()) === 0) test.skip(true, 'no tracked signals in this database');
    await card.click();
    await expect(page).toHaveURL(/\/chart\//);
    await expect(page.getByRole('button', { name: 'Interested', exact: true })).toBeVisible();
  });
});
