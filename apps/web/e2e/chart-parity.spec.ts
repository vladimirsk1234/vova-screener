import { expect, test } from '@playwright/test';

/**
 * Chart visual smoke. Requires API (+ Mongo) when hitting a real ticker.
 * Without API the page still mounts chrome; we assert UI shells that always exist.
 */
test.describe('chart parity UI', () => {
  test('settings sheet and drawing toolbar render', async ({ page }) => {
    await page.goto('/chart/AAPL');
    await expect(page.getByRole('toolbar', { name: 'Drawing tools' })).toBeVisible();
    await page.getByRole('button', { name: 'Settings' }).click();
    await expect(page.getByRole('dialog', { name: 'Chart settings' })).toBeVisible();
    await expect(page.getByRole('dialog', { name: 'Chart settings' }).getByText('Fibonacci').first()).toBeVisible();
    await page.getByRole('button', { name: 'Close' }).click();
    await expect(page.getByRole('dialog', { name: 'Chart settings' })).toHaveCount(0);
  });

  test('timeframe chips and fit control', async ({ page }) => {
    await page.goto('/chart/AAPL');
    await page.getByRole('button', { name: 'Weekly' }).click();
    await page.getByRole('button', { name: 'Fit' }).click();
    await expect(page.getByRole('link', { name: 'TradingView' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'FastGraph' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Interested' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Not Interested' })).toBeVisible();
  });

  test('drawing tool selection + undo idle state', async ({ page }) => {
    await page.goto('/chart/AAPL');
    const trend = page.getByRole('button', { name: 'Trend line' });
    await trend.click();
    await expect(trend).toHaveClass(/active/);
    await page.getByRole('button', { name: 'Select / pan' }).click();
    await expect(page.getByRole('button', { name: 'Undo' })).toBeDisabled();
  });

  test('default chart host uses Streamlit-grey canvas area', async ({ page }) => {
    await page.goto('/chart/AAPL');
    const host = page.locator('.chart-host');
    await expect(host).toBeVisible();
    await expect(host).toHaveCSS('background-color', 'rgb(112, 117, 133)');
    await page.screenshot({ path: `e2e/__snapshots__/chart-${test.info().project.name}.png`, fullPage: true });
  });
});
