import { expect, test } from '@playwright/test';

/**
 * Chart visual smoke. Requires API (+ Mongo) when hitting a real ticker.
 * Without API the page still mounts chrome; we assert UI shells that always exist.
 */
test.describe('chart parity UI', () => {
  test('settings sheet renders', async ({ page }) => {
    await page.goto('/chart/AAPL');
    await page.getByRole('button', { name: 'Settings' }).click();
    await expect(page.getByRole('dialog', { name: 'Chart settings' })).toBeVisible();
    await expect(
      page.getByRole('dialog', { name: 'Chart settings' }).getByText('Fibonacci').first(),
    ).toBeVisible();
    await page.getByRole('button', { name: 'Close' }).click();
    await expect(page.getByRole('dialog', { name: 'Chart settings' })).toHaveCount(0);
  });

  test('timeframe chips and outbound links', async ({ page }) => {
    await page.goto('/chart/AAPL');
    await page.getByRole('button', { name: 'Weekly' }).click();
    // Exact: Lightweight Charts injects its own "Charting by TradingView" attribution link.
    await expect(page.getByRole('link', { name: 'TradingView', exact: true })).toBeVisible();
    await expect(page.getByRole('link', { name: 'FastGraph' })).toBeVisible();
  });

  /** The mark lives on the tracked signal, so it is disabled for symbols nothing is tracking. */
  test('interest buttons are disabled for untracked symbols', async ({ page }) => {
    await page.goto('/chart/ZZ-NOT-TRACKED');
    await expect(page.getByRole('button', { name: 'Interested', exact: true })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Not Interested' })).toBeDisabled();
  });

  test('default chart host uses Streamlit-grey canvas area', async ({ page }) => {
    await page.goto('/chart/AAPL');
    const host = page.locator('.chart-host');
    await expect(host).toBeVisible();
    await expect(host).toHaveCSS('background-color', 'rgb(112, 117, 133)');
    // The reference image is rewritten on every run, so let the bars land first: a shot of the
    // loading line is worth nothing to whoever opens it next.
    await expect(page.getByText('Loading bars…')).toHaveCount(0);
    await page.screenshot({
      path: `e2e/__snapshots__/chart-${test.info().project.name}.png`,
      fullPage: true,
    });
  });
});
