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
    await expect(page.getByRole('tab', { name: 'TA', exact: true })).toBeVisible();
    await expect(page.getByRole('tab', { name: 'Fundamentals', exact: true })).toBeVisible();
  });

  /** The mark lives on the tracked signal, so it is disabled for symbols nothing is tracking. */
  test('interest buttons are disabled for untracked symbols', async ({ page }) => {
    await page.goto('/chart/ZZ-NOT-TRACKED');
    await expect(page.getByRole('button', { name: 'Interested', exact: true })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Not Interested' })).toBeDisabled();
  });

  test('TA / Fundamentals toggle keeps the selected timeframe', async ({ page }) => {
    await page.goto('/chart/AAPL');
    await page.getByRole('button', { name: 'Weekly' }).click();
    await expect(page.getByRole('button', { name: 'Weekly' })).toHaveClass(/active/);

    await page.getByRole('tab', { name: 'Fundamentals', exact: true }).click();
    await expect(page).toHaveURL(/\/chart\/AAPL\?view=fundamentals/);
    await expect(page.locator('.chart-stage .chart-host')).toBeVisible();
    await expect(page.locator('.chart-host')).toHaveCount(1);
    await expect(page.locator('.fund-chart-host')).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Summary' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'DCF', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'EPS', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: '5Y', exact: true })).toBeVisible();
    await expect(page.getByText('Fair value', { exact: true })).toBeVisible();
    await expect(page.getByText('Normal P/E').first()).toBeVisible();
    await expect(page.locator('.chart-watermark')).toHaveCount(0);
    await expect(page.getByText('ATR:')).toHaveCount(0);
    await expect(page.getByText('D: Seq')).toHaveCount(0);
    const stage = page.locator('.chart-stage');
    await expect(stage).toBeVisible();
    const box = await stage.boundingBox();
    expect(box?.height ?? 0).toBeGreaterThan(280);

    await page.getByRole('tab', { name: 'TA', exact: true }).click();
    await expect(page).toHaveURL(/\/chart\/AAPL$/);
    await expect(page.getByRole('button', { name: 'Weekly' })).toHaveClass(/active/);
    await expect(page.getByRole('link', { name: 'TradingView', exact: true })).toBeVisible();
  });

  test('legacy fundamentals URL redirects into the chart window', async ({ page }) => {
    await page.goto('/fundamentals/AAPL');
    await expect(page).toHaveURL(/\/chart\/AAPL\?view=fundamentals/);
    await expect(page.getByRole('tab', { name: 'Fundamentals', exact: true })).toHaveClass(/active/);
    await expect(page.locator('.chart-stage .chart-host')).toBeVisible();
    await expect(page.locator('.fund-chart-host')).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'DCF', exact: true })).toBeVisible();
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
