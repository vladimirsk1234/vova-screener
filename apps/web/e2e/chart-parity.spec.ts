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

  /** Ticker-level mark when the chart is not opened on a tracked signal (Value tab). */
  test('interest buttons stay enabled without a tracked signal', async ({ page }) => {
    await page.goto('/chart/ZZ-NOT-TRACKED');
    await expect(page.getByRole('button', { name: 'Interested', exact: true })).toBeEnabled();
    await expect(page.getByRole('button', { name: 'Not Interested' })).toBeEnabled();
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
    await expect(page.getByRole('button', { name: 'Interested', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Not Interested' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Daily' })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Weekly' })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Monthly' })).toHaveCount(0);
    const periodChips = page.locator('.chart-period-chips');
    await expect(periodChips.getByRole('button', { name: 'MAX', exact: true })).toBeVisible();
    await expect(periodChips.getByRole('button', { name: '15Y', exact: true })).toBeVisible();
    await expect(periodChips.getByRole('button', { name: '10Y', exact: true })).toBeVisible();
    await expect(periodChips.getByRole('button', { name: '1Y', exact: true })).toBeVisible();
    await expect(periodChips.getByRole('button', { name: '3Y', exact: true })).toBeVisible();
    await expect(periodChips.getByRole('button', { name: '5Y', exact: true })).toBeVisible();
    await expect(periodChips.getByRole('button', { name: '8Y', exact: true })).toBeVisible();
    await expect(periodChips.getByRole('button', { name: '19Y', exact: true })).toHaveCount(0);
    await expect(periodChips.getByRole('button', { name: '18Y', exact: true })).toHaveCount(0);
    await expect(periodChips.getByRole('button', { name: '2Y', exact: true })).toHaveCount(0);
    await expect(periodChips.getByRole('button', { name: '4Y', exact: true })).toHaveCount(0);
    await expect(periodChips.getByRole('button', { name: '7Y', exact: true })).toHaveCount(0);
    await expect(page.locator('.chart-fund-metrics').getByRole('button', { name: '1Y', exact: true })).toHaveCount(0);
    const periodBox = await periodChips.boundingBox();
    const chipOverflow = await periodChips.locator('.chip-row').evaluate((el) => ({
      scrollWidth: el.scrollWidth,
      clientWidth: el.clientWidth,
    }));
    expect(chipOverflow.scrollWidth).toBeLessThanOrEqual(chipOverflow.clientWidth + 1);
    const stage = page.locator('.chart-stage');
    await expect(stage).toBeVisible();
    const box = await stage.boundingBox();
    expect(periodBox?.y ?? Number.POSITIVE_INFINITY).toBeLessThan(box?.y ?? 0);
    const fundRow = page.getByTestId('fund-metrics-row');
    await expect(fundRow).toBeVisible();
    await expect(fundRow.getByText('FV', { exact: true })).toBeVisible();
    await expect(fundRow.getByText('Growth', { exact: true })).toBeVisible();
    await expect(fundRow.getByText('LT D/C', { exact: true })).toBeVisible();
    const fundRowBox = await fundRow.boundingBox();
    expect(fundRowBox?.y ?? Number.POSITIVE_INFINITY).toBeLessThan(box?.y ?? 0);
    await expect(page.getByRole('button', { name: 'Summary' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'DCF', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'EPS', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Op. EPS', exact: true })).toBeVisible();
    const legend = page.locator('.chart-fund-hud');
    await expect(legend.getByText('Fair value', { exact: true })).toBeVisible();
    await expect(legend.getByText('Normal P/E').first()).toBeVisible();
    await expect(page.locator('.chart-stage .chart-fund-hud')).toHaveCount(0);
    const legendBox = await legend.boundingBox();
    const toggleBox = await page.locator('.chart-view-toggle').boundingBox();
    expect(legendBox?.y ?? Number.POSITIVE_INFINITY).toBeGreaterThan(box?.y ?? 0);
    expect(legendBox?.y ?? Number.POSITIVE_INFINITY).toBeLessThan(toggleBox?.y ?? 0);
    await expect(page.locator('.chart-watermark')).toHaveCount(0);
    await expect(page.getByText('ATR:')).toHaveCount(0);
    await expect(page.getByText('D: Seq')).toHaveCount(0);
    expect(box?.height ?? 0).toBeGreaterThan(280);
    await expect(page.locator('.chart-page--fundamentals')).toHaveCSS('overflow-y', 'hidden');
    await expect(page.locator('.chart-page--fundamentals .chart-page-body')).toHaveCSS(
      'overflow-y',
      'auto',
    );
    await expect(page.locator('.chart-fund-metrics')).toHaveCSS('max-height', 'none');

    const chrome = page.locator('.chart-top-chrome');
    const chromeBefore = await chrome.boundingBox();
    await page.locator('.chart-page--fundamentals .chart-page-body').evaluate((el) => {
      el.scrollTop = 240;
    });
    const chromeAfter = await chrome.boundingBox();
    expect(chromeAfter?.y ?? Number.POSITIVE_INFINITY).toBe(chromeBefore?.y ?? -1);
    await expect(page.getByRole('button', { name: 'Interested', exact: true })).toBeInViewport();
    await expect(page.getByRole('button', { name: 'Not Interested' })).toBeInViewport();

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
