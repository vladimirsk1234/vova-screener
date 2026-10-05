/**
 * Runs the iPhone app's bundled web UI in Chromium at iPhone size, wired to the same on-device
 * /api (SQLite + Yahoo) the WebView bridge uses, and saves screenshots. Not part of the app.
 *
 *   node node_modules/tsx/dist/cli.mjs apps/ios/scripts/preview-harness.mts <outDir> [stocks] [etfs]
 */
import { mkdirSync, mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { chromium } from '@playwright/test';
import { createCachedStore } from '../../../packages/device/src/cachedStore';
import { createDeviceApi } from '../../../packages/device/src/api';
import { openNodeStore } from '../../../packages/device/src/nodeStore.test-helper';
import { parseUniverse } from '../../../packages/device/src/scan';

const root = path.resolve(import.meta.dirname, '../../..');
const outDir = path.resolve(process.argv[2] ?? '/tmp/vova-preview');
const stockCount = Number(process.argv[3] ?? 40);
const etfCount = Number(process.argv[4] ?? 15);
const BASE_URL = 'https://sv-screener.app/';

const generated = readFileSync(path.join(root, 'apps/ios/src/webApp.generated.ts'), 'utf8');
const html = JSON.parse(generated.slice(generated.indexOf('= ') + 2, generated.lastIndexOf(';'))) as string;
const full = parseUniverse(
  readFileSync(path.join(root, 'STOCK-TICKERS.txt'), 'utf8'),
  readFileSync(path.join(root, 'TV-LIST-ETF.txt'), 'utf8'),
);
const universe = { Stocks: full.Stocks.slice(0, stockCount), ETF: full.ETF.slice(0, etfCount) };

mkdirSync(outDir, { recursive: true });
const disk = await openNodeStore(path.join(mkdtempSync(path.join(tmpdir(), 'vova-preview-')), 'vova.db'));
const browser = await chromium.launch();
const context = await browser.newContext({
  viewport: { width: 390, height: 844 },
  deviceScaleFactor: 2,
  isMobile: true,
  hasTouch: true,
});
const page = await context.newPage();
const errors: string[] = [];
page.on('pageerror', (err) => errors.push(String(err)));
page.on('console', (msg) => {
  if (msg.type() === 'error') errors.push(msg.text());
});

let calls = 0;
const api = createDeviceApi({
  store: createCachedStore(disk),
  universe,
  onDataChanged: () => void page.evaluate('window.__vovaDataChanged && window.__vovaDataChanged()').catch(() => undefined),
});
await page.exposeFunction('__nativePost', async (raw: string) => {
  const msg = JSON.parse(raw) as { id: number; method: string; path: string; body: string | null };
  calls += 1;
  const res = await api.handle(msg.method, msg.path, msg.body);
  // Same call App.tsx injects into the WebView. Strings, because tsx rewrites serialized functions.
  await page.evaluate(`window.__vovaReply(${msg.id}, ${res.status}, ${JSON.stringify(res.body)})`);
});
await page.addInitScript({
  content: 'window.ReactNativeWebView = { postMessage: function (m) { window.__nativePost(m); } };',
});
await page.route(`${BASE_URL}**`, (route) => route.fulfill({ status: 200, contentType: 'text/html', body: html }));

const shot = async (name: string) => {
  await page.waitForTimeout(600);
  await page.screenshot({ path: path.join(outDir, `${name}.png`) });
  console.log(`saved ${name}.png`);
};

await page.goto(BASE_URL);
await page.waitForSelector('.app-header');
await shot('01-results-empty');

// Settings → Run scan now (All), exactly as on the web.
await page.getByRole('button', { name: 'Settings' }).click();
await page.getByRole('button', { name: 'Run scan now' }).click();
await shot('02-settings-scanning');
const started = Date.now();
while (Date.now() - started < 240_000) {
  const busy = await page.getByRole('button', { name: /Scanning…/ }).count();
  if (!busy) break;
  await page.waitForTimeout(1000);
}
// Settings → Rebuild history (confirm dialog), so History has closed trades to chart.
page.on('dialog', (dialog) => void dialog.accept());
await page.getByRole('button', { name: 'Rebuild history' }).click();
const rebuildStarted = Date.now();
while (Date.now() - rebuildStarted < 240_000) {
  if (!(await page.getByRole('button', { name: /Rebuilding…/ }).count())) break;
  await page.waitForTimeout(500);
}
await shot('02b-settings-rebuilt');
await page.getByRole('button', { name: 'Close' }).click();

await page.goto(`${BASE_URL}#/results/Stocks/Weekly/closed`);
await page.waitForSelector('.results-head');
await shot('04-results-stocks-weekly-closed');
await page.goto(`${BASE_URL}#/results/Stocks/Weekly/valid`);
await page.waitForSelector('.signal-card');
await shot('03-results-stocks-weekly-valid');

const card = page.locator('.signal-card').first();
if (await card.count()) {
  const before = calls;
  await card.click();
  await page.waitForSelector('.chart-host canvas', { timeout: 20_000 });
  await shot('05-chart-from-results');
  console.log(`chart open used ${calls - before} bridge calls`);
  await page.getByRole('button', { name: 'Back' }).click();
  const back = calls;
  await page.waitForSelector('.results-head');
  await page.waitForTimeout(300);
  console.log(`back to Results used ${calls - back} bridge calls`);
}

await page.locator('.bottom-nav a', { hasText: 'History' }).click();
await page.waitForSelector('.history-stats', { timeout: 20_000 }).catch(() => undefined);
await shot('06-history');
await page.locator('.sparkline').first().scrollIntoViewIfNeeded().catch(() => undefined);
await shot('07-history-equity');
await page.locator('.tf-growth').first().scrollIntoViewIfNeeded().catch(() => undefined);
await shot('07b-history-growth-by-timeframe');
const historyCard = page.locator('.signal-card').first();
if (await historyCard.count()) {
  await historyCard.scrollIntoViewIfNeeded();
  await shot('07c-history-closed-cards');
  const before = calls;
  await historyCard.click();
  await page.waitForSelector('.chart-host canvas', { timeout: 20_000 });
  await page.mouse.move(5, 5);
  await shot('08-chart-from-history-snapshot');
  console.log(`history chart open used ${calls - before} bridge calls`);
  await page.getByRole('button', { name: 'Back' }).click();
  const back = calls;
  await page.waitForSelector('.history-stats');
  await page.waitForTimeout(300);
  console.log(`back to History used ${calls - back} bridge calls`);
}

console.log(`bridge calls: ${calls}`);
console.log(errors.length ? `page errors:\n${errors.join('\n')}` : 'page errors: none');
await browser.close();
