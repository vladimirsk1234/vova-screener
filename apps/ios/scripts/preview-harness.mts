/**
 * Runs the iPhone app's bundled web UI in Chromium at iPhone size, wired to the same on-device
 * /api (SQLite + Yahoo) the WebView bridge uses, and saves screenshots. Not part of the app.
 * A second launch on the same SQLite file checks that saved settings come back after a restart.
 *
 *   node node_modules/tsx/dist/cli.mjs apps/ios/scripts/preview-harness.mts <outDir> [stocks] [etfs]
 */
import { mkdirSync, mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { chromium, devices, webkit, type Browser, type Page } from '@playwright/test';
import { createCachedStore } from '../../../packages/device/src/cachedStore';
import { createDeviceApi, type Notifier } from '../../../packages/device/src/api';
import { openNodeStore } from '../../../packages/device/src/nodeStore.test-helper';
import { parseUniverse } from '../../../packages/device/src/scan';

const root = path.resolve(import.meta.dirname, '../../..');
const outDir = path.resolve(process.argv[2] ?? '/tmp/vova-preview');
const stockCount = Number(process.argv[3] ?? 40);
const etfCount = Number(process.argv[4] ?? 15);
const BASE_URL = 'https://sv-screener.app/';
const dbFile = path.join(mkdtempSync(path.join(tmpdir(), 'vova-preview-')), 'sequence-vova.db');

const generated = readFileSync(path.join(root, 'apps/ios/src/webApp.generated.ts'), 'utf8');
const html = JSON.parse(generated.slice(generated.indexOf('= ') + 2, generated.lastIndexOf(';'))) as string;
const full = parseUniverse(
  readFileSync(path.join(root, 'STOCK-TICKERS.txt'), 'utf8'),
  readFileSync(path.join(root, 'TV-LIST-ETF.txt'), 'utf8'),
);
const universe = { Stocks: full.Stocks.slice(0, stockCount), ETF: full.ETF.slice(0, etfCount) };
mkdirSync(outDir, { recursive: true });

let permission: 'granted' | 'undetermined' = 'undetermined';
const notifications: string[] = [];
const notifier: Notifier = {
  permission: async () => permission,
  request: async () => {
    permission = 'granted';
    return permission;
  },
  notify: async (title, body) => {
    notifications.push(`${title} — ${body}`);
  },
  scheduleReminders: async (reminders) => {
    console.log(`scheduled reminders: ${reminders.map((r) => `${r.id}@${r.at.toISOString()}`).join(', ') || 'none'}`);
  },
};

/** One app launch: fresh WebView and a fresh device API over the same SQLite file. */
async function launch(browser: Browser, label: string) {
  const context = await browser.newContext(
    process.env.ENGINE === 'webkit'
      ? { ...devices['iPhone 14'] }
      : { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true },
  );
  const page = await context.newPage();
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(String(err)));
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(msg.text());
  });
  page.on('dialog', (dialog) => void dialog.accept());
  const counter = { calls: 0 };
  const api = createDeviceApi({
    store: createCachedStore(await openNodeStore(dbFile)),
    universe,
    notifier,
    onDataChanged: () => void page.evaluate('window.__vovaDataChanged && window.__vovaDataChanged()').catch(() => undefined),
  });
  await api.migratePresets();
  await api.restoreAlerts();
  await page.exposeFunction('__nativePost', async (raw: string) => {
    const msg = JSON.parse(raw) as { id: number; method: string; path: string; body: string | null };
    counter.calls += 1;
    const res = await api.handle(msg.method, msg.path, msg.body);
    // Same call App.tsx injects into the WebView. Strings, because tsx rewrites serialized functions.
    await page.evaluate(`window.__vovaReply(${msg.id}, ${res.status}, ${JSON.stringify(res.body)})`);
  });
  await page.addInitScript({ content: 'window.ReactNativeWebView = { postMessage: function (m) { window.__nativePost(m); } };' });
  await page.route(`${BASE_URL}**`, (route) => route.fulfill({ status: 200, contentType: 'text/html', body: html }));
  await page.goto(BASE_URL);
  await page.waitForSelector('.app-header');
  console.log(`launch ${label}`);
  return { page, context, errors, counter };
}

async function shot(page: Page, name: string) {
  await page.waitForTimeout(600);
  await page.screenshot({ path: path.join(outDir, `${name}.png`) });
  console.log(`saved ${name}.png`);
}

async function waitWhile(page: Page, name: RegExp, ms: number) {
  const started = Date.now();
  while (Date.now() - started < ms) {
    if (!(await page.getByRole('button', { name }).count())) return;
    await page.waitForTimeout(500);
  }
}

// ENGINE=webkit runs WebKit, the engine behind WKWebView on the iPhone.
const browser = await (process.env.ENGINE === 'webkit' ? webkit : chromium).launch();
const first = await launch(browser, '1');
const page = first.page;

// Settings → Alerts: turn some on and Save (asks for permission), then run a scan.
await page.getByRole('button', { name: 'Settings' }).click();
await page.getByText('New signals', { exact: true }).click();
await page.getByText('Sell to close', { exact: true }).click();
await page.getByText('Scan finished', { exact: true }).click();
await page.getByText('Weekly close reminder (Fri 16:15 ET)', { exact: true }).click();
await page.getByRole('button', { name: 'Save alerts' }).click();
await page.getByRole('button', { name: 'Saved' }).waitFor();
await page.getByText('Alerts', { exact: true }).scrollIntoViewIfNeeded();
await shot(page, '01-settings-alerts-saved');
await page.getByRole('button', { name: 'Run scan now' }).click();
await waitWhile(page, /Scanning…/, 240_000);
await page.getByRole('button', { name: 'Rebuild history' }).click();
await waitWhile(page, /Rebuilding…/, 240_000);
await page.getByRole('button', { name: 'Close' }).click();
console.log(`notifications raised during the scan:\n  ${notifications.join('\n  ') || 'none'}`);

await page.goto(`${BASE_URL}#/results/Stocks/Weekly/valid`);
await page.waitForSelector('.signal-card');
await shot(page, '02-results-valid');
await page.locator('.signal-card').first().click();
await page.waitForSelector('.chart-host canvas', { timeout: 20_000 });
await page.mouse.move(5, 5);
await shot(page, '03-chart-defaults');
const ticker = decodeURIComponent(new URL(page.url()).hash.split('/chart/')[1] ?? '');

// Visibility switches under the chart: turn Fibonacci and Bollinger Bands on, then Save preset.
await page.locator('.chart-visibility').getByText('Fibonacci').click();
await page.locator('.chart-visibility').getByText('Bollinger Bands').click();
await page.locator('.chart-visibility input[type="number"]').first().fill('9');
await shot(page, '04-chart-visibility-on');
await page.getByRole('button', { name: 'Settings' }).click();
await shot(page, '05-chart-settings-no-visibility');
await page.getByRole('button', { name: 'Save preset' }).click();
await page.getByRole('button', { name: 'Saved' }).waitFor();
await page.getByRole('button', { name: 'Close' }).click();
// Same session: open another chart and come back — the saved preset must be what loads.
await page.getByRole('button', { name: 'Back' }).click();
await page.waitForSelector('.results-head');
await page.locator('.signal-card').first().click();
await page.waitForSelector('.chart-visibility');
await page.waitForTimeout(800);
const readControls = (p: Page) =>
  p.locator('.chart-visibility input').evaluateAll((els) =>
    els.map((el) => ((el as HTMLInputElement).type === 'checkbox' ? (el as HTMLInputElement).checked : (el as HTMLInputElement).value)),
  );
const sameSession = await readControls(page);
console.log(`same session, next chart: visibility ${JSON.stringify(sameSession)}`);
console.log(`page errors (launch 1): ${first.errors.join(' | ') || 'none'}`);
await first.context.close();

// Relaunch on the same SQLite file: what a cold start or an app update sees.
const second = await launch(browser, '2 (same SQLite file)');
await second.page.goto(`${BASE_URL}#/chart/${encodeURIComponent(ticker)}`);
await second.page.waitForSelector('.chart-host canvas', { timeout: 20_000 });
await second.page.mouse.move(5, 5);
await second.page.waitForTimeout(800);
const afterRelaunch = await readControls(second.page);
console.log(`after relaunch: visibility ${JSON.stringify(afterRelaunch)}`);
await shot(second.page, '06-chart-after-relaunch');
await second.page.getByRole('button', { name: 'Back' }).click().catch(() => undefined);
await second.page.goto(BASE_URL);
await second.page.getByRole('button', { name: 'Settings' }).click();
await second.page.getByText('Alerts', { exact: true }).scrollIntoViewIfNeeded();
const alertState = await second.page.locator('.switch-row input').evaluateAll((els) =>
  els.map((el) => `${(el.parentElement?.textContent ?? '').trim()}=${(el as HTMLInputElement).checked}`),
);
console.log(`alerts after relaunch: ${alertState.join(', ')}`);
await shot(second.page, '07-settings-alerts-after-relaunch');
console.log(`page errors (launch 2): ${second.errors.join(' | ') || 'none'}`);
await browser.close();
