import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: 'list',
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:5173',
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'pixel7',
      use: { ...devices['Pixel 7'] },
    },
    {
      name: 'desktop',
      use: { ...devices['Desktop Chrome'], viewport: { width: 1280, height: 720 } },
    },
  ],
  webServer: process.env.PLAYWRIGHT_NO_SERVER
    ? undefined
    : {
        command: 'npm run dev -w @vova/web',
        url: 'http://127.0.0.1:5173',
        reuseExistingServer: true,
        timeout: 120_000,
      },
});
