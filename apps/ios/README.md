# Sequence Vova for iPhone

The app runs the web UI from `apps/web` inside a WebView: the same React screens, `styles.css`, Lightweight Charts `mountSequenceChart`, chart watermark position, History statistics and sparklines, and the same React Query caching when you move between pages. Its `/api` requests do not go to a server. They cross a bridge into `packages/device/src/api.ts`, which answers the same routes the NestJS API did from a SQLite file on the phone (`sequence-vova.db`) and Yahoo bars. The app does not call Railway and does not use MongoDB.

iPhone-only chart changes: chart background and watermark text default to black, the right price-scale numbers use the grid colour, and the Visibility switches (Fibonacci, Short EMA, Center EMA, SMA, Bollinger Bands, TP / SL lines) sit under the chart instead of in the chart settings sheet.

## Saved settings

Chart **Save preset** and **Save alerts** write to the `presets` table in `Documents/SQLite/sequence-vova.db` (expo-sqlite's default directory). Keys are fixed names (`chart`, `device:alerts`), not tied to an app version, so an App Store or TestFlight update keeps them; only deleting the app removes them. A cold start reads them back from that file.

## Alerts

Settings → Alerts (local notifications only, no push server). Saving with an alert on asks iOS for permission.

| Alert | When it fires |
|-------|---------------|
| New signals, Sell to close, Closing on the bar in progress | After **Run scan now** finishes a universe × timeframe, for what changed in New / Closed / CLOSING. Only while the app is open — no scan runs while iOS has the app suspended. |
| Scan finished | When a **Run scan now** pass completes. App open only. |
| Weekly close reminder | Friday 16:15 ET, scheduled with iOS — arrives with the app closed. Reminds you to scan; scans nothing. |
| Monthly close reminder | Last weekday of the month 16:15 ET, same as above. Market holidays are not modelled. |

Scope switches (Stocks, ETF, Weekly, Monthly, Only Interested) narrow the scan alerts.

The icon is the web app's own icon (`apps/web/public/icon-512.png`), scaled to 1024×1024 for iOS.

## What is left out

Everything that comes from Financial Modeling Prep: the Value tab, the Fundamentals chart view, EPS / FCF / DCF premia on cards, the undervalued filter and UV sort, and the EPS / UV tagging in History. The iPhone build of the web UI (`VITE_TARGET=device`) leaves those screens and controls out, and the device API answers fundamentals routes with 404. The chart watermark keeps its lines and its place, but without the FMP market cap and the `PE / Earn` line.

## What cannot live only on the phone

- **Unattended hourly scans.** **Run scan now** in Settings downloads Stocks and ETF onto the phone while the app is open. iOS suspends a backgrounded app, so the old 09:05–17:05 ET cron cannot keep running on its own.
- **The old trade journal.** Those rows lived in MongoDB (`legacyTrades`) and are not in this repository. History on the phone starts from trades this app closes, plus **Rebuild history** over bars already stored on the device.

## Working on it

After changing anything under `apps/web` or `packages/engine`, rebuild the bundled UI:

```bash
npm run build:web-app -w @vova/ios
```

CI checks that `src/webApp.generated.ts` matches the web sources.

To see the iPhone UI against the on-device API without a phone (Chromium at iPhone size, real Yahoo data, screenshots):

```bash
npx playwright install chromium
node node_modules/tsx/dist/cli.mjs apps/ios/scripts/preview-harness.mts /tmp/vova-preview 40 15
```

## TestFlight

`expo.ios.appleTeamId` in `app.json` and `submit.testflight.ios.appleTeamId` in `eas.json` are `8F5M9YCRZG`. No other Apple id is set. There is no App Store listing, and the repo stays private.

```bash
cd apps/ios
npx eas build -p ios --profile testflight
```

A build needs `EXPO_TOKEN` (or `npx eas login`). A non-interactive submit also needs an App Store Connect API key (key id, issuer id, and `.p8`). Do not commit them.
