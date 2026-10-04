# Sequence Vova for iPhone

The screener runs on the phone. Bars, tracked signals, settings, manual-scan history and the S&P series are stored in a local SQLite file (`sequence-vova.db`) and stay there after the app restarts. The app does not call Railway and does not use MongoDB.

Technical analysis and the existing History statistics are included: win rate, P&L, average R, RR at entry, hold, equity curve, capital pool, and S&P alpha from Yahoo `SPY` (then `^GSPC` if SPY fails). Fundamental fields that come from Financial Modeling Prep are not requested and are not shown. That includes the Value tab, EPS / FCF / DCF premia, P/E, market cap, earnings dates, the undervalued filter, and UV sort.

## What cannot live only on the phone

- **Unattended hourly scans.** A full Stocks + ETF pass downloads Yahoo bars while the app is in the foreground. iOS suspends a backgrounded app, so the old server cron (09:05–17:05 ET) cannot keep running after you leave the app. Start **Run scan now** in Settings and leave Sequence Vova open until it finishes. Bars already stored are reused for six hours, so a stopped pass can be continued later.
- **The old trade journal.** Those rows lived in MongoDB (`legacyTrades`) and are not in this repository. History on the phone starts from trades this app closes, plus **Rebuild history** over bars already on the device. Railway is not kept as a back door to that journal.

## TestFlight

This run did **not** upload a build. These secrets are not in the environment:

- `EXPO_TOKEN` (Expo account token)
- Apple Team ID (`appleTeamId`)
- App Store Connect API key (key id, issuer id, and `.p8`) for a non-interactive TestFlight submit

There is no App Store listing in this repo. After the secrets exist:

```bash
cd apps/ios
npx eas login
npx eas init
npx eas build -p ios --profile testflight
```

`eas init` writes the Expo project id into `app.json`. Do not invent one. `eas build -p ios --profile testflight` is the upload path (`distribution: store` produces the IPA EAS can send to TestFlight). A submit still needs the Apple credentials above; without them the build cannot be attached to TestFlight.

The repo stays private. CI on Linux typechecks this package and runs the device tests. It does not compile an iOS binary (that needs macOS or EAS).
