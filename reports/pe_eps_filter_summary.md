# PE/EPS vs Sequence Vova closed trades

Generated: 2026-08-05T04:36:14Z

## Caveat (look-ahead)

Fundamentals come from **current** Yahoo Finance `.info` via yfinance (`trailingPE`, `forwardPE`, `trailingEps`, `forwardEps`). They are **not** point-in-time at trade entry. Results are a hypothesis for a live filter on **new** signals, not a strict historical backtest.

## Data

- Closed trades: **3446** ({'ledger': 3446})
- Sample: first **600** tickers of **3290** in `STOCK-TICKERS.txt` (~18% of list order)
- Universe: Stocks only (`STOCK-TICKERS.txt`); ETFs excluded
- Ledger params: `min_rr=1.5`, `risk_dollars=100.0`, `no_rr_req=False`
- Windows: Daily ~2y daily bars; Weekly/Monthly resampled from 10y daily (Yahoo `10y` / `1d`)
- Win definition: `pnl_usd > 0`
- Improve criterion: win rate ↑ AND (net P&L ≥ baseline OR P&L/trade ↑ with P&L ≥ 90% baseline) AND cut losses > cut wins; min 30 trades

## Baseline

| TF | Trades | Win rate | Net P&L | Avg R | Avg RR entry |
|---|---:|---:|---:|---:|---:|
| All | 3446 | 34.27% | 27961.23 | 0.081 | 2.668 |
| Daily | 1716 | 35.55% | 9792.54 | 0.057 | 2.574 |
| Weekly | 1407 | 34.75% | 21940.52 | 0.156 | 2.67 |
| Monthly | 323 | 25.39% | -3771.83 | -0.117 | 3.162 |

## PE buckets (All TF)

| Bucket | Trades | Wins | Win rate | Net P&L | Avg R |
|---|---:|---:|---:|---:|---:|
| (0,15] | 1190 | 378 | 31.76% | -3905.64 | -0.033 |
| (15,25] | 674 | 241 | 35.76% | 8528.07 | 0.127 |
| (25,40] | 504 | 211 | 41.87% | 11539.62 | 0.229 |
| >40 | 725 | 240 | 33.10% | 10694.45 | 0.148 |
| missing | 353 | 111 | 31.44% | 1104.73 | 0.031 |

## Trailing EPS buckets (All TF)

| Bucket | Trades | Wins | Win rate | Net P&L | Avg R |
|---|---:|---:|---:|---:|---:|
| <=0 | 353 | 111 | 31.44% | 1104.73 | 0.031 |
| >0 | 3071 | 1067 | 34.74% | 27424.64 | 0.089 |
| missing | 22 | 3 | 13.64% | -568.13 | -0.258 |

## Forward EPS buckets (All TF)

| Bucket | Trades | Wins | Win rate | Net P&L | Avg R |
|---|---:|---:|---:|---:|---:|
| <=0 | 328 | 83 | 25.30% | -3699.51 | -0.113 |
| >0 | 2657 | 960 | 36.13% | 31823.13 | 0.12 |
| missing | 461 | 138 | 29.93% | -162.39 | -0.004 |

## Filter simulation vs baseline

| TF | Filter | Rule | Trades | Win rate | Δ WR | Net P&L | Δ P&L | P&L/trade | Cut wins | Cut losses | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| All | F1 | trailingPE > 0 | 3093 | 34.59% | 0.32 | 26856.5 | -1104.73 | 8.68 | 111 | 242 | improves |
| All | F2 | trailingEps > 0 | 3071 | 34.74% | 0.47 | 27424.64 | -536.59 | 8.93 | 114 | 261 | improves |
| All | F3 | trailingPE in [5, 25] | 1659 | 34.54% | 0.27 | 6347.3 | -21613.93 | 3.83 | 608 | 1179 | wr_up_pnl_down |
| All | F4 | forwardPE < trailingPE (both > 0) | 2092 | 36.28% | 2.01 | 20508.45 | -7452.78 | 9.8 | 422 | 932 | wr_up_pnl_down |
| All | F5 | F1 ∧ F2 | 3071 | 34.74% | 0.47 | 27424.64 | -536.59 | 8.93 | 114 | 261 | improves |
| Daily | F1 | trailingPE > 0 | 1530 | 36.14% | 0.59 | 12287.48 | 2494.94 | 8.03 | 57 | 129 | improves |
| Daily | F2 | trailingEps > 0 | 1519 | 36.27% | 0.72 | 12463.41 | 2670.87 | 8.21 | 59 | 138 | improves |
| Daily | F3 | trailingPE in [5, 25] | 816 | 36.40% | 0.85 | 2470.11 | -7322.43 | 3.03 | 313 | 587 | wr_up_pnl_down |
| Daily | F4 | forwardPE < trailingPE (both > 0) | 1030 | 36.70% | 1.15 | 3977.26 | -5815.28 | 3.86 | 232 | 454 | wr_up_pnl_down |
| Daily | F5 | F1 ∧ F2 | 1519 | 36.27% | 0.72 | 12463.41 | 2670.87 | 8.21 | 59 | 138 | improves |
| Weekly | F1 | trailingPE > 0 | 1266 | 34.91% | 0.16 | 19097.7 | -2842.82 | 15.09 | 47 | 94 | wr_up_pnl_down |
| Weekly | F2 | trailingEps > 0 | 1257 | 35.08% | 0.33 | 19413.4 | -2527.12 | 15.44 | 48 | 102 | wr_up_pnl_down |
| Weekly | F3 | trailingPE in [5, 25] | 688 | 34.88% | 0.13 | 8068.25 | -13872.27 | 11.73 | 249 | 470 | wr_up_pnl_down |
| Weekly | F4 | forwardPE < trailingPE (both > 0) | 854 | 37.35% | 2.6 | 19916.38 | -2024.14 | 23.32 | 170 | 383 | improves |
| Weekly | F5 | F1 ∧ F2 | 1257 | 35.08% | 0.33 | 19413.4 | -2527.12 | 15.44 | 48 | 102 | wr_up_pnl_down |
| Monthly | F1 | trailingPE > 0 | 297 | 25.25% | -0.14 | -4528.68 | -756.85 | -15.25 | 7 | 19 | no_improve |
| Monthly | F2 | trailingEps > 0 | 295 | 25.42% | 0.03 | -4452.18 | -680.35 | -15.09 | 7 | 21 | wr_up_pnl_down |
| Monthly | F3 | trailingPE in [5, 25] | 155 | 23.23% | -2.16 | -4191.07 | -419.24 | -27.04 | 46 | 122 | no_improve |
| Monthly | F4 | forwardPE < trailingPE (both > 0) | 208 | 29.81% | 4.42 | -3385.18 | 386.65 | -16.27 | 20 | 95 | improves |
| Monthly | F5 | F1 ∧ F2 | 295 | 25.42% | 0.03 | -4452.18 | -680.35 | -15.09 | 7 | 21 | wr_up_pnl_down |

## Verdict

Filters that met the improve criterion:

- **F1** on All: WR 34.59% (Δ 0.32), P&L 26856.5 (Δ -1104.73), cut 242 losses / 111 wins
- **F2** on All: WR 34.74% (Δ 0.47), P&L 27424.64 (Δ -536.59), cut 261 losses / 114 wins
- **F5** on All: WR 34.74% (Δ 0.47), P&L 27424.64 (Δ -536.59), cut 261 losses / 114 wins
- **F1** on Daily: WR 36.14% (Δ 0.59), P&L 12287.48 (Δ 2494.94), cut 129 losses / 57 wins
- **F2** on Daily: WR 36.27% (Δ 0.72), P&L 12463.41 (Δ 2670.87), cut 138 losses / 59 wins
- **F5** on Daily: WR 36.27% (Δ 0.72), P&L 12463.41 (Δ 2670.87), cut 138 losses / 59 wins
- **F4** on Weekly: WR 37.35% (Δ 2.6), P&L 19916.38 (Δ -2024.14), cut 383 losses / 170 wins
- **F4** on Monthly: WR 29.81% (Δ 4.42), P&L -3385.18 (Δ 386.65), cut 95 losses / 20 wins

Stronger hits (WR Δ ≥ 0.5pp **and** net P&L not down): F1/Daily, F2/Daily, F5/Daily, F4/Monthly.

Practical read: **F2 / F5 (`trailingEps > 0`)** and **F1 (`trailingPE > 0`)** help most on **Daily** (WR and net P&L both up). **F3 value band [5,25]** cuts too much winner P&L. **F4** lifts WR on Weekly/Monthly but often sacrifices total P&L on All/Daily. Bucket note: PE (25,40] has the highest raw WR here; cheap PE (0,15] underperforms — so a tight value band is not supported. Forward EPS > 0 also shows a cleaner WR/P&L split than trailing alone (see Forward EPS buckets).

Even with uplift, persist PE/EPS at signal open before trusting History segmentation — current Yahoo values still embed look-ahead. Prefer optional scan gates over hard defaults until point-in-time data exists.

### Source recommendation

**yfinance / Yahoo `.info` is the right free source** for this stack (already used for watermark and gap-scan `trailingPE > 0`). Do not add Finviz scraping for this analysis.

Trade-level CSV: `reports/pe_eps_trade_analysis.csv`

