/** On-device SQLite journal: scan history, trades, monthly P&L. */
import * as SQLite from 'expo-sqlite';
import type { BuyRow, ResultRow, ScanParams, SellRow } from '../types';
import { fetchYahooOhlc } from '../yahoo/client';

export type ScanRunRow = {
  id: number;
  created_at: string;
  tf: string;
  direction: string;
  source: string;
  min_rr: number;
  risk: number;
  as_of: string | null;
  params_json: string;
  signal_count: number;
};

export type TradeRow = {
  id: number;
  symbol: string;
  yahoo_ticker: string;
  tf: string;
  opened_at: string;
  as_of: string | null;
  entry: number;
  tp: number;
  sl: number;
  rr_at_entry: number;
  shares: number;
  risk_usd: number;
  status: 'open' | 'closed';
  exit_price: number | null;
  exit_date: string | null;
  exit_reason: string | null;
  pnl_usd: number | null;
  pnl_r: number | null;
};

let dbPromise: Promise<SQLite.SQLiteDatabase> | null = null;

export async function getDb() {
  if (!dbPromise) {
    dbPromise = (async () => {
      const db = await SQLite.openDatabaseAsync('vova_journal.db');
      await db.execAsync(`
        PRAGMA journal_mode = WAL;
        CREATE TABLE IF NOT EXISTS scan_runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at TEXT NOT NULL,
          tf TEXT NOT NULL,
          direction TEXT NOT NULL,
          source TEXT NOT NULL,
          min_rr REAL NOT NULL,
          risk REAL NOT NULL,
          as_of TEXT,
          params_json TEXT NOT NULL,
          signal_count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS signals (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id INTEGER NOT NULL,
          symbol TEXT NOT NULL,
          yahoo_ticker TEXT,
          company TEXT,
          payload_json TEXT NOT NULL,
          FOREIGN KEY(run_id) REFERENCES scan_runs(id)
        );
        CREATE TABLE IF NOT EXISTS trades (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          symbol TEXT NOT NULL,
          yahoo_ticker TEXT NOT NULL,
          tf TEXT NOT NULL,
          opened_at TEXT NOT NULL,
          as_of TEXT,
          entry REAL NOT NULL,
          tp REAL NOT NULL,
          sl REAL NOT NULL,
          rr_at_entry REAL NOT NULL,
          shares INTEGER NOT NULL,
          risk_usd REAL NOT NULL,
          status TEXT NOT NULL,
          exit_price REAL,
          exit_date TEXT,
          exit_reason TEXT,
          pnl_usd REAL,
          pnl_r REAL
        );
        CREATE INDEX IF NOT EXISTS idx_trades_open ON trades(symbol, tf, status);
      `);
      return db;
    })();
  }
  return dbPromise;
}

export async function saveScanRun(
  params: ScanParams,
  rows: ResultRow[],
  asOf: string | null,
): Promise<number> {
  const db = await getDb();
  const dataRows = rows.filter((r) => !(r as SellRow)._is_summary);
  const created = new Date().toISOString();
  const result = await db.runAsync(
    `INSERT INTO scan_runs (created_at, tf, direction, source, min_rr, risk, as_of, params_json, signal_count)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    created,
    params.tf,
    params.scanDirection,
    params.source,
    params.minRr,
    params.riskPerTrade,
    asOf,
    JSON.stringify(params),
    dataRows.length,
  );
  const runId = Number(result.lastInsertRowId);
  for (const row of dataRows) {
    await db.runAsync(
      `INSERT INTO signals (run_id, symbol, yahoo_ticker, company, payload_json) VALUES (?, ?, ?, ?, ?)`,
      runId,
      row.tv_symbol,
      row.yahoo_ticker || '',
      row['Company Name'] || '',
      JSON.stringify(row),
    );
  }
  return runId;
}

export async function journalNewBuySignals(
  rows: ResultRow[],
  params: ScanParams,
  asOf: string | null,
  strongOnly = false,
): Promise<number> {
  if (params.scanDirection !== 'buy') return 0;
  const db = await getDb();
  let opened = 0;
  const created = new Date().toISOString();
  for (const row of rows) {
    const buy = row as BuyRow;
    if (buy.New !== 1) continue;
    if (strongOnly && buy.Strong !== 1) continue;
    const openExists = await db.getFirstAsync<{ c: number }>(
      `SELECT COUNT(*) as c FROM trades WHERE symbol = ? AND tf = ? AND status = 'open'`,
      buy.tv_symbol,
      params.tf,
    );
    if (openExists && openExists.c > 0) continue;
    const entry =
      buy.RR > 0 ? (buy.TP + buy.RR * buy.SL) / (buy.RR + 1) : buy.TP;
    await db.runAsync(
      `INSERT INTO trades
       (symbol, yahoo_ticker, tf, opened_at, as_of, entry, tp, sl, rr_at_entry, shares, risk_usd, status)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')`,
      buy.tv_symbol,
      buy.yahoo_ticker,
      params.tf,
      created,
      asOf,
      Math.round(entry * 100) / 100,
      buy.TP,
      buy.SL,
      buy.RR,
      buy['Position Size (shares)'],
      params.riskPerTrade,
    );
    opened += 1;
  }
  return opened;
}

export async function listScanRuns(limit = 50): Promise<ScanRunRow[]> {
  const db = await getDb();
  return db.getAllAsync<ScanRunRow>(
    `SELECT * FROM scan_runs ORDER BY id DESC LIMIT ?`,
    limit,
  );
}

export async function listSignalsForRun(runId: number): Promise<ResultRow[]> {
  const db = await getDb();
  const rows = await db.getAllAsync<{ payload_json: string }>(
    `SELECT payload_json FROM signals WHERE run_id = ?`,
    runId,
  );
  return rows.map((r) => JSON.parse(r.payload_json) as ResultRow);
}

export async function listTrades(status?: 'open' | 'closed'): Promise<TradeRow[]> {
  const db = await getDb();
  if (status) {
    return db.getAllAsync<TradeRow>(
      `SELECT * FROM trades WHERE status = ? ORDER BY id DESC`,
      status,
    );
  }
  return db.getAllAsync<TradeRow>(`SELECT * FROM trades ORDER BY id DESC`);
}

/**
 * Update open trades: walk Yahoo bars after as_of.
 * SL-first same-bar rule: low <= SL before high >= TP → SL; else TP; else stay open.
 */
export async function updateOpenTrades(
  onProgress?: (done: number, total: number) => void,
): Promise<{ closed: number; stillOpen: number }> {
  const db = await getDb();
  const opens = await listTrades('open');
  let closed = 0;
  let stillOpen = 0;
  let done = 0;
  for (const trade of opens) {
    done += 1;
    onProgress?.(done, opens.length);
    try {
      const bars = await fetchYahooOhlc(trade.yahoo_ticker, trade.tf as 'Daily' | 'Weekly' | 'Monthly');
      if (!bars?.length) {
        stillOpen += 1;
        continue;
      }
      const startIdx = trade.as_of
        ? bars.findIndex((b) => b.date > trade.as_of!)
        : 0;
      const from = startIdx < 0 ? bars.length : startIdx;
      let exitPrice: number | null = null;
      let exitDate: string | null = null;
      let reason: string | null = null;
      for (let i = from; i < bars.length; i++) {
        const b = bars[i];
        const hitSl = b.low <= trade.sl;
        const hitTp = b.high >= trade.tp;
        if (hitSl && hitTp) {
          exitPrice = trade.sl;
          exitDate = b.date;
          reason = 'SL';
          break;
        }
        if (hitSl) {
          exitPrice = trade.sl;
          exitDate = b.date;
          reason = 'SL';
          break;
        }
        if (hitTp) {
          exitPrice = trade.tp;
          exitDate = b.date;
          reason = 'TP';
          break;
        }
      }
      if (exitPrice != null && exitDate && reason) {
        const pnl_usd = (exitPrice - trade.entry) * trade.shares;
        const risk = trade.entry - trade.sl;
        const pnl_r = risk > 0 ? (exitPrice - trade.entry) / risk : 0;
        await db.runAsync(
          `UPDATE trades SET status='closed', exit_price=?, exit_date=?, exit_reason=?, pnl_usd=?, pnl_r=? WHERE id=?`,
          exitPrice,
          exitDate,
          reason,
          Math.round(pnl_usd * 100) / 100,
          Math.round(pnl_r * 100) / 100,
          trade.id,
        );
        closed += 1;
      } else {
        stillOpen += 1;
      }
    } catch {
      stillOpen += 1;
    }
  }
  return { closed, stillOpen };
}

export async function manualCloseTrade(
  id: number,
  exitPrice: number,
  exitDate: string,
): Promise<void> {
  const db = await getDb();
  const trade = await db.getFirstAsync<TradeRow>(`SELECT * FROM trades WHERE id = ?`, id);
  if (!trade || trade.status !== 'open') return;
  const pnl_usd = (exitPrice - trade.entry) * trade.shares;
  const risk = trade.entry - trade.sl;
  const pnl_r = risk > 0 ? (exitPrice - trade.entry) / risk : 0;
  await db.runAsync(
    `UPDATE trades SET status='closed', exit_price=?, exit_date=?, exit_reason='MANUAL', pnl_usd=?, pnl_r=? WHERE id=?`,
    exitPrice,
    exitDate,
    Math.round(pnl_usd * 100) / 100,
    Math.round(pnl_r * 100) / 100,
    id,
  );
}

export type MonthlyStats = {
  month: string;
  closed: TradeRow[];
  open: TradeRow[];
  winRate: number;
  sumPnl: number;
  avgPnlR: number;
  avgRr: number;
};

export async function monthlyStats(month: string): Promise<MonthlyStats> {
  // month = YYYY-MM
  const db = await getDb();
  const closed = await db.getAllAsync<TradeRow>(
    `SELECT * FROM trades WHERE status='closed' AND exit_date IS NOT NULL AND substr(exit_date,1,7)=?`,
    month,
  );
  const open = await listTrades('open');
  const wins = closed.filter((t) => (t.pnl_usd ?? 0) > 0).length;
  const sumPnl = closed.reduce((a, t) => a + (t.pnl_usd ?? 0), 0);
  const avgPnlR =
    closed.length > 0
      ? closed.reduce((a, t) => a + (t.pnl_r ?? 0), 0) / closed.length
      : 0;
  const avgRr =
    closed.length > 0
      ? closed.reduce((a, t) => a + t.rr_at_entry, 0) / closed.length
      : 0;
  return {
    month,
    closed,
    open,
    winRate: closed.length ? (wins / closed.length) * 100 : 0,
    sumPnl: Math.round(sumPnl * 100) / 100,
    avgPnlR: Math.round(avgPnlR * 100) / 100,
    avgRr: Math.round(avgRr * 100) / 100,
  };
}

export function tradesToCsv(trades: TradeRow[]): string {
  const header =
    'id,symbol,tf,status,entry,tp,sl,shares,rr,exit_price,exit_date,exit_reason,pnl_usd,pnl_r';
  const lines = trades.map((t) =>
    [
      t.id,
      t.symbol,
      t.tf,
      t.status,
      t.entry,
      t.tp,
      t.sl,
      t.shares,
      t.rr_at_entry,
      t.exit_price ?? '',
      t.exit_date ?? '',
      t.exit_reason ?? '',
      t.pnl_usd ?? '',
      t.pnl_r ?? '',
    ].join(','),
  );
  return [header, ...lines].join('\n');
}
