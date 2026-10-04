import {
  HISTORY_GROUP_BYS,
  HISTORY_RANGES,
  HISTORY_TFS,
  UNIVERSES,
  fetchYahooDailyCloses,
  historyReport,
  historyTrades,
  setInterest,
  type HistoryGroupBy,
  type HistoryRange,
  type HistoryTf,
  type HistoryTradeSort,
  type SortDir,
  type Universe,
} from '@vova/device';
import { useEffect, useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { money, num, pct, periodLabel, signedMoney, signedMultiple } from './format';
import { useScreener } from './state';
import { colors } from './theme';
import { Chips, SignalCard, Sparkline, Stat, toneOf } from './ui';

const TRADE_SORTS: HistoryTradeSort[] = ['date', 'pnl', 'r', 'rr', 'interest', 'symbol'];

export function HistoryScreen({ onOpenChart }: { onOpenChart: (ticker: string, tf: 'Weekly' | 'Monthly', asOf?: string) => void }) {
  const { store, rev, refresh, settings } = useScreener();
  const [universe, setUniverse] = useState<Universe>('Stocks');
  const [tf, setTf] = useState<HistoryTf>('All');
  const [groupBy, setGroupBy] = useState<HistoryGroupBy>('Weekly');
  const [range, setRange] = useState<HistoryRange>('all');
  const [tradeSort, setTradeSort] = useState<HistoryTradeSort>('date');
  const [tradeDir, setTradeDir] = useState<SortDir>('desc');
  const [openPeriod, setOpenPeriod] = useState<string | null>(null);
  const [data, setData] = useState<Awaited<ReturnType<typeof load>> | null>(null);

  useEffect(() => {
    if (!store) return;
    let cancelled = false;
    load(store, { universe, tf, groupBy, range, tradeSort, tradeDir, openPeriod, minRr: settings.minRr, maxRiskUsd: settings.maxRiskUsd }).then((page) => {
      if (!cancelled) setData(page);
    });
    return () => {
      cancelled = true;
    };
  }, [store, universe, tf, groupBy, range, tradeSort, tradeDir, openPeriod, settings, rev]);

  const report = data?.report;
  const totals = report?.totals;
  return (
    <ScrollView style={styles.page} contentContainerStyle={{ padding: 12, paddingBottom: 32 }}>
      <Text style={styles.h1}>History</Text>
      <Text style={styles.muted}>
        Closed sell-to-close trades. P&L follows the current Max risk. Min RR filters what counts.
      </Text>
      <Chips label="Universe" value={universe} options={UNIVERSES} onChange={setUniverse} />
      <Chips label="Timeframe" value={tf} options={HISTORY_TFS} onChange={(next) => { setTf(next); setOpenPeriod(null); if (next !== 'All') setGroupBy(next); }} />
      <Chips label="Group by" value={groupBy} options={HISTORY_GROUP_BYS} onChange={(next) => { setGroupBy(next); setOpenPeriod(null); }} />
      <Chips label="Range" value={range} options={HISTORY_RANGES} onChange={(next) => { setRange(next); setOpenPeriod(null); }} format={(v) => v.toUpperCase()} />
      {!data ? <ActivityIndicator color={colors.accent} /> : null}
      {totals && report ? (
        <View style={styles.card}>
          <Text style={styles.h2}>Outcome</Text>
          <View style={styles.grid}>
            <Stat label="Win rate" value={`${totals.winRatePct}%`} />
            <Stat label={`Avg hold (${report.holdUnit})`} value={num(totals.avgHold)} />
            <Stat label="Avg trade size" value={money(totals.avgTradeSizeUsd)} />
            <Stat label="Avg winner" value={pct(totals.avgWinPct)} tone="up" />
            <Stat label="Avg loser" value={pct(totals.avgLossPct)} tone="down" />
            <Stat label="Avg R" value={num(totals.avgR)} />
            <Stat label="RR at entry" value={num(totals.avgRrEntry)} />
          </View>
          <Text style={styles.h2}>Vs Max risk</Text>
          <View style={styles.grid}>
            <Stat label="Net P&L" value={signedMoney(totals.pnlUsd)} tone={toneOf(totals.pnlUsd)} />
            <Stat label="Risk / trade" value={money(totals.maxRiskUsd)} />
            <Stat label="Closed / active" value={`${totals.closed} / ${totals.active}`} />
            <Stat label="Total risked" value={money(totals.totalRiskUsd)} />
            <Stat label="Profit / risk" value={signedMultiple(totals.profitToRisk)} tone={toneOf(totals.profitToRisk)} />
            <Stat label="Closed invested" value={money(totals.invested)} />
          </View>
          <Text style={styles.h2}>Capital pool</Text>
          <View style={styles.grid}>
            <Stat label="Peak capital" value={money(totals.peakCapitalUsd)} />
            <Stat label="Peaked" value={totals.peakCapitalAsOf ? periodLabel(totals.peakCapitalAsOf) : '—'} />
            <Stat label="Open now" value={money(totals.openCapitalUsd)} />
            <Stat label="ROI on peak" value={pct(totals.roiOnPeakPct)} tone={toneOf(totals.roiOnPeakPct)} />
            <Stat label="Avg capital" value={money(totals.avgCapitalUsd)} />
            <Stat label="ROI on avg" value={pct(totals.roiOnAvgPct)} tone={toneOf(totals.roiOnAvgPct)} />
            <Stat label={`S&P (${totals.benchmarkSymbol ?? 'SPY'})`} value={pct(totals.benchmarkReturnPct)} tone={toneOf(totals.benchmarkReturnPct)} />
            <Stat label="Alpha vs S&P (peak)" value={pct(totals.alphaVsBenchmarkPct)} tone={toneOf(totals.alphaVsBenchmarkPct)} />
            <Stat label="Alpha vs S&P (avg)" value={pct(totals.alphaOnAvgPct)} tone={toneOf(totals.alphaOnAvgPct)} />
          </View>
          <Sparkline points={report.equity} />
          {report.exitReasons.length ? (
            <Text style={styles.muted}>{report.exitReasons.map((r) => `${r.reason} ${r.count}`).join(' · ')}</Text>
          ) : null}
        </View>
      ) : null}
      {report ? (
        <View style={styles.card}>
          <Text style={styles.h2}>Growth by timeframe</Text>
          {report.timeframes.filter((row) => row.closed > 0).map((row) => (
            <View key={row.tf} style={{ marginBottom: 8 }}>
              <Text style={styles.symbol}>{row.tf} <Text style={toneOf(row.pnlUsd) === 'down' ? styles.down : styles.up}>{signedMoney(row.pnlUsd)}</Text></Text>
              <Sparkline points={row.equity} />
              <Text style={styles.muted}>
                {row.closed} closed · {row.winRatePct}% won · {money(row.avgTradeSizeUsd)} · {pct(row.avgWinPct)} / {pct(row.avgLossPct)}
              </Text>
            </View>
          ))}
          {report.timeframes.every((row) => row.closed === 0) ? <Text style={styles.muted}>No closed trades yet.</Text> : null}
        </View>
      ) : null}
      {report ? (
        <View style={styles.card}>
          <Text style={styles.h2}>Periods</Text>
          {report.periods.length === 0 ? <Text style={styles.muted}>No closed signals yet.</Text> : null}
          {report.periods.map((period) => (
            <Pressable key={period.periodKey} onPress={() => setOpenPeriod(openPeriod === period.periodKey ? null : period.periodKey)} style={styles.period}>
              <Text style={styles.symbol}>{periodLabel(period.periodKey)}</Text>
              <Text style={styles.muted}>
                {signedMoney(period.pnlUsd)} · {period.trades} · {period.winRatePct}% · R {num(period.avgR)} · hold {num(period.avgHold)}
              </Text>
            </Pressable>
          ))}
        </View>
      ) : null}
      <Text style={styles.h2}>{openPeriod ? periodLabel(openPeriod) : 'All closed'} · {data?.trades.total ?? 0}</Text>
      <Chips
        label="Sort closed signals"
        value={tradeSort}
        options={TRADE_SORTS}
        onChange={(next) => {
          if (next === tradeSort) setTradeDir(tradeDir === 'desc' ? 'asc' : 'desc');
          else setTradeSort(next);
        }}
      />
      {data?.trades.rows.map((row) => (
        <SignalCard
          key={row.id}
          row={row}
          bucket="closed"
          onOpen={() => onOpenChart(row.yahooTicker, row.tf, row.exitDate ?? undefined)}
          onInterest={async (next) => {
            if (!store) return;
            const all = await store.listSignals();
            await store.saveSignals(setInterest(all, row.id, next));
            refresh();
          }}
        />
      ))}
    </ScrollView>
  );
}

async function load(
  store: NonNullable<ReturnType<typeof useScreener>['store']>,
  opts: {
    universe: Universe;
    tf: HistoryTf;
    groupBy: HistoryGroupBy;
    range: HistoryRange;
    tradeSort: HistoryTradeSort;
    tradeDir: SortDir;
    openPeriod: string | null;
    minRr: number;
    maxRiskUsd: number;
  },
) {
  const signals = await store.listSignals();
  let benchmark = await store.getBenchmark();
  const stale = !benchmark || Date.now() - Date.parse(benchmark.fetchedAt) > 24 * 60 * 60 * 1000;
  if (stale) {
    const spy = await fetchYahooDailyCloses('SPY');
    const gspc = spy?.length ? null : await fetchYahooDailyCloses('^GSPC');
    const bars = spy?.length ? spy : gspc;
    if (bars?.length) {
      benchmark = { symbol: spy?.length ? 'SPY' : '^GSPC', bars, fetchedAt: new Date().toISOString() };
      await store.putBenchmark(benchmark);
    }
  }
  const report = historyReport({
    signals,
    universe: opts.universe,
    tf: opts.tf,
    groupBy: opts.groupBy,
    range: opts.range,
    minRr: opts.minRr,
    maxRiskUsd: opts.maxRiskUsd,
    benchmark: benchmark ? { symbol: benchmark.symbol, bars: benchmark.bars } : null,
  });
  const trades = historyTrades({
    signals,
    universe: opts.universe,
    tf: opts.tf,
    groupBy: opts.groupBy,
    range: opts.range,
    periodKey: opts.openPeriod ?? undefined,
    sort: opts.tradeSort,
    dir: opts.tradeDir,
    minRr: opts.minRr,
    maxRiskUsd: opts.maxRiskUsd,
    limit: 200,
  });
  return { report, trades };
}

const styles = StyleSheet.create({
  page: { flex: 1, backgroundColor: colors.bg },
  h1: { color: colors.text, fontSize: 22, fontWeight: '700', marginBottom: 4 },
  h2: { color: colors.text, fontSize: 16, fontWeight: '700', marginVertical: 8 },
  muted: { color: colors.muted, fontSize: 12, marginBottom: 8 },
  card: { backgroundColor: colors.surface, borderRadius: 12, padding: 12, marginBottom: 12 },
  grid: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between' },
  symbol: { color: colors.text, fontWeight: '700' },
  up: { color: colors.up },
  down: { color: colors.down },
  period: { paddingVertical: 8, borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: colors.chip },
});
