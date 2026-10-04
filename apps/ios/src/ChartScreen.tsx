import { buildChart, fetchYahooDailyCloses, fetchYahooOhlc, type ChartPayload, type UserTf } from '@vova/device';
import { useEffect, useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { WebView } from 'react-native-webview';
import { chartHtml } from './chartHtml';
import { money, num } from './format';
import { useScreener } from './state';
import { colors } from './theme';
import { Chips } from './ui';

export function ChartScreen({
  ticker,
  tf,
  asOf,
  onBack,
}: {
  ticker: string;
  tf: UserTf;
  asOf?: string;
  onBack: () => void;
}) {
  const { store } = useScreener();
  const [frame, setFrame] = useState<UserTf>(tf);
  const [payload, setPayload] = useState<ChartPayload | null>(null);
  const [prefs, setPrefs] = useState({ showFib: false, showEma: false, showBb: false, showTpSl: true });
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!store) return;
    let cancelled = false;
    setPayload(null);
    setError(null);
    (async () => {
      const saved = await store.getChartSettings();
      if (!cancelled) setPrefs(saved);
      let cached = await store.getBars(ticker, frame);
      if (!cached?.bars.length) {
        const pulled = await fetchYahooOhlc(ticker, frame);
        if (!pulled.bars) {
          if (!cancelled) setError('Yahoo returned no bars for this symbol.');
          return;
        }
        cached = {
          yahooTicker: ticker,
          tf: frame,
          bars: pulled.bars,
          fetchedAt: new Date().toISOString(),
          companyName: pulled.companyName,
        };
        await store.putBars(cached);
      }
      const other = frame === 'Weekly' ? 'Monthly' : 'Weekly';
      const otherBars = await store.getBars(ticker, other);
      let dailyClose: number | null = null;
      let prevDailyClose: number | null = null;
      const daily = await fetchYahooDailyCloses(ticker, { range: '5d' });
      if (daily && daily.length >= 2) {
        dailyClose = daily[daily.length - 1].close;
        prevDailyClose = daily[daily.length - 2].close;
      }
      const chart = buildChart({
        yahooTicker: ticker,
        symbol: cached.yahooTicker,
        companyName: cached.companyName || ticker,
        tf: frame,
        bars: cached.bars,
        asOf: asOf ?? null,
        dailyClose,
        prevDailyClose,
        weeklyBars: frame === 'Weekly' ? cached.bars : otherBars?.bars,
        monthlyBars: frame === 'Monthly' ? cached.bars : otherBars?.bars,
      });
      if (!cancelled) setPayload(chart);
    })().catch((err: Error) => {
      if (!cancelled) setError(err.message);
    });
    return () => {
      cancelled = true;
    };
  }, [store, ticker, frame, asOf]);

  const toggle = async (key: 'showFib' | 'showEma' | 'showBb' | 'showTpSl') => {
    const next = { ...prefs, [key]: !prefs[key] };
    setPrefs(next);
    await store?.putChartSettings(next);
  };

  return (
    <View style={styles.page}>
      <View style={styles.head}>
        <Pressable onPress={onBack} style={styles.back}><Text style={styles.backText}>Back</Text></Pressable>
        <Text style={styles.title}>{payload?.symbol ?? ticker}</Text>
      </View>
      <Chips label="Timeframe" value={frame} options={['Weekly', 'Monthly']} onChange={setFrame} format={(v) => (v === 'Weekly' ? 'W' : 'M')} />
      <View style={styles.toggles}>
        <Toggle label="Fib" on={prefs.showFib} onPress={() => void toggle('showFib')} />
        <Toggle label="EMA" on={prefs.showEma} onPress={() => void toggle('showEma')} />
        <Toggle label="BB" on={prefs.showBb} onPress={() => void toggle('showBb')} />
        <Toggle label="TP/SL" on={prefs.showTpSl} onPress={() => void toggle('showTpSl')} />
      </View>
      {payload?.pine ? (
        <Text style={styles.metrics}>
          {payload.pine.valid ? (payload.pine.isNew ? 'NEW' : 'VALID') : 'NO SETUP'} · RR {num(payload.pine.rr)} · TP {money(payload.pine.tp)} · SL {money(payload.pine.sl)}
        </Text>
      ) : null}
      {error ? <Text style={styles.error}>{error}</Text> : null}
      {!payload && !error ? <ActivityIndicator color={colors.accent} style={{ marginTop: 24 }} /> : null}
      {payload ? (
        <WebView
          originWhitelist={['*']}
          source={{ html: chartHtml(payload, prefs) }}
          style={styles.chart}
          scrollEnabled={false}
        />
      ) : null}
      <ScrollView style={styles.wm}>
        {(payload?.watermarkLines ?? []).map((line) => (
          <Text key={line} style={styles.wmLine}>{line}</Text>
        ))}
      </ScrollView>
    </View>
  );
}

function Toggle({ label, on, onPress }: { label: string; on: boolean; onPress: () => void }) {
  return (
    <Pressable onPress={onPress} style={[styles.toggle, on && styles.toggleOn]}>
      <Text style={styles.toggleText}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  page: { flex: 1, backgroundColor: colors.bg, padding: 12 },
  head: { flexDirection: 'row', alignItems: 'center', gap: 12, marginBottom: 8 },
  back: { minHeight: 44, justifyContent: 'center' },
  backText: { color: colors.accent, fontSize: 16 },
  title: { color: colors.text, fontSize: 20, fontWeight: '700' },
  toggles: { flexDirection: 'row', gap: 8, marginBottom: 8 },
  toggle: { minHeight: 36, paddingHorizontal: 10, borderRadius: 8, backgroundColor: colors.chip, justifyContent: 'center' },
  toggleOn: { backgroundColor: colors.accent },
  toggleText: { color: colors.text, fontSize: 13 },
  metrics: { color: colors.text, marginBottom: 8 },
  chart: { flex: 1, backgroundColor: '#131722', borderRadius: 12, minHeight: 280 },
  wm: { maxHeight: 120, marginTop: 8 },
  wmLine: { color: colors.muted, fontSize: 12 },
  error: { color: colors.down, marginVertical: 8 },
});
