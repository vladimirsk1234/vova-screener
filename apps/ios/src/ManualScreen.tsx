import { evaluateBars, fetchYahooOhlc, parseListEntry, type Rejection, type UserTf } from '@vova/device';
import { useEffect, useState } from 'react';
import { Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import { money, num } from './format';
import { useScreener } from './state';
import { colors } from './theme';
import { Chips } from './ui';

export function ManualScreen({
  onBack,
  onOpenChart,
}: {
  onBack: () => void;
  onOpenChart: (ticker: string, tf: UserTf) => void;
}) {
  const { store, settings } = useScreener();
  const [ticker, setTicker] = useState('');
  const [history, setHistory] = useState<string[]>([]);
  const [tf, setTf] = useState<UserTf>('Weekly');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ symbol: string; yahoo: string; name: string; reason: string | null; detail: Rejection['detail']; entry?: number; tp?: number; sl?: number; rr?: number | null } | null>(null);

  useEffect(() => {
    if (!store) return;
    void store.getManualHistory().then(setHistory);
  }, [store]);

  const start = async () => {
    if (!store) return;
    const parsed = parseListEntry(ticker.trim());
    if (!parsed) {
      setError('Enter one ticker.');
      return;
    }
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const pulled = await fetchYahooOhlc(parsed.yahoo, tf);
      if (pulled.bars) {
        await store.putBars({
          yahooTicker: parsed.yahoo,
          tf,
          bars: pulled.bars,
          fetchedAt: new Date().toISOString(),
          companyName: pulled.companyName || parsed.name,
        });
      }
      const outcome = evaluateBars({
        bars: pulled.bars,
        yahooTicker: parsed.yahoo,
        tvSymbol: parsed.tv,
        companyName: pulled.companyName || parsed.name || parsed.yahoo,
        tf,
        riskPerTrade: settings.maxRiskUsd,
      });
      const nextHistory = [parsed.yahoo, ...history.filter((item) => item !== parsed.yahoo)].slice(0, 10);
      await store.putManualHistory(nextHistory);
      setHistory(nextHistory);
      setResult({
        symbol: outcome.symbol,
        yahoo: parsed.yahoo,
        name: outcome.companyName,
        reason: outcome.buy ? null : outcome.reason,
        detail: outcome.detail,
        entry: outcome.buy?.entry,
        tp: outcome.buy?.tp,
        sl: outcome.buy?.sl,
        rr: outcome.buy?.rr,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <View style={styles.page}>
      <Pressable onPress={onBack} style={styles.back}><Text style={styles.backText}>Back</Text></Pressable>
      <Text style={styles.h1}>Manual scan</Text>
      <Text style={styles.muted}>One symbol, scored on this iPhone from Yahoo bars. No fundamental data is requested.</Text>
      <TextInput
        value={ticker}
        onChangeText={setTicker}
        autoCapitalize="characters"
        autoCorrect={false}
        placeholder="AAPL"
        placeholderTextColor={colors.muted}
        style={styles.input}
      />
      <View style={styles.history}>
        {history.map((item) => (
          <Pressable key={item} onPress={() => setTicker(item)} style={styles.hist}>
            <Text style={styles.histText}>{item}</Text>
          </Pressable>
        ))}
      </View>
      <Chips label="Timeframe" value={tf} options={['Weekly', 'Monthly']} onChange={setTf} />
      <Pressable style={styles.primary} disabled={busy} onPress={() => void start()}>
        <Text style={styles.primaryText}>{busy ? 'Scanning…' : 'Start scan'}</Text>
      </Pressable>
      {error ? <Text style={styles.error}>{error}</Text> : null}
      {result ? (
        <View style={styles.card}>
          <Text style={styles.h1}>{result.symbol}</Text>
          <Text style={styles.muted}>{result.name}</Text>
          {result.reason ? <Text style={styles.error}>{result.reason}</Text> : (
            <Text style={styles.body}>E {money(result.entry)} · TP {money(result.tp)} · SL {money(result.sl)} · RR {num(result.rr)}</Text>
          )}
          {result.detail ? (
            <Text style={styles.muted}>
              bar {result.detail.barDate ?? '—'} · close {money(result.detail.close)} · critical {money(result.detail.criticalLevel)} · RR {num(result.detail.rr)}
            </Text>
          ) : null}
          <Pressable style={styles.secondary} onPress={() => onOpenChart(result.yahoo, tf)}>
            <Text style={styles.primaryText}>Technical chart</Text>
          </Pressable>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  page: { flex: 1, backgroundColor: colors.bg, padding: 12 },
  back: { minHeight: 44, justifyContent: 'center' },
  backText: { color: colors.accent, fontSize: 16 },
  h1: { color: colors.text, fontSize: 22, fontWeight: '700' },
  muted: { color: colors.muted, marginVertical: 6 },
  body: { color: colors.text, marginVertical: 6 },
  input: {
    minHeight: 48,
    borderRadius: 10,
    backgroundColor: colors.surface,
    color: colors.text,
    paddingHorizontal: 12,
    fontSize: 18,
    marginVertical: 8,
  },
  history: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginBottom: 8 },
  hist: { backgroundColor: colors.chip, borderRadius: 8, paddingHorizontal: 10, minHeight: 36, justifyContent: 'center' },
  histText: { color: colors.text },
  primary: { minHeight: 48, borderRadius: 10, backgroundColor: colors.accent, alignItems: 'center', justifyContent: 'center', marginTop: 8 },
  secondary: { minHeight: 44, borderRadius: 10, backgroundColor: colors.chip, alignItems: 'center', justifyContent: 'center', marginTop: 8 },
  primaryText: { color: '#fff', fontWeight: '700' },
  error: { color: colors.down, marginTop: 8 },
  card: { backgroundColor: colors.surface, borderRadius: 12, padding: 12, marginTop: 16 },
});
