import React, { useState } from 'react';
import {
  Dimensions,
  Linking,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  View,
} from 'react-native';
import { CandleChart } from '../components/CandleChart';
import { useScan } from '../store/scanStore';
import { buildChartUrl, tfToTvInterval } from '../tradingview';

export function ChartScreen() {
  const { selectedSymbol, ohlcCache, chartParams, setChartParams, params } = useScan();
  const [showSettings, setShowSettings] = useState(false);
  const entry = selectedSymbol ? ohlcCache[selectedSymbol] : null;
  const width = Dimensions.get('window').width - 24;

  if (!entry) {
    return (
      <View style={styles.root}>
        <Text style={styles.empty}>Select a result row to open the chart.</Text>
      </View>
    );
  }

  const tvUrl = buildChartUrl(entry.symbol, tfToTvInterval(params.tf));

  return (
    <ScrollView style={styles.root} contentContainerStyle={{ padding: 12 }}>
      <Text style={styles.title}>{entry.symbol}</Text>
      <Text style={styles.sub}>{entry.tf} · {entry.yahoo_ticker}</Text>
      <CandleChart
        bars={entry.bars}
        tf={entry.tf}
        params={chartParams}
        width={width}
        height={320}
      />
      <Pressable style={styles.tvBtn} onPress={() => Linking.openURL(tvUrl)}>
        <Text style={styles.tvText}>Open in TradingView</Text>
      </Pressable>
      <Pressable onPress={() => setShowSettings((s) => !s)}>
        <Text style={styles.settingsToggle}>
          {showSettings ? 'Hide' : 'Show'} chart settings
        </Text>
      </Pressable>
      {showSettings && (
        <View style={styles.settings}>
          {(
            [
              ['show_crit_level', 'Critical level'],
              ['show_tp_sl', 'TP / SL'],
              ['show_breaks', 'Breaks'],
              ['show_hhll', 'HH/LL'],
            ] as const
          ).map(([key, label]) => (
            <View key={key} style={styles.switchRow}>
              <Text style={styles.label}>{label}</Text>
              <Switch
                value={Boolean(chartParams[key])}
                onValueChange={(v) => setChartParams((p) => ({ ...p, [key]: v }))}
              />
            </View>
          ))}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#1e222d' },
  empty: { color: '#9aa4b2', padding: 24, textAlign: 'center' },
  title: { color: '#fff', fontSize: 18, fontWeight: '700' },
  sub: { color: '#9aa4b2', marginBottom: 8 },
  tvBtn: {
    marginTop: 12,
    backgroundColor: '#2962ff',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  tvText: { color: '#fff', fontWeight: '600' },
  settingsToggle: { color: '#4ea1ff', marginTop: 16 },
  settings: { marginTop: 8, backgroundColor: '#2a2e39', borderRadius: 8, padding: 12 },
  switchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginVertical: 6,
  },
  label: { color: '#cfd6e0' },
});
