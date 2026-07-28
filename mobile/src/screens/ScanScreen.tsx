import React from 'react';
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
} from 'react-native';
import { useScan } from '../store/scanStore';
import type { SourceLabel, Timeframe } from '../types';

const SOURCES: SourceLabel[] = ['Stocks', 'ETF', 'MANUAL SCAN'];
const TFS: Timeframe[] = ['Daily', 'Weekly', 'Monthly'];

export function ScanScreen({ navigation }: { navigation: any }) {
  const { params, setParams, scanning, progress, startScan, stopScan, results, asOf } =
    useScan();

  const chip = (active: boolean) => [styles.chip, active && styles.chipOn];

  return (
    <ScrollView style={styles.root} contentContainerStyle={styles.content}>
      <Text style={styles.title}>CONFIGURATION</Text>

      <Text style={styles.label}>SOURCE</Text>
      <View style={styles.row}>
        {SOURCES.map((s) => (
          <Pressable
            key={s}
            disabled={scanning}
            style={chip(params.source === s)}
            onPress={() => setParams((p) => ({ ...p, source: s }))}
          >
            <Text style={styles.chipText}>{s === 'MANUAL SCAN' ? 'Manual' : s}</Text>
          </Pressable>
        ))}
      </View>

      {params.source === 'MANUAL SCAN' && (
        <>
          <Text style={styles.label}>TICKERS</Text>
          <TextInput
            editable={!scanning}
            style={styles.input}
            value={params.manualTickers}
            onChangeText={(t) => setParams((p) => ({ ...p, manualTickers: t }))}
            placeholder="AAPL, TSLA, NVDA"
            autoCapitalize="characters"
          />
        </>
      )}

      <Text style={styles.label}>$ RISK PER TRADE</Text>
      <TextInput
        editable={!scanning}
        style={styles.input}
        keyboardType="decimal-pad"
        value={String(params.riskPerTrade)}
        onChangeText={(t) =>
          setParams((p) => ({ ...p, riskPerTrade: Math.max(1, Number(t) || 1) }))
        }
      />

      <Text style={styles.label}>MIN RR (&gt;=1.5)</Text>
      <TextInput
        editable={!scanning}
        style={styles.input}
        keyboardType="decimal-pad"
        value={String(params.minRr)}
        onChangeText={(t) =>
          setParams((p) => ({ ...p, minRr: Math.max(0.1, Number(t) || 1.5) }))
        }
      />

      <Text style={styles.label}>SCAN DIRECTION</Text>
      <View style={styles.row}>
        <Pressable
          disabled={scanning}
          style={chip(params.scanDirection === 'buy')}
          onPress={() => setParams((p) => ({ ...p, scanDirection: 'buy' }))}
        >
          <Text style={styles.chipText}>BUY TO OPEN</Text>
        </Pressable>
        <Pressable
          disabled={scanning}
          style={chip(params.scanDirection === 'sell')}
          onPress={() => setParams((p) => ({ ...p, scanDirection: 'sell' }))}
        >
          <Text style={styles.chipText}>SELL TO CLOSE</Text>
        </Pressable>
      </View>

      {params.scanDirection === 'buy' && (
        <View style={styles.switchRow}>
          <Text style={styles.labelInline}>Use last HL in SL</Text>
          <Switch
            disabled={scanning}
            value={params.useLastHlSl}
            onValueChange={(v) => setParams((p) => ({ ...p, useLastHlSl: v }))}
          />
        </View>
      )}

      <Text style={styles.label}>TIMEFRAME</Text>
      <View style={styles.row}>
        {TFS.map((tf) => (
          <Pressable
            key={tf}
            disabled={scanning}
            style={chip(params.tf === tf)}
            onPress={() => setParams((p) => ({ ...p, tf }))}
          >
            <Text style={styles.chipText}>{tf}</Text>
          </Pressable>
        ))}
      </View>

      <View style={styles.switchRow}>
        <Text style={styles.labelInline}>NEW SIGNALS ONLY</Text>
        <Switch
          disabled={scanning}
          value={params.newOnly}
          onValueChange={(v) => setParams((p) => ({ ...p, newOnly: v }))}
        />
      </View>

      {!scanning ? (
        <Pressable style={styles.startBtn} onPress={startScan}>
          <Text style={styles.startText}>▶ START</Text>
        </Pressable>
      ) : (
        <Pressable style={styles.stopBtn} onPress={stopScan}>
          <Text style={styles.startText}>⏹ STOP</Text>
        </Pressable>
      )}

      {scanning && (
        <View style={styles.progressBox}>
          <ActivityIndicator color="#fff" />
          <Text style={styles.progressText}>{progress.message}</Text>
          <Text style={styles.progressText}>
            DL {progress.downloadPct}% · PROC {progress.processPct}%
          </Text>
        </View>
      )}

      {!scanning && results.length > 0 && (
        <Pressable
          style={styles.linkBtn}
          onPress={() => navigation.navigate('Results')}
        >
          <Text style={styles.linkText}>
            View results ({results.length}){asOf ? ` · as of ${asOf}` : ''}
          </Text>
        </Pressable>
      )}

      <Text style={styles.hint}>
        Streamlit web screener stays active for side-by-side comparison. Data saves only on
        this iPhone (SQLite).
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#1e222d' },
  content: { padding: 16, paddingBottom: 40 },
  title: { color: '#fff', fontSize: 18, fontWeight: '700', marginBottom: 12 },
  label: { color: '#9aa4b2', marginTop: 12, marginBottom: 6, fontSize: 12 },
  labelInline: { color: '#cfd6e0', fontSize: 14 },
  row: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  chip: {
    backgroundColor: '#2a2e39',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#3a4150',
  },
  chipOn: { backgroundColor: '#2962ff', borderColor: '#2962ff' },
  chipText: { color: '#fff', fontSize: 13 },
  input: {
    backgroundColor: '#2a2e39',
    color: '#fff',
    borderRadius: 8,
    padding: 12,
    borderWidth: 1,
    borderColor: '#3a4150',
  },
  switchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 14,
  },
  startBtn: {
    marginTop: 20,
    backgroundColor: '#089981',
    padding: 14,
    borderRadius: 10,
    alignItems: 'center',
  },
  stopBtn: {
    marginTop: 20,
    backgroundColor: '#f23645',
    padding: 14,
    borderRadius: 10,
    alignItems: 'center',
  },
  startText: { color: '#fff', fontWeight: '700', fontSize: 16 },
  progressBox: { marginTop: 16, alignItems: 'center', gap: 8 },
  progressText: { color: '#cfd6e0' },
  linkBtn: {
    marginTop: 16,
    padding: 12,
    backgroundColor: '#2a2e39',
    borderRadius: 8,
  },
  linkText: { color: '#4ea1ff', textAlign: 'center' },
  hint: { color: '#6b7380', marginTop: 24, fontSize: 12, lineHeight: 18 },
});
