import React, { useState } from 'react';
import {
  Pressable,
  ScrollView,
  Share,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { monthlyStats, tradesToCsv, type MonthlyStats } from '../db/journal';

function currentMonth() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`;
}

export function MonthlyScreen() {
  const [month, setMonth] = useState(currentMonth());
  const [stats, setStats] = useState<MonthlyStats | null>(null);

  const load = async () => {
    setStats(await monthlyStats(month));
  };

  return (
    <ScrollView style={styles.root} contentContainerStyle={{ padding: 16 }}>
      <Text style={styles.label}>Month (YYYY-MM)</Text>
      <TextInput
        style={styles.input}
        value={month}
        onChangeText={setMonth}
        autoCapitalize="none"
      />
      <Pressable style={styles.btn} onPress={load}>
        <Text style={styles.btnText}>Load report</Text>
      </Pressable>
      {stats && (
        <View style={styles.box}>
          <Text style={styles.title}>Closed this month: {stats.closed.length}</Text>
          <Text style={styles.line}>Win rate: {stats.winRate.toFixed(1)}%</Text>
          <Text style={styles.line}>Sum P&L: ${stats.sumPnl}</Text>
          <Text style={styles.line}>Avg P&L R: {stats.avgPnlR}</Text>
          <Text style={styles.line}>Avg RR at entry: {stats.avgRr}</Text>
          <Text style={[styles.title, { marginTop: 16 }]}>
            Still open: {stats.open.length}
          </Text>
          {stats.open.slice(0, 20).map((t) => (
            <Text key={t.id} style={styles.line}>
              {t.symbol} · entry {t.entry} · RR {t.rr_at_entry}
            </Text>
          ))}
          <Pressable
            style={[styles.btn, { marginTop: 16 }]}
            onPress={async () => {
              const csv = tradesToCsv([...stats.closed, ...stats.open]);
              await Share.share({ message: csv, title: `vova-${month}.csv` });
            }}
          >
            <Text style={styles.btnText}>Share CSV (local)</Text>
          </Pressable>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#1e222d' },
  label: { color: '#9aa4b2', marginBottom: 6 },
  input: {
    backgroundColor: '#2a2e39',
    color: '#fff',
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
  },
  btn: {
    backgroundColor: '#2962ff',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  btnText: { color: '#fff', fontWeight: '600' },
  box: { marginTop: 16, backgroundColor: '#2a2e39', borderRadius: 10, padding: 14 },
  title: { color: '#fff', fontWeight: '700', marginBottom: 6 },
  line: { color: '#cfd6e0', marginTop: 4 },
});
