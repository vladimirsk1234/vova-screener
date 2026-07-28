import React from 'react';
import {
  FlatList,
  Linking,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { useScan } from '../store/scanStore';
import type { BuyRow, SellRow } from '../types';

export function ResultsScreen({ navigation }: { navigation: any }) {
  const { results, params, asOf, rejected, setSelectedSymbol } = useScan();
  const isSell = params.scanDirection === 'sell';

  return (
    <View style={styles.root}>
      <Text style={styles.caption}>
        {isSell ? 'SELL TO CLOSE' : 'BUY TO OPEN'} · {params.tf}
        {asOf ? ` · as of ${asOf}` : ''} · {results.length} rows
      </Text>
      {rejected.length > 0 && (
        <Pressable onPress={() => navigation.navigate('Rejected')} style={styles.rejectLink}>
          <Text style={styles.rejectText}>Rejected / errors: {rejected.length}</Text>
        </Pressable>
      )}
      <FlatList
        data={results}
        keyExtractor={(_, i) => String(i)}
        contentContainerStyle={{ paddingBottom: 40 }}
        renderItem={({ item }) => {
          if (isSell) {
            const r = item as SellRow;
            return (
              <Pressable
                style={[styles.card, r._is_summary && styles.summary]}
                onPress={() => {
                  if (r._is_summary) return;
                  setSelectedSymbol(r.tv_symbol);
                  navigation.navigate('Chart');
                }}
              >
                <View style={styles.cardTop}>
                  <Text style={styles.sym}>{r.tv_symbol}</Text>
                  <Text
                    style={[
                      styles.pnl,
                      { color: r['P&L ($)'] >= 0 ? '#089981' : '#f23645' },
                    ]}
                  >
                    {r['P&L ($)']} ({r['P&L (%)']}%)
                  </Text>
                </View>
                <Text style={styles.sub}>{r['Company Name']}</Text>
                <Text style={styles.meta}>
                  Entry {r.Entry} → Exit {r.Exit} · Size {r['Position Size (shares)']} · RR{' '}
                  {r['RR at Entry']}/{r['RR at Close']}
                </Text>
                {!r._is_summary && (
                  <Pressable onPress={() => Linking.openURL(r.Symbol)}>
                    <Text style={styles.tv}>Open TradingView</Text>
                  </Pressable>
                )}
              </Pressable>
            );
          }
          const r = item as BuyRow;
          return (
            <Pressable
              style={styles.card}
              onPress={() => {
                setSelectedSymbol(r.tv_symbol);
                navigation.navigate('Chart');
              }}
            >
              <View style={styles.cardTop}>
                <Text style={styles.sym}>{r.tv_symbol}</Text>
                <Text style={styles.rr}>RR {r.RR}</Text>
              </View>
              <Text style={styles.sub}>{r['Company Name']}</Text>
              <Text style={styles.meta}>
                TP {r.TP} · SL {r.SL} · Size {r['Position Size (shares)']} · $
                {r['Position Value ($)']}
              </Text>
              <Text style={styles.flags}>
                New {r.New} · Valid {r.Valid} · Strong {r.Strong}
              </Text>
              <Pressable onPress={() => Linking.openURL(r.Symbol)}>
                <Text style={styles.tv}>Open TradingView</Text>
              </Pressable>
            </Pressable>
          );
        }}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#1e222d' },
  caption: { color: '#9aa4b2', padding: 12, fontSize: 12 },
  rejectLink: { paddingHorizontal: 12, marginBottom: 4 },
  rejectText: { color: '#ffb74d' },
  card: {
    marginHorizontal: 12,
    marginBottom: 10,
    backgroundColor: '#2a2e39',
    borderRadius: 10,
    padding: 12,
  },
  summary: { borderColor: '#2962ff', borderWidth: 1 },
  cardTop: { flexDirection: 'row', justifyContent: 'space-between' },
  sym: { color: '#fff', fontWeight: '700', fontSize: 16 },
  rr: { color: '#4ea1ff', fontWeight: '600' },
  pnl: { fontWeight: '700' },
  sub: { color: '#9aa4b2', marginTop: 4 },
  meta: { color: '#cfd6e0', marginTop: 6, fontSize: 13 },
  flags: { color: '#9aa4b2', marginTop: 4, fontSize: 12 },
  tv: { color: '#4ea1ff', marginTop: 8 },
});
