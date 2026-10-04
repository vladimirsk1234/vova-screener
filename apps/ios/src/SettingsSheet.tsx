import { rebuildHistory, TIMEFRAMES, type UserTf } from '@vova/device';
import { useState } from 'react';
import { Alert, Modal, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { useScreener } from './state';
import { colors } from './theme';
import { Chips } from './ui';

export function SettingsSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { settings, saveSettings, runScan, cancelScan, scan, store, refresh, universe } = useScreener();
  const [risk, setRisk] = useState(String(settings.maxRiskUsd));
  const [rr, setRr] = useState(String(settings.minRr));
  const [scanTf, setScanTf] = useState<UserTf | 'all'>('all');
  const [note, setNote] = useState<string | null>(null);

  const commit = () => {
    const maxRiskUsd = Number(risk);
    const minRr = Number(rr);
    if (Number.isFinite(maxRiskUsd) && maxRiskUsd > 0 && Number.isFinite(minRr) && minRr >= 0) {
      void saveSettings({ maxRiskUsd, minRr });
    }
  };

  return (
    <Modal visible={open} animationType="slide" transparent onShow={() => { setRisk(String(settings.maxRiskUsd)); setRr(String(settings.minRr)); }}>
      <View style={styles.backdrop}>
        <View style={styles.sheet}>
          <View style={styles.head}>
            <Text style={styles.title}>Settings</Text>
            <Pressable onPress={onClose} style={styles.close}><Text style={styles.link}>Close</Text></Pressable>
          </View>
          <ScrollView>
            <Text style={styles.label}>Max risk per signal ($)</Text>
            <TextInput value={risk} onChangeText={setRisk} onBlur={commit} keyboardType="decimal-pad" style={styles.input} />
            <Text style={styles.muted}>Position size is this divided by the distance to the stop. Open signals resize immediately. Closed rows in Results keep the size they closed at. History recomputes under the new risk.</Text>
            <Text style={styles.label}>Min RR</Text>
            <TextInput value={rr} onChangeText={setRr} onBlur={commit} keyboardType="decimal-pad" style={styles.input} />
            <Text style={styles.muted}>Floor on live RR for New and Valid, and on entry RR for Closed and History. 0 shows everything. Scans still record every signal.</Text>
            <Chips
              label="Rescan"
              value={scanTf}
              options={['all', ...TIMEFRAMES] as const}
              onChange={setScanTf}
              format={(v) => (v === 'all' ? 'All' : v === 'Weekly' ? 'W' : 'M')}
            />
            <Pressable
              style={styles.primary}
              onPress={() => void (scan?.phase === 'running' ? cancelScan() : runScan(scanTf))}
            >
              <Text style={styles.primaryText}>{scan?.phase === 'running' ? 'Stop scan' : 'Run scan now'}</Text>
            </Pressable>
            <Text style={styles.muted}>
              {scan ? `${scan.label}: ${scan.message}` : 'Stocks and ETF, Weekly and Monthly, downloaded from Yahoo onto this iPhone.'}
              {' '}Leave the app in the foreground until the pass finishes. iOS suspends a backgrounded app, so an unattended hourly scan cannot keep running the way the old server cron did.
            </Text>
            <Text style={styles.muted}>
              {universe.Stocks.length} stocks · {universe.ETF.length} ETFs in the bundled lists.
            </Text>
            <Pressable
              style={styles.secondary}
              onPress={() => {
                if (!store) return;
                void rebuildHistory(store, { maxRiskUsd: settings.maxRiskUsd }).then((counts) => {
                  setNote(`Rebuild inserted ${counts.inserted}, skipped ${counts.skipped}, no bars ${counts.noBars}.`);
                  refresh();
                });
              }}
            >
              <Text style={styles.primaryText}>Rebuild history</Text>
            </Pressable>
            <Text style={styles.muted}>Replays the close-scan ledger on bars already stored on this phone and adds missing closed trades. It does not delete anything.</Text>
            {note ? <Text style={styles.muted}>{note}</Text> : null}
            <Pressable
              style={styles.danger}
              onPress={() => {
                if (!store) return;
                Alert.alert('Reset tracked signals', 'Delete every tracked signal stored on this iPhone?', [
                  { text: 'Cancel', style: 'cancel' },
                  {
                    text: 'Delete',
                    style: 'destructive',
                    onPress: () => {
                      void store.saveSignals([]).then(() => {
                        setNote('Tracked signals cleared.');
                        refresh();
                      });
                    },
                  },
                ]);
              }}
            >
              <Text style={styles.primaryText}>Reset tracked signals</Text>
            </Pressable>
          </ScrollView>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: { flex: 1, backgroundColor: 'rgba(0,0,0,0.55)', justifyContent: 'flex-end' },
  sheet: { maxHeight: '88%', backgroundColor: colors.bg, borderTopLeftRadius: 16, borderTopRightRadius: 16, padding: 16 },
  head: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  title: { color: colors.text, fontSize: 20, fontWeight: '700' },
  close: { minHeight: 44, justifyContent: 'center' },
  link: { color: colors.accent, fontSize: 16 },
  label: { color: colors.text, marginTop: 10, marginBottom: 4 },
  input: { minHeight: 44, borderRadius: 8, backgroundColor: colors.surface, color: colors.text, paddingHorizontal: 12 },
  muted: { color: colors.muted, fontSize: 12, marginVertical: 8 },
  primary: { minHeight: 48, borderRadius: 10, backgroundColor: colors.accent, alignItems: 'center', justifyContent: 'center' },
  secondary: { minHeight: 44, borderRadius: 10, backgroundColor: colors.chip, alignItems: 'center', justifyContent: 'center', marginTop: 8 },
  danger: { minHeight: 44, borderRadius: 10, backgroundColor: '#6b2430', alignItems: 'center', justifyContent: 'center', marginVertical: 12 },
  primaryText: { color: '#fff', fontWeight: '700' },
});
