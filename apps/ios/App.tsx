import { NavigationContainer, DarkTheme } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { StatusBar } from 'expo-status-bar';
import { useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { ChartScreen } from './src/ChartScreen';
import { HistoryScreen } from './src/HistoryScreen';
import { ManualScreen } from './src/ManualScreen';
import { ResultsScreen } from './src/ResultsScreen';
import { SettingsSheet } from './src/SettingsSheet';
import { ScreenerProvider, useScreener } from './src/state';
import { colors } from './src/theme';
import type { UserTf } from '@vova/device';

type Route =
  | { name: 'tabs' }
  | { name: 'chart'; ticker: string; tf: UserTf; asOf?: string }
  | { name: 'manual' };

const Tabs = createBottomTabNavigator();

function Shell() {
  const { ready, error } = useScreener();
  const [route, setRoute] = useState<Route>({ name: 'tabs' });
  const [settingsOpen, setSettingsOpen] = useState(false);
  const openChart = (ticker: string, tf: UserTf, asOf?: string) => setRoute({ name: 'chart', ticker, tf, asOf });

  if (error) {
    return (
      <SafeAreaView style={styles.page}>
        <Text style={styles.error}>{error}</Text>
      </SafeAreaView>
    );
  }
  if (!ready) {
    return (
      <SafeAreaView style={styles.page}>
        <Text style={styles.muted}>Opening the local database…</Text>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.page} edges={['top']}>
      <StatusBar style="light" />
      {route.name === 'tabs' ? (
        <>
          <View style={styles.header}>
            <Text style={styles.brand}>SV Screener</Text>
            <Pressable onPress={() => setSettingsOpen(true)} style={styles.gear} accessibilityLabel="Settings">
              <Text style={styles.gearText}>⚙</Text>
            </Pressable>
          </View>
          <NavigationContainer theme={{ ...DarkTheme, colors: { ...DarkTheme.colors, background: colors.bg, card: colors.bg, text: colors.text, border: colors.chip, primary: colors.accent } }}>
            <Tabs.Navigator
              screenOptions={{
                headerShown: false,
                tabBarStyle: { backgroundColor: colors.surface, borderTopColor: colors.chip },
                tabBarActiveTintColor: colors.accent,
                tabBarInactiveTintColor: colors.muted,
              }}
            >
              <Tabs.Screen name="Results">
                {() => <ResultsScreen onOpenChart={openChart} onManual={() => setRoute({ name: 'manual' })} />}
              </Tabs.Screen>
              <Tabs.Screen name="History">
                {() => <HistoryScreen onOpenChart={openChart} />}
              </Tabs.Screen>
            </Tabs.Navigator>
          </NavigationContainer>
        </>
      ) : null}
      {route.name === 'chart' ? (
        <ChartScreen ticker={route.ticker} tf={route.tf} asOf={route.asOf} onBack={() => setRoute({ name: 'tabs' })} />
      ) : null}
      {route.name === 'manual' ? (
        <ManualScreen onBack={() => setRoute({ name: 'tabs' })} onOpenChart={openChart} />
      ) : null}
      <SettingsSheet open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </SafeAreaView>
  );
}

export function App() {
  return (
    <SafeAreaProvider>
      <ScreenerProvider>
        <Shell />
      </ScreenerProvider>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  page: { flex: 1, backgroundColor: colors.bg },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingBottom: 4 },
  brand: { color: colors.text, fontSize: 20, fontWeight: '700' },
  gear: { minWidth: 44, minHeight: 44, alignItems: 'center', justifyContent: 'center' },
  gearText: { color: colors.text, fontSize: 22 },
  muted: { color: colors.muted, padding: 16 },
  error: { color: colors.down, padding: 16 },
});
