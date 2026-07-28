import 'react-native-gesture-handler';
import React from 'react';
import { NavigationContainer, DarkTheme } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { StatusBar } from 'expo-status-bar';
import { Text } from 'react-native';
import { ScanProvider } from './src/store/scanStore';
import { ScanScreen } from './src/screens/ScanScreen';
import { ResultsScreen } from './src/screens/ResultsScreen';
import { ChartScreen } from './src/screens/ChartScreen';
import { RejectedScreen } from './src/screens/RejectedScreen';
import { HistoryScreen } from './src/screens/HistoryScreen';
import { TradesScreen } from './src/screens/TradesScreen';
import { MonthlyScreen } from './src/screens/MonthlyScreen';

const Tab = createBottomTabNavigator();
const ScanStack = createNativeStackNavigator();

const navTheme = {
  ...DarkTheme,
  colors: {
    ...DarkTheme.colors,
    background: '#1e222d',
    card: '#2a2e39',
    primary: '#2962ff',
    text: '#ffffff',
    border: '#3a4150',
  },
};

function ScanStackNav() {
  return (
    <ScanStack.Navigator
      screenOptions={{
        headerStyle: { backgroundColor: '#2a2e39' },
        headerTintColor: '#fff',
      }}
    >
      <ScanStack.Screen name="ScanHome" component={ScanScreen} options={{ title: 'Scan' }} />
      <ScanStack.Screen name="Results" component={ResultsScreen} />
      <ScanStack.Screen name="Chart" component={ChartScreen} />
      <ScanStack.Screen name="Rejected" component={RejectedScreen} />
    </ScanStack.Navigator>
  );
}

function TabIcon({ label, focused }: { label: string; focused: boolean }) {
  return (
    <Text style={{ color: focused ? '#4ea1ff' : '#9aa4b2', fontSize: 11 }}>{label}</Text>
  );
}

export default function App() {
  return (
    <ScanProvider>
      <NavigationContainer theme={navTheme}>
        <StatusBar style="light" />
        <Tab.Navigator
          screenOptions={{
            headerStyle: { backgroundColor: '#2a2e39' },
            headerTintColor: '#fff',
            tabBarStyle: { backgroundColor: '#2a2e39', borderTopColor: '#3a4150' },
            tabBarActiveTintColor: '#4ea1ff',
            tabBarInactiveTintColor: '#9aa4b2',
          }}
        >
          <Tab.Screen
            name="ScanTab"
            component={ScanStackNav}
            options={{
              headerShown: false,
              title: 'Scan',
              tabBarIcon: ({ focused }) => <TabIcon label="Scan" focused={focused} />,
            }}
          />
          <Tab.Screen
            name="History"
            component={HistoryScreen}
            options={{
              tabBarIcon: ({ focused }) => <TabIcon label="Hist" focused={focused} />,
            }}
          />
          <Tab.Screen
            name="Trades"
            component={TradesScreen}
            options={{
              tabBarIcon: ({ focused }) => <TabIcon label="Trades" focused={focused} />,
            }}
          />
          <Tab.Screen
            name="Monthly"
            component={MonthlyScreen}
            options={{
              tabBarIcon: ({ focused }) => <TabIcon label="P&L" focused={focused} />,
            }}
          />
        </Tab.Navigator>
      </NavigationContainer>
    </ScanProvider>
  );
}
