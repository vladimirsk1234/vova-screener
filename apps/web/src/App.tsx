import { Suspense, lazy, useState } from 'react';
import { NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom';
import { HistoryPage } from './pages/HistoryPage';
import { ManualPage } from './pages/ManualPage';
import { RejectedPage } from './pages/RejectedPage';
import { ResultsPage } from './pages/ResultsPage';
import { SettingsSheet } from './components/SettingsSheet';

// Lightweight Charts is the heaviest dependency and only the chart screen needs it.
const ChartPage = lazy(() =>
  import('./pages/ChartPage').then((m) => ({ default: m.ChartPage })),
);
const FundamentalsPage = lazy(() =>
  import('./pages/FundamentalsPage').then((m) => ({ default: m.FundamentalsPage })),
);

const TABS = [
  { to: '/results', label: 'Results' },
  { to: '/history', label: 'History' },
];

export function App() {
  const { pathname } = useLocation();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const isChart = pathname.startsWith('/chart/');
  const isFund = pathname.startsWith('/fundamentals/');
  const hideChrome = isChart || isFund;

  return (
    <div className={`app-shell${hideChrome ? ' app-shell--chart' : ''}`}>
      {!hideChrome && (
        <header className="app-header">
          <div className="app-header-row">
            <h1>Sequence Vova</h1>
            <button
              type="button"
              className="chart-icon-btn"
              aria-label="Settings"
              onClick={() => setSettingsOpen(true)}
            >
              ⚙
            </button>
          </div>
        </header>
      )}

      <main className={`app-main${hideChrome ? ' app-main-flush app-main--chart' : ''}`}>
        <Suspense fallback={<p className="empty">Loading…</p>}>
          <Routes>
            <Route path="/" element={<Navigate to="/results/Stocks/Daily/new" replace />} />
            <Route path="/results" element={<Navigate to="/results/Stocks/Daily/new" replace />} />
            <Route path="/results/manual" element={<ManualPage />} />
            <Route path="/results/manual/rejected/:runId" element={<RejectedPage />} />
            <Route path="/results/:universe" element={<ResultsPage />} />
            <Route path="/results/:universe/:tf" element={<ResultsPage />} />
            <Route path="/results/:universe/:tf/:bucket" element={<ResultsPage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/chart/:ticker" element={<ChartPage />} />
            <Route path="/fundamentals/:ticker" element={<FundamentalsPage />} />
            <Route path="*" element={<Navigate to="/results/Stocks/Daily/new" replace />} />
          </Routes>
        </Suspense>
      </main>

      {!hideChrome && (
        <nav className="bottom-nav" aria-label="Primary">
          {TABS.map((tab) => (
            <NavLink key={tab.to} to={tab.to}>
              <span className="dot" aria-hidden />
              {tab.label}
            </NavLink>
          ))}
        </nav>
      )}

      <SettingsSheet open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}
