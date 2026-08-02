import { Suspense, lazy, useEffect, useState } from 'react';
import { NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom';
import { HistoryPage } from './pages/HistoryPage';
import { ManualPage } from './pages/ManualPage';
import { RejectedPage } from './pages/RejectedPage';
import { ResultsPage } from './pages/ResultsPage';
import { SettingsSheet } from './components/SettingsSheet';
import { DEFAULT_RESULTS_PATH, lastResultsPath, rememberResultsPath } from './lib/tabMemory';

// Lightweight Charts is the heaviest dependency and only the chart screen needs it.
const ChartPage = lazy(() =>
  import('./pages/ChartPage').then((m) => ({ default: m.ChartPage })),
);

export function App() {
  const { pathname, search } = useLocation();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [resultsTo, setResultsTo] = useState(lastResultsPath);
  const isChart = pathname.startsWith('/chart/');

  useEffect(() => {
    rememberResultsPath(pathname, search);
    const next = lastResultsPath();
    setResultsTo((prev) => (prev === next ? prev : next));
  }, [pathname, search]);

  return (
    <div className={`app-shell${isChart ? ' app-shell--chart' : ''}`}>
      {!isChart && (
        <header className="app-header">
          <div className="app-header-row">
            <div className="brand">
              <img
                className="brand-mark"
                src="/icon-192.png"
                width={32}
                height={32}
                alt=""
                decoding="async"
              />
              <h1>SV Screener</h1>
            </div>
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

      <main className={`app-main${isChart ? ' app-main-flush app-main--chart' : ''}`}>
        <Suspense fallback={<p className="empty">Loading…</p>}>
          <Routes>
            <Route path="/" element={<Navigate to={resultsTo} replace />} />
            <Route path="/results" element={<Navigate to={resultsTo} replace />} />
            <Route path="/results/manual" element={<ManualPage />} />
            <Route path="/results/manual/rejected/:runId" element={<RejectedPage />} />
            <Route path="/results/:universe" element={<ResultsPage />} />
            <Route path="/results/:universe/:tf" element={<ResultsPage />} />
            <Route path="/results/:universe/:tf/:bucket" element={<ResultsPage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/chart/:ticker" element={<ChartPage />} />
            <Route path="*" element={<Navigate to={DEFAULT_RESULTS_PATH} replace />} />
          </Routes>
        </Suspense>
      </main>

      {!isChart && (
        <nav className="bottom-nav" aria-label="Primary">
          <NavLink
            to={resultsTo}
            className={() => (pathname.startsWith('/results') ? 'active' : undefined)}
          >
            <span className="dot" aria-hidden />
            Results
          </NavLink>
          <NavLink
            to="/history"
            className={() => (pathname.startsWith('/history') ? 'active' : undefined)}
          >
            <span className="dot" aria-hidden />
            History
          </NavLink>
        </nav>
      )}

      <SettingsSheet open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}
