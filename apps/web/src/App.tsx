import { Suspense, lazy, useEffect, useRef, useState } from 'react';
import { NavLink, Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import { HistoryPage } from './pages/HistoryPage';
import { ManualPage } from './pages/ManualPage';
import { RejectedPage } from './pages/RejectedPage';
import { ResultsPage } from './pages/ResultsPage';
import { ValuePage } from './pages/ValuePage';
import { SettingsSheet } from './components/SettingsSheet';
import {
  isChartLocation,
  isChartReturnSourcePath,
  lastAppPath,
  lastResultsPath,
  rememberAppPath,
  rememberChartReturn,
  rememberResultsPath,
} from './lib/tabMemory';

// Lightweight Charts is the heaviest dependency and only the chart screen needs it.
const ChartPage = lazy(() =>
  import('./pages/ChartPage').then((m) => ({ default: m.ChartPage })),
);
const FundamentalsPage = lazy(() =>
  import('./pages/FundamentalsPage').then((m) => ({ default: m.FundamentalsPage })),
);

/**
 * Restores the remembered route on a cold open. A plain replace would leave the restored screen as
 * the only history entry, so Back would exit the app; seeding Results underneath keeps Back inside.
 */
function RestoreEntry({ appTo, resultsTo }: { appTo: string; resultsTo: string }) {
  const navigate = useNavigate();
  const done = useRef(false);

  useEffect(() => {
    if (done.current) return;
    done.current = true;
    if (appTo === resultsTo) {
      navigate(appTo, { replace: true });
      return;
    }
    navigate(resultsTo, { replace: true });
    navigate(appTo);
  }, [appTo, resultsTo, navigate]);

  return null;
}

export function App() {
  const { pathname, search } = useLocation();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [resultsTo, setResultsTo] = useState(lastResultsPath);
  const [appTo, setAppTo] = useState(lastAppPath);
  const isChart = isChartLocation(pathname);
  const prevLocationRef = useRef(`${pathname}${search}`);
  const scrollYRef = useRef(0);

  // Track scroll while on a list page — after navigate to chart, window.scrollY is already 0.
  useEffect(() => {
    if (isChartLocation(pathname)) return;
    const onScroll = () => {
      scrollYRef.current = window.scrollY;
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, [pathname, search]);

  useEffect(() => {
    const next = `${pathname}${search}`;
    const prev = prevLocationRef.current;
    const q = prev.indexOf('?');
    const prevPath = q === -1 ? prev : prev.slice(0, q);

    if (isChartReturnSourcePath(prevPath) && isChartLocation(pathname)) {
      rememberChartReturn(prev, scrollYRef.current);
    }

    prevLocationRef.current = next;
    rememberResultsPath(pathname, search);
    rememberAppPath(pathname, search);
    const nextResults = lastResultsPath();
    setResultsTo((prevTo) => (prevTo === nextResults ? prevTo : nextResults));
    const nextApp = lastAppPath();
    setAppTo((prevTo) => (prevTo === nextApp ? prevTo : nextApp));
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
            <Route path="/" element={<RestoreEntry appTo={appTo} resultsTo={resultsTo} />} />
            <Route path="/results" element={<Navigate to={resultsTo} replace />} />
            <Route path="/results/value" element={<ValuePage />} />
            <Route path="/results/manual" element={<ManualPage />} />
            <Route path="/results/manual/rejected/:runId" element={<RejectedPage />} />
            <Route path="/results/:universe" element={<ResultsPage />} />
            <Route path="/results/:universe/:tf" element={<ResultsPage />} />
            <Route path="/results/:universe/:tf/:bucket" element={<ResultsPage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/chart/:ticker" element={<ChartPage />} />
            <Route path="/fundamentals/:ticker" element={<FundamentalsPage />} />
            <Route path="*" element={<RestoreEntry appTo={appTo} resultsTo={resultsTo} />} />
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
