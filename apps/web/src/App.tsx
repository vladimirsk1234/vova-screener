import { NavLink, Route, Routes, useLocation } from 'react-router-dom';
import { ScanPage } from './pages/ScanPage';
import { HistoryPage } from './pages/HistoryPage';
import { TradesPage } from './pages/TradesPage';
import { PnlPage } from './pages/PnlPage';
import { ResultsPage } from './pages/ResultsPage';
import { RejectedPage } from './pages/RejectedPage';
import { ChartPage } from './pages/ChartPage';

const tabs = [
  { to: '/', label: 'Scan', end: true },
  { to: '/history', label: 'History', end: false },
  { to: '/trades', label: 'Trades', end: false },
  { to: '/pnl', label: 'P&L', end: false },
];

export function App() {
  const { pathname } = useLocation();
  const isChart = pathname.startsWith('/chart/');

  return (
    <div className="app-shell">
      {!isChart && (
        <header className="app-header">
          <h1>Sequence Vova</h1>
          <p>Mobile-first screener</p>
        </header>
      )}

      <main className={`app-main${isChart ? ' app-main-flush' : ''}`}>
        <Routes>
          <Route path="/" element={<ScanPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/trades" element={<TradesPage />} />
          <Route path="/pnl" element={<PnlPage />} />
          <Route path="/runs/:runId" element={<ResultsPage />} />
          <Route path="/runs/:runId/rejected" element={<RejectedPage />} />
          <Route path="/chart/:ticker" element={<ChartPage />} />
        </Routes>
      </main>

      <nav className="bottom-nav" aria-label="Primary">
        {tabs.map((tab) => (
          <NavLink key={tab.to} to={tab.to} end={tab.end}>
            <span className="dot" aria-hidden />
            {tab.label}
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
