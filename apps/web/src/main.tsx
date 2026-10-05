import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, HashRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { App } from './App';
import { IS_DEVICE } from './lib/platform';
import './styles.css';

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false, staleTime: 15_000 } },
});

// The iPhone WebView loads this page from an inline HTML string, so path routing has no server
// behind it; hash routing keeps every web URL working there.
const Router = IS_DEVICE ? HashRouter : BrowserRouter;

if (IS_DEVICE) {
  // A background scan or rebuild on the phone finished: same lists the Settings poll refreshes.
  (window as Window & { __vovaDataChanged?: () => void }).__vovaDataChanged = () => {
    for (const key of ['results', 'results-summary', 'history', 'history-trades', 'tracked-signal', 'chart']) {
      void queryClient.invalidateQueries({ queryKey: [key] });
    }
  };
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <Router>
        <App />
      </Router>
    </QueryClientProvider>
  </StrictMode>,
);
