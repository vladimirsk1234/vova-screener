import { Navigate, useParams, useSearchParams } from 'react-router-dom';

/** Legacy `/fundamentals/:ticker` links land on the unified chart window. */
export function FundamentalsPage() {
  const { ticker = '' } = useParams();
  const [search] = useSearchParams();
  const params = new URLSearchParams(search);
  params.set('view', 'fundamentals');
  return <Navigate to={`/chart/${encodeURIComponent(ticker)}?${params.toString()}`} replace />;
}
