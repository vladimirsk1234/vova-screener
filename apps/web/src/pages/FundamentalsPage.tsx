import { Navigate, useParams } from 'react-router-dom';

/** Legacy `/fundamentals/:ticker` links land on the unified chart window. */
export function FundamentalsPage() {
  const { ticker = '' } = useParams();
  return <Navigate to={`/chart/${encodeURIComponent(ticker)}?view=fundamentals`} replace />;
}
