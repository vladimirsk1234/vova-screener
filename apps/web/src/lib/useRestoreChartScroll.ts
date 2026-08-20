import { useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import { takeChartReturnScroll } from './tabMemory';

/**
 * After Back from a chart card, restore the list scroll once content is ready.
 * Scroll is cleared after apply so a later visit to the same URL does not jump.
 */
export function useRestoreChartScroll(ready: boolean): void {
  const { pathname, search } = useLocation();
  const applied = useRef(false);

  useEffect(() => {
    applied.current = false;
  }, [pathname, search]);

  useEffect(() => {
    if (!ready || applied.current) return;
    const y = takeChartReturnScroll(`${pathname}${search}`);
    applied.current = true;
    if (y == null) return;
    requestAnimationFrame(() => {
      window.scrollTo(0, y);
    });
  }, [ready, pathname, search]);
}
