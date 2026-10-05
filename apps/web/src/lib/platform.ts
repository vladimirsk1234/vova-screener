/**
 * The iPhone build runs this UI inside the app's WebView. `/api` is then answered on the phone
 * (SQLite + Yahoo) through the native bridge, and screens that need FMP fundamentals are left out.
 */
export const IS_DEVICE = import.meta.env.VITE_TARGET === 'device';
