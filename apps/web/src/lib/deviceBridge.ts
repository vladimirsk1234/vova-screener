/** Request/response channel to the iPhone app that hosts this UI. */

type Pending = { resolve: (res: { status: number; body: string }) => void };

declare global {
  interface Window {
    ReactNativeWebView?: { postMessage: (message: string) => void };
    __vovaReply?: (id: number, status: number, body: string) => void;
  }
}

const pending = new Map<number, Pending>();
let nextId = 1;

window.__vovaReply = (id, status, body) => {
  const hit = pending.get(id);
  if (!hit) return;
  pending.delete(id);
  hit.resolve({ status, body });
};

export function deviceFetch(
  method: string,
  path: string,
  body?: string,
): Promise<{ status: number; body: string }> {
  const host = window.ReactNativeWebView;
  if (!host) return Promise.resolve({ status: 503, body: '{"message":"iPhone bridge unavailable"}' });
  const id = nextId++;
  return new Promise((resolve) => {
    pending.set(id, { resolve });
    host.postMessage(JSON.stringify({ id, method, path, body: body ?? null }));
  });
}
