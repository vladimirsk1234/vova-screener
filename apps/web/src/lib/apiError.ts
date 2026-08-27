/** Turn Nest / proxy error bodies into a single user-facing sentence. */

export function parseApiErrorBody(status: number, text: string): string {
  const fromJson = (raw: string): string | null => {
    try {
      const body = JSON.parse(raw) as { message?: unknown };
      if (typeof body.message === 'string' && body.message.trim()) return body.message.trim();
      if (Array.isArray(body.message) && body.message.length) {
        return body.message.map(String).join('; ');
      }
    } catch {
      /* keep looking */
    }
    return null;
  };

  const direct = fromJson(text);
  if (direct) return direct;

  const wrapped = text.match(/^\s*\d{3}\s*:\s*(\{[\s\S]*\})\s*$/);
  if (wrapped) {
    const inner = fromJson(wrapped[1]);
    if (inner) return inner;
  }

  return text.trim() || `${status}`;
}

export function isFundamentalsPendingError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err ?? '');
  return /still loading|Wait for the EOD|EOD refresh|Updating \d+\s*\/\s*\d+/i.test(msg);
}
