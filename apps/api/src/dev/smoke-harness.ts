/** Shared plumbing for the dev smoke scripts: an isolated database and a check/report pair. */
import { mongo } from 'mongoose';
import { EMBEDDED_PORT, resolveMongoUri, stopEmbedded } from '../db/local-mongo';

let failures = 0;

/**
 * A port that accepts connections is not the same as a server that answers, and the difference
 * costs five minutes of server selection when a previous run left a socket behind.
 */
async function reachable(uri: string): Promise<boolean> {
  const client = new mongo.MongoClient(uri, { serverSelectionTimeoutMS: 2000 });
  try {
    await client.db('admin').command({ ping: 1 });
    return true;
  } catch {
    return false;
  } finally {
    await client.close().catch(() => undefined);
  }
}

/**
 * Fixtures need a database of their own: a real background scan sitting next to them would look
 * like live data to the services under test. The embedded Mongo has a persistent data directory
 * shared with the dev server, hence a separate database name rather than a separate server.
 */
export async function useSmokeDatabase(name: string) {
  const running = `mongodb://127.0.0.1:${EMBEDDED_PORT}/vova?directConnection=true`;
  const base =
    process.env.MONGO_URI ?? ((await reachable(running)) ? running : await resolveMongoUri());
  const uri = new URL(base);
  uri.pathname = `/${name}`;
  process.env.MONGO_URI = uri.toString();
}

export function check(label: string, actual: unknown, expected: unknown) {
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  if (!ok) failures += 1;
  console.log(
    `${ok ? 'ok  ' : 'FAIL'} ${label}: ${JSON.stringify(actual)}` +
      (ok ? '' : ` (want ${JSON.stringify(expected)})`),
  );
}

/** Releases the embedded server before exiting, so smokes can be run back to back. */
export async function finish(label: string): Promise<never> {
  await stopEmbedded();
  if (failures) {
    console.error(`\n${failures} check(s) failed`);
    process.exit(1);
  }
  console.log(`\n${label} OK`);
  process.exit(0);
}
