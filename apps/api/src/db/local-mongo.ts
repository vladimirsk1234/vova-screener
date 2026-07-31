/**
 * Local MongoDB resolution.
 *
 * MONGO_URI wins (Railway / Docker / installed mongod). Otherwise a persistent
 * single-node replica set is started from the mongodb-memory-server binary so the
 * local machine needs no Docker or MongoDB install. Replica set (not standalone)
 * because transactions and change streams require one.
 */
import * as fs from 'node:fs';
import * as path from 'node:path';

export const REPO_ROOT = path.resolve(__dirname, '..', '..', '..', '..');
const DATA_DIR = path.join(REPO_ROOT, '.data', 'mongo');

let started: Promise<string> | null = null;

async function startEmbedded(): Promise<string> {
  fs.mkdirSync(DATA_DIR, { recursive: true });
  const { MongoMemoryReplSet } = await import('mongodb-memory-server');
  const rs = await MongoMemoryReplSet.create({
    replSet: { count: 1, dbName: 'vova', storageEngine: 'wiredTiger' },
    instanceOpts: [{ dbPath: DATA_DIR, port: 27019 }],
  });
  process.on('SIGINT', () => void rs.stop({ doCleanup: false }));
  process.on('SIGTERM', () => void rs.stop({ doCleanup: false }));
  return `${rs.getUri('vova')}`;
}

export function resolveMongoUri(): Promise<string> {
  if (process.env.MONGO_URI) return Promise.resolve(process.env.MONGO_URI);
  if (process.env.NODE_ENV === 'production') {
    throw new Error(
      'MONGO_URI is required in production (embedded mongodb-memory-server is local-only)',
    );
  }
  if (!started) started = startEmbedded();
  return started;
}
