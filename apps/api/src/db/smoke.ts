/** One-off check that the embedded MongoDB starts and accepts writes. */
import mongoose from 'mongoose';
import { resolveMongoUri } from './local-mongo';

async function main() {
  const uri = await resolveMongoUri();
  console.log('uri:', uri);
  await mongoose.connect(uri);
  const Model = mongoose.model('smoke', new mongoose.Schema({ at: Date }));
  await Model.create({ at: new Date() });
  console.log('docs:', await Model.countDocuments());
  await mongoose.disconnect();
  console.log('MONGO OK');
  process.exit(0);
}

main().catch((e) => {
  console.error('MONGO FAIL', e);
  process.exit(1);
});
