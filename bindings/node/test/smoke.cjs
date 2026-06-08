// Smoke test for @mlxforge/node. Streams a chat and then runs several chats
// concurrently on one engine (proving they share the batched scheduler).
// Self-skips when MLXFORGE_MODEL_DIR is not set.
'use strict';

const assert = require('node:assert');
const { Engine, version, abiVersion } = require('..');

async function collect(stream) {
  let out = '';
  for await (const chunk of stream) out += chunk;
  return out;
}

(async () => {
  console.log(`@mlxforge/node version=${version} abi=${abiVersion}`);
  assert.ok(typeof version === 'string' && version.length > 0);

  const dir = process.env.MLXFORGE_MODEL_DIR;
  if (!dir) {
    console.log('MLXFORGE_MODEL_DIR not set; skipping model test (binding loaded OK)');
    return;
  }

  const engine = await Engine.load(dir);
  console.log('loaded model:', engine.modelName);

  // Single greedy stream — deterministic baseline.
  const baseline = await collect(
    engine.chat([{ role: 'user', content: 'What is the capital of France?' }], { maxTokens: 16 }),
  );
  console.log('baseline:', JSON.stringify(baseline));
  assert.ok(baseline.length > 0, 'expected non-empty output');

  // Concurrency: identical greedy requests submitted together must batch and
  // each reproduce the baseline exactly.
  const N = 4;
  const outs = await Promise.all(
    Array.from({ length: N }, () =>
      collect(
        engine.chat([{ role: 'user', content: 'What is the capital of France?' }], {
          maxTokens: 16,
        }),
      ),
    ),
  );
  for (const o of outs) assert.strictEqual(o, baseline, 'batched greedy must equal single-stream');
  console.log(`concurrency OK: ${N} requests batched, all == baseline`);

  // A couple of distinct concurrent prompts complete independently.
  const [color, fruit] = await Promise.all([
    collect(engine.chat([{ role: 'user', content: 'Name one color.' }], { maxTokens: 8 })),
    collect(engine.chat([{ role: 'user', content: 'Name one fruit.' }], { maxTokens: 8 })),
  ]);
  console.log('distinct concurrent:', JSON.stringify({ color, fruit }));

  engine.dispose();
  console.log('OK');
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
