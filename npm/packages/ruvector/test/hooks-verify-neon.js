#!/usr/bin/env node
/**
 * `ruvector hooks verify` in neon mode (RUVECTOR_STORE=neon): the intelligence store is Neon, so the check
 * reads the store (through RVN_STORE_MODULE, here a stub) instead of looking for a local
 * .ruvector/intelligence.json, which never exists there and used to print a permanent
 * "Intelligence file: Will be created on first use" warning.
 */

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const CLI = path.join(__dirname, '..', 'bin', 'cli.js');
const work = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-verify-neon-'));
fs.mkdirSync(path.join(work, '.claude'));
fs.mkdirSync(path.join(work, '.ruvector'));
fs.writeFileSync(path.join(work, '.claude', 'settings.json'), '{}');
const stub = (body) => { const p = path.join(work, `stub-${Math.random().toString(36).slice(2)}.cjs`); fs.writeFileSync(p, `module.exports = { openDocStore: () => (${body}) };`); return p; };
const run = (env) => spawnSync(process.execPath, [CLI, 'hooks', 'verify'], { cwd: work, env: { ...process.env, HOME: work, NO_COLOR: '1', ...env }, encoding: 'utf8' }).stdout;
const neon = (body) => run({ RUVECTOR_STORE: 'neon', RVN_STORE_MODULE: stub(body) });

try {
  const full = neon(`{ exists: () => true, rows: { count: (c) => (c === 'memories' ? 12 : 34) } }`);
  assert.match(full, /✓ Intelligence store: Neon \(ruvector-intel\): 12 memories, 34 trajectories/, full);
  assert.doesNotMatch(full, /Intelligence file|Data directory/, 'no local-file check in neon mode');

  const empty = neon(`{ exists: () => false }`);
  assert.match(empty, /⚠ Intelligence store: Neon store is empty/, empty);

  const down = neon(`{ exists: () => { throw new Error('fetch failed'); } }`);
  assert.match(down, /✗ Intelligence store: Neon store check failed: fetch failed/, down);
  assert.match(down, /Verification failed/, 'an unreadable store fails the verification');

  // control: outside neon mode the local-file check is unchanged
  const local = run({ RUVECTOR_STORE: '' });
  assert.match(local, /✓ Data directory: \.ruvector\/ exists/, local);
  assert.match(local, /⚠ Intelligence file: Will be created on first use/, local);
  assert.doesNotMatch(local, /Intelligence store/, local);

  console.log('hooks-verify-neon: ok');
} finally {
  fs.rmSync(work, { recursive: true, force: true });
}
