#!/usr/bin/env node
/**
 * Neon mode (RUVECTOR_STORE=neon, the store reached through RVN_STORE_MODULE, here a stub) has no local
 * .ruvector/intelligence.json. Three commands assumed one and failed silently:
 *   1. `hooks export --include-all` exported 0 memories and 0 trajectories (they are lazy collections, never in
 *      intel.data): it now reads every row from the store, and says so when an export holds fewer than the store reports.
 *   2. `hooks subscribe` polled the missing file and never emitted: it now reads the store.
 *   3. The MCP server reloaded only when the file's mtime changed (always 0 here), so it never saw what CLI hooks learned
 *      after it started: in neon mode it now compares the store's fingerprint.
 * Each case below failed before the fix; the last case of each is a control.
 */

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn, spawnSync } = require('child_process');

const CLI = path.join(__dirname, '..', 'bin', 'cli.js');
const MCP = path.join(__dirname, '..', 'bin', 'mcp-server.js');
const work = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-neon-mode-'));
fs.mkdirSync(path.join(work, '.claude'));
fs.writeFileSync(path.join(work, '.claude', 'settings.json'), '{}');

// A stub Neon store whose state is a JSON file the test can change while a process runs ("another process wrote").
// doc: the non-lazy document; lazy: { memories: [...], trajectories: [...] }; count: optional overrides of rows.count.
const stateFile = path.join(work, 'store-state.json');
const setState = (s) => fs.writeFileSync(stateFile, JSON.stringify(s));
const STUB = path.join(work, 'stub-store.cjs');
fs.writeFileSync(STUB, `
const fs = require('fs');
const st = () => JSON.parse(fs.readFileSync(${JSON.stringify(stateFile)}, 'utf8'));
module.exports = { openDocStore: () => ({
  load: () => JSON.parse(JSON.stringify(st().doc)),
  save: () => {},
  exists: () => true,
  ensure: () => {},
  search: () => [],
  fpStmt: () => ({ text: 'fp', params: [] }),
  query: () => [{ n: 1, s: '1', v: String(st().version || 0) }],
  rows: {
    all: (c) => (st().lazy[c] || []).slice(0, st().allLimit ?? Infinity),
    count: (c) => (st().count && c in st().count ? st().count[c] : (st().lazy[c] || []).length),
    get: () => [], findBy: () => [], write: () => {},
  },
}) };`);
const env = { ...process.env, HOME: work, NO_COLOR: '1', RUVECTOR_STORE: 'neon', RVN_STORE_MODULE: STUB };
const cli = (args, extra = {}) => spawnSync(process.execPath, [CLI, ...args], { cwd: work, env: { ...env, ...extra }, encoding: 'utf8', timeout: 60000 });

const baseDoc = { patterns: { 'edit|coder': { q_value: 0.8 } }, errors: {}, stats: { total_patterns: 1 } };
const mems = [{ id: 'm1', content: 'first memory' }, { id: 'm2', content: 'second memory' }];
const trajs = [{ id: 't1' }, { id: 't2' }, { id: 't3' }];

function rpc(child, id, method, params, timeoutMs = 30000) {
  return new Promise((resolve, reject) => {
    let buf = '';
    const timer = setTimeout(() => reject(new Error(`timeout waiting for ${method}`)), timeoutMs);
    const onData = (chunk) => {
      buf += chunk.toString();
      for (const line of buf.split(/\r?\n/).filter(Boolean)) {
        let msg; try { msg = JSON.parse(line); } catch { continue; }
        if (msg.id === id) { clearTimeout(timer); child.stdout.off('data', onData); resolve(msg); return; }
      }
    };
    child.stdout.on('data', onData);
    child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`);
  });
}
const suggestions = (msg) => JSON.parse(msg.result.content[0].text).suggestions;

async function main() {
  try {
    // ── 1. export --include-all ──
    setState({ doc: baseDoc, lazy: { memories: mems, trajectories: trajs } });
    const out = path.join(work, 'export.json');
    const r1 = cli(['hooks', 'export', '--include-all', '-o', out]);
    assert.strictEqual(r1.status, 0, r1.stderr);
    const ex = JSON.parse(fs.readFileSync(out, 'utf8'));
    assert.strictEqual(ex.memories.length, 2, 'export --include-all holds every memory from the store');
    assert.strictEqual(ex.trajectories.length, 3, 'export --include-all holds every trajectory from the store');
    assert.doesNotMatch(r1.stderr, /but the store reports/, 'a complete export prints no warning');
    // an export that comes out short is said out loud
    setState({ doc: baseDoc, lazy: { memories: mems, trajectories: trajs }, allLimit: 1 });
    const r1b = cli(['hooks', 'export', '--include-all', '-o', out]);
    assert.match(r1b.stderr, /exported 1 memories, but the store reports 2/, r1b.stderr);
    // control: without --include-all nothing lazy is read or exported, as before
    setState({ doc: baseDoc, lazy: { memories: mems, trajectories: trajs } });
    cli(['hooks', 'export', '-o', out]);
    assert.strictEqual(JSON.parse(fs.readFileSync(out, 'utf8')).memories.length, 0, 'control: plain export has no memories');

    // ── 2. subscribe ──
    // subscribe runs until stopped: give it 3 s, then read what it emitted
    const sub = spawnSync(process.execPath, [CLI, 'hooks', 'subscribe', '--events', 'learn,memory', '--poll', '200'], { cwd: work, env, encoding: 'utf8', timeout: 3000 });
    const events = sub.stdout.split('\n').filter(Boolean).map((l) => JSON.parse(l));
    assert.ok(events.some((e) => e.type === 'learn' && e.data.total === 1), `subscribe emits the store's patterns: ${sub.stdout}`);
    assert.ok(events.some((e) => e.type === 'memory' && e.data.total === 2), `subscribe emits the store's memories: ${sub.stdout}`);

    // ── 3. MCP server sees what another process learned after it started ──
    setState({ doc: { ...baseDoc, errors: {} }, lazy: { memories: [], trajectories: [] }, version: 1 });
    const child = spawn(process.execPath, [MCP], { cwd: work, env: { ...env, RUVECTOR_MCP_ALLOW: '' }, stdio: ['pipe', 'pipe', 'pipe'] });
    let stderr = ''; child.stderr.on('data', (c) => { stderr += c.toString(); });
    try {
      await rpc(child, 1, 'initialize', { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'neon-mode-test', version: '1' } });
      const ask = (id) => rpc(child, id, 'tools/call', { name: 'hooks_error_suggest', arguments: { error: 'NEONTEST frobnicator exploded' } });
      assert.deepStrictEqual(suggestions(await ask(2)), [], 'no fix before it is recorded');
      // another process records a fix: the store's document and its fingerprint change
      setState({ doc: { ...baseDoc, errors: { 'NEONTEST frobnicator exploded': [{ fix: 'neontest-fix: reset it' }] } }, lazy: { memories: [], trajectories: [] }, version: 2 });
      assert.deepStrictEqual(suggestions(await ask(3)), ['neontest-fix: reset it'], `the running server sees the new fix (stderr: ${stderr.slice(-300)})`);
      // control: an unchanged fingerprint does not reload (the document changes, the token does not)
      setState({ doc: { ...baseDoc, errors: {} }, lazy: { memories: [], trajectories: [] }, version: 2 });
      assert.deepStrictEqual(suggestions(await ask(4)), ['neontest-fix: reset it'], 'control: no reload without a fingerprint change');
    } finally { child.kill(); }

    console.log('neon-mode-hooks: ok');
  } finally {
    fs.rmSync(work, { recursive: true, force: true });
  }
}

main().catch((e) => { console.error(e); fs.rmSync(work, { recursive: true, force: true }); process.exit(1); });
