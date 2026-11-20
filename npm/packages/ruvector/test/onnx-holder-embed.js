#!/usr/bin/env node
/**
 * neon-direct: the CLI's ONNX embedder asks the container's connection holder (NEON_HOLDER_SOCK) instead of loading
 * MiniLM in every process, and falls back to the local load whenever the holder cannot serve it.
 *
 * A fake holder (this file, role "holder") loads dist/core/onnx-embedder.js itself and answers the same requests as
 * digital-twin's neon-holder.mjs: {kind:'embed', fn:'info'|'identity'|embedQuery|embedPassage|…Batch, args}.
 * Every client runs in its own process (the module is a singleton) and is compared with a local load.
 * Before the change there is no holder path: getEmbedderVia() does not exist and every case fails.
 * Needs the built dist/ and the MiniLM model (cached after the first load), like the other embedder tests.
 */
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const net = require('net');
const crypto = require('crypto');
const { spawn, spawnSync } = require('child_process');

const DIST = process.env.RUVECTOR_DIST || path.join(__dirname, '..', 'dist');
const MOD = path.join(DIST, 'core', 'onnx-embedder.js');
const URL_ = 'postgres://holder-test.invalid/db';
const fp = (u) => crypto.createHash('sha256').update(String(u)).digest('hex').slice(0, 16);
const Q = 'what do we have next on the list';
const P = ['a passage about boats', 'another about bridges'];

const quiet = async (f) => { const l = console.log, e = console.error; console.log = console.error = () => {}; try { return await f(); } finally { console.log = l; console.error = e; } };

// ------------------------------------------------------------------------------------------- role: holder
async function holder(sock, mode, logFile) {
  process.env.RUVECTOR_EMBED_HOLDER = '0';
  const e = require(MOD);
  let loaded = null;
  const load = () => (loaded ||= quiet(() => e.initOnnxEmbedder()));
  const plain = (r) => (Array.isArray(r) ? r.map(plain) : ArrayBuffer.isView(r) ? Array.from(r)
    : r && typeof r === 'object' && r.embedding ? { ...r, embedding: Array.from(r.embedding) } : r);
  const handle = async (msg) => {
    fs.appendFileSync(logFile, `${msg.fn || msg.kind}\n`);
    if (msg.kind === 'shutdown') { setImmediate(() => { srv.close(); process.exit(0); }); return { ok: true }; }
    if (mode === 'error') throw new Error('holder refuses');
    await load();
    if (msg.fn === 'info') return { ready: e.isReady(), dim: e.getDimension() };
    if (msg.fn === 'identity') {
      if (mode === 'old') throw new Error('unknown embed function identity');   // a holder from before this change
      const identity = e.getEmbeddingSpaceIdentity();
      if (mode === 'badid') return { ready: true, dim: 384, modelId: 'all-MiniLM-L6-v2', identity: { ...identity, output_dimension: 256 } };
      return { ready: e.isReady(), dim: e.getDimension(), modelId: e.getActiveModelId(), identity };
    }
    if (mode === 'badvec') return { result: msg.fn.endsWith('Batch') ? msg.args[0].map(() => ({ embedding: [1, 2, 3] })) : { embedding: [1, 2, 3] } };
    return { result: plain(await e[msg.fn](...msg.args)) };
  };
  const srv = net.createServer((c) => {
    let buf = '';
    c.setEncoding('utf8');
    c.on('data', async (d) => {
      buf += d; const i = buf.indexOf('\n'); if (i < 0) return;
      const msg = JSON.parse(buf.slice(0, i)); buf = buf.slice(i + 1);
      if (mode === 'silent') { fs.appendFileSync(logFile, 'silent\n'); return; }   // accepts, never answers
      let r; try { r = { id: msg.id, ...(await handle(msg)) }; } catch (err) { r = { id: msg.id, error: String(err.message) }; }
      if (!c.destroyed) c.write(JSON.stringify(r) + '\n');
    });
    c.on('error', () => {});
  });
  await load();
  fs.writeFileSync(`${sock}.fp`, fp(URL_) + '\n');
  srv.listen(sock, () => { fs.writeFileSync(`${sock}.pid`, `${process.pid}\n`); process.stdout.write('ready\n'); });
}

// ------------------------------------------------------------------------------------------- role: client
async function client(scenario) {
  const e = require(MOD);
  const cfg = JSON.parse(process.env.T_CONFIG || '{}');
  if (process.env.T_SELFPID) fs.writeFileSync(`${process.env.NEON_HOLDER_SOCK}.pid`, `${process.pid}\n`);
  const t0 = Date.now();
  await quiet(() => e.initOnnxEmbedder(cfg));
  const out = { via0: e.getEmbedderVia ? e.getEmbedderVia() : 'no-holder-path', initMs: Date.now() - t0 };
  out.q = (await quiet(() => e.embedQuery(Q))).embedding;
  if (scenario === 'kill') {
    await new Promise((res) => { const c = net.connect(process.env.NEON_HOLDER_SOCK); c.on('connect', () => c.write('{"id":1,"kind":"shutdown"}\n')); c.on('data', () => { c.end(); res(); }); c.on('error', res); });
    await new Promise((r) => setTimeout(r, 300));
  }
  out.p = (await quiet(() => e.embed(P[0]))).embedding;
  out.b = (await quiet(() => e.embedPassageBatch(P))).map((r) => r.embedding);
  out.bq = (await quiet(() => e.embedQueryBatch(P))).map((r) => r.embedding);
  out.via1 = e.getEmbedderVia ? e.getEmbedderVia() : 'no-holder-path';
  out.ready = e.isReady(); out.dim = e.getDimension(); out.model = e.getActiveModelId();
  out.identity = e.getEmbeddingSpaceIdentity(); out.spaceId = e.getEmbeddingSpaceId(); out.prov = e.getEmbedderProvenance();
  out.holderEmbeds = e.getHolderEmbedCount ? e.getHolderEmbedCount() : -1;
  if (scenario === 'similarity') {
    out.sim = (await quiet(() => e.similarity(P[0], P[1]))).similarity;
    out.viaSim = e.getEmbedderVia();
  }
  process.stdout.write(JSON.stringify(out) + '\n');
  process.exit(0);
}

// ------------------------------------------------------------------------------------------- the test
if (process.argv[2] === 'holder') { holder(process.argv[3], process.argv[4], process.argv[5]); return; }
if (process.argv[2] === 'client') { client(process.argv[3]); return; }

const W = fs.mkdtempSync(path.join(os.tmpdir(), 'rvh-'));
const baseEnv = () => { const e = { ...process.env }; for (const k of ['NEON_HOLDER_SOCK', 'RVN_URL', 'PGSYNC_URL', 'RUVECTOR_EMBED_HOLDER', 'EMBED_SHARED', 'RUVECTOR_PARALLEL', 'MCP_SERVER']) delete e[k]; return e; };
function runClient(scenario, env) {
  const r = spawnSync(process.execPath, [__filename, 'client', scenario], { env: { ...baseEnv(), ...env }, encoding: 'utf8', timeout: 120000 });
  const line = (r.stdout || '').trim().split('\n').pop();
  try { return JSON.parse(line); } catch { throw new Error(`client ${scenario} gave no result: ${(r.stderr || '').slice(-300)}`); }
}
let n = 0;
function startHolder(mode) {
  const sock = path.join(W, `h${n++}.sock`), log = path.join(W, `h${n}.log`);
  fs.writeFileSync(log, '');
  const p = spawn(process.execPath, [__filename, 'holder', sock, mode, log], { env: baseEnv(), stdio: ['ignore', 'pipe', 'ignore'] });
  return new Promise((res, rej) => {
    const t = setTimeout(() => rej(new Error('fake holder did not start')), 120000);
    p.stdout.on('data', (d) => { if (String(d).includes('ready')) { clearTimeout(t); res({ sock, log, p, reqs: () => fs.readFileSync(log, 'utf8').split('\n').filter(Boolean) }); } });
  });
}
const env = (h, extra = {}) => ({ NEON_HOLDER_SOCK: h.sock, PGSYNC_URL: URL_, ...extra });
const embedReqs = (h) => h.reqs().filter((x) => x.startsWith('embed')).length;

let passed = 0, failed = 0;
async function t(name, fn) {
  try { await fn(); passed++; console.log(`  ok   ${name}`); }
  catch (e) { failed++; console.log(`  FAIL ${name}: ${e.message}`); }
}

(async () => {
  console.log('onnx-embedder through the neon-direct holder');
  const L = runClient('plain', {});   // the reference: a local load, no holder configured
  const same = (o, what) => {
    assert.deepStrictEqual(o.q, L.q, `${what}: query vector differs`);
    assert.deepStrictEqual(o.p, L.p, `${what}: passage vector differs`);
    assert.deepStrictEqual(o.b, L.b, `${what}: passage batch differs`);
    assert.deepStrictEqual(o.bq, L.bq, `${what}: query batch differs`);
  };
  const sameIdentity = (o) => {
    assert.deepStrictEqual(o.identity, L.identity, 'identity differs');
    assert.strictEqual(o.spaceId, L.spaceId, 'space id differs');
    assert.deepStrictEqual(o.prov, L.prov, 'provenance differs');
    assert.deepStrictEqual([o.ready, o.dim, o.model], [L.ready, L.dim, L.model]);
  };
  await t('reference: without a holder the embedder loads locally', () => assert.strictEqual(L.via1, 'local'));

  const H = await startHolder('normal');
  await t('a live holder for this database: every embed goes to it, no local model loaded', () => {
    const o = runClient('plain', env(H));
    assert.deepStrictEqual([o.via0, o.via1, o.holderEmbeds], ['holder', 'holder', 4]);
    assert.ok(embedReqs(H) >= 4, `holder saw ${embedReqs(H)} embed requests`);
  });
  await t('holder vectors are byte-identical to a local load (query, passage, both batches)', () => {
    const o = runClient('plain', env(H)); assert.strictEqual(o.via1, 'holder'); same(o, 'holder');
  });
  await t('identity, space id, provenance, model, dimension and readiness are identical to a local load', () => {
    const o = runClient('plain', env(H)); assert.strictEqual(o.via1, 'holder'); sameIdentity(o);
  });
  await t('a holder fingerprint for another database: local (never the wrong holder)', () => {
    const o = runClient('plain', { NEON_HOLDER_SOCK: H.sock, PGSYNC_URL: 'postgres://other.invalid/db' });
    assert.strictEqual(o.via1, 'local'); same(o, 'other db');
  });
  await t('RVN_URL wins over PGSYNC_URL, as in neon-holder-fp', () => {
    assert.strictEqual(runClient('plain', env(H, { RVN_URL: 'postgres://other.invalid/db' })).via1, 'local');
    assert.strictEqual(runClient('plain', env(H, { RVN_URL: URL_, PGSYNC_URL: 'postgres://x.invalid/y' })).via1, 'holder');
  });
  await t('RUVECTOR_EMBED_HOLDER=0 and EMBED_SHARED=0: local, the holder is not asked', () => {
    const before = H.reqs().length;
    assert.strictEqual(runClient('plain', env(H, { RUVECTOR_EMBED_HOLDER: '0' })).via1, 'local');
    assert.strictEqual(runClient('plain', env(H, { EMBED_SHARED: '0' })).via1, 'local');
    assert.strictEqual(H.reqs().length, before);
  });
  await t('the holder process itself (<sock>.pid is this process) never asks itself', () => {
    const o = runClient('plain', env(H, { T_SELFPID: '1' }));
    fs.writeFileSync(`${H.sock}.pid`, `${H.p.pid}\n`);
    assert.strictEqual(o.via1, 'local'); same(o, 'self');
  });
  for (const [what, cfg] of [['maxLength 64', { maxLength: 64 }], ['an explicit modelId', { modelId: 'all-MiniLM-L6-v2' }],
    ['a cacheDir', { cacheDir: path.join(os.homedir(), '.ruvector', 'models') }], ['enableParallel: true', { enableParallel: true }],
    ['normalize: false', { normalize: false }]]) {
    await t(`options that must stay local stay local: ${what}`, () => {
      const before = H.reqs().length;
      const o = runClient('plain', env(H, { T_CONFIG: JSON.stringify(cfg) }));
      assert.strictEqual(o.via0, 'local'); assert.strictEqual(H.reqs().length, before, 'the holder was asked');
    });
  }
  await t('RUVECTOR_PARALLEL=1 (the parallel pool): local', () => assert.strictEqual(runClient('plain', env(H, { RUVECTOR_PARALLEL: '1' })).via0, 'local'));
  await t('enableParallel: false and \'auto\' are still the plain default: holder', () => {
    assert.strictEqual(runClient('plain', env(H, { T_CONFIG: '{"enableParallel":false}' })).via1, 'holder');
    assert.strictEqual(runClient('plain', env(H, { T_CONFIG: '{"enableParallel":"auto"}' })).via1, 'holder');
  });
  await t('similarity() needs the local model: it loads it and answers as a local load does', () => {
    const o = runClient('similarity', env(H));
    const l = runClient('similarity', {});
    assert.strictEqual(o.viaSim, 'local'); assert.strictEqual(o.sim, l.sim); same(o, 'after similarity');
  });
  await t('the holder stops between two embeds: the next embed loads locally, vectors identical', () => {
    const o = runClient('kill', env(H));
    assert.deepStrictEqual([o.via0, o.via1], ['holder', 'local']); same(o, 'kill'); sameIdentity(o);
  });

  await t('fallback: NEON_HOLDER_SOCK names no socket → local, identical', () => {
    const o = runClient('plain', { NEON_HOLDER_SOCK: path.join(W, 'none.sock'), PGSYNC_URL: URL_ });
    assert.strictEqual(o.via1, 'local'); same(o, 'missing'); sameIdentity(o);
  });
  await t('fallback: a stale socket file whose holder is gone (connection refused) → local, identical', () => {
    const s = path.join(W, 'stale.sock');
    fs.writeFileSync(s, ''); fs.writeFileSync(`${s}.fp`, fp(URL_) + '\n');   // the path exists, nothing listens on it
    const o = runClient('plain', { NEON_HOLDER_SOCK: s, PGSYNC_URL: URL_ });
    assert.strictEqual(o.via1, 'local'); same(o, 'refused');
  });
  for (const [mode, what] of [['error', 'answers an error'], ['old', 'has no identity answer (a holder from before this change)'],
    ['badid', 'answers an unusable identity (256 dims)'], ['badvec', 'answers a malformed vector']]) {
    const h = await startHolder(mode);
    await t(`fallback: a holder that ${what} → local, identical`, () => {
      const o = runClient('plain', env(h));
      assert.strictEqual(o.via1, 'local'); same(o, mode); sameIdentity(o);
    });
    h.p.kill();
  }
  const S = await startHolder('silent');
  await t('fallback: a holder that never answers → local after RUVECTOR_EMBED_HOLDER_TIMEOUT_MS, identical', () => {
    const o = runClient('plain', env(S, { RUVECTOR_EMBED_HOLDER_TIMEOUT_MS: '300' }));
    assert.strictEqual(o.via1, 'local'); same(o, 'timeout');
    assert.ok(S.reqs().includes('silent'), 'the client never reached the silent holder');
  });
  S.p.kill(); H.p.kill();
  fs.rmSync(W, { recursive: true, force: true });
  console.log(`${passed} passed, ${failed} failed`);
  process.exit(failed ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
