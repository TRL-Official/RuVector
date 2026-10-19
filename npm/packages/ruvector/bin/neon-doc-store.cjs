'use strict';
// neon-doc-store.cjs — persist a JSON state document (ruvector's intelligence.json, ruflo's
// patterns.json, …) DIRECTLY in Neon, one row per record, from synchronous code.
//
// Why rows and not one JSONB blob: a blob is last-writer-wins, exactly like the file it replaces —
// two hook processes (or two Claude Code sessions) that load, add a record each and save would
// lose one record. Here save() is a three-way diff against what THIS process loaded:
//   • a record that is new or changed here          → upserted
//   • a record this process loaded and then removed → deleted
//   • a record some other process wrote meanwhile   → untouched (this process never saw it)
// so concurrent writers merge at record level. Same-record concurrent updates (since 2026-10-18, TODO D2):
// a record is updated only if Neon still holds the body this process loaded; if another process changed it
// meanwhile, its current body is read back and three-way merged (merge3: counters add both increments,
// timestamps keep the later, a key changed on one side keeps that change, arrays keep both sides' additions,
// otherwise ours) and written again under the same check. Until then the whole body was last-writer-wins:
// two hooks that each bumped a co-edit count or a pattern's success count kept one of the two.
//
// Why synchronous: ruvector's Intelligence.load()/save() and ruflo's loadFromDisk()/saveToDisk()
// are synchronous and called from dozens of sites. The call blocks in Atomics.wait while a worker
// thread does the HTTP round trip (neon-doc-worker.mjs) — measured ~119 ms, the same as an async
// query; a child process per call costs ~508 ms.
//
// Layout of a document → rows (coll, rid, body):
//   top-level array             → coll = key, rid = element.id when it is a string/number,
//                                  else sha1(element)#n          (order kept via seq)
//   top-level object in `keyed` → coll = key, rid = 'k:' + its sub-key  (patterns, errors, agents, …)
//   anything else               → coll = '_doc', rid = key       (stats, provenance, scalars)
// Unknown keys are kept — nothing is dropped, unlike ruvector's load() whitelist.
//
// Embeddings kept once (2026-10-19): a lazy record's embedding is copied into the pgvector column `vec` for search.
// With RVN_EMBED_IN_VEC=1 the copy in `body` is replaced by the marker "rvdirect:vec" — but only when Postgres
// confirms that `vec` gives back exactly the same numbers. Every read here rebuilds it from `vec`
// (REB below), with extra_float_digits pinned to 1: at 0 Postgres prints 15 digits and the rebuild would no longer
// be exact. Rows written before (embedding still in `body`) read as they are. A row holding the marker without
// `vec` is refused by the database (constraint rec_vec_mark, neon-embed-strip.mjs) and, if ever read, by EMB_OK.
//
// Env: RVN_URL (else PGSYNC_URL), RVN_SCHEMA (default rvdirect), PGSYNC_NEON_DRIVER,
//      RVN_TIMEOUT_MS (default 20000), RVN_EMBED_IN_VEC (1 = strip on write; default off).
const { Worker, MessageChannel, receiveMessageOnPort } = require('node:worker_threads');
const { createHash } = require('node:crypto');
const path = require('node:path');
const alert = require('./neon-alert.cjs'); // a failed call leaves a note the next prompt reports

function makeCaller() {
  const url = process.env.RVN_URL || process.env.PGSYNC_URL;
  if (!url) { const e = new Error('neon-doc-store: RVN_URL / PGSYNC_URL is not set — refusing to fall back to a file'); alert.record('doc-store', e); throw e; }
  const driver = process.env.PGSYNC_NEON_DRIVER || require.resolve('@neondatabase/serverless');
  const timeout = Number(process.env.RVN_TIMEOUT_MS) || 20000;
  // The worker inherits the host's execArgv; --input-type (node -e hosts) is refused for a file worker.
  const execArgv = process.execArgv.filter((a) => !a.startsWith('--input-type'));
  // the connection holder (neon-holder.mjs), when its socket exists; otherwise Neon directly
  // only a holder that serves THIS database (neon-holder-fp.cjs); otherwise connect directly to `url`
  const sock = require('./neon-holder-fp.cjs').holderFor(url);
  const worker = new Worker(path.join(__dirname, 'neon-doc-worker.mjs'), { workerData: { url, driver, sock }, execArgv });
  worker.unref(); // never keep a CLI process alive
  // NEON_TRACE=1: round trips and their time, per process, to stderr at exit (diagnostics only);
  // NEON_TRACE=2 also lists every round trip in order (statement start, store/param, ms)
  const tr = (process.env.NEON_TRACE === '1' || process.env.NEON_TRACE === '2') ? { n: 0, ms: 0, slow: [] } : null;
  if (tr) process.on('exit', () => process.env.NEON_TRACE === '2' && process.stderr.write(tr.slow.map(([d, w], i) => `[neon-trace #${i + 1}] ${d} ms ${w}`).join('\n') + '\n'));
  if (tr) process.on('exit', () => process.stderr.write(`[neon-trace doc kinds] ${JSON.stringify((tr.all || []).reduce((m, k) => (m[k] = (m[k] || 0) + 1, m), {}))}\n[neon-trace doc] ${tr.n} round trips, ${tr.ms} ms; slowest: ${tr.slow.sort((a, b) => b[0] - a[0]).slice(0, 5).map((x) => x.join(' ')).join(' | ')}\n`));
  return function call(stmts, cache) {
    const t0 = tr && Date.now();
    try { return call0(stmts, cache); } finally { if (tr) { const d = Date.now() - t0; tr.n++; tr.ms += d; tr.all = tr.all || []; tr.all.push(stmts[0].text.trim().split(/\s+/).slice(0, 3).join(' ') + (cache ? ':load' : '')); tr.slow.push([d, stmts[0].text.replace(/\s+/g, ' ').slice(0, 28) + ' [' + String((stmts[0].params || [])[0]).slice(0, 50) + ']' + (cache ? ' (cacheable)' : '')]); } }
  };
  function call0(stmts, cache) {
    const sab = new SharedArrayBuffer(4);
    const { port1, port2 } = new MessageChannel();
    worker.postMessage({ stmts, cache, sab, port: port2 }, [port2]);
    const r = Atomics.wait(new Int32Array(sab), 0, 0, timeout);
    const msg = receiveMessageOnPort(port1);
    port1.close();
    if (r === 'timed-out' || !msg) { const e = new Error(`neon-doc-store: Neon did not answer within ${timeout} ms`); alert.record('doc-store', e); throw e; }
    if (msg.message.error) { const e = new Error(`neon-doc-store: ${msg.message.error}`); e.pgCode = msg.message.code; alert.record('doc-store', e); throw e; }
    if (msg.message.raw !== undefined) return JSON.parse(msg.message.raw).rows;   // a holder reply passed through unparsed
    return msg.message.rows;
  };
}

// Three-way merge of one record: `base` = what the writer loaded (undefined: it created the record), `ours` =
// what it wants to write, `theirs` = what Neon holds now (written by another process meanwhile).
const J = (v) => JSON.stringify(v);
const isObj = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const COUNTER_MAX = 1e9;   // integers below this that both sides raised are counters: add both increments
function merge3(base, ours, theirs) {
  // NOT first "ours equals theirs": both sides raising a counter 0 → 1 is two increments, not one change
  if (base !== undefined && J(base) === J(ours)) return theirs;
  if (base !== undefined && J(base) === J(theirs)) return ours;
  if (isObj(ours) && isObj(theirs)) {
    const b = isObj(base) ? base : {}; const out = {};
    for (const k of new Set([...Object.keys(theirs), ...Object.keys(ours)])) {
      const inO = k in ours, inT = k in theirs, inB = k in b;
      if (inO && inT) out[k] = merge3(inB ? b[k] : undefined, ours[k], theirs[k]);
      else if (inO) { if (!(inB && J(b[k]) === J(ours[k]))) out[k] = ours[k]; }      // they removed it: keep only if we changed it
      else if (inT) { if (!(inB && J(b[k]) === J(theirs[k]))) out[k] = theirs[k]; }  // we removed it: keep only if they changed it
    }
    return out;
  }
  if (typeof ours === 'number' && typeof theirs === 'number' && typeof base === 'number') {
    const up = ours > base && theirs > base;
    if (up && Number.isInteger(base) && Number.isInteger(ours) && Number.isInteger(theirs) && Math.abs(ours) < COUNTER_MAX && Math.abs(theirs) < COUNTER_MAX)
      return base + (ours - base) + (theirs - base);
    if (up && Math.abs(ours) >= COUNTER_MAX && Math.abs(theirs) >= COUNTER_MAX) return Math.max(ours, theirs);   // timestamps (s or ms): the later
    return ours;
  }
  if (J(ours) === J(theirs)) return ours;   // the same change on both sides (a flag set, a value written)
  if (Array.isArray(ours) && Array.isArray(theirs)) {
    const b = Array.isArray(base) ? base : [];
    const inBase = new Set(b.map(J)), inOurs = new Set(ours.map(J));
    const out = theirs.filter((e) => !(inBase.has(J(e)) && !inOurs.has(J(e))));      // what we removed stays removed
    const have = new Set(out.map(J));
    for (const e of ours) if (!inBase.has(J(e)) && !have.has(J(e))) { out.push(e); have.add(J(e)); }  // what we added
    return out;
  }
  return ours;
}

// One worker per process, shared by every store opened in it (neon-fs.mjs opens one per file).
let sharedCall = null;

// --- embeddings kept once (see the header) ---
const MARK = 'rvdirect:vec';
// the body as written, with the embedding rebuilt from vec when it was stripped. `a` = the table alias.
// Exact only with extra_float_digits >= 1 in the same transaction: every statement list using it starts with PIN.
const REB = (a) => `CASE WHEN ${a}.body->>'embedding' = '${MARK}' AND ${a}.vec IS NOT NULL
  THEN jsonb_set(${a}.body, '{embedding}', to_jsonb(${a}.vec::real[]::float8[])) ELSE ${a}.body END`;
const PIN = { text: 'SET LOCAL extra_float_digits = 1' };
// a read must never hand the marker on: it means vec was lost, and callers would take a string for a vector
function EMB_OK(body, where) {
  if (body && typeof body === 'object' && body.embedding === MARK) {
    const e = new Error(`neon-doc-store: ${where} holds the embedding marker but no vector — refusing to return it`);
    alert.record('doc-store', e); throw e;
  }
  return body;
}

function openDocStore({ store, keyed = [], lazy = [], window = 0 }) {
  const schema = process.env.RVN_SCHEMA || 'rvdirect';
  if (!/^[a-z_][a-z0-9_]*$/.test(schema)) throw new Error(`neon-doc-store: bad schema name ${schema}`);
  const T = `${schema}.rec`;
  // keyed: the top-level objects stored one row per sub-key. '*' = every top-level object whose
  // values are all objects (generic stores whose shape is not known in advance — neon-fs.mjs).
  const AUTO = keyed === '*';
  const KEYED = new Set(AUTO ? [] : keyed);
  const isKeyed = (key, val) => (KEYED.has(key) && val && typeof val === 'object') || (AUTO && val && typeof val === 'object' && !Array.isArray(val)
    && Object.keys(val).length > 0 && Object.values(val).every((v) => v && typeof v === 'object'));
  // `lazy` collections (ruvector's memories) are never loaded whole — at real scale the document
  // exceeds Neon's 64 MB HTTP response limit (gate R9). Their rows carry a pgvector column, and
  // recall becomes search() — a query in Neon. Only the newest `window` rows are loaded, and since
  // a process deletes only rows it loaded, the unloaded rest is never at risk.
  const LAZY = [...lazy];
  const DIM = Number(process.env.RVN_DIM) || 384;
  const WINDOW = Number(process.env.RVN_WINDOW ?? window) || 0;
  const call = (sharedCall ??= makeCaller());
  // statements whose SQL uses REB: run after PIN, in one transaction; the PIN's own result is dropped
  const callPinned = (stmts) => call([PIN, ...stmts]).slice(1);
  const STRIP = process.env.RVN_EMBED_IN_VEC === '1';
  let snapshot = new Map(); // `${coll}\u0000${rid}` → JSON of body as loaded/saved
  // Row ids of lazy-collection elements, by object identity: an element this process loaded or saved
  // keeps its row; a NEW element always gets a row of its own. Ids alone cannot decide it — ruvector's
  // learn() names trajectories traj_<seconds>, so two hooks in one second (any session) share an id,
  // and with the collection not loaded the second would have overwritten the first row.
  const lazyRid = new WeakMap();
  const stats = { loads: 0, saves: 0, upserts: 0, deletes: 0, ms: 0, searchExact: 0, searchIndex: 0 };

  let uidN = 0;
  const uid = () => `${process.pid.toString(36)}${Date.now().toString(36)}${(uidN++).toString(36)}${Math.random().toString(36).slice(2, 6)}`;

  function toRows(doc) {
    const rows = [];
    for (const [key, val] of Object.entries(doc)) {
      if (val === undefined) continue;
      if (Array.isArray(val) && LAZY.includes(key)) {
        const used = new Set();
        for (const el of val) {
          const obj = el && typeof el === 'object';
          let rid = obj ? lazyRid.get(el) : undefined;
          if (!rid || used.has(rid)) {
            const base = obj && (typeof el.id === 'string' || typeof el.id === 'number') ? `id:${el.id}`
              : `h:${createHash('sha1').update(JSON.stringify(el)).digest('hex').slice(0, 20)}`;
            rid = `${base}~${uid()}`;
            if (obj) lazyRid.set(el, rid);
          }
          used.add(rid);
          rows.push({ coll: key, rid, body: el, kind: 'a' });
        }
        if (val.length === 0) rows.push({ coll: '_doc', rid: `[]${key}`, body: [], kind: 'e' });
      } else if (Array.isArray(val)) {
        // Elements without an id are identified by content. An element this process LOADED keeps the
        // row id it came with (first unclaimed row with the same content hash); a NEW element gets a
        // fresh unique id — so two sessions appending identical-looking entries (a history of
        // outcomes) both keep theirs, instead of colliding on the same content-derived id.
        // Matched by the element's exact JSON as loaded FIRST (2026-10-18): the rid's hash was taken from the
        // writer's key order, but a body comes back from jsonb in Postgres's key order, so the hash of a loaded
        // element often differs from its rid. Then the newest receipt of the policy chain was deleted and
        // re-inserted (identical) by every ruflo command, which made the holder reload all 14 MB every time.
        const avail = new Map(), byJson = new Map();
        for (const [k, j] of snapshot) {
          const [c, r] = k.split('\u0000');
          if (c !== key || !r.startsWith('h:')) continue;
          const h = r.slice(2, 22); (avail.get(h) || avail.set(h, []).get(h)).push(r);
          (byJson.get(j) || byJson.set(j, []).get(j)).push(r);
        }
        const claimed = new Set();
        const take = (list) => { while (list && list.length) { const r = list.shift(); if (!claimed.has(r)) { claimed.add(r); return r; } } return undefined; };
        const seen = new Map();
        for (const el of val) {
          let rid;
          if (el && typeof el === 'object' && (typeof el.id === 'string' || typeof el.id === 'number')) {
            // Real stores repeat ids (the live trajectories do), so a repeat gets #n: both copies are kept.
            const base = `id:${el.id}`;
            const n = (seen.get(base) || 0) + 1; seen.set(base, n);
            rid = n === 1 ? base : `${base}#${n}`;
          } else {
            const j = JSON.stringify(el);
            const h = createHash('sha1').update(j).digest('hex').slice(0, 20);
            rid = take(byJson.get(j)) ?? take(avail.get(h)) ?? `h:${h}~${uid()}`;
          }
          rows.push({ coll: key, rid, body: el, kind: 'a' });
        }
        if (val.length === 0) rows.push({ coll: '_doc', rid: `[]${key}`, body: [], kind: 'e' }); // keep empty arrays
      } else if (isKeyed(key, val)) {
        for (const [k, v] of Object.entries(val)) if (v !== undefined) rows.push({ coll: key, rid: `k:${k}`, body: v, kind: 'o' });
        if (Object.keys(val).length === 0) rows.push({ coll: '_doc', rid: `{}${key}`, body: {}, kind: 'e' });
      } else {
        rows.push({ coll: '_doc', rid: key, body: val, kind: 'd' });
      }
    }
    return rows;
  }

  const DDL = [
    { text: `CREATE SCHEMA IF NOT EXISTS ${schema}` },
    { text: `CREATE TABLE IF NOT EXISTS ${T} (store text NOT NULL, coll text NOT NULL, rid text NOT NULL, body jsonb NOT NULL,
               seq bigserial, ver int NOT NULL DEFAULT 1, updated_at timestamptz NOT NULL DEFAULT now(),
               vec vector(${DIM}), PRIMARY KEY (store, coll, rid))` },
    { text: `CREATE INDEX IF NOT EXISTS rec_vec_hnsw ON ${T} USING hnsw (vec vector_cosine_ops)` },
    // a stripped embedding (the marker) is only ever stored together with its vector
    { text: `DO $c$ BEGIN ALTER TABLE ${T} ADD CONSTRAINT rec_vec_mark CHECK (body->>'embedding' IS DISTINCT FROM '${MARK}' OR vec IS NOT NULL);
               EXCEPTION WHEN duplicate_object THEN NULL; END $c$` },
    // NOTHING IS EVER LOST: every deleted row — by any tool, any session (ruflo pruning its patterns,
    // a JSON store rewritten without an entry, a file unlinked) — is copied here first, by Postgres.
    { text: `CREATE TABLE IF NOT EXISTS ${schema}.rec_deleted (store text NOT NULL, coll text NOT NULL, rid text NOT NULL, body jsonb,
               vec vector(${DIM}), seq bigint, ver int, deleted_at timestamptz NOT NULL DEFAULT now())` },
    { text: `CREATE INDEX IF NOT EXISTS rec_deleted_store ON ${schema}.rec_deleted (store, coll)` },
    { text: `CREATE OR REPLACE FUNCTION ${schema}.rec_keep_deleted() RETURNS trigger LANGUAGE plpgsql AS $f$
               BEGIN INSERT INTO ${schema}.rec_deleted (store, coll, rid, body, vec, seq, ver) VALUES (OLD.store, OLD.coll, OLD.rid, OLD.body, OLD.vec, OLD.seq, OLD.ver);
               RETURN OLD; END $f$` },
    { text: `CREATE OR REPLACE TRIGGER rec_keep_deleted BEFORE DELETE ON ${T} FOR EACH ROW EXECUTE FUNCTION ${schema}.rec_keep_deleted()` },
  ];
  // DDL only when the table is missing: CREATE INDEX IF NOT EXISTS still locks the table, and
  // every hook process loads.
  function withTable(fn) {
    try { return fn(); }
    catch (e) {
      if (!/does not exist/.test(e.message)) throw e;
      try { call(DDL); } catch { /* a concurrent first-time creator won the race */ }
      return fn();
    }
  }

  // A repeat load in the same process (2026-10-18): ruflo reads the policy chain (14 MB, 12,797 receipts) two
  // or three times per command — exists, stat, read, then again to report the mode — 0.45–2.2 s each. The
  // rows of the last full load are kept with the store's fingerprint (non-lazy rows: count, max seq, sum of
  // ver — any insert, update or delete changes one of them), computed from those same rows, so it describes
  // exactly what was loaded. A repeat load asks Neon for the fingerprint only (one small query) and, when it
  // is unchanged, rebuilds the document from memory. RVN_NO_MEMO=1 turns it off.
  let memo = null;   // { fp, rows: [{ coll, rid, j }], primedAt? }
  // prime(rows): rows another query already fetched (neon-fs reads the small files along with its file list,
  // 2026-10-18). The FIRST load within PRIME_MS uses them as they are — a snapshot from a moment ago, as a load
  // a moment earlier would have been; writes stay checked, so it cannot overwrite anything newer. Later loads
  // check the fingerprint as usual.
  const PRIME_MS = Number(process.env.RVN_PRIME_MS ?? 3000);
  const MEMO = !WINDOW && process.env.RVN_NO_MEMO !== '1';
  const fpOf = (rows) => { let n = 0, s = 0n, v = 0n; for (const r of rows) { if (LAZY.includes(r.coll)) continue; if (r.seq == null || r.ver == null) return null; n++; const q = BigInt(Math.round(Number(r.seq))); if (q > s) s = q; v += BigInt(r.ver); } return `${n}:${s}:${v}`; };
  stats.memoHits = 0;
  // load() also records what this process has SEEN (the snapshot save() diffs against). peek=true reads
  // without that — for a stat or an existence check, which must not widen what a later save may delete.
  function load({ peek = false } = {}) {
    const t = Date.now();
    let rows = null, fresh = false;
    if (MEMO && memo && memo.primedAt && Date.now() - memo.primedAt < PRIME_MS) {
      rows = memo.rows.map((r) => ({ coll: r.coll, rid: r.rid, body: JSON.parse(r.j), j: r.j })); stats.primeHits = (stats.primeHits || 0) + 1;
      memo = { fp: memo.fp, rows: memo.rows };   // used once: the next load checks the fingerprint
    } else if (MEMO && memo) {
      const [[f]] = withTable(() => call([{ text: `SELECT count(*)::int AS n, coalesce(max(seq), 0)::text AS s, coalesce(sum(ver), 0)::text AS v
                   FROM ${T} WHERE store = $1 AND NOT (coll = ANY($2::text[]))`, params: [store, LAZY] }]));
      if (`${f.n}:${f.s}:${f.v}` === memo.fp) { rows = memo.rows.map((r) => ({ coll: r.coll, rid: r.rid, body: JSON.parse(r.j), j: r.j })); stats.memoHits++; }
    }
    if (!rows) {
      // lazy rows (the window) may carry a stripped embedding: rebuilt from vec (REB, pinned)
      const SELECT = [
        { text: `SELECT coll, rid, body, seq, ver FROM (
                   SELECT coll, rid, body, seq, ver FROM ${T} WHERE store = $1 AND NOT (coll = ANY($2::text[]))
                   UNION ALL
                   SELECT coll, rid, body, seq, ver FROM (SELECT coll, rid, ${REB('l')} AS body, seq, ver FROM ${T} l
                     WHERE store = $1 AND coll = ANY($2::text[]) ORDER BY seq DESC LIMIT $3) w
                 ) d ORDER BY seq`, params: [store, LAZY, WINDOW] },
      ];
      // the holder may answer from memory when the store's fingerprint is unchanged (no session wrote to
      // it since); a window of lazy rows changes with every append, so only window-less loads are cached.
      // A window-less load holds no lazy rows (LIMIT 0), so it needs no PIN, and the holder's cached answer
      // (non-lazy rows only, neon-holder.mjs) is the same document.
      const cache = WINDOW ? undefined : { key: `${schema}\u0000${store}\u0000${LAZY.join(',')}`, store, table: T, lazy: LAZY };
      [rows] = withTable(() => (cache ? call(SELECT, cache) : callPinned(SELECT)));
      fresh = true;
    }
    stats.loads++; stats.ms += Date.now() - t;
    const seen = new Map();
    if (!peek) snapshot = seen;
    if (!rows.length) { if (fresh && MEMO) memo = { fp: '0:0:0', rows: [] }; return null; } // an empty store: the caller starts from its defaults
    const doc = {};
    const keep = fresh && MEMO ? [] : null; const fp = keep && fpOf(rows);
    for (const { coll, rid, body, j: j0 } of rows) {
      EMB_OK(body, `${store} ${coll}/${rid}`);
      const j = j0 ?? JSON.stringify(body);
      seen.set(`${coll}\u0000${rid}`, j);
      if (keep && fp) keep.push({ coll, rid, j });
      if (coll === '_doc') {
        if (rid.startsWith('[]')) { doc[rid.slice(2)] ??= []; continue; }
        if (rid.startsWith('{}')) { doc[rid.slice(2)] ??= {}; continue; }
        doc[rid] = body;
      } else if (rid.startsWith('k:')) {
        (doc[coll] ??= {})[rid.slice(2)] = body;
      } else {
        (doc[coll] ??= []).push(body);
        if (LAZY.includes(coll) && body && typeof body === 'object') lazyRid.set(body, rid);
      }
    }
    if (keep) memo = fp ? { fp, rows: keep } : null;
    return doc;
  }

  // Upsert `up` [{coll, rid, body, base?}] and delete `del` [{coll, rid, base?}].
  // `base` makes a write CHECKED (TODO D2): a JSON string = the body the writer loaded — the row is updated
  // (or deleted) only if Neon still holds it; null = a new record — inserted only if nobody created it
  // meanwhile. A checked upsert that finds the row changed reads it back, merges (merge3) and tries again.
  // No `base` key: a plain upsert/delete, as before.
  const VEC = `CASE WHEN x.coll = ANY($3::text[]) AND jsonb_typeof(x.body->'embedding') = 'array'
                     AND jsonb_array_length(x.body->'embedding') = ${DIM}
                    THEN (x.body->>'embedding')::vector END`;
  // the body to store: with RVN_EMBED_IN_VEC=1, the embedding becomes the marker, but ONLY when the vector gives back
  // exactly these numbers (the same expression REB reads with, under the same PIN); otherwise the full body
  const SB = !STRIP ? 'x.body' : `CASE WHEN (${VEC}) IS NOT NULL
                     AND x.body->'embedding' = to_jsonb(((x.body->>'embedding')::vector)::real[]::float8[])
                    THEN jsonb_set(x.body, '{embedding}', to_jsonb('${MARK}'::text)) ELSE x.body END`;
  // a writer's `base` (the body as it loaded it, embedding included) against the stored row: equal when the stored
  // body, embedding rebuilt, is the same — read under PIN
  const SAME = (t) => `${REB(t)} = x.base::jsonb`;
  // e->'body', not jsonb_to_recordset's body column: that turns a JSON null into SQL NULL,
  // and a record whose value is null ({"lastScan": null}) failed the NOT NULL column
  const X = `SELECT a.n AS ord, a.e->>'coll' AS coll, a.e->>'rid' AS rid, coalesce(a.e->'body', 'null'::jsonb) AS body,
                    a.e->>'base' AS base, a.e ? 'base' AS checked
             FROM jsonb_array_elements($2::jsonb) WITH ORDINALITY AS a(e, n)`;
  const conflicts = { merged: 0, rounds: 0 };
  // plain rows: upsert; checked rows: update-if-unchanged / insert-if-absent. RETURNING the checked rows written.
  const write = (rows) => ({
    text: `WITH x AS (${X}),
           plain AS (INSERT INTO ${T} (store, coll, rid, body, vec)
             SELECT $1, x.coll, x.rid, ${SB}, ${VEC} FROM x WHERE NOT x.checked ORDER BY x.ord
             ON CONFLICT (store, coll, rid) DO UPDATE SET body = excluded.body, vec = excluded.vec,
                    ver = ${T}.ver + 1, updated_at = now()
             -- vec too: with the embedding stripped, a changed embedding leaves the body (the marker) unchanged
             WHERE ${T}.body IS DISTINCT FROM excluded.body OR ${T}.vec IS DISTINCT FROM excluded.vec),
           upd AS (UPDATE ${T} t SET body = ${SB}, vec = ${VEC}, ver = t.ver + 1, updated_at = now()
             FROM x WHERE x.checked AND x.base IS NOT NULL AND t.store = $1 AND t.coll = x.coll AND t.rid = x.rid
               AND ${SAME('t')}
             RETURNING t.coll, t.rid),
           ins AS (INSERT INTO ${T} (store, coll, rid, body, vec)
             SELECT $1, x.coll, x.rid, ${SB}, ${VEC} FROM x WHERE x.checked AND x.base IS NULL ORDER BY x.ord
             ON CONFLICT (store, coll, rid) DO NOTHING
             RETURNING coll, rid)
           SELECT coll, rid FROM upd UNION ALL SELECT coll, rid FROM ins`,
    params: [store, JSON.stringify(rows), LAZY],
  });
  function writeRows(up, del = []) {
    if (!up.length && !del.length) return;
    memo = null;   // any write: the next load reads Neon again
    up = up.map((u, i) => ({ ord: i, coll: u.coll, rid: u.rid, body: u.body, ...('base' in u ? { base: u.base } : {}) }));
    // One transaction per ~8 MB chunk: Neon's HTTP endpoint caps a request/response at 64 MB, and
    // the first save of a real store is 63 MB. Rows are independent, so chunking loses nothing
    // a reader could observe as inconsistent within one record.
    const CHUNK = Number(process.env.RVN_CHUNK_BYTES) || 8 * 1024 * 1024;
    const chunks = [];
    let cur = [], size = 0;
    for (const u of up) {
      const n = JSON.stringify(u.body).length + (typeof u.base === 'string' ? u.base.length : 0) + 64;
      if (cur.length && size + n > CHUNK) { chunks.push(cur); cur = []; size = 0; }
      cur.push(u); size += n;
    }
    if (cur.length) chunks.push(cur);
    const delStmt = del.length && {
      // a checked delete removes the row only if Neon still holds what the deleter loaded: an update another
      // process made meanwhile wins over the delete (nothing is lost either way — rec_deleted keeps a copy)
      text: `DELETE FROM ${T} t USING (SELECT a.e->>'coll' AS coll, a.e->>'rid' AS rid, a.e->>'base' AS base, a.e ? 'base' AS checked
                                       FROM jsonb_array_elements($2::jsonb) AS a(e)) x
             WHERE t.store = $1 AND t.coll = x.coll AND t.rid = x.rid AND (NOT x.checked OR x.base IS NULL OR ${SAME('t')})`,
      params: [store, JSON.stringify(del)],
    };
    if (!chunks.length) withTable(() => callPinned([delStmt]));
    chunks.forEach((c, i) => {
      const res = withTable(() => callPinned(i === chunks.length - 1 && delStmt ? [write(c), delStmt] : [write(c)]));
      resolveConflicts(c, res[0]);
    });
  }
  // the checked rows of a chunk that were not written: someone changed (or created, or deleted) them since
  // they were loaded. Read them back, merge, and write again under the same check — at most 30 rounds, with a random back-off.
  function resolveConflicts(chunk, written) {
    const done = new Set(written.map((r) => `${r.coll}\u0000${r.rid}`));
    let pending = chunk.filter((u) => 'base' in u && !done.has(`${u.coll}\u0000${u.rid}`));
    for (let round = 0; pending.length; round++) {
      if (round >= 30) { const e = new Error(`neon-doc-store: ${pending.length} record(s) of ${store} kept changing under concurrent writers — not saved`); alert.record('doc-store', e); throw e; }
      // writers that collided would collide again in lockstep: wait a random, growing moment (synchronous code)
      if (round) Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, Math.floor(Math.random() * Math.min(400, 20 * 2 ** Math.min(round, 5))));
      conflicts.rounds++;
      const [now] = withTable(() => callPinned([{ text: `SELECT coll, rid, ${REB('t')} AS body FROM ${T} t JOIN jsonb_to_recordset($2::jsonb) AS x(coll text, rid text) USING (coll, rid) WHERE t.store = $1`,
        params: [store, JSON.stringify(pending.map(({ coll, rid }) => ({ coll, rid })))] }]));
      const cur = new Map(now.map((r) => [`${r.coll}\u0000${r.rid}`, EMB_OK(r.body, `${store} ${r.coll}/${r.rid}`)]));
      const retry = [];
      for (const u of pending) {
        const k = `${u.coll}\u0000${u.rid}`;
        const base = typeof u.base === 'string' ? JSON.parse(u.base) : undefined;
        if (!cur.has(k)) { retry.push({ ...u, base: null }); continue; }   // deleted meanwhile: ours is re-created (the delete's copy is in rec_deleted)
        const theirs = cur.get(k);
        const body = merge3(base, u.body, theirs);
        conflicts.merged++;
        if (J(body) === J(theirs)) continue;                                // theirs already holds everything
        retry.push({ ...u, body, base: J(theirs) });
      }
      if (!retry.length) return;
      const [w] = withTable(() => callPinned([write(retry)]));
      const ok = new Set(w.map((r) => `${r.coll}\u0000${r.rid}`));
      pending = retry.filter((u) => !ok.has(`${u.coll}\u0000${u.rid}`));
    }
  }

  function save(doc) {
    const t = Date.now();
    const rows = toRows(doc);
    const now = new Map();
    const up = [];
    let ord = 0;
    for (const r of rows) {
      const k = `${r.coll}\u0000${r.rid}`;
      const j = JSON.stringify(r.body);
      now.set(k, j);
      if (snapshot.get(k) !== j) up.push({ ord: ord++, coll: r.coll, rid: r.rid, body: r.body, base: snapshot.has(k) ? snapshot.get(k) : null });
    }
    const del = [];
    for (const k of snapshot.keys()) if (!now.has(k)) { const [coll, rid] = k.split('\u0000'); del.push({ coll, rid, base: snapshot.get(k) }); }
    if (!up.length && !del.length) return { upserts: 0, deletes: 0 };
    writeRows(up, del);
    memo = null;   // what Neon holds now is not what was loaded
    snapshot = now;
    stats.saves++; stats.upserts += up.length; stats.deletes += del.length; stats.ms += Date.now() - t;
    return { upserts: up.length, deletes: del.length };
  }

  function prime(rows) {
    if (!MEMO || memo || snapshot.size) return false;   // only a store this process has not read or written yet
    const fp = fpOf(rows); if (!fp) return false;
    memo = { fp, rows: rows.map((r) => ({ coll: r.coll, rid: r.rid, j: JSON.stringify(r.body) })), primedAt: Date.now() };
    return true;
  }

  // Nearest rows of a lazy collection by cosine similarity: [{ ...body, score }], best first.
  // opts.eq = { field: value } filters on top-level body fields; opts.threshold drops weak matches.
  function search(coll, vector, k = 5, opts = {}) {
    if (!Array.isArray(vector) && !ArrayBuffer.isView(vector)) return [];
    if (vector.length !== DIM) return []; // e.g. the 64-dim hash fallback: nothing comparable stored
    const t = Date.now();
    const params = [store, `[${Array.from(vector).join(',')}]`, coll, k];
    let where = '';
    for (const [f, v] of Object.entries(opts.eq || {})) { params.push(f, String(v)); where += ` AND body->>$${params.length - 1} = $${params.length}`; }
    if (opts.threshold != null) { params.push(opts.threshold); where += ` AND 1 - (vec <=> $2::vector) >= $${params.length}`; }
    // Complete recall (2026-10-19). Every store shares ONE HNSW index, and the store filter was applied to its
    // candidates afterwards: when the nearest candidates belonged to other stores, a search came back short or
    // empty (2 of 50 memory searches in a rehearsal). Default now: an EXACT scan of the store's rows (ties by row id)
    // — complete and truly nearest; ~20-25 ms on the server for 7,000 memories (the HNSW scan: <1 ms; the round
    // trip ~105 ms either way). RVN_SEARCH=index keeps the index for large stores: pgvector >= 0.8's iterative scan
    // (strict order; a guarded block, so an older pgvector skips it) keeps scanning until enough rows pass the
    // filter, and a result still short of k is redone exactly. Counts in stats: searchIndex, searchExact.
    const SELECT = { text: `SELECT ${REB('s')} AS body, 1 - (vec <=> $2::vector) AS score FROM ${T} s
               WHERE store = $1 AND coll = $3 AND vec IS NOT NULL${where} ORDER BY vec <=> $2::vector, rid LIMIT $4`, params };
    const exact = () => { stats.searchExact = (stats.searchExact || 0) + 1; return withTable(() => callPinned([{ text: 'SET LOCAL enable_indexscan = off' }, SELECT]))[1]; };
    let rows;
    if (process.env.RVN_SEARCH === 'index') {
      stats.searchIndex = (stats.searchIndex || 0) + 1;
      [, , rows] = withTable(() => callPinned([
        { text: `SET LOCAL hnsw.ef_search = ${Math.max(40, k * 4)}` },
        { text: `DO $i$ BEGIN PERFORM set_config('hnsw.iterative_scan', 'strict_order', true); EXCEPTION WHEN OTHERS THEN NULL; END $i$` },
        SELECT,
      ]));
      if (rows.length < k) rows = exact();
    } else rows = exact();
    stats.ms += Date.now() - t;
    return rows.map((r) => ({ ...EMB_OK(r.body, `${store} ${coll} (search)`), score: Number(r.score) }));
  }

  // Row-level access for stores that are never loaded as a document (ruflo's patterns).
  // every body read here has its embedding rebuilt (REB, pinned) and is checked (EMB_OK)
  const ok = (list, coll) => { for (const r of list) EMB_OK(r.body, `${store} ${coll}/${r.rid ?? r.seq}`); return list; };
  const rows = {
    get: (coll, rids) => rids.length ? ok(withTable(() => callPinned([{ text: `SELECT rid, ${REB('g')} AS body FROM ${T} g WHERE store = $1 AND coll = $2 AND rid = ANY($3::text[])`, params: [store, coll, rids] }]))[0], coll) : [],
    findBy: (coll, field, value) => ok(withTable(() => callPinned([{ text: `SELECT rid, ${REB('f')} AS body FROM ${T} f WHERE store = $1 AND coll = $2 AND body->>$3 = $4 ORDER BY seq LIMIT 1`, params: [store, coll, field, String(value)] }]))[0], coll),
    count: (coll) => Number(withTable(() => call([{ text: `SELECT count(*) AS n FROM ${T} WHERE store = $1 AND coll = $2`, params: [store, coll] }]))[0][0].n),
    // every row, oldest first, in pages — a single response is capped at 64 MB
    all: (coll, eq = {}) => {
      const out = []; let after = 0;
      const [f, v] = Object.entries(eq)[0] || [];
      for (;;) {
        const page = ok(withTable(() => callPinned([{ text: `SELECT seq, ${REB('p')} AS body FROM ${T} p WHERE store = $1 AND coll = $2 AND seq > $3
          ${f ? 'AND body->>$5 = $6' : ''} ORDER BY seq LIMIT $4`, params: f ? [store, coll, after, 2000, f, String(v)] : [store, coll, after, 2000] }]))[0], coll);
        for (const r of page) out.push(r.body);
        if (page.length < 2000) return out;
        after = page[page.length - 1].seq;
      }
    },
    // an upsert may carry `base` (checked write, see writeRows); a delete may be { rid, base } instead of a rid
    write: (coll, upserts, deletes = []) => writeRows(upserts.map((u) => ({ coll, ...u })), deletes.map((d) => (typeof d === 'object' ? { coll, ...d } : { coll, rid: d }))),
    // retention, only when a cap is explicitly asked for: keep the newest `keep` rows
    retain: (coll, keep) => (memo = null, withTable(() => call([{ text: `DELETE FROM ${T} WHERE store = $1 AND coll = $2 AND seq <= (
          SELECT seq FROM ${T} WHERE store = $1 AND coll = $2 ORDER BY seq DESC OFFSET $3 LIMIT 1)`, params: [store, coll, keep] }]))),
    clear: (coll) => (memo = null, withTable(() => call([{ text: `DELETE FROM ${T} WHERE store = $1 AND coll = $2`, params: [store, coll] }]))),
  };

  // Remove the whole store (a deleted file).
  function drop() { withTable(() => call([{ text: `DELETE FROM ${T} WHERE store = $1`, params: [store] }])); snapshot = new Map(); memo = null; }
  // Does the store hold anything? (one indexed lookup, cheaper than load)
  function exists() { return withTable(() => call([{ text: `SELECT 1 FROM ${T} WHERE store = $1 LIMIT 1`, params: [store] }]))[0].length > 0; }

  // ensure(): the full DDL, idempotent — for a store table created before the deleted-rows archive
  function ensure() { call(DDL); }
  return { load, save, prime, search, rows, drop, exists, ensure, stats, conflicts, query: (text, params) => withTable(() => call([{ text, params }])[0]) };
}

module.exports = { openDocStore, merge3 };
