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
// so concurrent writers merge at record level. Same-record concurrent updates: last writer wins
// for that one record (the sync layer's per-record clocks are the upgrade path — see the doc).
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
// Env: RVN_URL (else PGSYNC_URL), RVN_SCHEMA (default rvdirect), PGSYNC_NEON_DRIVER,
//      RVN_TIMEOUT_MS (default 20000).
const { Worker, MessageChannel, receiveMessageOnPort } = require('node:worker_threads');
const { createHash } = require('node:crypto');
const path = require('node:path');

function makeCaller() {
  const url = process.env.RVN_URL || process.env.PGSYNC_URL;
  if (!url) throw new Error('neon-doc-store: RVN_URL / PGSYNC_URL is not set — refusing to fall back to a file');
  const driver = process.env.PGSYNC_NEON_DRIVER || require.resolve('@neondatabase/serverless');
  const timeout = Number(process.env.RVN_TIMEOUT_MS) || 20000;
  // The worker inherits the host's execArgv; --input-type (node -e hosts) is refused for a file worker.
  const execArgv = process.execArgv.filter((a) => !a.startsWith('--input-type'));
  const worker = new Worker(path.join(__dirname, 'neon-doc-worker.mjs'), { workerData: { url, driver }, execArgv });
  worker.unref(); // never keep a CLI process alive
  return function call(stmts) {
    const sab = new SharedArrayBuffer(4);
    const { port1, port2 } = new MessageChannel();
    worker.postMessage({ stmts, sab, port: port2 }, [port2]);
    const r = Atomics.wait(new Int32Array(sab), 0, 0, timeout);
    const msg = receiveMessageOnPort(port1);
    port1.close();
    if (r === 'timed-out' || !msg) throw new Error(`neon-doc-store: Neon did not answer within ${timeout} ms`);
    if (msg.message.error) throw new Error(`neon-doc-store: ${msg.message.error}`);
    return msg.message.rows;
  };
}

function openDocStore({ store, keyed = [], lazy = [], window = 0 }) {
  const schema = process.env.RVN_SCHEMA || 'rvdirect';
  if (!/^[a-z_][a-z0-9_]*$/.test(schema)) throw new Error(`neon-doc-store: bad schema name ${schema}`);
  const T = `${schema}.rec`;
  const KEYED = new Set(keyed);
  // `lazy` collections (ruvector's memories) are never loaded whole — at real scale the document
  // exceeds Neon's 64 MB HTTP response limit (gate R9). Their rows carry a pgvector column, and
  // recall becomes search() — a query in Neon. Only the newest `window` rows are loaded, and since
  // a process deletes only rows it loaded, the unloaded rest is never at risk.
  const LAZY = [...lazy];
  const DIM = Number(process.env.RVN_DIM) || 384;
  const WINDOW = Number(process.env.RVN_WINDOW ?? window) || 0;
  const call = makeCaller();
  let snapshot = new Map(); // `${coll}\u0000${rid}` → JSON of body as loaded/saved
  const stats = { loads: 0, saves: 0, upserts: 0, deletes: 0, ms: 0 };

  function toRows(doc) {
    const rows = [];
    for (const [key, val] of Object.entries(doc)) {
      if (val === undefined) continue;
      if (Array.isArray(val)) {
        const seen = new Map();
        for (const el of val) {
          let rid;
          // Real stores repeat ids (the live trajectories do), so a repeat gets #n: both copies are kept.
          const base = el && typeof el === 'object' && (typeof el.id === 'string' || typeof el.id === 'number')
            ? `id:${el.id}` : `h:${createHash('sha1').update(JSON.stringify(el)).digest('hex').slice(0, 20)}`;
          const n = (seen.get(base) || 0) + 1; seen.set(base, n);
          rid = n === 1 ? base : `${base}#${n}`;
          rows.push({ coll: key, rid, body: el, kind: 'a' });
        }
        if (val.length === 0) rows.push({ coll: '_doc', rid: `[]${key}`, body: [], kind: 'e' }); // keep empty arrays
      } else if (KEYED.has(key) && val && typeof val === 'object') {
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

  function load() {
    const t = Date.now();
    const SELECT = [
      { text: `SELECT coll, rid, body FROM (
                 SELECT coll, rid, body, seq FROM ${T} WHERE store = $1 AND NOT (coll = ANY($2::text[]))
                 UNION ALL
                 SELECT coll, rid, body, seq FROM (SELECT coll, rid, body, seq FROM ${T}
                   WHERE store = $1 AND coll = ANY($2::text[]) ORDER BY seq DESC LIMIT $3) w
               ) d ORDER BY seq`, params: [store, LAZY, WINDOW] },
    ];
    const [rows] = withTable(() => call(SELECT));
    stats.loads++; stats.ms += Date.now() - t;
    snapshot = new Map();
    if (!rows.length) return null; // an empty store: the caller starts from its defaults
    const doc = {};
    for (const { coll, rid, body } of rows) {
      snapshot.set(`${coll}\u0000${rid}`, JSON.stringify(body));
      if (coll === '_doc') {
        if (rid.startsWith('[]')) { doc[rid.slice(2)] ??= []; continue; }
        if (rid.startsWith('{}')) { doc[rid.slice(2)] ??= {}; continue; }
        doc[rid] = body;
      } else if (rid.startsWith('k:')) {
        (doc[coll] ??= {})[rid.slice(2)] = body;
      } else {
        (doc[coll] ??= []).push(body);
      }
    }
    return doc;
  }

  // Upsert `up` [{coll, rid, body}] and delete `del` [{coll, rid}].
  function writeRows(up, del = []) {
    if (!up.length && !del.length) return;
    up = up.map((u, i) => ({ ord: i, coll: u.coll, rid: u.rid, body: u.body }));
    // One transaction per ~8 MB chunk: Neon's HTTP endpoint caps a request/response at 64 MB, and
    // the first save of a real store is 63 MB. Rows are independent, so chunking loses nothing
    // a reader could observe as inconsistent within one record.
    const CHUNK = Number(process.env.RVN_CHUNK_BYTES) || 8 * 1024 * 1024;
    const chunks = [];
    let cur = [], size = 0;
    for (const u of up) {
      const n = JSON.stringify(u.body).length + 64;
      if (cur.length && size + n > CHUNK) { chunks.push(cur); cur = []; size = 0; }
      cur.push(u); size += n;
    }
    if (cur.length) chunks.push(cur);
    const upsert = (rows) => ({
      // the vector column of lazy rows comes from the same payload; a wrong-dimension vector stays
      // NULL (not searchable) rather than failing the save — the provenance guard owns that rule.
      text: `INSERT INTO ${T} (store, coll, rid, body, vec)
             SELECT $1, x.coll, x.rid, x.body,
                    CASE WHEN x.coll = ANY($3::text[]) AND jsonb_typeof(x.body->'embedding') = 'array'
                          AND jsonb_array_length(x.body->'embedding') = ${DIM}
                         THEN (x.body->>'embedding')::vector END
             FROM jsonb_to_recordset($2::jsonb) AS x(ord int, coll text, rid text, body jsonb) ORDER BY x.ord
             ON CONFLICT (store, coll, rid) DO UPDATE SET body = excluded.body, vec = excluded.vec,
                    ver = ${T}.ver + 1, updated_at = now()
             WHERE ${T}.body IS DISTINCT FROM excluded.body`,
      params: [store, JSON.stringify(rows), LAZY],
    });
    const delStmt = del.length && {
      text: `DELETE FROM ${T} t USING jsonb_to_recordset($2::jsonb) AS x(coll text, rid text)
             WHERE t.store = $1 AND t.coll = x.coll AND t.rid = x.rid`,
      params: [store, JSON.stringify(del)],
    };
    if (!chunks.length) withTable(() => call([delStmt]));
    chunks.forEach((c, i) => withTable(() => call(i === chunks.length - 1 && delStmt ? [upsert(c), delStmt] : [upsert(c)])));
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
      if (snapshot.get(k) !== j) up.push({ ord: ord++, coll: r.coll, rid: r.rid, body: r.body });
    }
    const del = [];
    for (const k of snapshot.keys()) if (!now.has(k)) { const [coll, rid] = k.split('\u0000'); del.push({ coll, rid }); }
    if (!up.length && !del.length) return { upserts: 0, deletes: 0 };
    writeRows(up, del);
    snapshot = now;
    stats.saves++; stats.upserts += up.length; stats.deletes += del.length; stats.ms += Date.now() - t;
    return { upserts: up.length, deletes: del.length };
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
    const [, rows] = withTable(() => call([
      { text: `SET LOCAL hnsw.ef_search = ${Math.max(40, k * 4)}` },
      { text: `SELECT body, 1 - (vec <=> $2::vector) AS score FROM ${T}
               WHERE store = $1 AND coll = $3 AND vec IS NOT NULL${where} ORDER BY vec <=> $2::vector LIMIT $4`, params },
    ]));
    stats.ms += Date.now() - t;
    return rows.map((r) => ({ ...r.body, score: Number(r.score) }));
  }

  // Row-level access for stores that are never loaded as a document (ruflo's patterns).
  const rows = {
    get: (coll, rids) => rids.length ? withTable(() => call([{ text: `SELECT rid, body FROM ${T} WHERE store = $1 AND coll = $2 AND rid = ANY($3::text[])`, params: [store, coll, rids] }]))[0] : [],
    findBy: (coll, field, value) => withTable(() => call([{ text: `SELECT rid, body FROM ${T} WHERE store = $1 AND coll = $2 AND body->>$3 = $4 ORDER BY seq LIMIT 1`, params: [store, coll, field, String(value)] }]))[0],
    count: (coll) => Number(withTable(() => call([{ text: `SELECT count(*) AS n FROM ${T} WHERE store = $1 AND coll = $2`, params: [store, coll] }]))[0][0].n),
    // every row, oldest first, in pages — a single response is capped at 64 MB
    all: (coll, eq = {}) => {
      const out = []; let after = 0;
      const [f, v] = Object.entries(eq)[0] || [];
      for (;;) {
        const page = withTable(() => call([{ text: `SELECT seq, body FROM ${T} WHERE store = $1 AND coll = $2 AND seq > $3
          ${f ? 'AND body->>$5 = $6' : ''} ORDER BY seq LIMIT $4`, params: f ? [store, coll, after, 2000, f, String(v)] : [store, coll, after, 2000] }]))[0];
        for (const r of page) out.push(r.body);
        if (page.length < 2000) return out;
        after = page[page.length - 1].seq;
      }
    },
    write: (coll, upserts, deletes = []) => writeRows(upserts.map((u) => ({ coll, ...u })), deletes.map((rid) => ({ coll, rid }))),
    // retention, only when a cap is explicitly asked for: keep the newest `keep` rows
    retain: (coll, keep) => withTable(() => call([{ text: `DELETE FROM ${T} WHERE store = $1 AND coll = $2 AND seq <= (
          SELECT seq FROM ${T} WHERE store = $1 AND coll = $2 ORDER BY seq DESC OFFSET $3 LIMIT 1)`, params: [store, coll, keep] }])),
    clear: (coll) => withTable(() => call([{ text: `DELETE FROM ${T} WHERE store = $1 AND coll = $2`, params: [store, coll] }])),
  };

  return { load, save, search, rows, stats, query: (text, params) => withTable(() => call([{ text, params }])[0]) };
}

module.exports = { openDocStore };
