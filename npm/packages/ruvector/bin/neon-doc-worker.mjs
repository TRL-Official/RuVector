// neon-doc-worker.mjs — the async half of neon-doc-store.cjs.
//
// Runs in a worker thread. The main thread posts {stmts, cache, sab, port}; this runs the statements
// against Neon (one round trip — a transaction when there is more than one), posts the rows (or the
// error) back on `port`, then wakes the main thread, which is blocked in Atomics.wait. That is what
// lets a synchronous CLI (ruvector's Intelligence.load/save) talk to Postgres.
// Through the connection holder (neon-holder.mjs, socket NEON_HOLDER_SOCK) when it runs: no driver
// import and no TLS per process, and `cache` lets the holder answer a load from memory when the store
// is unchanged. When the holder cannot be reached it talks to Neon directly, as it always did. Falling
// back mid-request is safe: the store only upserts and deletes by key, so running a statement twice
// changes nothing.
import { parentPort, workerData } from 'node:worker_threads';
import net from 'node:net';
import fs from 'node:fs';

let direct = null;
const viaDriver = async (stmts) => {
  direct ??= await import(workerData.driver).then(({ neon }) => neon(workerData.url));
  return stmts.length === 1 ? [await direct.query(stmts[0].text, stmts[0].params || [])]
    : direct.transaction(stmts.map((s) => direct.query(s.text, s.params || [])));
};

// the holder connection: one socket, newline-delimited JSON, replies in order
// newline-delimited JSON over a socket: scan each chunk ONCE for the newline (rescanning the growing
// buffer on every 64 KB chunk was quadratic — ~1.3 s for a 13 MB document)
const lineReader = (onLine) => { let parts = []; return (d) => {
  let s = typeof d === 'string' ? d : d.toString('utf8');
  for (let i; (i = s.indexOf('\n')) >= 0; s = s.slice(i + 1)) { parts.push(s.slice(0, i)); const line = parts.join(''); parts = []; onLine(line); }
  if (s) parts.push(s);
}; };
let sock = null, waiting = [], nextId = 1;
const viaHolder = (stmts, cache) => new Promise((resolve, reject) => {
  if (!sock) {
    sock = net.connect(workerData.sock); sock.unref();
    sock.setEncoding('utf8');
    // a rows reply ({"id":N,"rows":…}, the holder's key order) goes to the main thread as the raw text: a string
    // crosses threads as one copy, while parsing here AND structured-cloning the object graph cost ~1.8 s for
    // the 14 MB policy chain (2026-10-18). Errors and transport failures are parsed here as before.
    sock.on('data', lineReader((line) => { const w = waiting.shift(); if (w) w(/^\{"id":\d+,"rows":/.test(line.slice(0, 40)) ? { raw: line } : JSON.parse(line)); }));
    const fail = (e) => {
      // a socket file without a holder behind it: remove it, so the next wrapper call starts one
      if (e && e.code === 'ECONNREFUSED') { try { fs.unlinkSync(workerData.sock); } catch { /* gone */ } }
      for (const w of waiting.splice(0)) w({ transport: String(e && e.code || e) });
      sock = null; workerData.sock = null;
    };
    sock.on('error', fail); sock.on('close', () => fail('closed'));
  }
  waiting.push((m) => (m.transport ? reject(Object.assign(new Error(m.transport), { transport: true })) : resolve(m)));
  sock.write(JSON.stringify({ id: nextId++, kind: 'doc', stmts, cache }) + '\n');
});

parentPort.on('message', async ({ stmts, cache, sab, port }) => {
  let reply;
  try {
    let rows;
    if (workerData.sock) {
      try { const m = await viaHolder(stmts, cache); if (m.error) throw new Error(m.error); if (m.raw !== undefined) reply = { raw: m.raw }; else rows = m.rows; }
      catch (e) { if (!e.transport) throw e; rows = await viaDriver(stmts); }
    } else rows = await viaDriver(stmts);
    reply ??= { rows };
  } catch (e) {
    reply = { error: String(e && e.message || e), code: e && e.code };
  }
  port.postMessage(reply);
  port.close();
  const flag = new Int32Array(sab);
  Atomics.store(flag, 0, 1);
  Atomics.notify(flag, 0);
});
// Messages posted before this listener existed are queued by Node and delivered now, so the
// main thread needs no "ready" handshake.
