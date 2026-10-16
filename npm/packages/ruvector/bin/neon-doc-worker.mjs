// neon-doc-worker.mjs — the async half of neon-doc-store.cjs.
//
// Runs in a worker thread. The main thread posts {stmts, sab, port}; this runs the statements
// against Neon (one HTTP round trip — a transaction when there is more than one), posts the rows
// (or the error) back on `port`, then wakes the main thread, which is blocked in Atomics.wait.
// That is what lets a synchronous CLI (ruvector's Intelligence.load/save) talk to Postgres.
import { parentPort, workerData } from 'node:worker_threads';

const { neon } = await import(workerData.driver);
const sql = neon(workerData.url);

parentPort.on('message', async ({ stmts, sab, port }) => {
  let reply;
  try {
    const rows = stmts.length === 1
      ? [await sql.query(stmts[0].text, stmts[0].params || [])]
      : await sql.transaction(stmts.map((s) => sql.query(s.text, s.params || [])));
    reply = { rows };
  } catch (e) {
    reply = { error: String(e && e.message || e) };
  }
  port.postMessage(reply);
  port.close();
  const flag = new Int32Array(sab);
  Atomics.store(flag, 0, 1);
  Atomics.notify(flag, 0);
});
// Messages posted before this listener existed are queued by Node and delivered now, so the
// main thread needs no "ready" handshake.
