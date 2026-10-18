// neon-alert.cjs — remember that Neon failed, so the next prompt says so (2026-10-18).
//
// Every neon-direct call that cannot reach Neon (no answer, refused, auth, a server error) throws —
// nothing is ever written locally instead. But the learning hooks run with `2>/dev/null || true`, so a
// throw inside a hook reached nobody. record() appends a line to <git dir>/.neon-alert.json (when, which component, the error with any
// connection string removed); read() sums them up (first, last, count, per component);
// .claude/scripts/neon-sentinel.mjs reads it on the next prompt, re-probes Neon and reports either
// "not responding" or "back again". Never throws: recording a failure must not cause another.
//   record(source, err)  · isOutage(err)  · file()  · read()  · clear() → the cleared note or null
// Env: NEON_ALERT_FILE (tests), REPO_ROOT / DIGITAL_TWIN_ROOT (else the checkout around the cwd).
'use strict';
const fs = require('node:fs');
const path = require('node:path');

function gitDir(start) {
  let d = start;
  for (let i = 0; i < 40; i++) {
    const g = path.join(d, '.git');
    try {
      const st = fs.statSync(g);
      if (st.isDirectory()) return g;
      const m = fs.readFileSync(g, 'utf8').match(/^gitdir:\s*(.+)$/m); // a worktree
      if (m) return path.resolve(d, m[1].trim());
    } catch { /* not here */ }
    const up = path.dirname(d);
    if (up === d) return null;
    d = up;
  }
  return null;
}
function file() {
  if (process.env.NEON_ALERT_FILE) return process.env.NEON_ALERT_FILE;
  const g = gitDir(process.env.REPO_ROOT || process.env.DIGITAL_TWIN_ROOT || process.cwd());
  return g ? path.join(g, '.neon-alert.json') : null;
}
// connection strings and passwords never reach the note
const scrub = (s) => String(s).replace(/postgres(?:ql)?:\/\/[^\s'"`]+/gi, '<connection-string>')
  .replace(/(password|passwd|pwd)\s*[=:]\s*[^\s,;'"]+/gi, '$1=<hidden>').replace(/\s+/g, ' ').slice(0, 300);

// An outage, not an ordinary SQL error: no SQLSTATE at all (transport), or a class that means the
// server or the connection is the problem — 08 connection, 28 auth, 3D no database, 53 resources,
// 57 shut down / cancelled, 58 system, XX internal.
const OUTAGE_TEXT = /did not answer|timed? ?out|fetch failed|ECONN|ENOTFOUND|EAI_AGAIN|ETIMEDOUT|EHOSTUNREACH|socket|connect|password authentication|authentication failed|too many connections|compute|endpoint|quota|HTTP status [45]\d\d|server error|Unexpected token|not valid JSON|Failed to parse URL/i;
function isOutage(err) {
  // pgCode: the SQLSTATE a facade attached (undefined there = a transport failure); else the error's own code
  const code = err && (Object.prototype.hasOwnProperty.call(err, 'pgCode') ? err.pgCode : err.code);
  if (typeof code === 'string' && /^[0-9A-Z]{5}$/.test(code)) return /^(08|28|3D|53|57|58|XX)/.test(code);
  return OUTAGE_TEXT.test(`${err && err.message || err} ${code || ''}`) || !code;
}
// One line per failure, appended (2026-10-18): concurrent writers — a hook, the holder draining the
// queue, the sentinel — each append a short line; read-modify-write of one JSON object lost entries
// when two failed at the same moment (seen in neon-failure-test F3). read() adds the lines up.
const CAP = 1 << 20; // past 1 MB of notes (thousands of failures) the count stops growing; the alert stands
function read() {
  let text; try { text = fs.readFileSync(file(), 'utf8'); } catch { return null; }
  const a = { count: 0, sources: {} };
  for (const line of text.split('\n')) {
    if (!line) continue;
    let x; try { x = JSON.parse(line); } catch { continue; } // a torn last line is skipped
    a.first ??= x.t; a.last = x.t; a.count++; a.error = x.error;
    a.sources[x.source] = (a.sources[x.source] || 0) + 1;
  }
  return a.count ? a : null;
}
function record(source, err) {
  try {
    const f = file(); if (!f || !isOutage(err)) return;
    try { if (fs.statSync(f).size > CAP) return; } catch { /* first failure */ }
    const line = JSON.stringify({ t: new Date().toISOString(), source, error: scrub(err && err.message || err) }) + '\n';
    fs.appendFileSync(f, line, { mode: 0o600 });
  } catch { /* never throw from here */ }
}
function clear() {
  const f = file(); if (!f) return null;
  // rename first: a failure appended after this starts a new note instead of being deleted unread
  const moved = `${f}.${process.pid}.cleared`;
  try { fs.renameSync(f, moved); } catch { return null; }
  let a = null; try { const prev = process.env.NEON_ALERT_FILE; process.env.NEON_ALERT_FILE = moved; a = read(); if (prev == null) delete process.env.NEON_ALERT_FILE; else process.env.NEON_ALERT_FILE = prev; } catch { /* unreadable */ }
  try { fs.unlinkSync(moved); } catch { /* gone */ }
  return a;
}
module.exports = { record, isOutage, file, read, clear, scrub };
