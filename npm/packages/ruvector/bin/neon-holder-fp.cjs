// neon-holder-fp.cjs — which database a connection holder serves, so a client never uses the wrong one.
//
// The holder (neon-holder.mjs) answers on NEON_HOLDER_SOCK with ITS OWN connection. Until 2026-10-18 a
// client used whatever holder that variable named, so a process asking for another database (RVN_URL)
// was silently served by the holder's — found when the "Neon unreachable" gate (neon-fs-test N7) saw a
// write succeed against an unreachable address. Now the holder writes a fingerprint of its URL beside
// its socket (<sock>.fp: 16 hex chars of sha256, never the URL itself) and a client uses the holder
// only when the fingerprint matches its own URL; otherwise it connects directly to the URL it was given.
'use strict';
const crypto = require('node:crypto');
const fs = require('node:fs');

const fp = (url) => crypto.createHash('sha256').update(String(url)).digest('hex').slice(0, 16);

// The holder socket to use for `url`, or null when there is none for this database.
function holderFor(url, sock = process.env.NEON_HOLDER_SOCK) {
  if (!url || !sock) return null;
  try { return fs.existsSync(sock) && fs.readFileSync(`${sock}.fp`, 'utf8').trim() === fp(url) ? sock : null; }
  catch { return null; }
}

module.exports = { fp, holderFor };
