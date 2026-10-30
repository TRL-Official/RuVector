// Community detection on graphs with planted groups: louvainCommunities() must end and find them, and
// spectralClustering() must find them the same way every run. Each case runs in a child process with a time
// limit, so a non-terminating implementation fails here instead of hanging the suite.
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const modulePath = require.resolve('../dist/core/graph-algorithms');

// Run `body` (a function source taking (g, planted)) in a child process; returns its JSON result.
function inChild(body, timeoutMs = 10000) {
  const src = `
    const g = require(${JSON.stringify(modulePath)});
    // Planted partition: k groups of n/k nodes, dense inside, sparse across; seeded, so every run sees the same graph.
    function planted(n, k, seed = 7) {
      let s = seed;
      const rnd = () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648;
      const group = (i) => Math.floor((i * k) / n);
      const nodes = Array.from({ length: n }, (_, i) => 'n' + i);
      const pIn = Math.min(0.5, 12 / (n / k)), pOut = pIn / 40;
      const edges = [];
      for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++)
          if (rnd() < (group(i) === group(j) ? pIn : pOut)) edges.push({ from: nodes[i], to: nodes[j] });
      return { nodes, edges, group };
    }
    // true when every planted group has one label of its own
    function recovered(labelOf, p, k) {
      const labels = Array.from({ length: k }, (_, c) => new Set(p.nodes.filter((_, i) => p.group(i) === c).map(labelOf)));
      return labels.every((s) => s.size === 1) && new Set(labels.map((s) => [...s][0])).size === k;
    }
    const out = (${body})(g, planted, recovered);
    process.stdout.write(JSON.stringify(out));`;
  const r = spawnSync(process.execPath, ['-e', src], { encoding: 'utf8', timeout: timeoutMs });
  if (r.error && r.error.code === 'ETIMEDOUT') return { timedOut: true };
  if (r.status !== 0) return { error: (r.stderr || '').split('\n').slice(0, 3).join(' ') };
  return JSON.parse(r.stdout);
}

const cases = [
  ['louvain ends and finds two planted groups of 25 (50 nodes)', () => {
    const r = inChild(`(g, planted, recovered) => { const p = planted(50, 2); const c = g.louvainCommunities(g.buildGraph(p.nodes, p.edges));
      return { recovered: recovered((n) => c.get(n), p, 2), communities: new Set(c.values()).size }; }`);
    assert.ok(!r.timedOut, 'louvainCommunities did not return within 10 s');
    assert.ok(!r.error, r.error);
    assert.deepEqual(r, { recovered: true, communities: 2 });
  }],
  ['louvain ends and finds two planted groups of 200 (400 nodes)', () => {
    const r = inChild(`(g, planted, recovered) => { const p = planted(400, 2); const c = g.louvainCommunities(g.buildGraph(p.nodes, p.edges));
      return { recovered: recovered((n) => c.get(n), p, 2), communities: new Set(c.values()).size }; }`);
    assert.ok(!r.timedOut, 'louvainCommunities did not return within 10 s');
    assert.ok(!r.error, r.error);
    assert.deepEqual(r, { recovered: true, communities: 2 });
  }],
  ['louvain: an edgeless graph keeps every node alone; a triangle is one community', () => {
    const r = inChild(`(g) => {
      const e = g.louvainCommunities(g.buildGraph(['a', 'b', 'c'], []));
      const t = g.louvainCommunities(g.buildGraph(['a', 'b', 'c'], [{ from: 'a', to: 'b' }, { from: 'b', to: 'c' }, { from: 'c', to: 'a' }]));
      return { edgeless: new Set(e.values()).size, triangle: new Set(t.values()).size }; }`);
    assert.ok(!r.timedOut && !r.error, r.error || 'timed out');
    assert.deepEqual(r, { edgeless: 3, triangle: 1 });
  }],
  ['spectral finds two planted groups (50 nodes) and three (90 nodes), smallest eigenvalue ~0', () => {
    const r = inChild(`(g, planted, recovered) => {
      const p2 = planted(50, 2), s2 = g.spectralClustering(g.buildGraph(p2.nodes, p2.edges), 2);
      const p3 = planted(90, 3), s3 = g.spectralClustering(g.buildGraph(p3.nodes, p3.edges), 3);
      return { two: recovered((n) => s2.clusters.get(n), p2, 2), three: recovered((n) => s3.clusters.get(n), p3, 3),
               firstEigenvalueNearZero: Math.abs(s2.eigenvalues[0]) < 1e-6 }; }`);
    assert.ok(!r.timedOut && !r.error, r.error || 'timed out');
    assert.deepEqual(r, { two: true, three: true, firstEigenvalueNearZero: true });
  }],
  ['spectral gives the same clusters on every run', () => {
    const r = inChild(`(g, planted) => { const p = planted(60, 2);
      const a = g.spectralClustering(g.buildGraph(p.nodes, p.edges), 2), b = g.spectralClustering(g.buildGraph(p.nodes, p.edges), 2);
      return { same: p.nodes.every((n) => a.clusters.get(n) === b.clusters.get(n)) }; }`);
    assert.ok(!r.timedOut && !r.error, r.error || 'timed out');
    assert.deepEqual(r, { same: true });
  }],
];

const failures = [];
for (const [name, fn] of cases) {
  try { fn(); console.log(`  PASS  ${name}`); } catch (error) { failures.push(`${name}: ${error.message}`); console.log(`  FAIL  ${name}`); }
}
assert.deepEqual(failures, []);
