/**
 * ONNX WASM Embedder - Semantic embeddings for hooks
 *
 * Provides real transformer-based embeddings using all-MiniLM-L6-v2
 * running in pure WASM (no native dependencies).
 *
 * Uses bundled ONNX WASM files from src/core/onnx/
 *
 * Features:
 * - 384-dimensional semantic embeddings
 * - Real semantic understanding (not hash-based)
 * - Cached model loading (downloads from HuggingFace on first use)
 * - Batch embedding support
 * - Optional parallel workers for 3.8x batch speedup
 */

import * as path from 'path';
import * as fs from 'fs';
import { pathToFileURL } from 'url';
import { createRequire } from 'module';
import {
  EmbeddingProvenance,
  EmbeddingSpaceIdentity,
  EmbedTextKind,
  assertEmbeddingSpaceIdentity,
  embedderKindForModel,
  embeddingSpaceId,
  getModelPrefixSpec,
  prefixText,
  promptTemplateHash,
  requireModelPrefixSpec,
} from './embedding-provenance';
import { createHash } from 'crypto';
import * as net from 'net';

// Extend globalThis type for ESM require compatibility
declare global {
  // eslint-disable-next-line no-var
  var __ruvector_require: NodeRequire | undefined;
}

// Set up ESM-compatible require for WASM module (fixes Windows/ESM compatibility)
// The WASM bindings use module.require for Node.js crypto, this provides a fallback
if (typeof globalThis !== 'undefined' && !globalThis.__ruvector_require) {
  try {
    // In ESM context, use createRequire with __filename
    globalThis.__ruvector_require = createRequire(__filename);
  } catch {
    // Fallback: require should be available in CommonJS
    try {
      globalThis.__ruvector_require = require;
    } catch {
      // Neither available - WASM will fall back to crypto.getRandomValues
    }
  }
}

// Force native dynamic import (avoids TypeScript transpiling to require)
// eslint-disable-next-line @typescript-eslint/no-implied-eval
const dynamicImport = new Function('specifier', 'return import(specifier)') as (specifier: string) => Promise<any>;

// Types
export interface OnnxEmbedderConfig {
  modelId?: string;
  maxLength?: number;
  normalize?: boolean;
  cacheDir?: string;
  /**
   * Enable parallel workers for batch operations
   * - 'auto' (default): Enable for long-running processes, skip for CLI
   * - true: Always enable workers
   * - false: Never use workers
   */
  enableParallel?: boolean | 'auto';
  /** Number of worker threads (default: CPU cores - 1) */
  numWorkers?: number;
  /** Minimum batch size to use parallel processing (default: 4) */
  parallelThreshold?: number;
  /** Required override for a custom pinned artifact revision. */
  embeddingSpaceIdentity?: EmbeddingSpaceIdentity;
}

// Capability detection
let simdAvailable = false;
let parallelAvailable = false;

export interface EmbeddingResult {
  embedding: number[];
  dimension: number;
  timeMs: number;
}

export interface SimilarityResult {
  similarity: number;
  timeMs: number;
}

// Lazy-loaded module state
let wasmModule: any = null;
let embedder: any = null;
let parallelEmbedder: any = null;
let loadError: Error | null = null;
let loadPromise: Promise<void> | null = null;
let isInitialized = false;
let parallelEnabled = false;
let parallelThreshold = 4;

// Captured at init so the bundled worker pool can reuse the loaded model bytes
// (shared to workers via SharedArrayBuffer) instead of re-downloading per worker.
let loadedModelBytes: Uint8Array | null = null;
let loadedTokenizerJson: string | null = null;
let loadedMaxLength = 256;
let bundledPool: any = null;

// ADR-210: identity of the loaded model, for prefix policies (D4) and the
// embedding-provenance record (D0).
let loadedModelId: string | null = null;
let loadedNormalize = true;
let loadedEmbeddingSpaceIdentity: EmbeddingSpaceIdentity | null = null;

// Default model
const DEFAULT_MODEL = 'all-MiniLM-L6-v2';

// ---------------------------------------------------------------------------------------------------------------
// neon-direct (TRL-Official fork, 2026-11-20): the container's connection holder already keeps this module loaded
// and answers {kind:'embed', fn, args} on its unix socket (digital-twin neon-direct/neon-holder.mjs). A CLI process
// that loads MiniLM itself pays ~0.65 s per call; asking the holder costs one round trip. The holder runs THIS module
// with the same default options, so its vectors are byte-identical; its identity answer is the one a local load
// derives. Used only when:
//   - NEON_HOLDER_SOCK names a socket whose <sock>.fp is sha256(RVN_URL || PGSYNC_URL)[:16] (the rule of
//     digital-twin's neon-holder-fp.cjs: never a holder of another database), and <sock>.pid is not this process
//     (the holder itself loads this module: it never asks itself);
//   - the init asks for the plain default model (no modelId, maxLength, normalize, cacheDir, identity override,
//     parallel workers): anything else loads locally, as before;
//   - RUVECTOR_EMBED_HOLDER and EMBED_SHARED are not '0'.
// Whenever the holder is absent, refuses, answers an error, an unusable identity or a malformed vector, or does not
// answer within RUVECTOR_EMBED_HOLDER_TIMEOUT_MS (default 5000), the model loads locally exactly as before, for good
// in this process. Never the hash fallback. similarity() and the parallel pool always need the local model.
let holderSock: string | null = null;   // embeds go to this holder while set
let holderOff = false;                  // the holder failed once: local from now on
let holderEmbeds = 0;

function holderTimeoutMs(): number {
  const n = Number(process.env.RUVECTOR_EMBED_HOLDER_TIMEOUT_MS);
  return Number.isFinite(n) && n > 0 ? n : 5000;
}

/** The holder socket serving this process's database, or null (same fingerprint rule as neon-holder-fp.cjs). */
export function embedHolderSocket(env: NodeJS.ProcessEnv = process.env): string | null {
  if (env.RUVECTOR_EMBED_HOLDER === '0' || env.EMBED_SHARED === '0') return null;
  const sock = env.NEON_HOLDER_SOCK;
  const url = env.RVN_URL || env.PGSYNC_URL;
  if (!sock || !url) return null;
  try {
    if (!fs.existsSync(sock)) return null;
    const fp = createHash('sha256').update(String(url)).digest('hex').slice(0, 16);
    if (fs.readFileSync(`${sock}.fp`, 'utf8').trim() !== fp) return null;
  } catch {
    return null;
  }
  try {
    if (fs.readFileSync(`${sock}.pid`, 'utf8').trim() === String(process.pid)) return null;
  } catch {
    /* no pid file: not this process */
  }
  return sock;
}

function askHolder(sock: string, msg: Record<string, unknown>): Promise<any> {
  return new Promise((resolve, reject) => {
    let buf = '';
    let done = false;
    const c = net.connect(sock);
    const finish = (err: Error | null, value?: any) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      c.destroy();
      if (err) reject(err); else resolve(value);
    };
    const timer = setTimeout(() => finish(new Error('holder timeout')), holderTimeoutMs());
    c.setEncoding('utf8');
    c.on('connect', () => c.write(JSON.stringify({ id: 1, ...msg }) + '\n'));
    c.on('data', (d: string) => {
      buf += d;
      const i = buf.indexOf('\n');
      if (i < 0) return;
      let r: any;
      try { r = JSON.parse(buf.slice(0, i)); } catch (e: any) { return finish(e); }
      if (r && r.error) finish(new Error(String(r.error))); else finish(null, r);
    });
    c.on('error', (e: Error) => finish(e));
    c.on('close', () => finish(new Error('holder closed the connection')));
  });
}

function isDefaultOnnxConfig(config: OnnxEmbedderConfig): boolean {
  for (const [key, value] of Object.entries(config || {})) {
    if (value === undefined) continue;
    if (key === 'enableParallel' && (value === false || value === 'auto')) continue;
    return false;
  }
  return true;
}

/** Initialise from the holder: true when every embed of this process can go there. */
async function initFromHolder(sock: string): Promise<boolean> {
  try {
    const r = await askHolder(sock, { kind: 'embed', fn: 'identity' });
    const id = r && r.identity;
    if (!r || r.ready !== true || r.dim !== 384 || r.modelId !== DEFAULT_MODEL || !id) return false;
    assertEmbeddingSpaceIdentity(id);
    if (id.model_id !== DEFAULT_MODEL || id.output_dimension !== 384 || id.normalize !== true
      || id.provider !== 'ruvector-onnx-wasm' || id.prefix_policy !== requireModelPrefixSpec(DEFAULT_MODEL).prefixPolicy) return false;
    loadedModelId = DEFAULT_MODEL;
    loadedNormalize = true;
    loadedMaxLength = id.truncation_tokens;
    loadedEmbeddingSpaceIdentity = id;
    holderSock = sock;
    isInitialized = true;
    return true;
  } catch {
    return false;
  }
}

/** Load the model in this process (the holder failed, or the caller needs the local model). */
async function ensureLocalEmbedder(): Promise<void> {
  holderOff = true;
  if (embedder) return;
  if (holderSock) {
    holderSock = null;
    isInitialized = false;
    loadPromise = null;
  }
  await initOnnxEmbedder();
}

const isVector = (v: any): boolean =>
  Array.isArray(v) && v.length === 384 && v.every((x: any) => typeof x === 'number' && Number.isFinite(x));

/** Where this process's embeddings come from: 'holder', 'local', or null before init. */
export function getEmbedderVia(): 'holder' | 'local' | null {
  return holderSock ? 'holder' : embedder ? 'local' : null;
}

/** Embeds this process sent to the holder. */
export function getHolderEmbedCount(): number {
  return holderEmbeds;
}

export function deriveOnnxEmbeddingSpaceIdentity(
  modelId: string,
  modelBytes: Uint8Array,
  tokenizerJson: string,
  maxLength: number,
  normalize: boolean,
  dimension: number,
): EmbeddingSpaceIdentity {
  const modelSha = createHash('sha256').update(modelBytes).digest('hex');
  const tokenizerSha = createHash('sha256').update(tokenizerJson, 'utf8').digest('hex');
  const prefixSpec = requireModelPrefixSpec(modelId);
  const identity: EmbeddingSpaceIdentity = {
    schema_version: 1,
    provider: 'ruvector-onnx-wasm',
    model_id: modelId,
    model_artifact_sha256: modelSha,
    model_graph_sha256: modelSha,
    tokenizer_sha256: tokenizerSha,
    prompt_template_sha256: promptTemplateHash(prefixSpec),
    pooling_strategy: 'mean',
    normalize,
    truncation_tokens: maxLength,
    output_dimension: dimension,
    output_dtype: 'f32',
    runtime_revision: 'ruvector-onnx-wasm-v1',
    distance_metric: 'cosine',
    role_policy: prefixSpec.queryPrefix === prefixSpec.passagePrefix ? 'symmetric' : 'asymmetric',
    prefix_policy: prefixSpec.prefixPolicy,
    prefix_policy_version: 1,
  };
  assertEmbeddingSpaceIdentity(identity);
  return identity;
}

/**
 * Check if the ONNX embedder is *available* — i.e. the bundled WASM files are
 * present and the embedder can be initialized.
 *
 * NOTE: This is a capability check, NOT a readiness check. It returns `true`
 * before `initOnnxEmbedder()` has run (so callers can decide whether to init).
 * To check whether the model has actually been loaded, use `isOnnxInitialized()`
 * or `isReady()`. See https://github.com/ruvnet/RuVector/issues/523.
 */
export function isOnnxAvailable(): boolean {
  try {
    const pkgPath = path.join(__dirname, 'onnx', 'pkg', 'ruvector_onnx_embeddings_wasm.js');
    return fs.existsSync(pkgPath);
  } catch {
    return false;
  }
}

/**
 * Check whether the bundled parallel worker pool can be loaded — i.e. the
 * `onnx/bundled-parallel.mjs` file ships in the package. This reflects the
 * *bundled* pool (the only parallel implementation), NOT the unpublished
 * external `ruvector-onnx-embeddings-wasm/parallel` package, which was rejected
 * in ADR-194. See https://github.com/ruvnet/RuVector/issues/531.
 */
function detectParallelAvailable(): boolean {
  try {
    const poolPath = path.join(__dirname, 'onnx', 'bundled-parallel.mjs');
    parallelAvailable = fs.existsSync(poolPath);
    return parallelAvailable;
  } catch {
    parallelAvailable = false;
    return false;
  }
}

/**
 * Check if SIMD is available (from WASM module)
 */
function detectSimd(): boolean {
  try {
    if (wasmModule && typeof wasmModule.simd_available === 'function') {
      simdAvailable = wasmModule.simd_available();
      return simdAvailable;
    }
  } catch {}
  return false;
}

/**
 * Initialize the bundled, zero-dependency worker pool for batch throughput.
 *
 * Opt-in only (`enableParallel === true`) so the default/'auto' path does not
 * silently spawn worker threads for existing callers. Output vectors are
 * bit-identical to the single-thread path (issue #523).
 *
 * The previously-referenced external package
 * `ruvector-onnx-embeddings-wasm/parallel` was never published and was rejected
 * in ADR-194; the bundled pool (`onnx/bundled-parallel.mjs`) is the only
 * parallel implementation. See https://github.com/ruvnet/RuVector/issues/531.
 */
async function tryInitParallel(config: OnnxEmbedderConfig): Promise<boolean> {
  // Skip unless parallelism is explicitly requested (covers false and 'auto').
  if (config.enableParallel !== true) {
    parallelAvailable = false;
    return false;
  }
  if (!detectParallelAvailable()) {
    console.error('Parallel embedder not available: bundled worker pool (onnx/bundled-parallel.mjs) missing');
    return false;
  }
  try {
    if (!loadedModelBytes || !loadedTokenizerJson) {
      throw new Error('model bytes unavailable for bundled pool');
    }
    const poolUrl = pathToFileURL(path.join(__dirname, 'onnx', 'bundled-parallel.mjs')).href;
    const { ParallelEmbedder } = await dynamicImport(poolUrl);
    const pool = new ParallelEmbedder({
      modelBytes: loadedModelBytes,
      tokenizerJson: loadedTokenizerJson,
      maxLength: loadedMaxLength,
      dimension: embedder ? embedder.dimension() : 384,
      numWorkers: config.numWorkers,
    });
    await pool.init();
    parallelEmbedder = pool;
    parallelThreshold = config.parallelThreshold || 4;
    parallelEnabled = true;
    parallelAvailable = true;
    console.error(`Parallel embedder ready (bundled): ${pool.numWorkers} workers, SIMD: ${simdAvailable}`);
    return true;
  } catch (e: any) {
    parallelAvailable = false;
    console.error(`Parallel embedder not available: ${e.message}`);
    return false;
  }
}

/**
 * Initialize the ONNX embedder (downloads model if needed)
 */
export async function initOnnxEmbedder(config: OnnxEmbedderConfig = {}): Promise<boolean> {
  if (isInitialized) return true;
  if (loadError) throw loadError;
  if (loadPromise) {
    await loadPromise;
    return isInitialized;
  }

  loadPromise = (async () => {
    // neon-direct: the holder's model, for the plain default model only (see embedHolderSocket above)
    if (!holderOff && isDefaultOnnxConfig(config) && !wantsParallel(config)) {
      const sock = embedHolderSocket();
      if (sock && await initFromHolder(sock)) return;
      holderOff = true;
    }
    try {
      // Paths to bundled ONNX files
      const bgJsPath = path.join(__dirname, 'onnx', 'pkg', 'ruvector_onnx_embeddings_wasm_bg.js');
      const wasmPath = path.join(__dirname, 'onnx', 'pkg', 'ruvector_onnx_embeddings_wasm_bg.wasm');
      const loaderPath = path.join(__dirname, 'onnx', 'loader.js');

      if (!fs.existsSync(bgJsPath) || !fs.existsSync(wasmPath)) {
        throw new Error('ONNX WASM files not bundled. The onnx/ directory is missing.');
      }

      // Load the bg.js module directly (avoids the ESM `import * as wasm from "*.wasm"`
      // in the main .js shim which requires --experimental-wasm-modules on Node 18-24).
      const bgUrl = pathToFileURL(bgJsPath).href;
      const loaderUrl = pathToFileURL(loaderPath).href;
      wasmModule = await dynamicImport(bgUrl);

      // Instantiate the .wasm bytes via WebAssembly API (no --experimental-wasm-modules needed).
      const wasmBytes = fs.readFileSync(wasmPath);
      const wasmResult = await WebAssembly.instantiate(wasmBytes, { './ruvector_onnx_embeddings_wasm_bg.js': wasmModule });
      const wasmExports = wasmResult.instance.exports;
      if (typeof wasmModule.__wbg_set_wasm === 'function') {
        wasmModule.__wbg_set_wasm(wasmExports);
      }
      if (typeof (wasmExports as any).__wbindgen_start === 'function') {
        (wasmExports as any).__wbindgen_start();
      }

      const loaderModule = await dynamicImport(loaderUrl);
      const { ModelLoader } = loaderModule;

      // Create model loader with caching
      const modelLoader = new ModelLoader({
        cache: true,
        cacheDir: config.cacheDir || path.join(process.env.HOME || '/tmp', '.ruvector', 'models'),
      });

      // Load model (downloads from HuggingFace on first use)
      const modelId = config.modelId || DEFAULT_MODEL;
      console.error(`Loading ONNX model: ${modelId}...`);

      const { modelBytes, tokenizerJson, config: modelConfig } = await modelLoader.loadModel(modelId);

      // Retain for the bundled parallel worker pool (see initParallelEmbedder).
      loadedModelBytes = modelBytes;
      loadedTokenizerJson = tokenizerJson;
      loadedMaxLength = config.maxLength || modelConfig.maxLength || 256;
      loadedModelId = modelId;
      loadedNormalize = config.normalize !== false;

      // Create embedder with config
      const embedderConfig = new wasmModule.WasmEmbedderConfig()
        .setMaxLength(config.maxLength || modelConfig.maxLength || 256)
        .setNormalize(config.normalize !== false)
        .setPooling(0); // Mean pooling

      embedder = wasmModule.WasmEmbedder.withConfig(modelBytes, tokenizerJson, embedderConfig);

      const derivedIdentity = deriveOnnxEmbeddingSpaceIdentity(
        modelId,
        modelBytes,
        tokenizerJson,
        loadedMaxLength,
        config.normalize !== false,
        embedder.dimension(),
      );
      const supplied = config.embeddingSpaceIdentity;
      if (supplied) {
        assertEmbeddingSpaceIdentity(supplied);
        for (const key of [
          'model_id', 'model_artifact_sha256', 'model_graph_sha256', 'tokenizer_sha256',
          'prompt_template_sha256', 'normalize', 'truncation_tokens', 'output_dimension',
          'pooling_strategy', 'role_policy', 'prefix_policy',
        ] as const) {
          if (JSON.stringify(supplied[key]) !== JSON.stringify(derivedIdentity[key])) {
            throw new Error(`embeddingSpaceIdentity disagrees with loaded ONNX field: ${key}`);
          }
        }
        loadedEmbeddingSpaceIdentity = supplied;
      } else {
        assertEmbeddingSpaceIdentity(derivedIdentity);
        loadedEmbeddingSpaceIdentity = derivedIdentity;
      }

      // Detect SIMD capability
      detectSimd();
      console.error(`ONNX embedder ready: ${embedder.dimension()}d, SIMD: ${simdAvailable}`);

      isInitialized = true;

      // Determine if we should use parallel workers
      // - true: always enable
      // - false: never enable
      // - 'auto'/undefined: enable for long-running processes (MCP, servers), skip for CLI
      if (wantsParallel(config)) {
        await tryInitParallel(config);
      }
    } catch (e: any) {
      loadError = new Error(`Failed to initialize ONNX embedder: ${e.message}`);
      throw loadError;
    }
  })();

  await loadPromise;
  return isInitialized;
}

/**
 * Whether init starts parallel workers: true always, false never; 'auto'/undefined: for long-running processes
 * (MCP, servers) or RUVECTOR_PARALLEL=1, skipped for the CLI.
 */
function wantsParallel(config: OnnxEmbedderConfig): boolean {
  if (config.enableParallel === true) return true;
  if (config.enableParallel === false) return false;
  // Auto-detect: check if running as CLI hook or long-running process
  const isCLI = process.argv[1]?.includes('cli.js') ||
                process.argv[1]?.includes('bin/ruvector') ||
                process.env.RUVECTOR_CLI === '1';
  const isMCP = process.env.MCP_SERVER === '1' ||
                process.argv.some(a => a.includes('mcp'));
  const forceParallel = process.env.RUVECTOR_PARALLEL === '1';
  // Enable parallel for MCP/servers or if explicitly requested, skip for CLI
  return forceParallel || (isMCP && !isCLI);
}

async function embedKind(kind: EmbedTextKind, text: string): Promise<EmbeddingResult> {
  if (!isInitialized) {
    await initOnnxEmbedder();
  }
  if (holderSock) {
    try {
      const r = await askHolder(holderSock, { kind: 'embed', fn: kind === 'query' ? 'embedQuery' : 'embedPassage', args: [text] });
      const v = r && r.result;
      if (v && isVector(v.embedding)) {
        holderEmbeds++;
        return { embedding: v.embedding, dimension: v.embedding.length, timeMs: Number(v.timeMs) || 0 };
      }
    } catch {
      /* the holder went away: local from now on */
    }
    await ensureLocalEmbedder();
  }
  if (!embedder) {
    throw new Error('ONNX embedder not initialized');
  }

  // ADR-210 D4: apply the model's registered query/passage prefix. MiniLM has
  // empty prefixes, so the default model's output is byte-identical to before.
  const prepared = prefixText(loadedModelId ?? DEFAULT_MODEL, kind, text);

  const start = performance.now();
  const embedding = embedder.embedOne(prepared);
  const timeMs = performance.now() - start;

  return {
    embedding: Array.from(embedding),
    dimension: embedding.length,
    timeMs,
  };
}

/**
 * Generate embedding for text. Equivalent to `embedPassage()` (ADR-210 D4):
 * stored/passage text is the default; use `embedQuery()` for search queries.
 */
export async function embed(text: string): Promise<EmbeddingResult> {
  return embedKind('passage', text);
}

/** Embed a search query, applying the model's registered query prefix (D4). */
export async function embedQuery(text: string): Promise<EmbeddingResult> {
  return embedKind('query', text);
}

/** Embed a passage/document, applying the model's registered passage prefix (D4). */
export async function embedPassage(text: string): Promise<EmbeddingResult> {
  return embedKind('passage', text);
}

export async function embedFor(role: EmbedTextKind, text: string): Promise<EmbeddingResult> {
  if (role !== 'query' && role !== 'passage') throw new TypeError(`unknown embedding role: ${role}`);
  return embedKind(role, text);
}

/**
 * Generate embeddings for multiple texts
 * Uses parallel workers automatically for batches >= parallelThreshold
 */
export async function embedBatch(texts: string[]): Promise<EmbeddingResult[]> {
  return embedBatchFor('passage', texts);
}

export async function embedQueryBatch(texts: string[]): Promise<EmbeddingResult[]> {
  return embedBatchFor('query', texts);
}

export async function embedPassageBatch(texts: string[]): Promise<EmbeddingResult[]> {
  return embedBatchFor('passage', texts);
}

export async function embedBatchFor(kind: EmbedTextKind, texts: string[]): Promise<EmbeddingResult[]> {
  if (kind !== 'query' && kind !== 'passage') throw new TypeError(`unknown embedding role: ${kind}`);
  if (!isInitialized) {
    await initOnnxEmbedder();
  }
  if (holderSock) {
    try {
      const r = await askHolder(holderSock, { kind: 'embed', fn: kind === 'query' ? 'embedQueryBatch' : 'embedPassageBatch', args: [texts] });
      const v = r && r.result;
      if (Array.isArray(v) && v.length === texts.length && v.every((x: any) => x && isVector(x.embedding))) {
        holderEmbeds++;
        return v.map((x: any) => ({ embedding: x.embedding, dimension: x.embedding.length, timeMs: Number(x.timeMs) || 0 }));
      }
    } catch {
      /* the holder went away: local from now on */
    }
    await ensureLocalEmbedder();
  }
  if (!embedder) {
    throw new Error('ONNX embedder not initialized');
  }

  // ADR-210 D4: batch embedding is the passage path (embed() === embedPassage()).
  const prepared = texts.map(t => prefixText(loadedModelId ?? DEFAULT_MODEL, kind, t));

  const start = performance.now();

  // Use parallel workers for large batches
  if (parallelEnabled && parallelEmbedder && prepared.length >= parallelThreshold) {
    const batchResults = await parallelEmbedder.embedBatch(prepared);
    const totalTime = performance.now() - start;
    const dimension = parallelEmbedder.dimension || 384;

    return batchResults.map((emb: number[]) => ({
      embedding: Array.from(emb),
      dimension,
      timeMs: totalTime / texts.length,
    }));
  }

  // Sequential fallback
  const batchEmbeddings = embedder.embedBatch(prepared);
  const totalTime = performance.now() - start;

  const dimension = embedder.dimension();
  const results: EmbeddingResult[] = [];

  for (let i = 0; i < prepared.length; i++) {
    const embedding = batchEmbeddings.slice(i * dimension, (i + 1) * dimension);
    results.push({
      embedding: Array.from(embedding),
      dimension,
      timeMs: totalTime / texts.length,
    });
  }

  return results;
}

/**
 * Calculate cosine similarity between two texts
 */
export async function similarity(text1: string, text2: string): Promise<SimilarityResult> {
  if (!isInitialized) {
    await initOnnxEmbedder();
  }
  if (holderSock) await ensureLocalEmbedder();   // the WASM similarity needs the local model
  if (!embedder) {
    throw new Error('ONNX embedder not initialized');
  }

  const start = performance.now();
  const sim = embedder.similarity(text1, text2);
  const timeMs = performance.now() - start;

  return { similarity: sim, timeMs };
}

/**
 * Calculate cosine similarity between two embeddings
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Embeddings must have same dimension');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  return magnitude === 0 ? 0 : dotProduct / magnitude;
}

/**
 * Get embedding dimension
 */
export function getDimension(): number {
  return embedder ? embedder.dimension() : 384;
}

/**
 * Check if the embedder has been initialized (model loaded) and is ready to
 * embed. Returns `false` until `initOnnxEmbedder()` (or the first `embed()`,
 * which auto-initializes) has completed successfully.
 */
export function isReady(): boolean {
  return isInitialized;
}

/**
 * Whether the ONNX embedder has been initialized (model loaded).
 *
 * Post-init counterpart to `isOnnxAvailable()` (which only checks that the
 * bundled files exist). Named distinctly from the WASM-core `isInitialized()`
 * export to avoid a barrel name collision. Equivalent to `isReady()`; provided
 * as a self-documenting gate so callers can distinguish "bundled" (available)
 * from "loaded" (initialized). See
 * https://github.com/ruvnet/RuVector/issues/523.
 */
export function isOnnxInitialized(): boolean {
  return isInitialized;
}

/** Model id of the loaded model, or null before init (ADR-210). */
export function getActiveModelId(): string | null {
  return loadedModelId;
}

/**
 * Embedding-provenance record (ADR-210 D0) describing vectors produced by the
 * loaded ONNX embedder, or null before the model is initialized.
 */
export function getEmbedderProvenance(): EmbeddingProvenance | null {
  if (!isInitialized) return null;
  const modelId = loadedModelId ?? DEFAULT_MODEL;
  return {
    embedderKind: embedderKindForModel(modelId),
    modelId,
    dimension: getDimension(),
    normalize: loadedNormalize,
    prefixPolicy: getModelPrefixSpec(modelId).prefixPolicy,
  };
}

/** Complete immutable ADR-274 identity, or null before model initialization. */
export function getEmbeddingSpaceIdentity(): EmbeddingSpaceIdentity | null {
  return loadedEmbeddingSpaceIdentity ? { ...loadedEmbeddingSpaceIdentity } : null;
}

export function getEmbeddingSpaceId(): string | null {
  return loadedEmbeddingSpaceIdentity
    ? embeddingSpaceId(loadedEmbeddingSpaceIdentity)
    : null;
}

/**
 * Get embedder stats including SIMD and parallel capabilities
 */
export function getStats(): {
  ready: boolean;
  dimension: number;
  model: string;
  simd: boolean;
  parallel: boolean;
  parallelWorkers: number;
  parallelThreshold: number;
} {
  return {
    ready: isInitialized,
    dimension: embedder ? embedder.dimension() : 384,
    model: DEFAULT_MODEL,
    simd: simdAvailable,
    parallel: parallelEnabled,
    parallelWorkers: parallelEmbedder?.numWorkers || 0,
    parallelThreshold,
  };
}

/**
 * Shutdown parallel workers (call on exit)
 */
export async function shutdown(): Promise<void> {
  if (parallelEmbedder) {
    await parallelEmbedder.shutdown();
    parallelEmbedder = null;
    parallelEnabled = false;
  }
  await shutdownParallelEmbedder();
}

/**
 * Initialize the bundled-WASM worker pool for high-throughput batch embedding
 * (issue #523 SOTA). Self-contained — uses Node worker_threads + the bundled
 * WASM over SharedArrayBuffer model bytes, no external dependency. Vectors are
 * identical to the single-thread path (cosine-equivalent).
 *
 * @param numWorkers number of worker threads (default: min(cpus-2, 16))
 */
export async function initParallelEmbedder(numWorkers?: number): Promise<boolean> {
  if (bundledPool) return true;
  if (!isInitialized) await initOnnxEmbedder();
  if (holderSock) await ensureLocalEmbedder();   // the pool shares the local model bytes
  if (!loadedModelBytes || !loadedTokenizerJson) {
    throw new Error('Model bytes unavailable; cannot start parallel embedder.');
  }
  const poolUrl = pathToFileURL(path.join(__dirname, 'onnx', 'bundled-parallel.mjs')).href;
  const { ParallelEmbedder } = await dynamicImport(poolUrl);
  const pool = new ParallelEmbedder({
    modelBytes: loadedModelBytes,
    tokenizerJson: loadedTokenizerJson,
    maxLength: loadedMaxLength,
    dimension: getDimension(),
    numWorkers,
  });
  await pool.init();
  bundledPool = pool;
  return true;
}

/**
 * Batch-embed via the bundled worker pool, sharded across CPU cores. Lazily
 * starts the pool on first use. Returns embeddings in input order.
 */
export async function embedBatchParallel(texts: string[]): Promise<number[][]> {
  if (!bundledPool) await initParallelEmbedder();
  // ADR-210 D4: bulk ingest is the passage path; MiniLM prefixes are empty.
  const prepared = texts.map(t => prefixText(loadedModelId ?? DEFAULT_MODEL, 'passage', t));
  return bundledPool.embedBatch(prepared);
}

/** Number of active pool workers (0 if the pool isn't started). */
export function getParallelWorkerCount(): number {
  return bundledPool ? bundledPool.numWorkers : 0;
}

/** Batches at or above this size route through the worker pool (ADR-210 D3). */
export const BULK_EMBED_THRESHOLD = 32;

let bulkPoolFallbackWarned = false;

/**
 * Default bulk-embedding path (ADR-210 D3): batches of `threshold`
 * (default 32) or more texts route through the bundled parallel worker pool
 * — fp32 model bytes shared across workers via SharedArrayBuffer, vectors
 * identical to the single-thread path. Smaller batches, and any batch when
 * pool startup fails (no worker_threads, no SharedArrayBuffer), use the
 * single-threaded batch path with one stderr note.
 *
 * INT8 STATUS (honest gap, ADR-210 D3): the registered int8 variants
 * (QUANTIZED_MODELS in onnx-optimized.ts) cannot run on the bundled WASM
 * runtime today — its graph analyzer rejects quantized MiniLM exports
 * ("Failed analyse for node /Unsqueeze", verified against both
 * Xenova/all-MiniLM-L6-v2 model_quantized.onnx and the official
 * sentence-transformers model_quint8_avx2.onnx exports). Bulk ingest
 * therefore defaults to parallel-fp32; int8 ingest needs a Rust-side
 * runtime upgrade in the ruvector-onnx-embeddings-wasm crate (tracked as
 * an ADR-210 follow-up). Single-query latency keeps fp32 either way.
 */
export async function embedBulk(
  texts: string[],
  opts: { threshold?: number } = {}
): Promise<number[][]> {
  if (!texts || texts.length === 0) return [];
  const threshold = opts.threshold ?? BULK_EMBED_THRESHOLD;
  if (!isInitialized) {
    await initOnnxEmbedder();
  }
  if (texts.length >= threshold) {
    try {
      return await embedBatchParallel(texts);
    } catch (e: any) {
      if (!bulkPoolFallbackWarned) {
        bulkPoolFallbackWarned = true;
        console.error(
          `ruvector: parallel bulk-embed pool unavailable (${e?.message ?? e}); ` +
          `using single-threaded batch embedding.`
        );
      }
    }
  }
  const results = await embedBatch(texts);
  return results.map(r => r.embedding);
}

/** Shut down the bundled worker pool and release its threads. */
export async function shutdownParallelEmbedder(): Promise<void> {
  if (bundledPool) {
    await bundledPool.shutdown();
    bundledPool = null;
  }
}

// Export class wrapper for compatibility
export class OnnxEmbedder {
  private config: OnnxEmbedderConfig;

  constructor(config: OnnxEmbedderConfig = {}) {
    this.config = config;
  }

  async init(): Promise<boolean> {
    return initOnnxEmbedder(this.config);
  }

  /** Equivalent to embedPassage() — ADR-210 D4. */
  async embed(text: string): Promise<number[]> {
    const result = await embed(text);
    return result.embedding;
  }

  /** Embed a search query with the model's registered query prefix (D4). */
  async embedQuery(text: string): Promise<number[]> {
    const result = await embedQuery(text);
    return result.embedding;
  }

  /** Embed a passage/document with the model's registered passage prefix (D4). */
  async embedPassage(text: string): Promise<number[]> {
    const result = await embedPassage(text);
    return result.embedding;
  }

  async embedFor(role: EmbedTextKind, text: string): Promise<number[]> {
    const result = await embedFor(role, text);
    return result.embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const results = await embedBatch(texts);
    return results.map(r => r.embedding);
  }

  async embedBatchFor(role: EmbedTextKind, texts: string[]): Promise<number[][]> {
    const results = await embedBatchFor(role, texts);
    return results.map(r => r.embedding);
  }

  async embedQueryBatch(texts: string[]): Promise<number[][]> {
    return this.embedBatchFor('query', texts);
  }

  async embedPassageBatch(texts: string[]): Promise<number[][]> {
    return this.embedBatchFor('passage', texts);
  }

  async similarity(text1: string, text2: string): Promise<number> {
    const result = await similarity(text1, text2);
    return result.similarity;
  }

  get dimension(): number {
    return getDimension();
  }

  get ready(): boolean {
    return isReady();
  }
}

export default OnnxEmbedder;
