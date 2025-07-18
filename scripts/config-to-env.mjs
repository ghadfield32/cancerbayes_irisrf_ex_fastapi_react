#!/usr/bin/env node
/**
 * Generate web/.env from root config.yaml + selected environment.
 *
 * Usage:
 *   APP_ENV=staging node scripts/config-to-env.mjs
 *   node scripts/config-to-env.mjs prod
 *
 * Precedence:
 *   1. CLI arg (first non-flag) or APP_ENV env var (else 'dev')
 *   2. Merge: default + <env>
 *   3. For each key starting with VITE_ output to web/.env
 *
 * Guarantees VITE_API_URL ends with /api/v1 (without trailing slash duplication).
 *
 * Rationale:
 *   - Centralizes config (12-Factor: config not baked into code). 
 *   - Lets Vite statically inject values at build time (import.meta.env). 
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import * as yaml from 'yaml'; // using 'yaml' pkg

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);
const root       = path.resolve(__dirname, '..');
const webDir     = path.join(root, 'web');

function log(...args) {
  console.log('[config-to-env]', ...args);
}

function fail(msg) {
  console.error('❌ [config-to-env]', msg);
  process.exit(1);
}

const cliEnvArg = process.argv.slice(2).find(a => !a.startsWith('-'));
const targetEnv = process.env.APP_ENV || cliEnvArg || 'dev';

const cfgPath = path.join(root, 'config.yaml');
if (!fs.existsSync(cfgPath)) fail(`config.yaml not found at ${cfgPath}`);

const rawYaml = fs.readFileSync(cfgPath, 'utf8');
let doc;
try {
  doc = yaml.parse(rawYaml);
} catch (e) {
  fail(`YAML parse error: ${e.message}`);
}

if (!doc.default) fail('Missing "default" section in config.yaml');
if (!doc[targetEnv]) log(`⚠️  No explicit section "${targetEnv}" – using default only`);

const merged = { ...doc.default, ...(doc[targetEnv] || {}) };

// Extract VITE_* keys
const viteEntries = Object.entries(merged)
  .filter(([k]) => k.startsWith('VITE_'));

// Normalize API URL if present
function normalizeApi(url) {
  if (!url) return url;
  let base = url.trim();
  if (!/^https?:\/\//.test(base) && !base.startsWith('http://localhost') && !base.startsWith('http://127.0.0.1'))
    log(`⚠️  VITE_API_URL does not look absolute: "${base}"`);
  base = base.replace(/\/+$/, '');
  if (!/\/api\/v1$/.test(base)) base = base + '/api/v1';
  return base;
}

let apiBefore = null;
let apiAfter  = null;

const lines = viteEntries.map(([k, v]) => {
  if (k === 'VITE_API_URL') {
    apiBefore = v;
    const norm = normalizeApi(v);
    apiAfter = norm;
    return `${k}=${norm}`;
  }
  return `${k}=${v}`;
});

// Ensure VITE_API_URL exists (fail-fast for staging/prod)
if (!merged.VITE_API_URL && ['staging','prod','production'].includes(targetEnv)) {
  fail(`VITE_API_URL missing for environment "${targetEnv}"`);
}

// Write .env
const outEnvPath = path.join(webDir, '.env');
const envContent = lines.join('\n') + '\n';
log(`Writing to: ${outEnvPath}`);
log(`Content: ${envContent}`);
fs.writeFileSync(outEnvPath, envContent, 'utf8');

// Copy config.yaml into web (for inspection – not strictly required)
const webCfgPath = path.join(webDir, 'config.yaml');
fs.copyFileSync(cfgPath, webCfgPath);

log(`Environment: ${targetEnv}`);
if (apiBefore) {
  log(`VITE_API_URL (raw)  : ${apiBefore}`);
  log(`VITE_API_URL (final): ${apiAfter}`);
} else {
  log('⚠️  No VITE_API_URL key found in merged config (frontend may rely on fallback)');
}
log(`Wrote ${outEnvPath} with ${viteEntries.length} VITE_ keys.`); 
