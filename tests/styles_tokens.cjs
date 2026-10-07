// Run with node tests/styles_tokens.cjs. No packages or live jobs.
// Every class and id that styles.css selects must be produced by the page,
// by a browser script (vendor libraries included: KaTeX emits .katex-display)
// or by the server-rendered markup in skynet_app/. Tests do not keep CSS alive.
const { readFileSync, readdirSync } = require('node:fs');
const { join } = require('node:path');
const assert = require('node:assert/strict');
const { test } = require('node:test');

const ROOT = join(__dirname, '..');

// Class names assembled at runtime from a prefix and a state value.
const DYNAMIC = new Map([
  ['is-online', 'setConnection() in static/app.js sets `connection-dot is-${state}`'],
]);

function filesUnder(directory, pattern) {
  const found = [];
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      if (!['__pycache__', 'node_modules'].includes(entry.name)) found.push(...filesUnder(path, pattern));
    } else if (pattern.test(entry.name)) found.push(path);
  }
  return found;
}

function referenceWords() {
  const files = [
    join(ROOT, 'static/index.html'),
    ...filesUnder(join(ROOT, 'static'), /\.js$/),
    ...filesUnder(join(ROOT, 'skynet_app'), /\.py$/),
  ];
  const words = new Set();
  for (const file of files) for (const word of readFileSync(file, 'utf8').split(/[^A-Za-z0-9_-]+/)) words.add(word);
  return words;
}

// Map each class/id token to the selectors that use it. Preludes are the text
// before every "{" that is not an at-rule; declarations end at ";" or "}".
function selectorTokens(css) {
  const source = css.replace(/\/\*[\s\S]*?\*\//g, ' ');
  const tokens = new Map();
  let start = 0;
  for (let index = 0; index < source.length; index++) {
    const character = source[index];
    if (character === '{') {
      const prelude = source.slice(start, index).trim();
      if (!prelude.startsWith('@')) {
        const selectors = prelude.replace(/\[[^\]]*\]/g, '');
        for (const [, token] of selectors.matchAll(/[.#]([A-Za-z_][A-Za-z0-9_-]*)/g)) {
          if (!tokens.has(token)) tokens.set(token, new Set());
          tokens.get(token).add(prelude.replace(/\s+/g, ' '));
        }
      }
      start = index + 1;
    } else if (character === '}' || character === ';') {
      start = index + 1;
    }
  }
  return tokens;
}

test('every class and id in styles.css is produced by the page, a script or the server', () => {
  const tokens = selectorTokens(readFileSync(join(ROOT, 'static/styles.css'), 'utf8'));
  assert.ok(tokens.size > 250, `parsed only ${tokens.size} tokens`);
  assert.ok(tokens.has('table-panel') && tokens.has('runs-body'), 'parser finds classes and ids');
  const words = referenceWords();
  const unreferenced = [...tokens]
    .filter(([token]) => !words.has(token) && !DYNAMIC.has(token))
    .map(([token, selectors]) => `${token}: ${[...selectors].slice(0, 3).join(' | ')}`);
  assert.deepEqual(unreferenced, [], `styles.css selects tokens nothing produces:\n${unreferenced.join('\n')}`);
  for (const [token, reason] of DYNAMIC) assert.ok(tokens.has(token), `${token} left styles.css; drop it from DYNAMIC (${reason})`);
});

test('every custom property styles.css reads is declared there and read without a per-use fallback', () => {
  const css = readFileSync(join(ROOT, 'static/styles.css'), 'utf8').replace(/\/\*[\s\S]*?\*\//g, ' ');
  const declared = new Set([...css.matchAll(/(--[A-Za-z0-9-]+)\s*:/g)].map(([, name]) => name));
  const reads = [...css.matchAll(/var\(\s*(--[A-Za-z0-9-]+)\s*(,[^)]*)?\)/g)];
  assert.ok(reads.length > 50, `parsed only ${reads.length} var() reads`);
  assert.deepEqual(reads.filter(([, name]) => !declared.has(name)).map(([read]) => read), [], 'styles.css reads undeclared custom properties');
  assert.deepEqual(reads.filter(([, , fallback]) => fallback).map(([read]) => read), [], 'a token defined once needs no per-use fallback');
  assert.match(css, /:root \{[^}]*--accent:/, 'the accent colour is defined on :root');
});
