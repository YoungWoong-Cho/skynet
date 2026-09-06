// Run with node --test tests/collection_browser.cjs. No network or live jobs.
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const { test } = require('node:test');

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}
function recording(id, right = 20, warnings = []) {
  return {sha256: id, provider: 'visionpro-local', summary: {
    header: {task: `Recording ${id}`, created_at: '2026-09-04'},
    frames: 40, duration_seconds: 2, tracked_hand_frames: {left: 20, right}, head_tracked_frames: warnings.length ? 0 : 40, warnings,
  }};
}
function page(api) {
  const nodes = new Map();
  const node = selector => {
    if (!nodes.has(selector)) nodes.set(selector, {
      value: '', textContent: '', innerHTML: '', hidden: selector === '#collection', disabled: false, checked: false, dataset: {}, files: [], handlers: {},
      addEventListener(event, handler) { (this.handlers[event] ||= []).push(handler); },
      async emit(event, properties = {}) { for (const handler of this.handlers[event] || []) await handler({preventDefault() {}, currentTarget: this, target: this, ...properties}); },
      querySelectorAll() { return []; }, scrollIntoView() {}, focus() {},
    });
    return nodes.get(selector);
  };
  const document = {querySelector: node, querySelectorAll: () => [], addEventListener() {}};
  const context = vm.createContext({document, window: {addEventListener() {}}, location: {hash: ''}, api,
    loadedTabs: new Set(), escapeHtml: value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;'),
    formatDate: String, queueReasonLabel: () => '', setTimeout: () => 1, clearTimeout() {},
  });
  for (const filename of ['local-capture.js', 'capture-processing.js']) vm.runInContext(readFileSync(join(__dirname, '../static', filename), 'utf8'), context);
  return {node, context};
}
const nextTurn = () => new Promise(resolve => setImmediate(resolve));

test('recordings become usable while the independent headset setup check is still pending', async () => {
  const setup = deferred();
  const {node, context} = page(path => path.includes('/setup') ? setup.promise : Promise.resolve({captures: [recording('fresh')]}));
  const loading = context.loadLocalCollection();
  await nextTurn();
  assert.match(node('#local-capture-body').innerHTML, /Recording fresh/);
  assert.equal(node('#capture-cycle-start').disabled, false);
  assert.equal(node('#local-capture-refresh').disabled, true);
  setup.reject(new Error('Xcode unavailable'));
  await loading;
  assert.match(node('#local-capture-checks').textContent, /Xcode unavailable/);
  assert.equal(node('#capture-cycle-start').disabled, false);
});

test('a stale refresh cannot replace the newer recording list', async () => {
  const older = deferred();
  let reads = 0;
  const {node, context} = page(path => path.includes('/setup') ? Promise.resolve({checks: [], storage_path: '/local'}) : ++reads === 1 ? older.promise : Promise.resolve({captures: [recording('new')]}));
  const first = context.loadLocalCollection();
  await context.loadLocalCollection();
  older.resolve({captures: [recording('old')]});
  await first;
  assert.match(node('#local-capture-body').innerHTML, /Recording new/);
  assert.doesNotMatch(node('#local-capture-body').innerHTML, /Recording old/);
});

test('a failed recording refresh keeps an explicit error and prevents stale submissions', async () => {
  let fail = false;
  const {node, context} = page(path => path.includes('/setup') ? Promise.resolve({checks: [], storage_path: '/local'}) : fail ? Promise.reject(new Error('connection lost')) : Promise.resolve({captures: [recording('saved')]}));
  await context.loadLocalCollection();
  fail = true;
  await context.loadLocalCollection();
  assert.equal(node('#local-capture-error').hidden, false);
  assert.match(node('#local-capture-error').textContent, /connection lost/);
  assert.match(node('#local-capture-body').innerHTML, /Recording saved/);
  assert.match(node('#local-capture-body').innerHTML, /data-use-capture="saved" disabled/);
  assert.equal(node('#capture-cycle-start').disabled, true);
});

test('preferred imports select the right recording and expose tracking warnings without a fallback', async () => {
  const captures = [recording('first'), recording('head-missing', 20, ['No head poses were tracked.']), recording('unsupported', 0, ['No right hand frames were tracked.'])];
  const {node, context} = page(path => Promise.resolve(path.includes('/setup') ? {checks: [], storage_path: '/local'} : {captures}));
  await context.loadLocalCollection(false, 'head-missing');
  assert.equal(node('#capture-cycle-recording').value, 'head-missing');
  assert.match(node('#capture-cycle-selection').innerHTML, /No head poses/);
  await context.loadLocalCollection(false, 'unsupported');
  assert.equal(node('#capture-cycle-recording').value, '');
  assert.equal(node('#capture-cycle-start').disabled, true);
});

test('background refresh cannot re-enable a submission or create a second request', async () => {
  const submit = deferred();
  let posts = 0;
  const captures = [recording('ready')];
  const {node, context} = page((path, options) => {
    if (options?.method === 'POST') { posts++; return submit.promise; }
    return Promise.resolve(path.endsWith('/jobs') ? {jobs: []} : path.includes('/setup') ? {checks: [], storage_path: '/local'} : {captures});
  });
  await context.loadLocalCollection();
  const pending = node('#capture-cycle-form').emit('submit');
  await context.loadLocalCollection();
  assert.equal(node('#capture-cycle-start').disabled, true);
  await node('#capture-cycle-form').emit('submit');
  assert.equal(posts, 1);
  submit.resolve({id: 'cycle', state: 'PREPARING'});
  await pending;
  assert.equal(node('#capture-cycle-start').disabled, false);
});

test('import controls reject missing, empty and oversized files before submitting', () => {
  const {node, context} = page(() => { throw new Error('No API call expected'); });
  for (const file of [null, {name: 'recording.csv', size: 100}, {name: 'empty.jsonl', size: 0}, {name: 'large.jsonl', size: 513 * 1024 * 1024}]) {
    node('#local-capture-file').files = file ? [file] : [];
    assert.equal(context.updateCaptureImportSelection(), false);
    assert.equal(node('#local-capture-upload').disabled, true);
  }
  node('#local-capture-file').files = [{name: 'recording.jsonl', size: 100}];
  assert.equal(context.updateCaptureImportSelection(), true);
  assert.equal(node('#local-capture-upload').disabled, false);
  assert.match(node('#local-capture-import-status').textContent, /Ready to import/);
});
