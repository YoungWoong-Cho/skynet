// State/race regressions for the inspector. Rendering is also checked in-browser.
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {test} = require('node:test');
const source = readFileSync(join(__dirname, '../static/tensor-trace.js'), 'utf8');
const settle = () => new Promise(setImmediate);

function harness() {
  class Element {
    constructor() { this.events = {}; this.children = []; this.dataset = {}; this.attributes = {}; this.value = '0'; }
    addEventListener(name, listener) { this.events[name] = listener; }
    setAttribute(key, value) { this.attributes[key] = value; }
    removeAttribute(key) { delete this.attributes[key]; }
    replaceChildren() { this.children = []; }
    set innerHTML(value) {
      this.html = value;
      this.children = [...value.matchAll(/data-tensor-step="(\d+)"/g)].map(match => {
        const button = new Element(); button.dataset.tensorStep = match[1]; return button;
      });
    }
    get innerHTML() { return this.html; }
    querySelectorAll() { return this.children; }
    closest() { return this; }
    scrollIntoView() {}
    focus() {}
  }
  const nodes = new Map();
  const element = key => {
    if (!nodes.has(key)) nodes.set(key, new Element());
    return nodes.get(key);
  };
  const requests = [];
  const api = (url, options) => new Promise((resolve, reject) => requests.push({url, options, resolve, reject}));
  const context = vm.createContext({api, escapeHtml: s => String(s).replaceAll('<', '&lt;').replaceAll('>', '&gt;'), AbortController, setTimeout, Blob, URL, document: {getElementById: element, querySelectorAll: () => []}});
  vm.runInContext(source, context);
  const click = id => element(id).events.click({target: element(id)});
  const open = (id='cycle') => {
    const trigger = new Element(); trigger.dataset = {tensorCycle: id, tensorName: 'Fixture cycle', tensorFrames: '3'};
    element('capture-cycle-jobs').events.click({target: trigger});
  };
  return {element, requests, click, open};
}
function trace(frame=0) {
  return {frame, frame_count: 3, source_index: frame * 2, parameter_count: 24732, provenance: 'Fixture', checkpoint_sha256: 'abc', dataset_sha256: 'def', changed_action_indices: [], action_mae: 0, demonstrated_action: [0], steps: ['input', 'normalize', 'layer-0', 'layer-1', 'layer-2', 'layer-3', 'layer-4', 'denormalize', 'clamp'].map(id => ({id, title: id, shape: [1,1], values: [123000], dtype:'float32', stats:{min:123000, max:123000, mean:123000}, parameters:[], formula:'x', explanation:'fixture'}))};
}

test('Previous/Next traverse every step without new requests and stop at both ends', async () => {
  const h = harness(); h.open(); h.requests[0].resolve(trace()); await settle();
  assert.equal(h.element('tensor-previous').disabled, true);
  for (let i=0;i<12;i++) h.click('tensor-next');
  assert.equal(h.element('tensor-position').textContent, 'Step 9 of 9');
  assert.equal(h.element('tensor-next').disabled, true);
  for (let i=0;i<12;i++) h.click('tensor-previous');
  assert.equal(h.element('tensor-position').textContent, 'Step 1 of 9');
  assert.equal(h.element('tensor-previous').disabled, true);
  assert.equal(h.requests.length, 1);
  assert.match(h.element('tensor-step-detail').innerHTML, /123000/); // meaningful trailing zeroes survive formatting
});

test('Late responses cannot replace a newer selected cycle', async () => {
  const h = harness(); h.open('old'); h.open('new');
  assert.equal(h.requests[0].options.signal.aborted, true);
  h.requests[1].resolve(trace(2)); await settle();
  h.requests[0].resolve(trace(0)); await settle();
  assert.equal(h.element('tensor-frame').value, '2');
  assert.match(h.element('tensor-source').textContent, /new/);
  assert.match(h.element('tensor-status').textContent, /Frame 2 loaded/);
});

test('Close invalidates pending responses and errors leave a retry path with no old values', async () => {
  const h = harness(); h.open(); h.click('tensor-close');
  h.requests[0].resolve(trace()); await settle();
  assert.equal(h.element('tensor-inspector').hidden, true);
  h.open(); h.requests[1].reject(new Error('Runtime unavailable')); await settle();
  assert.equal(h.element('tensor-load').disabled, false);
  assert.equal(h.element('tensor-content').hidden, true);
  assert.match(h.element('tensor-status').textContent, /Runtime unavailable.*try again/);
  h.element('tensor-frame').value = '2';
  h.element('tensor-frame-form').events.submit({preventDefault(){}});
  h.requests[2].resolve(trace(2)); await settle();
  assert.equal(h.element('tensor-content').hidden, false);
  assert.equal(h.element('tensor-position').textContent, 'Step 1 of 9');
});

test('Invalid frame never triggers a trace request', () => {
  const h = harness(); h.open();
  for (const value of ['-1','3','1.5']) {
    h.element('tensor-frame').value = value;
    h.element('tensor-frame-form').events.submit({preventDefault(){}});
  }
  assert.equal(h.requests.length, 1);
  assert.match(h.element('tensor-status').textContent, /whole-number frame from 0 to 2/);
});
