// Navigation uses the real application activation code, without network or jobs.
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const assert = require('node:assert/strict');
const { test } = require('node:test');
const { JSDOM } = require('jsdom');
function page(view = 'live') {
  const dom = new JSDOM(readFileSync(join(__dirname, '../static/index.html'), 'utf8'), {
    url: `http://localhost:8080/?collection_view=${view}#collection`, runScripts: 'outside-only', pretendToBeVisual: true,
  });
  const w = dom.window, el = id => w.document.getElementById(id);
  let loads = [];
  w.loadActiveTab = tab => { loads.push(tab); return Promise.resolve(); };
  w.scrollTo = w.HTMLElement.prototype.scrollIntoView = () => {};
  w.stopClusterAutoRefresh = w.closeActiveDisclosure = w.resetRunAttemptContext = () => {};
  w.eval('var activeTab = "cluster"; var activeRunAttemptDisclosure = null; var elements = {toast: document.getElementById("toast")};');
  w.eval(readFileSync(join(__dirname, '../static/workspace-navigation.js'), 'utf8'));
  const source = readFileSync(join(__dirname, '../static/app.js'), 'utf8');
  const start = source.indexOf('function activateTab('), end = source.indexOf('function restoreSshGatewayPreference(', start);
  w.eval(source.slice(start,end));
  w.activateTab('collection', false);
  return { w, el, loads };
}
test('one row opens setup, collect, recording and registry with keyboard focus', () => {
  const {w, el, loads} = page();
  assert.equal(w.document.querySelectorAll('[data-data-tab]').length, 4);
  assert.equal(w.document.querySelectorAll('[aria-label="Data sections"]').length, 0);
  assert.equal(el('data').hidden, false);
  assert.equal(el('collection-view-live').hidden, false);
  el('data-tab-recording').click();
  assert.equal(el('collection-view-recordings').hidden, false);
  assert.equal(el('collection-view-live').hidden, true);
  el('data-tab-recording').dispatchEvent(new w.KeyboardEvent('keydown', {key: 'ArrowRight'}));
  assert.equal(el('datasets').hidden, false);
  assert.equal(el('collection').hidden, true);
  assert.equal(w.document.activeElement.id, 'data-tab-registry');
  assert.equal(loads.at(-1), 'datasets');
  el('data-tab-registry').dispatchEvent(new w.KeyboardEvent('keydown', {key: 'Home'}));
  assert.equal(el('collection-view-setup').hidden, false);
  assert.equal(w.document.activeElement.id, 'data-tab-setup');
  assert.equal(el('refresh-data-registry').hidden, true);
  w.close();
});
test('help and legacy links resolve to a single canonical Data URL', () => {
  const {w, el} = page('cycles');
  assert.equal(el('collection-view-setup').hidden, false);
  assert.equal(w.location.hash, '#data');
  assert.equal(new URLSearchParams(w.location.search).get('data_view'), 'setup');
  assert.ok(!w.location.search.includes('collection_view'));
  w.activateTab('datasets');
  assert.equal(el('datasets').hidden, false);
  w.document.querySelector('[data-collection-guide="record"]').click();
  assert.equal(el('collection-view-setup').hidden, false);
  assert.equal(w.document.activeElement.id, 'vision-pro-guide-title');
  w.history.pushState(null, '', '?collection_view=recordings#collection');
  w.dispatchEvent(new w.PopStateEvent('popstate'));
  assert.equal(el('collection-view-recordings').hidden, false);
  w.close();
});
test('Back/Forward restores Data view and never exposes both content panels', async () => {
  const {w, el} = page();
  el('data-tab-recording').click();
  el('data-tab-registry').click();
  const back = new Promise(resolve => w.addEventListener('popstate', resolve, {once:true}));
  w.history.back(); await back;
  assert.equal(el('collection-view-recordings').hidden, false);
  assert.equal(el('datasets').hidden, true);
  const forward = new Promise(resolve => w.addEventListener('popstate', resolve, {once:true}));
  w.history.forward(); await forward;
  assert.equal(el('datasets').hidden, false);
  assert.equal(el('collection').hidden, true);
  w.activateTab('experiments');
  assert.equal(el('data').hidden, true);
  assert.equal(el('experiments').hidden, false);
  w.close();
});
