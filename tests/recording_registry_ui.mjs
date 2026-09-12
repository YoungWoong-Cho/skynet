import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {JSDOM} from 'jsdom';
const w = new JSDOM(await readFile(new URL('../static/index.html', import.meta.url), 'utf8'), {
  runScripts: 'outside-only', pretendToBeVisual: true,
  url: 'http://localhost:8080/?data_view=recording#data',
}).window;
const observers = [], Observer = w.MutationObserver;
w.MutationObserver = class extends Observer { constructor(fn) { super(fn); observers.push(this); } };
w.fetch = () => new Promise(() => {});
w.scrollTo = w.HTMLElement.prototype.scrollIntoView = () => {};
w.matchMedia = () => ({matches: false, addEventListener() {}, removeEventListener() {}});
w.HTMLDialogElement.prototype.showModal = function() { this.open = true; };
w.HTMLDialogElement.prototype.close = function() { if (this.open) { this.open = false; this.dispatchEvent(new w.Event('close')); } };
const el = id => w.document.getElementById(id);
const flush = async () => { for (let i = 0; i < 5; i++) await new Promise(resolve => setImmediate(resolve)); };
try {
  for (const file of ['dialogs.js', 'workspace-navigation.js', 'connection-settings.js', 'app.js', 'live-conversion.js']) {
    w.eval(await readFile(new URL('../static/' + file, import.meta.url), 'utf8'));
  }
  const originalId = '6a257afb-aaaa-4000-8000-000000000000';
  const copyId = '5f95eb65-bbbb-4000-8000-000000000000';
  const original = {id: originalId, created_at: '2026-09-08T18:27:14Z', profile: {task: 'Cube', robot: 'Shadow'}, recordings: ['a', 'b']};
  const copy = {...original, id: copyId, recordings: ['a']};
  const empty = {...original, id: 'unregistered'};
  let resources = [
    {id: 'original-data', provider: 'collection', name: 'Full dataset', metadata: {session_id: originalId}, versions: [{id:'v1'}, {id:'v2'}, {id:'v3'}]},
    {id: 'second-data', provider: 'collection', name: 'Another dataset', metadata: {session_id: originalId}, versions: []},
    {id: 'archived-data', provider: 'collection', name: 'Archived dataset', metadata: {session_id: originalId}, archived_at: '2026-09-10', versions: []},
    {id: 'copy-data', provider: 'collection', name: 'One episode', metadata: {session_id: originalId, recording_session_id: copyId}, versions: []},
    {id: 'external', provider: 'huggingface', name: '<b>External</b>', metadata: {session_id: originalId}, versions: []},
  ];
  const requests = [];
  w.api = async path => {
    requests.push(path);
    if (path.startsWith('/api/data/resources?')) return {resources};
    return {items: []};
  };
  w.renderSimulationRecordings([original, copy, empty]);
  assert.equal(el('simulation-recordings-body').rows[0].cells[4].textContent, '—', 'unknown counts must not appear as zero');
  await w.loadDataRegistry(true);
  const recordingRows = () => [...el('simulation-recordings-body').rows];
  const registered = index => recordingRows()[index].cells[4].querySelector('button');
  assert.equal(registered(0).textContent, '3', 'count Registry resources, not their versions; include archived resources');
  assert.equal(registered(1).textContent, '1', 'copy ownership overrides original source membership');
  assert.equal(registered(2).textContent, '0');
  assert.equal(recordingRows()[0].cells[2].textContent, w.formatDate(original.created_at));
  assert.equal(recordingRows()[0].cells[0].textContent.includes(w.formatDate(original.created_at)), false);
  assert.deepEqual([...recordingRows()[0].querySelectorAll('.row-actions button')].map(x => x.textContent), ['View recordings', 'Register']);
  assert.equal(el('show-data-resource-form').textContent, 'New');
  const controls = [...el('show-data-resource-form').parentElement.children];
  assert.ok(controls.indexOf(el('data-resource-search').parentElement) < controls.indexOf(el('show-data-resource-form')));
  assert.ok(controls.indexOf(el('data-show-archived').parentElement) < controls.indexOf(el('show-data-resource-form')));
  registered(0).click(); await flush();
  assert.equal(new URL(w.location.href).searchParams.get('data_view'), 'registry');
  assert.equal(w.location.hash, '#data');
  assert.equal(el('data-resource-search').value, originalId);
  assert.equal(el('data-show-archived').checked, true);
  const resourceIds = () => [...el('data-resources-body').querySelectorAll('[data-resource-id]')].map(x => x.dataset.resourceId);
  assert.deepEqual(resourceIds(), ['original-data', 'second-data', 'archived-data']);
  assert.equal(el('data-resources-body').rows[0].cells[1].textContent, originalId.slice(0, 8) + '...');
  assert.equal(el('data-resources-body').rows[0].cells[1].querySelector('span').title, originalId);
  await w.loadDataRegistry(true);
  assert.deepEqual(resourceIds(), ['original-data', 'second-data', 'archived-data'], 'refresh preserves the recording filter');
  registered(1).click(); await flush();
  assert.deepEqual(resourceIds(), ['copy-data']);
  assert.equal(el('data-show-archived').checked, false);
  registered(2).click(); await flush();
  assert.match(el('data-resources-body').textContent, /No resources match/);
  assert.equal(el('data-resources-body').rows[0].cells[0].colSpan, 9);
  el('data-resource-search').value = 'HUGGINGFACE';
  el('data-resource-search').dispatchEvent(new w.Event('input'));
  assert.deepEqual(resourceIds(), ['external'], 'typing replaces the exact recording filter with case-insensitive search');
  assert.equal(el('data-resources-body').rows[0].cells[1].textContent, '', 'external resources have no recording ID');
  assert.equal(el('data-resources-body').querySelector('b'), null, 'resource names are escaped');
  el('data-resource-search').value = '';
  el('data-resource-search').dispatchEvent(new w.Event('input'));
  assert.equal(resourceIds().length, 4);
  el('data-show-archived').checked = true;
  el('data-show-archived').dispatchEvent(new w.Event('change')); await flush();
  assert.equal(resourceIds().length, 5);
  el('show-data-resource-form').click();
  assert.equal(el('data-resource-form-dialog').open, true);
  el('close-data-resource-form').click();
  assert.equal(el('show-data-resource-form').textContent, 'New', 'closing the common modal does not restore the old label');
  resources = resources.filter(x => x.id !== 'second-data');
  await w.loadDataRegistry(true);
  assert.equal(registered(0).textContent, '2', 'deletion refreshes Registered without reloading the page');
  const workingApi = w.api;
  w.api = async () => { throw new Error('Registry offline'); };
  await w.loadDataRegistry(true);
  assert.equal(recordingRows()[0].cells[4].textContent, '—', 'failed refresh must not claim an accurate count');
  w.api = workingApi;
  await w.loadDataRegistry(true);
  assert.equal(registered(0).textContent, '2');
  console.log('Recording / Registry: resource counts, source ownership, archive visibility, navigation, filtering, refresh and failure recovery passed.');
} finally { for (const observer of observers) observer.disconnect(); await flush(); w.close(); }
