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
  for (const file of ['dialogs.js', 'workspace-navigation.js', 'connection-settings.js', 'app.js', 'live-conversion.js', 'maintenance.js']) {
    w.eval(await readFile(new URL('../static/' + file, import.meta.url), 'utf8'));
  }
  const originalId = '6a257afb-aaaa-4000-8000-000000000000';
  const copyId = '5f95eb65-bbbb-4000-8000-000000000000';
  const original = {id: originalId, created_at: '2026-09-08T18:27:14Z', profile: {task: 'Cube', robot: 'Shadow'}, recordings: ['a', 'b']};
  const copy = {...original, id: copyId, recordings: ['a']};
  const empty = {...original, id: 'unregistered'};
  let resources = [
    {id: 'original-data', category: 'dataset', provider: 'collection', display_name: 'Full dataset', source_key: 'Full dataset', metadata: {session_id: originalId}, versions: [{id:'v1'}, {id:'v2'}, {id:'v3'}]},
    {id: 'second-data', category: 'dataset', provider: 'collection', display_name: 'Another dataset', source_key: 'Another dataset', metadata: {session_id: originalId}, versions: []},
    {id: 'archived-data', category: 'dataset', provider: 'collection', display_name: 'Archived dataset', source_key: 'Archived dataset', metadata: {session_id: originalId}, archived_at: '2026-09-10', versions: []},
    {id: 'copy-data', category: 'dataset', provider: 'collection', display_name: 'One episode', source_key: 'One episode', metadata: {session_id: originalId, recording_session_id: copyId}, versions: []},
    {id: 'external', category: 'dataset', provider: 'huggingface', display_name: '<b>External</b>', source_key: '<b>External</b>', metadata: {session_id: originalId}, versions: []},
  ];
  const requests = [];
  w.api = async path => {
    requests.push(path);
    if (path.startsWith('/api/data/resources?')) return {resources};
    return {items: []};
  };
  w.renderSimulationRecordings([original, copy, empty]);
  assert.equal(el('simulation-recordings-body').rows[0].cells[3].textContent, '—', 'unknown counts must not appear as zero');
  await w.loadDataRegistry(true);
  const recordingRows = () => [...el('simulation-recordings-body').rows];
  const registered = index => recordingRows()[index].cells[3].querySelector('button');
  assert.equal(registered(0).textContent, '3 datasets', 'count Registry resources, not their versions; include archived resources');
  assert.equal(registered(1).textContent, '1 dataset', 'copy ownership overrides original source membership');
  assert.equal(registered(2).textContent, '0 datasets');
  assert.equal(recordingRows()[0].cells[2].textContent, w.formatDate(original.created_at));
  assert.equal(recordingRows()[0].cells[0].textContent.includes(w.formatDate(original.created_at)), false);
  assert.deepEqual([...recordingRows()[0].querySelectorAll('.row-actions button')].map(x => x.textContent), ['View', 'Convert', 'Delete']);
  assert.equal(el('show-data-resource-form').textContent.trim(), 'New');
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
  assert.equal(el('data-resources-body').rows[0].cells[1].textContent, '1 recording');
  assert.equal(el('data-resources-body').rows[0].cells[1].querySelector('button').dataset.resourceAction, 'recordings');
  await w.loadDataRegistry(true);
  assert.deepEqual(resourceIds(), ['original-data', 'second-data', 'archived-data'], 'refresh preserves the recording filter');
  registered(1).click(); await flush();
  assert.deepEqual(resourceIds(), ['copy-data']);
  assert.equal(el('data-show-archived').checked, false);
  registered(2).click(); await flush();
  assert.match(el('data-resources-body').textContent, /No datasets or file sets match/);
  assert.equal(el('data-resources-body').rows[0].cells[0].colSpan, 6);
  el('data-resource-search').value = 'HUGGINGFACE';
  el('data-resource-search').dispatchEvent(new w.Event('input'));
  assert.deepEqual(resourceIds(), ['external'], 'typing replaces the exact recording filter with case-insensitive search');
  assert.equal(el('data-resources-body').rows[0].cells[1].textContent, '0 recordings', 'external resources have no recording ID');
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
  assert.equal(el('show-data-resource-form').textContent.trim(), 'New', 'closing the common modal does not restore the old label');
  resources = resources.filter(x => x.id !== 'second-data');
  await w.loadDataRegistry(true);
  assert.equal(registered(0).textContent, '2 datasets', 'deletion refreshes Registered without reloading the page');
  const workingApi = w.api;
  w.api = async () => { throw new Error('Registry offline'); };
  await w.loadDataRegistry(true);
  assert.equal(recordingRows()[0].cells[3].textContent, '—', 'failed refresh must not claim an accurate count');
  w.api = workingApi;
  await w.loadDataRegistry(true);
  assert.equal(registered(0).textContent, '2 datasets');
  // Files never expose recording ownership, even for a stale response.
  resources.push({id:'assets', display_name:'Scene', source_key:'Scene', kind:'simulation_assets', category:'file',provider:'collection', metadata:{session_id:originalId, recording_session_id:originalId}, versions:[]});
  await w.loadDataRegistry(true);
  assert.equal(registered(0).textContent, '2 datasets', 'file sets do not count as registered datasets');
  await w.activateTab('data', true, 'files'); await flush();
  assert.equal(el('data-resource-recording-column').hidden, true);
  assert.deepEqual(resourceIds(), ['assets']);
  assert.equal(el('data-resources-body').rows[0].cells.length, 6);
  assert.equal(el('data-resources-body').rows[0].cells[1].textContent, 'simulation_assets');
  assert.equal(el('data-resources-body').querySelector('[data-entity-kind=recording]'), null);
  assert.equal(el('data-resource-search').placeholder, 'Filter name or source');
  el('data-resource-search').value = 'missing';
  el('data-resource-search').dispatchEvent(new w.Event('input'));
  assert.equal(el('data-resources-body').rows[0].cells[0].colSpan, 6);
  await w.activateTab('data', true, 'registry'); await flush();
  assert.equal(el('data-resource-recording-column').hidden, false);
  assert.equal(el('data-resources-body').rows[0].cells.length, 6);
  assert.equal(el('data-resource-count').hidden, true);
  resources.push({id:'multi',category:'dataset',provider:'collection',display_name:'Combined recordings', source_key:'Combined recordings',kind:'demonstrations',recording_ids:[originalId,copyId],metadata:{session_id:originalId},versions:[]});
  await w.loadDataRegistry(true);
  const multi=el('data-resources-body').querySelector('[data-resource-id=multi]');
  assert.equal(multi.cells[1].textContent,'2 recordings');
  multi.cells[1].querySelector('button').click();await flush();
  assert.equal(new URL(w.location.href).searchParams.get('data_view'),'recording');
  assert.deepEqual(recordingRows().map(r=>r.dataset.sessionId),[originalId,copyId]);
  w.renderSimulationRecordings([original,copy,empty]);
  assert.deepEqual(recordingRows().map(r=>r.dataset.sessionId),[originalId,copyId]);
  recordingRows()[1].cells[3].querySelector('button').click();await flush();
  assert.deepEqual(resourceIds(),['copy-data','multi'],'both directions use full source membership');
  el('new-recording').click();await flush();
  assert.equal(new URL(w.location.href).searchParams.get('data_view'),'collect');
  // Recording rows use the existing deletion dialog and confirmation contract.
  await w.activateTab('data', true, 'recording');
  const removal = recordingRows()[0].querySelector('[data-delete-kind=recording]');
  assert.equal(removal.dataset.deleteId, originalId);
  const endpoint = '/api/maintenance/history/recording/' + originalId;
  let blocked = true, completeDelete;
  const deletionCalls = [], refreshes = [];
  w.api = async (path, options = {}) => {
    if (!path.startsWith(endpoint)) return workingApi(path, options);
    deletionCalls.push({path, ...options});
    if (options.method === 'DELETE') return new Promise(resolve => {completeDelete = resolve;});
    return {label:'Cube', token:'a'.repeat(64), counts:{live_xr_sessions:1},
      files:[{path:'/cluster/recordings/' + originalId, size_bytes:12}],
      blockers:blocked ? [{kind:'dataset', id:'original-data', label:'Full dataset', reason:'Delete this dataset first'}] : []};
  };
  removal.click(); await flush();
  assert.equal(el('maintenance-dialog').open, true);
  assert.equal(el('maintenance-title').textContent, 'Delete recording');
  assert.equal(el('maintenance-confirm').disabled, true);
  assert.equal(el('maintenance-content').querySelector('[data-entity-kind=dataset]').dataset.entityId, 'original-data');
  assert.equal(el('maintenance-content').querySelector('[data-delete-kind=dataset]').dataset.deleteId, 'original-data');
  el('maintenance-dialog').querySelector('[data-dialog-close]').click();
  assert.equal(deletionCalls.filter(x => x.method === 'DELETE').length, 0, 'closing a preview never deletes');
  blocked = false;
  w.refreshRecordingsAfterDeletion = async () => {refreshes.push('recordings'); w.renderSimulationRecordings([copy]);};
  w.refreshPreparedDatasets = async () => {refreshes.push('datasets');};
  removal.click(); await flush();
  assert.equal(el('maintenance-confirm').disabled, false);
  assert.match(el('maintenance-content').textContent, /\/cluster\/recordings\//);
  el('maintenance-confirm').click(); el('maintenance-confirm').click();
  assert.equal(deletionCalls.filter(x => x.method === 'DELETE').length, 1);
  assert.equal(JSON.parse(deletionCalls.at(-1).body).token, 'a'.repeat(64));
  completeDelete({deleted:true}); await flush();
  assert.equal(el('maintenance-dialog').open, false);
  assert.deepEqual(refreshes.sort(), ['datasets','recordings']);
  assert.deepEqual(recordingRows().map(r => r.dataset.sessionId), [copyId]);
  console.log('Recording / Registry: resource counts, source ownership, archive visibility, navigation, filtering, refresh and failure recovery passed.');
} finally { for (const observer of observers) observer.disconnect(); await flush(); w.close(); }
