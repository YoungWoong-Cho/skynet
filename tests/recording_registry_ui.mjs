import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {JSDOM} from 'jsdom';
import {indexHtml} from './index_page.cjs';
const w = new JSDOM(indexHtml(), {
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
    {id: 'original-data', category: 'dataset', provider: 'collection', display_name: 'Full dataset', source_key: 'Full dataset', resource_id:'shared-source', recording_ids:[originalId], status:'READY', created_at:'2026-09-19', metadata:{num_episodes:51, adapter:{name:'UniDex'}, observations:['state','pointcloud']}},
    {id: 'second-data', category: 'dataset', provider: 'collection', display_name: 'Another dataset', source_key: 'Another dataset', resource_id:'shared-source', recording_ids:[originalId], status:'READY', created_at:'2026-09-18', metadata:{num_episodes:51, adapter:{name:'HPT'}, observations:['state','rgb']}},
    {id: 'archived-data', category: 'dataset', provider: 'collection', display_name: 'Archived dataset', source_key: 'Archived dataset', resource_id:'shared-source', recording_ids:[originalId], status:'READY', created_at:'2026-09-17', archived_at:'2026-09-10'},
    {id: 'copy-data', category: 'dataset', provider: 'collection', display_name: 'One episode', source_key: 'One episode', resource_id:'copy-source', recording_ids:[copyId], status:'READY', created_at:'2026-09-16'},
    {id: 'external', category: 'dataset', provider: 'huggingface', display_name: '<b>External</b>', source_key: '<b>External</b>', resource_id:'external-source', recording_ids:[], status:'READY', created_at:'2026-09-15'},
  ];
  const requests = [];
  w.api = async path => {
    requests.push(path);
    if (path.startsWith('/api/data/datasets?')) return {datasets:resources.filter(row=>row.category==='dataset')};
    if (path.startsWith('/api/data/resources?')) return {resources:resources.filter(row=>row.category==='file')};
    return {items: []};
  };
  // Replace the bootstrap coordinator whose initial requests intentionally never settle.
  w.stopSkynetLiveRefresh();
  w.initializeLiveRefresh();
  w.SkynetDatasetUI = {adapterLabel:row=>row.metadata?.adapter?.name || '—',inputModalities:metadata=>`<ul class=adapter-input-modalities>${(metadata?.observations||[]).map(item=>`<li>${item}</li>`).join('')}</ul>`};
  w.renderSimulationRecordings([original, copy, empty]);
  assert.equal(el('simulation-recordings-body').rows[0].cells[3].textContent, '—', 'unknown counts must not appear as zero');
  await w.loadDataRegistry(true);
  const recordingRows = () => [...el('simulation-recordings-body').rows];
  const registered = index => recordingRows()[index].cells[3].querySelector('button');
  assert.equal(registered(0).textContent, '3 datasets', 'count individual dataset versions even when all share one internal resource; include archived versions');
  assert.equal(registered(1).textContent, '1 dataset', 'exact source membership belongs to the dataset version');
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
  assert.deepEqual([...el('data-resources-body').closest('table').querySelectorAll('th')].filter(th=>!th.hidden).map(th=>th.textContent), ['Name','Recordings','Adapter','Input modalities','Episodes','Experiment presets','Created at','Actions']);
  const firstDataset = el('data-resources-body').rows[0];
  assert.equal(firstDataset.cells[2].textContent,'UniDex');
  assert.equal(firstDataset.cells[4].textContent,'51');
  assert.ok(firstDataset.cells[3].querySelector('.adapter-input-modalities'));
  assert.deepEqual([...firstDataset.querySelectorAll('.row-actions button')].map(button=>button.textContent), ['View','Use in experiment','Edit','Archive','Delete']);
  assert.equal(firstDataset.querySelector('[data-delete-kind=dataset]').dataset.deleteId,'original-data');
  let viewed;
  w.openPreparedDataset=async id=>{viewed=id;};
  firstDataset.querySelector('[data-resource-action=dataset]').click();await flush();
  assert.equal(viewed,'original-data','View receives exact version, never its shared source group');
  await w.loadDataRegistry(true);
  assert.deepEqual(resourceIds(), ['original-data', 'second-data', 'archived-data'], 'refresh preserves the recording filter');
  registered(1).click(); await flush();
  assert.deepEqual(resourceIds(), ['copy-data']);
  assert.equal(el('data-show-archived').checked, false);
  registered(2).click(); await flush();
  assert.match(el('data-resources-body').textContent, /No datasets match/);
  assert.equal(el('data-resources-body').rows[0].cells[0].colSpan, 8);
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
  // Editing and archiving target only one published version, never its source group.
  const registryApi = w.api, writes = [];
  let preparedSelection, registeredSelection;
  const readyJob = {id:'conversion',version_id:'original-data',state:'READY',training_ready:true};
  w.api = async (path, options={}) => {
    if (path === '/api/data/exports/jobs') return {exports:[readyJob]};
    if (path === '/api/data/datasets/original-data') {
      writes.push({path,...options});
      return {dataset:resources.find(row=>row.id==='original-data')};
    }
    return registryApi(path, options);
  };
  w.usePreparedDataset=async job=>{preparedSelection=job;};
  w.useRegisteredDataset=async id=>{registeredSelection=id;};
  await w.useDataset('original-data');
  assert.equal(preparedSelection.version_id,'original-data');
  await assert.rejects(w.useDataset('archived-data'),/archived/);
  await w.useDataset('external');
  assert.equal(registeredSelection,'external');
  await w.openDataResourceEditor('original-data');
  assert.equal(writes.at(-1).path,'/api/data/datasets/original-data');
  assert.equal(el('data-resource-source-details').hidden,true,'internal source identity is absent from dataset editing');
  el('data-resource-name').value='Renamed exact dataset';
  await w.updateDataResource('original-data');
  assert.equal(writes.at(-1).method,'PATCH');
  assert.equal(JSON.parse(writes.at(-1).body).display_name,'Renamed exact dataset');
  w.askUserDialog=async()=>true;
  await w.archiveDataResource('original-data');
  assert.equal(writes.at(-1).method,'PATCH');
  assert.deepEqual(JSON.parse(writes.at(-1).body),{archived:true});
  await w.restoreDataResource('original-data');
  assert.deepEqual(JSON.parse(writes.at(-1).body),{archived:false});
  assert.equal(writes.some(row=>row.path.includes('shared-source')),false);
  w.api = registryApi;
  const newVersion={id:'newly-published',category:'dataset',recording_ids:[],status:'READY',display_name:'New result'};
  resources.push(newVersion);
  const readsBefore=requests.filter(path=>path.startsWith('/api/data/datasets?')).length;
  const publication=new w.CustomEvent('dataset-preparation-changed',{detail:[{state:'READY',version_id:newVersion.id}]});
  w.document.dispatchEvent(publication);await flush();
  assert.equal(requests.filter(path=>path.startsWith('/api/data/datasets?')).length,readsBefore+1,'new READY version refreshes the open dataset table');
  w.document.dispatchEvent(publication);await flush();
  assert.equal(requests.filter(path=>path.startsWith('/api/data/datasets?')).length,readsBefore+1,'polling the same publication does not repeat list reads');
  resources=resources.filter(row=>row.id!==newVersion.id);
  w.document.dispatchEvent(new w.CustomEvent('dataset-preparation-changed',{detail:[{state:'READY',version_id:'original-data',training_ready:false}]}));
  assert.equal(el('data-resources-body').querySelector('[data-resource-id=original-data] [data-resource-action=train]'),null,'A known non-trainable conversion does not offer Use in experiment');
  await assert.rejects(w.useDataset('original-data'),/not currently available/);
  w.document.dispatchEvent(new w.CustomEvent('dataset-preparation-changed',{detail:[]}));
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
  assert.equal(el('data-resources-body').rows[0].cells.length, 8);
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
  console.log('Recording / Registry: exact dataset counts, source ownership, archive visibility, navigation, filtering, refresh and failure recovery passed.');
} finally { for (const observer of observers) observer.disconnect(); await flush(); w.close(); }
