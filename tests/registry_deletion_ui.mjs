import assert from 'node:assert/strict';
import {pageWindow, byId, flusher, spyMutationObservers, stubBrowserApis, polyfillDialogs, loadScripts} from './ui_harness.mjs';
const w = pageWindow({url:'http://localhost:8080/#evaluations'});
const el = byId(w);
const settle = flusher(12);
const observers = spyMutationObservers(w);
stubBrowserApis(w);
polyfillDialogs(w, {guarded:false, returnValue:false});
try {
  await loadScripts(w, ['dialogs.js', 'workspace-navigation.js', 'connection-settings.js', 'app.js', 'maintenance.js'],
    {'app.js': 'window.showCollectionAdapters=rows=>{collectionAdapterRows=rows;renderCollectionAdapters();};'});
  el('email-workspace-content').hidden=false;
  w.scheduleEvaluationTargetValidation=()=>{};
  let suites=[{id:'suite',name:'cube',label:'Cube simulation',evaluator_adapter:'isaac_lab',suite_version:'2',tasks:['cube'],can_delete:true,updated_at:'2026-09-11T12:00:00Z',config_json:{tasks:['cube'],task_options:[{id:'cube',label:'Pick <cube>'}],task_selection_reason:'Exactly one case per checkpoint.'}}];
  let adapters=[{id:'adapter',name:'Editable default',editable:true,shared:false,versions:[],latest_version_number:2,
    latest_version:{id:'version',version_number:2,manifest:{slug:'example',display_name:'Example'}}}];
  const calls=[];
  w.api=async (path, options={})=>{
    calls.push({path,...options});
    if(path==='/api/adapters/adapter') return {adapter:adapters[0]};
    if(path.startsWith('/api/adapters?')) return {adapters};
    if(path.startsWith('/api/evaluation-suites')) return {suites};
    if(path.startsWith('/api/maintenance/history/')) {
      if(options.method==='DELETE') {suites=[];return {deleted:true};}
      const isAdapter=path.includes('/adapter/');
      return {label:isAdapter?'Editable default':'Cube simulation',token:'x'.repeat(64),files:[],counts:{versions:2},
        blockers:isAdapter?[{kind:'experiment',id:'experiment',label:'Pinned consumer',reason:'Delete this experiment first'}]:[]};
    }
    return {};
  };
  await w.loadAdapters(true);
  assert.ok(el('adapters-body').querySelector('[data-adapter-action="edit"]'),'defaults have a direct Edit action');
  assert.doesNotMatch(el('adapters-body').textContent,/Shared|clone to customize/);
  await w.activateTab('experiments', true, 'adapters'); await settle();
  const adapterView=el('adapters-body').querySelector('[data-adapter-action="view"]');
  adapterView.focus(); adapterView.click(); await settle();
  assert.ok(el('adapter-editor-dialog').open, el('toast')?.textContent || el('adapter-editor-title').textContent);
  assert.ok(el('adapter-editor').closest('dialog.app-dialog[data-panel-dialog]'));
  assert.equal(el('adapter-manifest').readOnly,true);
  assert.equal(el('adapter-editor').closest('tr'),null);
  el('close-adapter-editor').click(); await settle();
  assert.equal(el('adapter-editor-dialog').open,false);
  assert.equal(w.document.activeElement,adapterView);
  const previousApi=w.api;
  let resolveAdapter;
  w.api=()=>new Promise(resolve=>{resolveAdapter=resolve;});
  adapterView.click(); await settle();
  assert.ok(el('adapter-editor-dialog').open,'loading uses the same modal');
  w.document.dispatchEvent(new w.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true}));
  resolveAdapter({adapter:adapters[0]}); await settle();
  assert.equal(el('adapter-editor-dialog').open,false,'late response cannot reopen a closed modal');
  w.api=previousApi;
  el('adapters-body').querySelector('[data-delete-kind="adapter"]').click();await settle();
  assert.equal(el('maintenance-title').textContent,'Delete adapter');
  assert.ok(el('maintenance-dialog').open);
  assert.ok(el('maintenance-confirm').disabled,'dependent adapter cannot be deleted');
  assert.match(el('maintenance-content').textContent,/Pinned consumer/);
  assert.equal(el('maintenance-content').querySelector('[data-delete-kind]').dataset.deleteKind,'experiment');
  el('maintenance-dialog').querySelector('[data-dialog-close]').click();

  // Both adapter registries archive and restore through one flow, each with its own wording.
  const lastToast=()=>[...el('toast').querySelectorAll('.notification-message')].at(-1)?.textContent;
  const confirmArchive=async message=>{
    await settle();
    const dialog=w.document.querySelector('dialog[data-app-confirmation][open]');
    assert.equal(dialog?.querySelector('.dialog-message').textContent,message);
    dialog.returnValue='confirm'; // this page's dialog polyfill records no close value
    dialog.querySelector('form').dispatchEvent(new w.Event('submit',{cancelable:true}));
    await settle();
  };
  const writes=()=>calls.filter(c=>c.method&&/^\/api\/(collection\/)?adapters\//.test(c.path)).map(c=>[c.method,c.path]);
  calls.length=0;
  el('adapters-body').querySelector('[data-adapter-action="archive"]').click();
  await confirmArchive('Archive adapter adapter? Existing experiment snapshots will not change.');
  assert.deepEqual(writes(),[['DELETE','/api/adapters/adapter']]);
  assert.equal(lastToast(),'Adapter archived.');
  const activeAdapters=adapters;
  adapters=[{...activeAdapters[0],archived_at:'2026-09-11T12:00:00Z'}];
  el('adapter-show-archived').checked=true;
  await w.loadAdapters(true);
  calls.length=0;
  el('adapters-body').querySelector('[data-adapter-action="restore"]').click();
  await confirmArchive('Restore adapter adapter? Existing experiment snapshots will not change.');
  assert.deepEqual(writes(),[['POST','/api/adapters/adapter/restore']]);
  assert.equal(lastToast(),'Adapter restored.');
  adapters=activeAdapters;
  el('adapter-show-archived').checked=false;
  await w.loadAdapters(true);
  let collectionReloads=0;
  w.loadCollection=async()=>{collectionReloads++;};
  const showCollectionAdapter=archived_at=>w.showCollectionAdapters([{id:'recorder',archived_at,manifest:{key:'recorder',version:'1',runnable:true,streams:[]}}]);
  showCollectionAdapter(null);
  calls.length=0;
  el('collection-adapters-body').querySelector('[data-collection-adapter-action="archive"]').click();
  await confirmArchive('Archive this collection adapter? Existing sessions will remain readable.');
  assert.deepEqual(writes(),[['DELETE','/api/collection/adapters/recorder']]);
  assert.equal(lastToast(),'Collection adapter archived.');
  assert.equal(collectionReloads,1);
  showCollectionAdapter('2026-09-11T12:00:00Z');
  calls.length=0;
  const collectionApi=w.api;
  w.api=async(path,options={})=>{calls.push({path,...options});throw new Error('Gone');};
  el('collection-adapters-body').querySelector('[data-collection-adapter-action="restore"]').click();
  await settle();
  assert.equal(w.document.querySelector('dialog[data-app-confirmation][open]'),null,'restoring a collection adapter asks nothing');
  assert.deepEqual(writes(),[['POST','/api/collection/adapters/recorder/restore']]);
  assert.equal(lastToast(),'Adapter restore failed: Gone');
  w.api=collectionApi;

  await w.loadEvaluationSuites(true,'');
  await w.loadEvaluationCatalog(true);
  await w.activateTab('evaluations',true,'suites');
  const suiteView=el('evaluation-suites-body').querySelector('[data-suite-view]');
  assert.equal(el('evaluation-suites-body').querySelector('tr').cells.length,6);
  assert.equal(el('evaluation-suites-body').querySelector('details'),null);
  assert.doesNotMatch(el('evaluation-suites-body').textContent,/Exactly one case/);
  assert.match(el('evaluation-suites-body').textContent,/Sep 11/);
  suiteView.focus(); suiteView.click(); await settle();
  assert.ok(el('evaluation-suite-detail-dialog').open);
  assert.ok(el('evaluation-suite-detail').closest('dialog.app-dialog[data-panel-dialog]'));
  assert.equal(el('evaluation-suite-description').textContent,'Exactly one case per checkpoint.');
  assert.equal(el('evaluation-suite-tasks').querySelector('tr').cells[0].textContent,'Pick <cube>');
  assert.equal(el('evaluation-suite-tasks').querySelector('cube'),null,'task text is escaped');
  assert.equal(el('evaluation-suite-tasks').querySelector('tr').cells[1].textContent,'cube');
  el('evaluation-suite-detail-dialog').querySelector('[data-dialog-close]').click(); await settle();
  assert.equal(w.document.activeElement,suiteView);
  el('evaluation-suite-search').value='Exactly one case';
  el('evaluation-suite-search').dispatchEvent(new w.Event('input'));
  assert.equal(el('evaluation-suite-count').textContent,'1 of 1 suites','hidden descriptions remain searchable');
  el('evaluation-suite-search').value='';
  el('evaluation-suite-search').dispatchEvent(new w.Event('input'));
  el('evaluation-suites-body').querySelector('[data-delete-kind="suite"]').click();await settle();
  assert.equal(el('maintenance-title').textContent,'Delete evaluation suite');
  assert.equal(el('maintenance-confirm').disabled,false);
  el('maintenance-confirm').click();el('maintenance-confirm').click();await settle();
  assert.equal(calls.filter(c=>c.method==='DELETE').length,1);
  assert.equal(el('maintenance-dialog').open,false);
  assert.equal(el('evaluation-suite-count').textContent,'0 suites');
  assert.doesNotMatch(el('evaluation-suite').innerHTML,/value="suite"/,'submit options cannot retain a deleted suite');
  assert.doesNotMatch(el('experiment-evaluation-suites').innerHTML,/value="suite"/,'experiment defaults cannot retain a deleted suite');
  assert.match(el('evaluation-suites-body').textContent,/No evaluation suites/);
  console.log('Registry deletion UI: shared modal, dependency guidance, direct editing, archive and restore, single submission and refreshed selectors passed.');
} finally {observers.forEach(o=>o.disconnect());await settle();w.close();}
