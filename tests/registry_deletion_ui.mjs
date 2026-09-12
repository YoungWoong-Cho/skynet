import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {JSDOM} from 'jsdom';
const w = new JSDOM(await readFile(new URL('../static/index.html', import.meta.url), 'utf8'), {
  runScripts:'outside-only', pretendToBeVisual:true, url:'http://localhost:8080/#evaluations',
}).window;
const el = id => w.document.getElementById(id);
const settle = async () => { for (let i=0;i<12;i++) await new Promise(r=>setImmediate(r)); };
const observers=[], Observer=w.MutationObserver;
w.MutationObserver=class extends Observer {constructor(callback){super(callback);observers.push(this);}};
w.fetch = () => new Promise(()=>{});
w.scrollTo = w.HTMLElement.prototype.scrollIntoView = () => {};
w.matchMedia = () => ({matches:false, addEventListener(){},removeEventListener(){}});
w.HTMLDialogElement.prototype.showModal = function(){this.open=true;};
w.HTMLDialogElement.prototype.close = function(){this.open=false;this.dispatchEvent(new w.Event('close'));};
try {
  for (const name of ['dialogs.js','workspace-navigation.js','connection-settings.js','app.js','maintenance.js']) {
    w.eval(await readFile(new URL('../static/'+name, import.meta.url),'utf8'));
  }
  el('email-workspace-content').hidden=false;
  w.scheduleEvaluationTargetValidation=()=>{};
  let suites=[{id:'suite',name:'cube',label:'Cube simulation',evaluator:'isaac_lab',version:'2',tasks:['cube'],can_delete:true,updated_at:'2026-09-11T12:00:00Z',config_json:{tasks:['cube'],task_options:[{id:'cube',label:'Pick <cube>'}],task_selection_reason:'Exactly one case per checkpoint.'}}];
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
  console.log('Registry deletion UI: shared modal, dependency guidance, direct editing, single submission and refreshed selectors passed.');
} finally {observers.forEach(o=>o.disconnect());await settle();w.close();}
