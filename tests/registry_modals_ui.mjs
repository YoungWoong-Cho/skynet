import assert from 'node:assert/strict';
import {pageWindow, byId, flusher, spyMutationObservers, stubBrowserApis, polyfillDialogs, readStatic} from './ui_harness.mjs';
const w = pageWindow({url:'http://localhost:8080/?data_view=registry#data'});
const observers = spyMutationObservers(w);
const el = byId(w);
stubBrowserApis(w);
polyfillDialogs(w);
const flush = flusher(3);
try {
  for(const file of ['dialogs.js','workspace-navigation.js','connection-settings.js','app.js']) w.eval((await readStatic(file)) + (file==='app.js' ? '\nwindow.setRegistryTestData=(resources,imports=[])=>{dataResourceRows=resources;dataImportRows=imports;renderDataResources();renderDataImports();};' : ''));
  w.activateTab('data',true,'files');
  w.setRegistryTestData([{id:'resource',category:'file',provider:'local',namespace:'test',display_name:'Example',source_key:'example-source',kind:'simulation_assets'}]);
  for(const [launch,panel] of [['show-data-resource-form','data-resource-form'],['show-data-derivation-form','data-derivation-form']]) {
    el(launch).click();
    assert.equal(el(panel+'-dialog').open,true,panel);
    assert.equal(el(panel).closest('tr'),null);
    assert.equal(el(launch).textContent.includes('Close'),false);
    el('close-'+panel).click();
    assert.equal(el(panel+'-dialog').open,false);
    assert.equal(el(panel).hidden,true);
  }
  el('data-resources-body').querySelector('[data-resource-action="version"]').click();
  assert.equal(el('data-version-form-dialog').open,true);
  assert.equal(el('data-version-resource-id').value,'resource');
  assert.equal(w.document.querySelectorAll('tr.row-disclosure-companion').length,0);
  el('close-data-version-form').click();
  el('show-data-derivation-form').click();
  w.showToast('A meaningful validation error',true);
  assert.match(el('data-derivation-form-dialog').querySelector('[role="alert"]').textContent,/meaningful/);
  el('close-data-derivation-form').click();
  let resolveEdit;
  w.api=()=>new Promise(resolve=>{resolveEdit=resolve;});
  el('data-resources-body').querySelector('[data-resource-action="edit"]').click();
  w.activateTab('experiments');
  resolveEdit({resource:{id:'resource',description:'late'}}); await flush();
  assert.equal(el('data-resource-form-dialog').open,false,'late edit response must not reopen a closed workflow');
  w.activateTab('datasets');
  w.setRegistryTestData([], [{id:'import',resource_id:'resource',state:'SUCCEEDED',gateway:'sky1',request:{subset:'demo'}}]);
  el('data-imports-body').querySelector('button').click();
  assert.equal(el('data-import-detail-dialog').open,true);
  assert.equal(el('data-imports-body').querySelectorAll('tr').length,1);
  assert.ok(el('data-import-detail-content').querySelector('[data-import-action="logs"]'));
  el('data-import-detail-dialog').querySelector('[data-dialog-close]').click();
  assert.equal(el('data-import-detail-dialog').open,false);
  console.log('Registry modals: existing form actions, row stability, closing, local errors, stale edit cancellation and import details passed.');
} catch(error) {console.error(error); process.exitCode=1;} finally {for(const observer of observers)observer.disconnect(); await flush(); w.close();}
