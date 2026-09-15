import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {JSDOM} from 'jsdom';
const w = new JSDOM(await readFile(new URL('../static/index.html',import.meta.url),'utf8'), {
  runScripts:'outside-only',pretendToBeVisual:true,url:'http://localhost:8080/?data_view=registry#data'
}).window;
const observers=[], Observer=w.MutationObserver;
w.MutationObserver=class extends Observer{constructor(fn){super(fn);observers.push(this);}};
w.fetch=()=>new Promise(()=>{});
w.scrollTo=w.HTMLElement.prototype.scrollIntoView=()=>{};
w.matchMedia=()=>({matches:false,addEventListener(){},removeEventListener(){}});
w.HTMLDialogElement.prototype.showModal=function(){this.open=true;};
w.HTMLDialogElement.prototype.close=function(){this.open=false;this.dispatchEvent(new w.Event('close'));};
const el=id=>w.document.getElementById(id), flush=async()=>{for(let i=0;i<5;i++)await new Promise(r=>setImmediate(r));};
try{
  for(const file of ['dialogs.js','workspace-navigation.js','connection-settings.js','app.js'])w.eval(await readFile(new URL('../static/'+file,import.meta.url),'utf8'));
  w.eval('window.seedPresets = rows => { experimentRows=rows; renderExperiments(); };');
  const types={dataset:{dataset:'Dataset',demonstrations:'Demonstrations'},file:{model:'Model',simulation_assets:'Simulation assets'}};
  const resources=[
    {id:'cube',category:'dataset',kind:'demonstrations',provider:'collection',display_name:'Cube',source_key:'Cube',recording_ids:['one','two'],versions:[
      {id:'source',format:'skynet.episodes/v1'}, {id:'z1',format:'egoverse-episodes-zarr/v1'},
      {id:'z2',format:'egoverse-episodes-zarr/v1'}, {id:'h1',format:'xpolicylab-act-hdf5/v1'}]},
    {id:'external',category:'dataset',kind:'dataset',provider:'huggingface',display_name:'External',source_key:'External',versions:[]},
    {id:'file',category:'file',kind:'model',provider:'huggingface',display_name:'Robot model',source_key:'Robot model',versions:[]},
    {id:'unclassified',kind:'demonstrations',provider:'test',display_name:'Unknown',source_key:'Unknown',versions:[]},
  ];
  let imports=[{id:'failed-import',resource_id:'external',state:'FAILED',error:'Download interrupted'}];
  w.api=async path=>path.startsWith('/api/data/resources?')?{resources,resource_types:types}:path==='/api/data/imports'?{imports}:{items:[]};
  await w.activateTab('data',true,'registry'); await w.loadDataRegistry(true);
  const row=id=>el('data-resources-body').querySelector(`[data-resource-id="${id}"]`);
  const headers=()=>[...el('data-resources-body').closest('table').querySelectorAll('thead th')].filter(x=>!x.hidden).map(x=>x.textContent.trim());
  assert.deepEqual(headers(),['Name','Recordings','Formats','Source','Updated','Actions']);
  assert.equal(row('unclassified'),null,'Missing category must not silently classify as a dataset');
  assert.equal(row('file'),null);
  assert.equal(row('cube').cells[2].textContent,'HDF5 × 1 · Zarr × 2');
  assert.equal(row('external').cells[2].textContent,'—');
  assert.equal(el('data-resource-count').hidden,true);
  let opened;w.openPreparedDataset=async id=>{opened=id;};
  row('cube').cells[2].querySelector('button').click();await flush();assert.equal(opened,'cube');
  row('external').querySelector('[data-resource-action="import-detail"]').click();await flush();
  assert.equal(el('data-import-detail-dialog').open,true);assert.match(el('data-import-detail-content').textContent,/Download interrupted/);
  w.SkynetDialog.close(el('data-import-detail-dialog'));
  w.document.dispatchEvent(new w.CustomEvent('dataset-preparation-changed',{detail:[
    {id:'failed',resource_id:'cube',state:'FAILED',error:'bad new config'},
    {id:'pending',resource_id:'cube',state:'QUEUED'},
    {id:'finished',resource_id:'cube',state:'READY'}
  ]}));
  assert.match(row('cube').cells[0].textContent,/Conversion failed/);
  assert.match(row('cube').cells[0].textContent,/Conversion: queued/);
  assert.doesNotMatch(row('cube').cells[0].textContent,/ready/i);
  assert.equal(row('cube').cells[2].textContent,'HDF5 × 1 · Zarr × 2','Failure does not replace previous successful formats');
  el('show-data-resource-form').click();
  assert.equal(el('data-resource-category').value,'dataset');
  assert.deepEqual([...el('data-resource-kind').options].map(x=>x.value),Object.keys(types.dataset));
  el('close-data-resource-form').click();
  await w.activateTab('data',true,'files');
  assert.deepEqual(headers(),['Name','Type','Formats','Source','Updated','Actions']);
  assert.equal(row('cube'),null);assert.equal(row('file').cells[1].textContent,'Model');
  el('show-data-resource-form').click();
  assert.equal(el('data-resource-category').value,'file');
  assert.deepEqual([...el('data-resource-kind').options].map(x=>x.value),Object.keys(types.file));
  el('data-resource-provider').value='huggingface';el('data-resource-namespace').value='org';el('data-resource-name').value='Robot';el('data-resource-source-key').value='robot-v1';el('data-resource-kind').value='model';
  let submitted;
  w.api=async (path,request={})=>{if(request.method==='POST'){submitted=JSON.parse(request.body);throw Error('Stop after capturing request');}return {};};
  await w.createDataResource({preventDefault(){}});
  assert.equal(submitted.category,'file');assert.equal(submitted.kind,'model');
  assert.equal(submitted.display_name,'Robot');assert.equal(submitted.source_key,'robot-v1');assert.equal('name' in submitted,false);
  // Reciprocal links carry exact IDs, while typing returns to ordinary search.
  const presets=[{id:'preset',name:'Cube preset',dataset_ids:['cube']},{id:'other',name:'Other preset',dataset_ids:['external']}];
  w.api=async path=>path.startsWith('/api/data/resources?')?{resources,resource_types:types}:path==='/api/data/imports'?{imports}:path==='/api/experiments'?{experiments:presets}:{};
  w.seedPresets(presets);
  await w.activateTab('experiments',true,'presets');
  assert.equal(el('experiments-body').closest('[data-tab-panel]').id,'presets');
  assert.ok(!el('presets').hidden);
  el('experiments-body').querySelector('[data-preset-datasets="preset"]').click();await flush();
  assert.equal(new URL(w.location.href).searchParams.get('data_view'),'registry');
  assert.ok(row('cube'));assert.equal(row('external'),null);
  assert.ok(row('cube').querySelector('[data-delete-kind="dataset"]'));
  el('data-resource-search').value='External';el('data-resource-search').dispatchEvent(new w.Event('input'));
  assert.equal(row('cube'),null);assert.ok(row('external'));
  const link=w.document.createElement('button');
  link.dataset.datasetPresets='cube';link.dataset.datasetLabel='Cube';link.dataset.presetIds='["preset"]';
  w.document.body.append(link);link.click();await flush();
  assert.equal(new URL(w.location.href).searchParams.get('experiment_view'),'presets');
  assert.equal(el('experiments-body').querySelectorAll('tr').length,1);
  assert.match(el('experiments-body').textContent,/Cube preset/);
  el('experiment-search').value='Other';el('experiment-search').dispatchEvent(new w.Event('input'));
  assert.match(el('experiments-body').textContent,/Other preset/);
  assert.doesNotMatch(el('experiments-body').textContent,/Cube preset/);
  el('data-resource-search').dataset.resourceIds='["cube"]';
  await w.openConvertedDataset({resource_id:'external'});
  assert.ok(row('external'));assert.equal(row('cube'),null,'A newly converted dataset clears the previous preset filter');
  console.log('Catalog: explicit categories, allowed types, Formats, inline failure/running work, modal navigation and category-specific creation passed.');
}finally{for(const observer of observers)observer.disconnect();await flush();w.close();}
