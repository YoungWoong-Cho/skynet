import assert from "node:assert/strict";
import {pageWindow, byId, flusher, polyfillDialogs, readStatic, appFunction} from "./ui_harness.mjs";
const w=pageWindow({url:'http://localhost:8080/#data'}), el=byId(w), calls=[];
let subscription;
w.activeTab='cluster';
w.SkynetRefresh={connected:true,register(key,topics,visible,refresh,invalidate){subscription={topics,visible,refresh,invalidate};}};
polyfillDialogs(w, {returnValue:false});
w.HTMLElement.prototype.scrollIntoView=()=>{};
const app=await readStatic('app.js');
for(const name of ['escapeHtml','formatDate','stateClass','statusPill','setHtmlIfChanged','tableColumnCount','emptyRow','patchTableRow','reconcileTableSequence','valueHtml','keyValueHtml','datasetEpisodeCount','datasetCanTrain']) w.eval(appFunction(app, name));
const sharedHistoryStart=app.indexOf('window.SkynetJobHistory =');
w.eval(app.slice(sharedHistoryStart,app.indexOf('\nfunction renderDataImports()',sharedHistoryStart)));
w.dataPreparationRows=[];
w.document.addEventListener('dataset-preparation-changed',event=>{w.dataPreparationRows=event.detail;});
let inspectedResource;
w.openDataInspection=id=>{inspectedResource=id;};
w.showToast=()=>{};
const previewEvents=[];
w.SkynetEpisodeViewer={openDataset:(host,id)=>previewEvents.push(['open',host,id]),closeDataset:host=>previewEvents.push(['close',host])};
const copied=[];w.navigator.clipboard={writeText:async value=>copied.push(value)};
let selectedDataset;
w.useDataset=async id=>{selectedDataset=id;};
const adapter=(id,name,observations=['state','rgb'])=>({id,adapter_id:id,name,adapter_version_id:id+'-v1',adapter_version_number:1,
  available:true,trainable:true,default_data_preset:'default',data_presets:[{id:'default',observations,observation_requirements:{streams:[]},minimum_episodes:1}]});
const adapters=[adapter('hpt','EgoVerse HPT'),adapter('unidex','UniDex',['state','point_cloud'])];
const datasets={
  hpt:{id:'hpt',resource_id:'source',category:'dataset',display_name:'Shadow HPT',description:'Joint observations',created_at:'2026-09-19',format:'skynet.recording-dataset/v1',metadata:{adapter:{name:'EgoVerse HPT'},observations:['state','rgb'],episodes:[{},{}],split:{train:[0],validation:[1],seed:42}},locations:[{kind:'cluster',status:'AVAILABLE',host:'sky2',path:'/shared/hpt'}]},
  unidex:{id:'unidex',resource_id:'source',category:'dataset',display_name:'Shadow UniDex',created_at:'2026-09-20',format:'skynet.recording-dataset/v1',metadata:{adapter:{name:'UniDex'},observations:['state','point_cloud'],episodes:[{},{}]},locations:[{kind:'cluster',status:'AVAILABLE',path:'/shared/unidex'}]},
};
let jobs=[
  {id:'hpt-job',version_id:'hpt',resource_id:'source',name:'Shadow HPT',session_id:'recording',adapter:{name:'EgoVerse HPT'},state:'READY',training_ready:true,execution:'cluster',locations:datasets.hpt.locations},
  {id:'unidex-job',version_id:'unidex',resource_id:'source',name:'Shadow UniDex',session_id:'recording',adapter:{name:'UniDex'},state:'READY',training_ready:true,execution:'cluster',locations:datasets.unidex.locations},
  {id:'failed-job',resource_id:'source',name:'Failed earlier attempt',session_id:'recording',adapter:{name:'UniDex'},state:'FAILED',error:'Rendering failed',execution:'cluster',cluster_job_id:'12'},
];
const sessions={recording:{id:'recording',name:'Shadow Pick up cube',eligible:true,episodes:51},one:{id:'one',name:'One episode',eligible:true,episodes:1}};
const api=async(path,request={})=>{
  calls.push([path,request]);
  if(path.startsWith('/api/data/exports/options/'))return structuredClone({session:sessions[path.split('/').at(-1)],adapters});
  if(path==='/api/data/exports/jobs')return structuredClone({exports:jobs});
  if(path.startsWith('/api/data/datasets/'))return {dataset:structuredClone(datasets[path.split('/').at(-1)])};
  if(path.startsWith('/api/data/resources/'))return {resource:{id:'file-group',display_name:'Robot assets',category:'file',versions:[{format:'URDF',locations:[{host:'sky2',path:'/assets/robot'}]}]}};
  if(path.endsWith('/retry')){jobs.find(j=>j.id==='failed-job').state='QUEUED';return {};}
  if(path==='/api/data/exports' && request.method==='POST'){
    const body=JSON.parse(request.body);
    const job={id:'new-job',name:body.name,resource_id:'source',adapter:{name:'UniDex'},session_id:'recording',state:'QUEUED',detail:'Waiting for worker'};
    jobs.push(job);return structuredClone(job);
  }
  throw new Error('Unexpected API '+path);
};
w.api=api;
const flush = flusher(8);
try{
  w.eval(await readStatic('dialogs.js'));
  w.eval(await readStatic('policy-exports.js'));
  await flush();
  assert.equal(calls.length,0,'Unrelated tabs do not load conversion history');
  await w.openPolicyExport('recording');
  assert.deepEqual(calls.map(([path])=>path),['/api/data/exports/options/recording']);
  assert.equal(el('policy-export-name').readOnly,false,'Each output has an independent editable name');
  assert.equal(el('policy-export-name').value,'Shadow Pick up cube · EgoVerse HPT');
  assert.doesNotMatch(el('policy-export-name-help').textContent,/Adds a prepared result/);
  el('policy-export-adapter').value='unidex';el('policy-export-adapter').dispatchEvent(new w.Event('change'));
  assert.equal(el('policy-export-name').value,'Shadow Pick up cube · UniDex');
  const modality=id=>el('policy-export-modalities').querySelector(`[data-input-modality="${id}"]`);
  assert.equal(modality('point_cloud').classList.contains('is-used'),true);
  assert.equal(modality('rgb').classList.contains('is-used'),false);
  assert.equal(modality('depth').classList.contains('is-used'),false);
  assert.doesNotMatch(el('policy-export-modalities').textContent,/Used|Not used|1,024|XYZRGB/);
  el('policy-export-name').value='My named dataset';
  el('policy-export-adapter').value='hpt';el('policy-export-adapter').dispatchEvent(new w.Event('change'));
  assert.equal(el('policy-export-name').value,'My named dataset','Changing adapter preserves an edited name');
  el('preparation-seed').value='17';
  adapters[0].available=false;subscription.invalidate(['adapters']);await subscription.refresh();
  assert.equal(el('create-policy-export').disabled,true);
  assert.equal(el('policy-export-name').value,'My named dataset');
  assert.equal(el('preparation-seed').value,'17');
  adapters[0].available=true;adapters[0].adapter_version_id='hpt-v2';adapters[0].adapter_version_number=2;
  subscription.invalidate(['adapters']);await subscription.refresh();
  assert.equal(el('policy-export-adapter').value,'','Changed adapter versions require explicit reselection');
  el('policy-export-adapter').value='hpt';el('policy-export-adapter').dispatchEvent(new w.Event('change'));
  el('policy-export-form').dispatchEvent(new w.Event('submit',{cancelable:true}));await flush();
  const submitted=JSON.parse(calls.find(([path,options])=>path==='/api/data/exports' && options.method==='POST')[1].body);
  assert.equal(submitted.name,'My named dataset');assert.equal(submitted.resource_id,undefined);
  assert.equal(submitted.adapter_version_id,'hpt-v2');
  assert.equal(el('policy-export-dialog').open,false);
  assert.equal(el('prepared-dataset-dialog').open,true);
  assert.match(el('prepared-dataset-content').textContent,/Waiting for worker/);
  assert.equal(el('prepared-dataset-content').querySelectorAll('tbody tr').length,1);
  jobs.find(j=>j.id==='new-job').state='READY';jobs.find(j=>j.id==='new-job').version_id='hpt';jobs.find(j=>j.id==='new-job').training_ready=true;
  await subscription.refresh();
  assert.equal(el('prepared-dataset-title').textContent,'Shadow HPT','Completed conversion opens its exact published result');
  assert.doesNotMatch(el('prepared-dataset-content').textContent,/Shadow UniDex|Failed earlier attempt/);
  assert.equal(el('prepared-dataset-content').querySelector('table'),null,'Dataset details contain no nested dataset table');
  assert.equal(el('prepared-dataset-content').querySelector('[data-data-history]'),null,'Hidden provenance groups are not exposed');
  assert.equal(el('prepared-dataset-actions').querySelector('[data-use-dataset]').dataset.useDataset,'hpt');
  assert.equal(el('prepared-dataset-content').querySelector('[data-input-modality="rgb"]').classList.contains('is-used'),true);
  el('prepared-dataset-actions').querySelector('[data-use-dataset]').click();await flush();assert.equal(selectedDataset,'hpt');
  await w.openPreparedDataset('unidex');
  assert.equal(el('prepared-dataset-preview').hidden,false);
  assert.deepEqual(previewEvents.at(-1),['open','prepared-dataset-preview','unidex'],'Dataset View embeds the shared episode preview');
  assert.equal(el('prepared-dataset-title').textContent,'Shadow UniDex');
  assert.equal(el('prepared-dataset-content').querySelector('[data-input-modality="rgb"]').classList.contains('is-used'),false);
  assert.equal(el('prepared-dataset-content').querySelector('[data-input-modality="point_cloud"]').classList.contains('is-used'),true);
  el('prepared-dataset-content').querySelector('[data-dataset-copy-path]').click();await flush();assert.deepEqual(copied,['/shared/unidex']);
  datasets.unidex.archived_at='2026-09-20';await subscription.refresh();
  assert.equal(el('prepared-dataset-actions').querySelector('[data-use-dataset]'),null);
  datasets.unidex.archived_at=null;jobs.find(j=>j.id==='unidex-job').training_ready=false;await subscription.refresh();
  assert.equal(el('prepared-dataset-actions').querySelector('[data-use-dataset]'),null,'Archived adapter cannot remain trainable');
  await w.openDatasetConversionHistory({versionId:'hpt'});
  assert.equal(el('prepared-dataset-title').textContent,'Conversion history');
  assert.doesNotMatch(el('prepared-dataset-content').textContent,/Failed earlier attempt|Shadow UniDex/);
  await w.openDatasetConversionHistory();
  assert.match(el('prepared-dataset-content').textContent,/Failed earlier attempt/);
  const failedRow=el('prepared-dataset-content').querySelector('[data-history-id="failed-job"]');
  const retryButton=failedRow.querySelector('[data-preparation-retry]');
  retryButton.focus();
  await subscription.refresh();
  assert.equal(el('prepared-dataset-content').querySelector('[data-history-id="failed-job"]'),failedRow,'Shared table preserves rows during background refresh');
  assert.equal(w.document.activeElement,retryButton,'Unchanged history controls keep focus');
  assert.equal(el('prepared-dataset-content').querySelector('.policy-export-job'),null);
  assert.ok(el('prepared-dataset-content').querySelector('a[href$="/export.log"]'));
  el('prepared-dataset-content').querySelector('[data-preparation-retry]').click();await flush();
  assert.ok(calls.some(([path,options])=>path==='/api/data/exports/failed-job/retry' && options.method==='POST'));
  assert.equal(failedRow.querySelector('[data-preparation-retry]'),null,'Changed state updates shared row actions');
  assert.match(failedRow.textContent,/Slurm job 12/);assert.doesNotMatch(failedRow.textContent,/sky\d/,'A job without a recorded gateway names no login host');
  assert.equal(el('policy-export-history'),null,'Conversion history has no separate inline panel');
  assert.equal(el('policy-export-jobs'),null);
  w.api=async(path,request={})=>{if(path==='/api/data/exports/jobs')throw new Error('History down');return api(path,request);};
  await w.openPreparedDataset('hpt');assert.equal(el('prepared-dataset-title').textContent,'Shadow HPT','Published data details are independent of history availability');
  await w.openFileResource('file-group');assert.equal(inspectedResource,'file-group','Files reuse the existing inspection controller');
  assert.equal(el('prepared-dataset-dialog').open,false);
  w.api=api;await w.openPolicyExport('one');assert.equal(el('preparation-validation').disabled,true);assert.equal(el('create-policy-export').disabled,false);
  let resolveOld;
  w.api=(path,request={})=>path.endsWith('/options/one')?new Promise(resolve=>{resolveOld=resolve;}):api(path,request);
  const old=w.openPolicyExport('one');await flush();await w.openPolicyExport('recording');resolveOld({session:sessions.one,adapters});await old;
  assert.match(el('policy-export-name').value,/Shadow/,'Late options cannot overwrite a newer form');
  w.api=api;
  w.SkynetDialog.close(el('policy-export-dialog'));w.activeTab='datasets';
  let fallback=0;const setTimeout=w.setTimeout.bind(w);w.setTimeout=(fn,delay,...args)=>delay===3000?(fallback++,-1):setTimeout(fn,delay,...args);
  await subscription.refresh();assert.equal(fallback,0);w.SkynetRefresh.connected=false;await subscription.refresh();assert.equal(fallback,1);
  assert.equal(calls.some(([path,request])=>path==='/api/data/exports' && !request.method),false,'Never fetch full legacy export catalog');
  console.log('Flat dataset details, conversion names, exact-result actions, history, stale requests and modality UI passed.');
}finally{w.close();}
