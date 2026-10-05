import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {JSDOM} from 'jsdom';
import {indexHtml} from './index_page.cjs';
const w = new JSDOM(indexHtml(), {runScripts:'outside-only',pretendToBeVisual:true,url:'http://localhost/#evaluations'}).window;
const observers=[],Observer=w.MutationObserver;
w.MutationObserver=class extends Observer {constructor(fn){super(fn);observers.push(this);}};
w.fetch=()=>new Promise(()=>{});w.scrollTo=w.HTMLElement.prototype.scrollIntoView=()=>{};
w.matchMedia=()=>({matches:false,addEventListener(){},removeEventListener(){}});w.CSS={escape:s=>s};
w.HTMLDialogElement.prototype.showModal=function(){this.open=true;};
w.HTMLDialogElement.prototype.close=function(){this.open=false;this.dispatchEvent(new w.Event('close'));};
const el=id=>w.document.getElementById(id),flush=async()=>{for(let i=0;i<8;i++)await new Promise(r=>setImmediate(r));};
try {
 for(const file of ['dialogs.js','workspace-navigation.js','connection-settings.js','app.js']) {
  let source=await readFile(new URL('../static/'+file,import.meta.url),'utf8');
  if(file==='app.js')source+='\nwindow.seedLinkTest=run=>{runRows=[run];renderRuns();}; window.seedDatasetLinkTest=(dataset,adapter)=>{if(adapter){adapterRows=[{id:"adapter",name:"Policy",latest_version:{id:"adapter-v1",version_number:1,manifest:{schema_version:"skynet.adapter/v1",slug:"policy",display_name:"Policy",train:{input_fields:[{path:"native.config.dataset",label:"Dataset",kind:"json",...(adapter==="takes data"?{data_binding:{role:"training_data"}}:{})}]}}}}];adaptersLoaded=true;populateExperimentAdapters();applySelectedAdapter({loadSource:false});}trainingDatasetRows=[dataset];populateExperimentDataBundles();};';
  w.eval(source);
 }
 const run={id:'run',experiment_name:'My experiment',experiment_id:'experiment',run_number:2,status:'SUCCEEDED',attempts:[],checkpoints:[{id:'cp',path:'/cluster/last.ckpt',status:'AVAILABLE'}],manual_actions:{}};
 w.seedLinkTest(run);w.activateTab=async()=>{};w.startRunDetailPolling=()=>{};
 let resolve;
 w.api=async path=>{assert.equal(path,'/api/runs/run?include_payloads=false');return new Promise(r=>resolve=r);};
 const launch=w.document.createElement('button');launch.dataset.entityKind='run';launch.dataset.entityId='run';launch.dataset.checkpointId='cp';launch.textContent='View checkpoint';w.document.body.append(launch);
 launch.click();await flush();
 assert.equal(el('run-detail-dialog').open,true);
 assert.match(el('run-detail-meta').textContent,/Loading run/);
 resolve({run});await flush();await new Promise(r=>setTimeout(r,35));
 assert.doesNotMatch(el('run-detail-meta').textContent,/Loading run/,'linked navigation uses the live disclosure token');
 assert.match(el('run-detail-meta').textContent,/My experiment/);
 assert.equal(w.document.activeElement.dataset.checkpointRow,'cp','linked checkpoints receive focus');
 assert.equal(el('run-detail-meta').querySelector('[data-entity-kind="experiment"]').dataset.entityId,'experiment');
 assert.ok(el('run-checkpoint-list').querySelector('[data-copy-value="/cluster/last.ckpt"]'));
 // The algorithm is chosen before its data and runtime.
 for (const later of ['training-data-section','training-runtime-section'])
  assert.equal(el('experiment-adapter').compareDocumentPosition(el(later)) & w.Node.DOCUMENT_POSITION_FOLLOWING, w.Node.DOCUMENT_POSITION_FOLLOWING, later);
 // Selecting data must not wait for unrelated page requests (history/tracking).
 w.activateTab=()=>new Promise(()=>{});
 w.refreshExperimentModelIO=w.showToast=()=>{};
 const dataset={id:'dataset',name:'One episode',format:'zarr',assignments:[]};
 w.loadTrainingInputs=async()=>w.seedDatasetLinkTest(dataset);
 await assert.rejects(()=>w.useRegisteredDataset('dataset'),/Choose an algorithm first/);
 w.loadTrainingInputs=async()=>w.seedDatasetLinkTest(dataset,'takes no data');
 await assert.rejects(()=>w.useRegisteredDataset('dataset'),/does not take a registered dataset/);
 w.loadTrainingInputs=async()=>w.seedDatasetLinkTest(dataset,'takes data');
 await w.useRegisteredDataset('dataset');
 assert.equal(el('experiment-data-bundle').value,'dataset');
 console.log('Entity navigation: linked run loading, checkpoint focus, escaped copy links and registered dataset selection passed.');
} finally {for(const observer of observers)observer.disconnect();w.close();}
