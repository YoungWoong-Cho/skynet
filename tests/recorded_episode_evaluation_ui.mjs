import assert from 'node:assert/strict';
import {pageWindow, byId, spyMutationObservers, stubBrowserApis, polyfillDialogs, loadScripts} from './ui_harness.mjs';

const w = pageWindow({url: 'http://localhost:8080/#runs'});
const observers = spyMutationObservers(w);
stubBrowserApis(w, {cssEscape: true});
polyfillDialogs(w, {guarded: false, returnValue: false, closeEvent: false});
const el = byId(w);
try {
  await loadScripts(w);

  w.scheduleEvaluationTargetValidation = () => {};
  const runId = el('evaluation-run-id');
  runId.value = 'one-episode';
  const simulation = {id:'same-episode', is_default:true, label:'Recorded training episode · simulator',
    evaluator_adapter:'isaac_lab', enabled:true, current:true, maximum_parallelism:1,
    allowed_gpu_types:['a40','l40s'],
    config_json:{initial_state:'single_training_episode', maximum_episodes_per_task:1,
      tasks:['Dexverse-PickCube-v0'],task_options:[{id:'Dexverse-PickCube-v0',label:'Pick up cube'}],
      task_selection_mode:'all_only',task_catalog_complete:true}};
  const other = {...simulation,id:'other',is_default:false};
  w.api=async()=>({suites:[other,simulation]});
  await w.loadEvaluationSuites(true);
  assert.equal(el('evaluation-suite').value,'same-episode','Single episode defaults to its saved initial scene');
  assert.equal(w.preferredEvaluationSuite({}).id,'same-episode');
  assert.equal(el('evaluation-episodes').max,'1');
  const gpu = el('evaluation-resource-gpu');
  const gpuValues = () => Array.from(gpu.options, option => option.value);
  assert.deepEqual(gpuValues(), ['a40','l40s'], 'Isaac evaluations only show configured compatible GPUs');
  gpu.value = 'l40s';
  w.updateEvaluationEnvironmentFromSuite();
  assert.equal(gpu.value, 'l40s', 'Refreshing a suite preserves an allowed GPU selection');
  const trainingGpuValues = Array.from(el('experiment-gpu-type').options, option => option.value);
  w.updateEvaluationGpuOptions({allowed_gpu_types:null});
  assert.ok(gpuValues().includes('rtx_6000'), 'Other evaluators retain their normal GPU choices');
  gpu.value = 'rtx_6000';
  w.updateEvaluationEnvironmentFromSuite();
  assert.deepEqual(gpuValues(), ['a40','l40s']);
  assert.equal(gpu.value, 'a40', 'Switching to Isaac replaces an incompatible selection');
  w.updateEvaluationGpuOptions({allowed_gpu_types:[]});
  assert.equal(gpu.disabled, true);
  assert.equal(w.validateEvaluationResources(), false, 'Missing placement configuration cannot be submitted');
  w.updateEvaluationEnvironmentFromSuite();
  assert.equal(gpu.disabled, false);
  assert.deepEqual(Array.from(el('experiment-gpu-type').options, option => option.value), trainingGpuValues);

  const incompatible = {...other, id:'different-robot', label:'Other robot suite',
    compatibility:{ready:false,status:'mapping_required',label:'Mapping required',messages:['Joint identities differ.']}};
  w.api=async()=>({suites:[incompatible,{...simulation,compatibility:{ready:true,status:'compatible',label:'Compatible',messages:[],executor:'recorded_simulator'}}]});
  await w.loadEvaluationSuites(true);
  assert.equal(el('evaluation-suite').options.length,3,'all suites remain selectable for inspection');
  assert.match(el('evaluation-suite').textContent,/Other robot suite — Mapping required/);
  assert.equal(w.preferredEvaluationSuite({}).id,'same-episode','incompatible suites are never auto-selected');
  el('evaluation-suite').value='different-robot';w.updateEvaluationEnvironmentFromSuite();
  assert.match(el('evaluation-suite-status').textContent,/Joint identities differ/);
  assert.equal(w.evaluationSuiteIsRunnable(incompatible),false);
  const targets = [
    {id:'wuji-target', name:'WUJI target', assignments:[{version:{metadata:{contract:'skynet.unidex-pointcloud-faas/v1'}}}]},
    {id:'hat-target', name:'HAT target', assignments:[{version:{metadata:{contract:'skynet.hat-rgb-fingertips/v1'}}}]},
    {id:'rgb-target', name:'RGB target', assignments:[{version:{metadata:{contract:'skynet.egoverse-rgb-joints/v1'}}}]}
  ];
  const unidexSuite = {...other, requires_target_dataset:true, target_dataset_contract:'skynet.unidex-pointcloud-faas/v1'};
  const requests = [];
  w.api=async endpoint=>{requests.push(endpoint);return endpoint === '/api/data/selections' ? {datasets:targets} : {suites:[unidexSuite]};};
  await w.loadDataBundles(true);
  await w.loadEvaluationSuites(true);
  assert.equal(el('evaluation-target-fields').hidden,false);
  assert.equal(el('evaluation-target-dataset').required,true);
  assert.deepEqual(Array.from(el('evaluation-target-dataset').options, o=>o.value),['','wuji-target']);
  el('evaluation-target-dataset').value='wuji-target';
  const unseenSignature=w.evaluationTargetSignature();
  await w.loadEvaluationSuites(false);
  assert.match(requests.at(-1),/target_dataset_id=wuji-target&unseen_embodiment=true/);
  assert.equal(w.evaluationDatasetInput().target_dataset_id,'wuji-target');
  el('evaluation-unseen-hand').checked=false;
  assert.notEqual(w.evaluationTargetSignature(),unseenSignature);
  await w.loadEvaluationSuites(false);
  assert.match(requests.at(-1),/unseen_embodiment=false/,'Target and unseen changes invalidate the suite cache scope');
  const hatSuite = {...other, requires_target_dataset:true, target_dataset_contract:'skynet.hat-rgb-fingertips/v1'};
  w.api=async()=>({suites:[hatSuite]});
  await w.loadEvaluationSuites(true);
  assert.equal(el('evaluation-target-fields').hidden,false,'HAT can select a held-out hand in the actual form');
  assert.deepEqual(Array.from(el('evaluation-target-dataset').options, o=>o.value),['','hat-target']);
  assert.equal(el('evaluation-target-dataset').value,'','Switching policy contracts clears an incompatible target');
  el('evaluation-target-dataset').value='hat-target';
  el('evaluation-unseen-hand').checked=true;
  assert.equal(w.evaluationDatasetInput().target_dataset_id,'hat-target');
  assert.equal(w.evaluationDatasetInput().unseen_embodiment,true);
  w.api=async()=>({suites:[simulation]});
  await w.loadEvaluationSuites(true);
  assert.equal(el('evaluation-target-fields').hidden,true);
  assert.equal(el('evaluation-target-dataset').disabled,true);
  assert.equal(el('evaluation-target-dataset').value,'');
  assert.equal(Object.keys(w.evaluationDatasetInput()).length,0,'Single embodiment loaders keep their existing input contract');
  runId.value='deleted-recording';
  w.api=async()=>({suites:[],unavailable_suites:[{id:'same-episode',reason:'The original recording is no longer registered'}]});
  await w.loadEvaluationSuites(true);
  assert.match(el('evaluation-suite-status').textContent,/original recording/);
  el('evaluation-suite-status').textContent='';
  await w.loadEvaluationSuites(false);
  assert.match(el('evaluation-suite-status').textContent,/original recording/,'Cached results preserve the actual reason');
  runId.value='other-user-run';
  w.api=async()=>{throw new Error('Connection failed');};
  await w.loadEvaluationSuites(true);
  assert.match(el('evaluation-suite-status').textContent,/Connection failed/);
  assert.doesNotMatch(el('evaluation-suite-status').textContent,/registry|original recording/);
  console.log('Single-episode evaluation: default selection, episode bound, cached reason and network-error isolation passed.');
} finally {
  for (const observer of observers) observer.disconnect();
  w.close();
}
