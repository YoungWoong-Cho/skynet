import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {JSDOM} from 'jsdom';

const w = new JSDOM(await readFile(new URL('../static/index.html', import.meta.url), 'utf8'), {
  runScripts: 'outside-only', pretendToBeVisual: true, url: 'http://localhost:8080/#runs',
}).window;
const observers = [], Observer = w.MutationObserver;
w.MutationObserver = class extends Observer {constructor(cb) {super(cb); observers.push(this);}};
w.fetch = () => new Promise(() => {});
w.scrollTo = w.HTMLElement.prototype.scrollIntoView = () => {};
w.matchMedia = () => ({matches: false, addEventListener() {}, removeEventListener() {}});
w.CSS = {escape: value => value};
w.HTMLDialogElement.prototype.showModal = function() {this.open = true;};
w.HTMLDialogElement.prototype.close = function() {this.open = false;};
const el = id => w.document.getElementById(id);
try {
  for (const file of ['dialogs.js', 'workspace-navigation.js', 'connection-settings.js', 'app.js']) {
    w.eval(await readFile(new URL('../static/' + file, import.meta.url), 'utf8'));
  }

  w.scheduleEvaluationTargetValidation = () => {};
  const runId = el('evaluation-run-id');
  runId.value = 'one-episode';
  const simulation = {id:'same-episode', is_default:true, label:'Recorded training episode · simulator',
    evaluator_adapter:'isaac_lab', enabled:true, current:true, maximum_parallelism:1,
    config_json:{initial_state:'single_training_episode', maximum_episodes_per_task:1,
      tasks:['Dexverse-PickCube-v0'],task_options:[{id:'Dexverse-PickCube-v0',label:'Pick up cube'}],
      task_selection_mode:'all_only',task_catalog_complete:true}};
  const other = {...simulation,id:'other',is_default:false};
  w.api=async()=>({suites:[other,simulation]});
  await w.loadEvaluationSuites(true);
  assert.equal(el('evaluation-suite').value,'same-episode','Single episode defaults to its saved initial scene');
  assert.equal(w.preferredEvaluationSuite({}).id,'same-episode');
  assert.equal(el('evaluation-episodes').max,'1');
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
