import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {JSDOM} from 'jsdom';

const w = new JSDOM(await readFile(new URL('../static/index.html', import.meta.url), 'utf8'), {
  runScripts: 'outside-only', pretendToBeVisual: true,
  url: 'http://localhost:8080/?experiment_view=submit#runs',
}).window;
const el = id => w.document.getElementById(id);
const visible = id => !el(id).closest('[hidden]');
const observers = [], Observer = w.MutationObserver;
w.MutationObserver = class extends Observer {constructor(callback) {super(callback); observers.push(this);}};
w.fetch = () => new Promise(() => {});
w.scrollTo = w.HTMLElement.prototype.scrollIntoView = () => {};
w.matchMedia = () => ({matches: false, addEventListener() {}, removeEventListener() {}});
const settle = () => new Promise(resolve => setTimeout(resolve, 15));
try {
  for (const file of ['dialogs.js', 'workspace-navigation.js', 'connection-settings.js', 'app.js']) {
    w.eval(await readFile(new URL('../static/' + file, import.meta.url), 'utf8'));
  }
  // Display the signed-in shell without involving real workspace credentials or a server.
  el('email-workspace-content').hidden = false;
  w.scheduleEvaluationTargetValidation = () => {};
  const requests = [];
  const compatible = {id: 'held-out', name: 'egoverse_held_out', label: 'Held-out predictions',
    evaluator: 'egoverse', version: 'v2', tasks: ['held_out']};
  const simulator = {id: 'simulator', name: 'dexverse_recorded', label: 'DexVerse simulator',
    evaluator: 'isaac_lab', version: 'v1', tasks: ['PickCube'], config_json: {tasks: ['PickCube']}};
  const run = {id: 'training', status: 'SUCCEEDED', checkpoints: [
    {path: '/cluster/last.ckpt', status: 'AVAILABLE', checkpoint_type: 'INFERENCE'},
  ]};
  w.api = async path => {
    requests.push(path);
    if (path === '/api/evaluation-suites') return {suites: [compatible, simulator]};
    if (path === '/api/evaluation-suites?run_id=training') return {suites: [compatible]};
    if (path === '/api/runs/training') return {run};
    if (path === '/api/runs') return {runs: [run]};
    if (path === '/api/evaluations') return {evaluations: []};
    return {};
  };
  assert.equal(w.location.hash, '#experiments', 'Legacy run links resolve to the experiment workspace');
  assert.equal(new URL(w.location.href).searchParams.get('experiment_view'), 'runs');
  assert.ok(visible('runs-body'));
  assert.equal(el('runs').closest('#experiment-workspace'), el('experiment-workspace'));
  assert.equal(w.document.querySelector('.tab-nav [data-tab-target="runs"]'), null);
  assert.equal(w.document.querySelector('.tab-nav [aria-selected="true"]').textContent, 'Experiments');
  for (const view of ['presets', 'submit', 'adapters', 'runs']) {
    el(`experiment-tab-${view}`).click();
    assert.equal(el(`experiment-tab-${view}`).getAttribute('aria-selected'), 'true');
    assert.ok(visible(view === 'submit' ? 'experiment-form' : view));
    assert.equal(w.document.querySelectorAll('#experiment-workspace .page-actions > button:not([hidden])').length, view === 'presets' ? 1 : 2,
      'Only the current tutorial and refresh are displayed');
  }

  await w.activateTab('evaluations', true, 'submit');
  el('evaluation-run-id').value = 'training';
  await w.loadEvaluationSuites(true);
  el('evaluation-suite').value = 'held-out';
  w.updateEvaluationEnvironmentFromSuite();
  const formOptions = el('evaluation-suite').innerHTML;
  await w.activateTab('evaluations', true, 'suites');
  assert.ok(visible('evaluation-suites-body'));
  assert.equal(visible('evaluation-form'), false);
  assert.equal(visible('evaluations-body'), false);
  assert.match(el('evaluation-suites-body').textContent, /DexVerse simulator/);
  assert.equal(el('evaluation-suite-count').textContent, '2 suites');
  assert.equal(el('evaluation-suite').innerHTML, formOptions, 'The catalog cannot overwrite run-compatible options');
  assert.equal(el('evaluation-suite').value, 'held-out');
  assert.equal(el('evaluation-environment').value, 'egoverse');
  el('evaluation-suite-search').value = 'isaac_lab';
  el('evaluation-suite-search').dispatchEvent(new w.Event('input'));
  assert.equal(el('evaluation-suite-count').textContent, '1 of 2 suites');
  assert.doesNotMatch(el('evaluation-suites-body').textContent, /Held-out predictions/);
  const count = requests.length;
  el('refresh-evaluations').click(); await settle();
  assert.deepEqual(requests.slice(count), ['/api/evaluation-suites'], 'Suites Refresh reloads the global catalog only');
  el('evaluation-tab-suites').dispatchEvent(new w.KeyboardEvent('keydown', {key: 'ArrowLeft', bubbles: true}));
  await settle();
  assert.equal(w.document.activeElement, el('evaluation-tab-runs'));
  assert.ok(visible('evaluations-body'));
  assert.equal(visible('evaluation-form'), false);
  assert.match(el('evaluation-runs').textContent, /Evaluation run history/);
  w.history.back(); await settle();
  assert.ok(visible('evaluation-suites-body'), 'Back restores the selected subtab');
  w.history.forward(); await settle();
  assert.ok(visible('evaluations-body'), 'Forward restores run history');
  await w.startEvaluationForRun('training');
  assert.ok(visible('evaluation-form'), 'Start evaluation always opens Submit, even from Runs');
  assert.equal(el('evaluation-run-id').value, 'training');
  assert.equal(new URL(w.location.href).searchParams.get('evaluation_view'), 'submit');
  console.log('Workspace tabs: legacy links, panel ownership, keyboard/history, global catalog isolation, search/refresh and Start evaluation passed.');
} finally {
  for (const observer of observers) observer.disconnect();
  w.close();
}
