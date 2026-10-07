import assert from 'node:assert/strict';
import {pageWindow, byId, flusher, spyMutationObservers, stubBrowserApis, polyfillDialogs, readStatic} from './ui_harness.mjs';

const w = pageWindow({url: 'http://localhost:8080/#experiments'});
const observers = spyMutationObservers(w);
stubBrowserApis(w, {cssEscape: true});
polyfillDialogs(w, {returnValue: false});
const el = byId(w);
const flush = flusher(5);

try {
  for (const file of ['dialogs.js', 'workspace-navigation.js', 'connection-settings.js', 'app.js']) {
    let source = await readStatic(file);
    if (file === 'app.js') source += `
      window.audit = {
        setAdapters(rows) { adapterRows = rows; populateExperimentAdapters(); },
        options() { return activeRepositoryInputOptions(); },
        setSuite(suite) { evaluationSuites = [suite]; elements.evaluationSuite.innerHTML = '<option value="' + suite.id + '">Suite</option>'; populateEvaluationTasks(suite); },
        async validateTarget() { const request = ++evaluationTargetValidationRequest; return validateEvaluationTarget(request, evaluationTargetSignature()); },
        setRuns(rows) { runRows = rows; renderRuns(); },
        tutorialSteps() { return tutorialTours.experiments.steps; },
      };`;
    w.eval(source);
  }

  // A single-task suite uses mutually exclusive controls and restores its real default.
  const suite = {id: 'single', config_json: {tasks: ['one', 'two'], task_selection_mode: 'single', default_tasks: ['one']}};
  w.audit.setSuite(suite);
  let tasks = [...el('evaluation-tasks-options').querySelectorAll('input')];
  assert.deepEqual(tasks.map(input => input.type), ['radio', 'radio']);
  assert.equal(tasks[0].checked, true);
  tasks[1].click();
  assert.deepEqual(tasks.map(input => input.checked), [false, true]);
  assert.equal(el('evaluation-tasks-filter').getAttribute('aria-invalid'), 'false');
  assert.equal(el('evaluation-tasks-all').hidden, true);
  assert.equal(el('evaluation-tasks-clear').hidden, false);
  el('evaluation-tasks-clear').click();
  assert.deepEqual(tasks.map(input => input.checked), [true, false]);
  w.audit.setSuite({...suite, config_json: {...suite.config_json, default_tasks: []}});
  assert.equal(el('evaluation-tasks-clear').hidden, true, 'no invalid Use default action when the suite has no default');
  assert.equal(el('evaluation-tasks-filter-label').textContent, 'Choose one task', 'an unselected required single task cannot claim a default');

  // Resource syntax and transport errors must not blame a valid run/checkpoint.
  el('evaluation-run-id').value = 'run';
  el('evaluation-checkpoint').value = '/runs/run/last.ckpt';
  el('evaluation-resource-time').value = 'bad';
  let requests = 0;
  w.api = async () => { requests++; throw new Error('Resource service unavailable'); };
  assert.equal(await w.audit.validateTarget(), false);
  assert.equal(requests, 0, 'invalid resource syntax is caught before an API request');
  assert.equal(el('evaluation-resource-time').getAttribute('aria-invalid'), 'true');
  assert.match(el('evaluation-resource-summary').textContent, /wall time/);
  el('evaluation-resource-time').value = '00:05:00';
  await w.audit.validateTarget();
  assert.equal(requests, 1);
  assert.equal(el('evaluation-run-id').hasAttribute('aria-invalid'), false);
  assert.equal(el('evaluation-checkpoint').hasAttribute('aria-invalid'), false);
  assert.match(el('evaluation-plan-status').textContent, /Resource service unavailable/);
  const scheduled = [];
  const schedule = w.scheduleEvaluationTargetValidation;
  w.scheduleEvaluationTargetValidation = options => scheduled.push(options);
  w.revalidateFinishedEvaluation([{run_id: 'run', status: 'RUNNING'}], [{run_id: 'run', status: 'CANCELLED'}]);
  w.revalidateFinishedEvaluation([{run_id: 'other', status: 'RUNNING'}], []);
  assert.equal(scheduled.length, 1, 'only the selected run leaving active evaluation revalidates the form');
  assert.equal(scheduled[0].immediate, true);
  w.scheduleEvaluationTargetValidation = schedule;

  // Returning to the same repository/commit restores its adapter-specific cached catalog.
  const commit = 'a'.repeat(40);
  const repository = 'https://github.com/example/shared';
  const adapter = (id, choices) => ({id, name: id, latest_version: {id: id + '-v1', version_number: 1, manifest: {
    schema_version: 'skynet.adapter/v1', slug: id, display_name: id, default_repository: repository,
    runtime: {allowed_backends: ['existing']}, capabilities: {supports_resume: false},
    train: {input_fields: choices ? [{path: 'native.config.config_name', kind: 'string', label: 'Config', choice_source: {kind: 'repository_config'}}] : []},
  }}});
  w.audit.setAdapters([adapter('openpi', true), adapter('generic', false)]);
  el('experiment-source').value = repository;
  el('experiment-workdir').value = '.';
  el('experiment-revision').innerHTML = '<option>' + commit + '</option>';
  const inspectionRequests = [];
  w.api = async path => {
    if (!path.startsWith('/api/source/inspect?')) return {};
    const id = new URL(path, w.location.origin).searchParams.get('adapter_id');
    inspectionRequests.push(id);
    return {commit, runtime_candidates: [{type: 'existing', runnable: true, confidence: 'strong'}], input_options: id === 'openpi' ? {'native.config.config_name': {choices: ['debug', 'train'], complete: true}} : {}};
  };
  el('experiment-adapter').value = 'adapter-version:openpi-v1';
  await w.inspectRepositoryRuntime();
  assert.deepEqual([...w.audit.options().get('native.config.config_name').choices], ['debug', 'train']);
  el('experiment-adapter').value = 'adapter-version:generic-v1';
  w.applySelectedAdapter();
  await flush();
  assert.equal(w.audit.options().size, 0);
  el('experiment-adapter').value = 'adapter-version:openpi-v1';
  w.applySelectedAdapter();
  await flush();
  assert.deepEqual([...w.audit.options().get('native.config.config_name').choices], ['debug', 'train']);
  assert.deepEqual(inspectionRequests, ['openpi', 'generic'], 'restored catalog reuses its original response');
  const configControl = el('adapter-field-native-config-config_name');
  configControl.value = JSON.stringify('train');
  configControl.dispatchEvent(new w.Event('change', {bubbles: true}));
  w.clearRepositoryInputOptions({render: true});
  assert.equal(el('adapter-field-native-config-config_name').value, JSON.stringify('train'), 'code reinspection retains the typed choice');
  assert.equal(w.validateAdapterDeclaredFields({focus: false, notify: false}), false, 'a retained but unavailable choice cannot validate');
  await w.inspectRepositoryRuntime();
  assert.equal(el('adapter-field-native-config-config_name').value, JSON.stringify('train'));
  assert.equal(w.validateAdapterDeclaredFields({focus: false, notify: false}), true);
  let releaseOld;
  w.api = () => new Promise(resolve => { releaseOld = resolve; });
  const staleInspection = w.inspectRepositoryRuntime(true);
  el('experiment-adapter').value = 'adapter-version:generic-v1';
  await w.inspectRepositoryRuntime();
  releaseOld({commit, input_options: {'native.config.config_name': {choices: ['stale'], complete: true}}});
  await staleInspection;
  assert.equal(w.audit.options().size, 0, 'the previous adapter response cannot overwrite the selected catalog');

  for (const [status, label] of [['QUEUED','Waiting to sync'],['ERROR','Sync failed'],['BLOCKED','Connection blocked']]) {
    const markup = w.trackingLinksHtml({tracking_links:[{provider:'wandb',status}]});
    assert.ok(markup.includes(label));
    assert.ok(!markup.includes('<a '));
  }
  assert.match(w.trackingLinksHtml({tracking_links:[{provider:'wandb',status:'CONNECTED',url:'https://wandb.ai/team/project/runs/one'}]}), /href="https:\/\/wandb.ai\/team\/project\/runs\/one"/);

  // Loading another run removes the previous external tracking destination immediately.
  w.audit.setRuns([{id: 'new-run', status: 'PENDING'}]);
  el('run-detail-tracking').innerHTML = '<a href="https://example.com/old">Old tracking</a>';
  el('run-checkpoint-list').innerHTML = '<p>previous-run-checkpoint</p>';
  el('run-checkpoints').hidden = false;
  let releaseRun;
  w.api = () => new Promise(resolve => { releaseRun = resolve; });
  const runRequest = w.viewRun('new-run', el('runs-body').querySelector('[data-run-action="view"]'));
  assert.equal(el('run-detail-tracking').textContent, '');
  assert.equal(el('run-checkpoint-list').textContent, '');
  assert.equal(el('run-checkpoints').hidden, true);
  releaseRun({run: {id: 'new-run', status: 'CANCELLED', attempts: []}});
  await runRequest;

  // Recovering the same tracking attachment removes its errors in both surfaces,
  // while errors for a different operation remain available to the user.
  const trackingButton = w.document.createElement('button');
  trackingButton.textContent = 'Connect W&B';
  el('run-detail-actions').append(trackingButton);
  w.api = async () => { throw new Error('temporary tracking failure'); };
  await w.attachRunTracking('new-run', 'wandb', '/api/runs/new-run/tracking/wandb/attach', trackingButton);
  assert.match(el('run-detail-dialog').querySelector('.dialog-notice').textContent, /temporary tracking failure/);
  assert.match(el('runs-error').textContent, /temporary tracking failure/);
  w.showToast('Other run remains blocked', true, {scope: 'run-resume:other'});
  w.showNotice(el('runs-error'), 'Other run remains blocked', {scope: 'run-resume:other'});
  w.api = async () => ({run: {id: 'new-run', status: 'CANCELLED', attempts: []}});
  await w.attachRunTracking('new-run', 'wandb', '/api/runs/new-run/tracking/wandb/attach', trackingButton);
  assert.doesNotMatch(el('run-detail-dialog').querySelector('.dialog-notice').textContent, /temporary tracking failure/);
  assert.doesNotMatch(el('runs-error').textContent, /temporary tracking failure/);
  assert.match(el('run-detail-dialog').querySelector('.dialog-notice').textContent, /Other run remains blocked/);
  assert.match(el('runs-error').textContent, /Other run remains blocked/);
  w.clearNotificationScope('run-resume:other');

  // A slow attachment belongs to its original run even if another detail opens.
  w.audit.setRuns([{id: 'new-run', status: 'CANCELLED'}, {id: 'other-run', status: 'CANCELLED'}]);
  w.api = async path => ({run: {id: path.endsWith('other-run') ? 'other-run' : 'new-run', status: 'CANCELLED', attempts: []}});
  await w.viewRun('new-run', el('runs-body').querySelector('[data-id="new-run"][data-run-action="view"]'));
  const pendingButton = w.document.createElement('button');
  pendingButton.textContent = 'Connect W&B';
  el('run-detail-actions').append(pendingButton);
  let finishAttachment;
  w.api = path => path.endsWith('/attach')
    ? new Promise(resolve => { finishAttachment = resolve; })
    : Promise.resolve({run: {id: 'other-run', status: 'CANCELLED', attempts: []}});
  const pendingAttachment = w.attachRunTracking('new-run', 'wandb', '/api/runs/new-run/tracking/wandb/attach', pendingButton);
  await w.viewRun('other-run', el('runs-body').querySelector('[data-id="other-run"][data-run-action="view"]'));
  const selectedDetail = el('run-detail-meta').textContent;
  finishAttachment({run: {id: 'new-run', status: 'CANCELLED', attempts: [], tracking_links: [{provider: 'wandb', url: 'https://example.com/original-run'}]}});
  await pendingAttachment;
  assert.equal(el('run-detail-title').textContent, 'other-run', 'late attachment must not replace the selected title');
  assert.equal(el('run-detail-meta').textContent, selectedDetail, 'late attachment must not replace the selected run');
  assert.doesNotMatch(el('run-detail-tracking').textContent, /original-run/);

  el('close-run-detail').click();

  // Filters changed during a slow refresh cannot hide the run just submitted.
  const opened = [];
  w.activateTab = async () => {};
  w.viewRun = async id => opened.push(id);
  let releaseList;
  w.loadRuns = () => new Promise(resolve => { releaseList = () => { w.audit.setRuns([{id: 'created-run', status: 'SUBMITTED'}]); resolve(); }; });
  const submission = w.openSubmittedTrainingRun({experimentId: 'experiment', responses: [{runs: [{id: 'created-run'}]}]});
  await flush();
  el('run-search').value = 'unrelated';
  el('run-status-filter').value = 'failed';
  releaseList();
  await submission;
  assert.deepEqual(opened, ['created-run']);
  assert.equal(el('run-search').value, '');
  assert.equal(el('run-status-filter').value, 'all');
  assert.equal(el('runs-error').hidden, true);

  // Tutorials do not ask for disabled fields; epoch-based adapters use their actual control.
  const steps = w.audit.tutorialSteps();
  el('hp-learning-rate').disabled = true;
  assert.equal(steps.find(step => step.id === 'learning-rate').skipIf(), true);
  el('hp-max-steps').disabled = true;
  el('adapter-declared-fields').insertAdjacentHTML('beforeend', '<input data-adapter-input-path="native.config.epochs">');
  const durationStep = steps.find(step => step.id === 'max-steps');
  assert.equal(durationStep.skipIf(), false);
  assert.equal(w.document.querySelector(durationStep.selector()).dataset.adapterInputPath, 'native.config.epochs');
  assert.equal(durationStep.useValue(), '1');
  const token = w.tutorialToken();
  assert.equal(token, token.toLowerCase(), 'tutorial identity survives backend slug normalization');

  el('hp-learning-rate').disabled = false;
  el('hp-learning-rate').value = '0';
  w.validateExperiment({notify: false});
  assert.equal(el('hp-learning-rate').validity.customError, true, 'invalid training values retain their error while an earlier step is incomplete');
  el('hp-learning-rate').value = '0.001';
  w.validateExperiment({notify: false});
  assert.equal(el('hp-learning-rate').validationMessage, '');
  const states = [...el('run-status-filter').options].map(option => option.value);
  for (const state of ['submitting', 'submitted', 'cancelling', 'retry_pending', 'submission_failed', 'blocked']) assert.ok(states.includes(state));
  w.audit.setRuns([{id: 'requeued-run', status: 'REQUEUED'}]);
  el('run-status-filter').value = 'requeued';
  w.renderRuns();
  assert.equal(el('runs-body').querySelector('[data-run-action="view"]').dataset.id, 'requeued-run', 'additional ledger states remain filterable');

  // Loaded historical origin remains distinct from the next immutable revision.
  w.hydrateExperimentConfiguration = async () => true;
  w.api = async () => ({experiment: {id: 'history', name: 'History', latest_revision_number: 6, revisions: [{revision_number: 2, requested_spec_json: {name: 'History'}}]}});
  w.updateBatchCompatibility = () => ({valid: true, errors: []});
  w.validateExperiment = () => true;
  w.experimentPayload = () => ({});
  w.matchingExperimentForPayload = () => ({id: 'history', latest_revision_number: 6});
  const successfulLoad = w.api;
  w.showToast('Unrelated action still needs attention', true, {scope: 'unrelated-action'});
  w.api = async () => { throw new Error('temporary configuration failure'); };
  await w.loadExperimentConfiguration('history', 2);
  assert.match(el('toast').textContent, /temporary configuration failure/);
  w.api = successfulLoad;
  await w.loadExperimentConfiguration('history', 2);
  assert.doesNotMatch(el('toast').textContent, /temporary configuration failure/);
  assert.match(el('toast').textContent, /Unrelated action still needs attention/);
  assert.match(el('experiment-revision-intent').textContent, /Loaded revision 2/);
  assert.match(el('experiment-revision-intent').textContent, /create revision 7/);
  w.api = async () => { throw new Error('temporary preview failure'); };
  await w.previewExperiment();
  assert.match(el('toast').textContent, /temporary preview failure/);
  assert.match(el('toast').textContent, /Unrelated action still needs attention/);
  w.api = async () => ({script: '#!/bin/bash\ntrue', scripts: ['#!/bin/bash\ntrue'], variant_count: 1, blockers: [], warnings: [], resolved_revision: commit});
  await w.previewExperiment();
  assert.doesNotMatch(el('toast').textContent, /temporary preview failure/);
  assert.match(el('toast').textContent, /Unrelated action still needs attention/);
  console.log('Training audit UI: task selection/defaults, validation attribution, terminal refresh, adapter cache restoration/stale responses, tracking reset and submission filter race passed.');
} finally {
  for (const observer of observers) observer.disconnect();
  w.close();
}
