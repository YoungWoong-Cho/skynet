// Run with node --test tests/browser_regressions.cjs. No packages or live jobs.
const { readFileSync } = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const { test } = require('node:test');
const source = readFileSync(require('node:path').join(__dirname, '../static/app.js'), 'utf8');
function load(names, globals = {}) {
  const context = vm.createContext(globals);
  for (const name of names) {
    const start = source.search(new RegExp(`^(?:async )?function ${name}\\(`, 'm'));
    assert.ok(start >= 0, name);
    const rest = source.slice(start);
    const end = rest.indexOf('\n}\n');
    vm.runInContext(rest.slice(0, end + 3), context);
  }
  return context;
}

test('Completed includes SUCCEEDED without including failed runs; Slurm search works', () => {
  const elements = { runSearch: { value: '' }, runStatusFilter: { value: 'completed' } };
  const c = load(['filteredRuns'], { elements, runRows: [
    { id: 'a', status: 'SUCCEEDED', latest_attempt: { slurm_job_id: '123' } },
    { id: 'b', status: 'COMPLETED' }, { id: 'c', status: 'FAILED' },
  ] });
  assert.equal(c.filteredRuns().length, 2);
  elements.runSearch.value = '123';
  assert.equal(c.filteredRuns()[0].id, 'a');
});
test('terminal progress is not described as waiting', () => {
  const c = load(['progressSummaryLabel']);
  assert.equal(c.progressSummaryLabel({ total: 100, unit: 'step', eta_state: 'complete' }), 'Recorded count unavailable');
  assert.equal(c.progressSummaryLabel({ total: 100, completed: 0 }), '0/100 items');
});
test('nested settings remain readable and sensitive values stay redacted', () => {
  const c = load(['flattenSettings'], { isSensitiveKey: (key) => key.includes('token') });
  const rows = c.flattenSettings({ profiles: [{ config: { nested: { queue: 'normal', token: 'secret' } } }] });
  assert.equal(rows.find(([key]) => key.endsWith('.queue'))[1], 'normal');
  assert.equal(rows.find(([key]) => key.endsWith('.token'))[1], 'configured');
  assert.ok(!JSON.stringify(rows).includes('secret'));
});
test('preview fingerprint expires when configuration changes', () => {
  let payload = { name: 'one', sweep: [1] };
  const c = load(['experimentPreviewIsCurrent'], { experimentPreviewSignature: JSON.stringify(payload), experimentPayload: () => payload });
  assert.equal(c.experimentPreviewIsCurrent(), true);
  payload = { name: 'one', sweep: [1, 2] };
  assert.equal(c.experimentPreviewIsCurrent(), false);
});
test('seed validation rejects malformed, empty and unsafe values', () => {
  const input = { value: '0,1,2', setCustomValidity(message) { this.message = message; } };
  const c = load(['evaluationSeedsValid'], { elements: { evaluationSeeds: input }, document: { querySelector: () => null } });
  assert.equal(c.evaluationSeedsValid(), true);
  for (const value of ['', '1,,2', '-1', '1.5', 'abc', '9007199254740992']) {
    input.value = value;
    assert.equal(c.evaluationSeedsValid(), false, value);
  }
});
test('submitted variant uses persisted latest attempt rather than NOT SUBMITTED', () => {
  const c = load(['normalizedRunState', 'runAttemptCount', 'runAttemptRecords', 'attemptHasSlurmSubmission', 'runHasExplicitPreflightFailure', 'variantRunDisplayState'], {
    explicitBoolean: () => null, firstValue: (...v) => v.find(x => x != null),
  });
  assert.equal(c.variantRunDisplayState({ status: 'FAILED', attempt_count: 1, latest_attempt: { slurm_job_id: '1', status: 'FAILED' } }), 'FAILED');
  assert.equal(c.variantRunDisplayState({ status: 'DRAFT', attempt_count: 0 }), 'NOT SUBMITTED');
});
test('adapter view uses the versions returned by the API', async () => {
  const field = () => ({ value: '', textContent: '' });
  const elements = Object.fromEntries(['adapterEditor','adapterEditorTitle','adapterEditorStatus','adapterEditorSlug','adapterEditorName','adapterManifest','adapterDescription','adapterChangeNote','adapterValidationRepository','adapterValidationRevision','experimentSource','experimentRevision'].map(name => [name, field()]));
  const versions = [{ version_number: 2 }];
  let rendered;
  let error;
  const c = load(['openAdapter'], { elements, currentRevealLauncher: () => null, disclosureToken: () => 1, disclosureTokenIsCurrent: () => true,
    api: async () => ({ adapter: { id: 'a', name: 'A', versions } }), entityFrom: (p,k) => p[k], adapterId: a => a.id,
    adapterManifest: () => ({}), repositoryDefaults: () => ({ url: '' }), formatManifest: JSON.stringify,
    renderAdapterVersions: v => { rendered = v; }, setAdapterEditorMode() {}, adapterArchived: () => false, revealPanel() {},
    closeDisclosurePanel() {}, showToast: message => { error = message; },
  });
  await c.openAdapter('a');
  assert.equal(error, undefined);
  assert.equal(rendered, versions);
});

test('submission feedback never reports failed transport as success', () => {
  const c = load(['normalizedRunState', 'submissionFeedback']);
  const result = c.submissionFeedback({ submitted: [{ run_id: 'run', status: 'FAILED', error: 'gateway unavailable' }] });
  assert.equal(result.error, true);
  assert.match(result.message, /gateway unavailable/);
  assert.match(c.submissionFeedback({ run_id: 'run', status: 'SUBMITTING' }).message, /still being confirmed/);
});
test('submission handoff accepts the actual backend submitted array', () => {
  const c = load(['submittedTrainingRunRecords', 'explicitlySubmittedTrainingRunRecords']);
  assert.equal(c.explicitlySubmittedTrainingRunRecords({ revision_number: 1, submitted: [{ run_id: 'new-run', status: 'FAILED' }] })[0].id, 'new-run');
});
test('bundle selection is unsupported without a declared binding', () => {
  const c = load(['experimentBundleCompatibility'], { selectedAdapter: () => ({}), declaredAdapterInputFields: () => [] });
  assert.equal(c.experimentBundleCompatibility({}).compatible, false);
});

test('zero quota and missing quota never invent a free GPU', () => {
  const c = load(['gpuAllocationCell'], { escapeHtml: String, userColor: () => '#000' });
  const zero = c.gpuAllocationCell({ usage: 0, limit: 0 });
  assert.match(zero, /No GPU quota/);
  assert.ok(!zero.includes('allocation-free-slot'));
  assert.ok(!c.gpuAllocationCell({ usage: 0 }).includes('allocation-free-slot'));
});
test('submission failures nested in revision responses stay visible', () => {
  const c = load(['normalizedRunState', 'submissionFeedback']);
  const feedback = c.submissionFeedback({ experiment: { runs: [{ id: 'new', status: 'FAILED', latest_attempt: { slurm_reason: 'resource check failed' } }] } });
  assert.equal(feedback.error, true);
  assert.match(feedback.message, /resource check failed/);
});

test('structured API errors retain their explanation and field location', () => {
  const c = load(['apiErrorMessage']);
  assert.equal(c.apiErrorMessage([{ loc: ['body', 'seeds', 0], msg: 'Must be an integer' }]), 'seeds.0: Must be an integer');
  assert.equal(c.apiErrorMessage({ code: 'UNSUPPORTED', message: 'This model has no compatible evaluator' }), 'This model has no compatible evaluator');
  assert.ok(!c.apiErrorMessage({ code: 'MISSING_RUNTIME', errors: ['Install runtime'] }).includes('[object Object]'));
});

test('bundle compatibility rejects local files and extra unconsumed data', () => {
  const c = load(['experimentBundleCompatibility'], { selectedAdapter: () => ({}), declaredAdapterInputFields: () => [{data_binding: {role:'training_data', position:0, formats:['lerobot-v2.0']}}] });
  const assignment = {role:'training_data', position:0, version:{format:'lerobot-v2.0',path:'/cluster/data',status:'READY'}};
  assert.equal(c.experimentBundleCompatibility({assignments:[assignment]}).compatible, true);
  assert.match(c.experimentBundleCompatibility({assignments:[{...assignment,version:{...assignment.version,metadata:{storage_location:'workstation'}}}]}).message, /collection workstation/);
  assert.equal(c.experimentBundleCompatibility({assignments:[{...assignment,version:{...assignment.version,status:'LOCAL'}}]}).compatible, false);
  assert.match(c.experimentBundleCompatibility({assignments:[assignment,{...assignment,position:1}]}).message, /cannot consume/);
  assert.match(c.experimentBundleCompatibility({assignments:[assignment,assignment]}).message, /duplicate/);
});

test('collection defaults preserve zero and false, respect gateway choice and require fresh evidence', () => {
  const c = load(['collectionSessionDefaults']);
  const manifest = {defaults: {config: {task: 'capture'}, capture: {timestamps_recorded: false, nominal_rate_hz: 90}, resources: {gateway: 'auto', gpu_count: 0}, capabilities: {stream: {status: 'verified'}}}, capabilities: [{id: 'stream', scope: 'network', default_status: 'UNKNOWN'}, {id: 'daemon', scope: 'compute_node', default_status: 'ADMIN_REQUIRED'}]};
  const before = JSON.stringify(manifest);
  const result = c.collectionSessionDefaults(manifest, 'sky1');
  assert.equal(result.resources.gateway, 'sky1');
  assert.equal(result.resources.gpu_count, 0);
  assert.equal(result.capture.timestamps_recorded, false);
  assert.equal(result.capabilities.stream.status, 'unknown');
  assert.equal(result.capabilities.daemon.status, 'admin_required');
  assert.equal(result.capabilities.stream.verified_by, null);
  result.config.task = 'changed';
  assert.equal(JSON.stringify(manifest), before);
});

test('a disconnected read expires with a useful error, while submission requests have no read timeout', async () => {
  let calls = 0;
  const c = load(['apiRequest'], {Headers, AbortController, setTimeout: cb => { queueMicrotask(cb); return 1; }, clearTimeout() {}, apiErrorMessage: String,
    fetch: async (path, options) => {
      calls++;
      if (options.method === 'POST') { assert.equal(options.signal, undefined); return {ok:true, text:async()=>'{"submitted":true}'}; }
      return new Promise((resolve,reject)=>options.signal.addEventListener('abort',()=>reject(new Error('aborted'))));
    },
  });
  await assert.rejects(c.apiRequest('/read'), /timed out after 60 seconds/);
  assert.equal((await c.apiRequest('/submit', {method:'POST'})).submitted, true);
  assert.equal(calls,2);
});


test('Slurm placeholder exit codes are not reported as final while an attempt runs', () => {
  const c = load(['runAttemptValue', 'attemptExitCodeLabel']);
  assert.equal(c.attemptExitCodeLabel({status: 'RUNNING', exit_code: '0:0'}), 'Available when the attempt ends');
  assert.equal(c.attemptExitCodeLabel({status: 'SUCCEEDED', exit_code: '0:0'}), '0:0');
  assert.equal(c.attemptExitCodeLabel({state: 'FAILED', exit_code: '1:0'}), '1:0');
});


test('queue waits are explained separately from failed attempts', () => {
  const c = load(['runAttemptValue', 'queueReasonLabel', 'attemptFailureDetail']);
  const pending = {status: 'PENDING', slurm_reason: 'QOSGrpGRES'};
  assert.match(c.queueReasonLabel(pending), /GPU quota/);
  assert.equal(c.attemptFailureDetail(pending), null);
  assert.equal(c.attemptFailureDetail({status: 'FAILED', slurm_reason: 'OutOfMemory'}), 'OutOfMemory');
  assert.equal(c.attemptFailureDetail({status: 'PENDING', error: 'Submission acknowledgement lost'}), 'Submission acknowledgement lost');
});


test('entering Collection refreshes recordings and cycle history even when adapter data is cached', async () => {
  const calls = [];
  const c = load(['loadCollection'], {loadedTabs: new Set(['collection']),
    loadLocalCollection: force => calls.push(['recordings', force]),
    loadCaptureCycles: () => calls.push(['cycles']),
  });
  await c.loadCollection();
  assert.deepEqual(calls, [['recordings', false], ['cycles']]);
});
