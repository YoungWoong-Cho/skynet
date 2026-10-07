import assert from 'node:assert/strict';
import {pageWindow, byId, spyMutationObservers, stubBrowserApis, polyfillDialogs, loadScripts} from './ui_harness.mjs';

const w = pageWindow({url: 'http://localhost:8080/#runs'});
const observers = spyMutationObservers(w);
stubBrowserApis(w, {cssEscape: true});
polyfillDialogs(w, {guarded: false, returnValue: false, closeEvent: false});
const el = byId(w);
try {
  await loadScripts(w);
  const attempt = {id: 'attempt', attempt_number: 1, status: 'SUBMITTING', display_status: 'SUBMISSION UNCONFIRMED',
    created_at: '2026-09-11T06:13:49Z', slurm_reason: 'Submission outcome unknown: SSH operation timed out'};
  const run = {id: 'run', status: 'SUBMITTING', display_status: 'SUBMISSION UNCONFIRMED',
    status_detail: 'Connection failed before Slurm acceptance could be confirmed. Recover the existing submission or cancel it.',
    latest_attempt: attempt, attempts: [attempt], manual_actions: {recover_submission: {enabled: true}, cancel: {enabled: true}}};
  w.api = async path => path === '/api/runs' ? {runs: [run]} : path === '/api/runs/run?include_payloads=false' ? {run} : {content: ''};
  await w.loadRuns(true);
  assert.match(el('runs-body').textContent, /SUBMISSION UNCONFIRMED/);
  assert.doesNotMatch(el('runs-body').textContent, /Waiting for epoch data|SUBMITTING/);
  el('run-status-filter').value = 'submission unconfirmed';
  w.renderRuns();
  assert.match(el('runs-body').textContent, /SUBMISSION UNCONFIRMED/);
  await w.viewRun('run', el('runs-body').querySelector('button'));
  assert.match(el('run-detail-meta').textContent, /SUBMISSION UNCONFIRMED/);
  assert.equal(el('run-detail').dataset.runState, 'SUBMITTING', 'Internal state continues polling and prevents duplicate submission');
  assert.match(el('attempts-body').textContent, /SUBMISSION UNCONFIRMED/);
  assert.equal(el('attempts-body').rows[0].cells[4].textContent.trim(), '-', 'Created time is not a training start time');
  assert.equal(el('run-detail-actions').querySelector('[data-run-action="recover_submission"]').disabled, false);
  const button = el('attempts-body').querySelector('button');
  await w.toggleRunAttemptDetail(button.dataset.attemptKey, button);
  assert.match(el('run-attempt-detail-meta').textContent, /SUBMISSION UNCONFIRMED/);
  assert.match(w.evaluationRowCells({...run, id: 'eval'}).join(''), /SUBMISSION UNCONFIRMED/);
  w.renderEvaluationAttempt(attempt);
  assert.match(el('evaluation-attempt-meta').textContent, /SUBMISSION UNCONFIRMED/);
  console.log('Unconfirmed submission: consistent list, detail, attempt, evaluation, filter and recover action passed.');
} finally {
  for (const observer of observers) observer.disconnect();
  w.close();
}
