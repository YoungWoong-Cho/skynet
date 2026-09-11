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
  const attempt = {id: 'attempt', attempt_number: 1, status: 'SUBMITTING', display_status: 'SUBMISSION UNCONFIRMED',
    created_at: '2026-09-11T06:13:49Z', slurm_reason: 'Submission outcome unknown: SSH operation timed out'};
  const run = {id: 'run', status: 'SUBMITTING', display_status: 'SUBMISSION UNCONFIRMED',
    status_detail: 'Connection failed before Slurm acceptance could be confirmed. Recover the existing submission or cancel it.',
    latest_attempt: attempt, attempts: [attempt], manual_actions: {recover_submission: {enabled: true}, cancel: {enabled: true}}};
  w.api = async path => path === '/api/runs' ? {runs: [run]} : path === '/api/runs/run' ? {run} : {content: ''};
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
