/* The server owns job state and immutable request deduplication. */
let captureCycleTimer = null;
let captureCycleLoading = false;
let latestCaptureCycleId = null;
let captureCyclePreviewSource = null;
const completedCaptureCycles = new Set();
function updateCaptureCycleRecordings(captures) {
  const select = document.querySelector('#capture-cycle-recording');
  const selected = select.value;
  const compatible = captures.filter(c => c.provider === 'visionpro-local' && c.summary.tracked_hand_frames.right > 1);
  select.innerHTML = compatible.map(c => `<option value="${escapeHtml(c.sha256)}">${escapeHtml(c.summary.header.task)} · ${escapeHtml(formatDate(c.summary.header.created_at))}</option>`).join('');
  if (compatible.some(c => c.sha256 === selected)) select.value = selected;
  document.querySelector('#capture-cycle-start').disabled = !compatible.length;
  const message = document.querySelector('#capture-cycle-message');
  const emptyMessage = 'Import a recording with right-hand tracking to begin.';
  if (!compatible.length) message.textContent = emptyMessage;
  else if (message.textContent === emptyMessage) message.textContent = '';
}
function captureArtifact(job, name) {
  return `/api/collection/processing/jobs/${encodeURIComponent(job.id)}/artifacts/${encodeURIComponent(name)}`;
}
function captureCycleStatus(state) {
  return {SUCCEEDED: 'Completed', FAILED: 'Failed', CANCELLED: 'Cancelled', RUNNING: 'Running', PENDING: 'Queued', PREPARING: 'Preparing', SUBMITTING: 'Submitting', SUBMISSION_UNKNOWN: 'Submission needs review'}[state] || state;
}
function renderCaptureCycle(job, expanded = false) {
  const active = !['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(job.state);
  const stageLabels = {simulation: 'Simulation & dataset', training: 'Training', evaluation: 'Evaluation'};
  const stages = ['simulation', 'training', 'evaluation'].filter(name => job.stages?.[name]).map(name => {
    const stage = job.stages[name];
    const details = [];
    if (stage.frames) details.push(`${stage.frames} frames`);
    if (stage.epoch) details.push(`Epoch ${stage.epoch}/${stage.total_epochs}`);
    if (stage.epochs) details.push(`${stage.epochs} epochs`);
    if (stage.task_success !== undefined) details.push(stage.task_success ? 'Task achieved' : 'Task not achieved');
    if (stage.successes !== undefined) details.push(`${stage.successes}/${stage.episodes} successful trials`);
    if (stage.detail) details.push(stage.detail);
    return `<li><span>${escapeHtml(stageLabels[name])}</span><strong>${escapeHtml(details.join(' · ') || captureCycleStatus(stage.status))}</strong>${stage.status !== 'SUCCEEDED' && details.length ? `<small>${escapeHtml(captureCycleStatus(stage.status))}</small>` : ''}</li>`;
  }).join('');
  const artifacts = job.result?.artifacts || {};
  const files = Object.keys(artifacts).filter(name => !name.endsWith('.mp4')).map(name => `<a href="${captureArtifact(job, name)}" download>${escapeHtml({'dataset-manifest.json': 'Dataset manifest', 'dataset.hdf5': 'Dataset (HDF5)', 'state-bc.pt': 'Policy checkpoint'}[name] || name)}</a>`).join('');
  const videos = Object.keys(artifacts).filter(name => name.endsWith('.mp4')).map(name => {
    const title = name === 'replay.mp4' ? 'Recording replay' : `Policy evaluation${Object.keys(artifacts).filter(key => key.startsWith('evaluation-')).length > 1 ? ` · trial ${Number(name.match(/evaluation-(\d+)/)?.[1]) + 1}` : ''}`;
    return `<button type="button" class="button button-outline" data-cycle-video="${escapeHtml(captureArtifact(job, name))}" data-video-title="${escapeHtml(title)}">Watch ${escapeHtml(title.toLowerCase())}</button>`;
  }).join('');
  const reason = job.state === 'PENDING' && job.scheduler?.Reason ? queueReasonLabel({slurm_reason: job.scheduler.Reason, status: job.state}) : '';
  const tone = job.state === 'SUCCEEDED' ? 'complete' : ['FAILED', 'SUBMISSION_UNKNOWN'].includes(job.state) ? 'error' : 'active';
  return `<article class="collection-cycle-card">
    <div class="collection-cycle-heading"><div><h4>${escapeHtml(job.name)}</h4><p>${escapeHtml(formatDate(job.created_at))} · Seed ${escapeHtml(job.config.seed)}</p></div><span class="collection-cycle-status ${tone}">${escapeHtml(captureCycleStatus(job.state))}</span></div>
    ${reason ? `<p>${escapeHtml(reason)}</p>` : ''}
    ${job.preparation ? `<p>${escapeHtml(job.preparation)}</p>` : ''}
    ${job.error ? `<p class="inline-alert" role="alert">${escapeHtml(job.error)}</p>` : ''}
    ${job.refresh_error ? `<p class="inline-alert" role="alert">Status could not refresh: ${escapeHtml(job.refresh_error)}. Last known state is shown.</p>` : ''}
    <ul class="collection-cycle-stages">${stages || '<li>Waiting for preparation and GPU availability.</li>'}</ul>
    <details class="collection-inline-details" data-cycle-details="${escapeHtml(job.id)}" ${expanded ? 'open' : ''}><summary>Results, videos &amp; job details</summary>
      <p class="collection-form-help">Cycle ${escapeHtml(job.id)}${job.job_id ? ` · Cluster job ${escapeHtml(job.job_id)}` : ''}</p>
      ${job.result ? `<p class="collection-form-help">Dataset: ${escapeHtml(job.result.dataset.frames)} frames. Source replay: ${job.result.dataset.capture_success ? 'task achieved' : 'task not achieved'}. Policy: state-based behavior cloning.</p><div class="collection-artifact-actions">${videos}</div><div class="collection-file-links">${files}<a href="#datasets">Dataset &amp; recording lineage</a></div>` : ''}
      ${job.root ? `<p><a href="/api/collection/processing/jobs/${encodeURIComponent(job.id)}/logs" target="_blank" rel="noopener">Read job log</a></p>` : ''}
    </details>
    ${active && job.job_id ? `<button type="button" class="button button-outline" data-cycle-action="cancel" data-cycle-id="${escapeHtml(job.id)}" ${job.cancellation_requested ? 'disabled' : ''}>${job.cancellation_requested ? 'Cancellation requested' : 'Cancel cycle'}</button>` : ''}
    ${['FAILED', 'SUBMISSION_UNKNOWN'].includes(job.state) && !job.job_id ? `<button type="button" class="button button-outline" data-cycle-action="recover" data-cycle-id="${escapeHtml(job.id)}">Recover submission / retry setup</button>` : ''}
    ${job.state === 'FAILED' && job.job_id ? '<p>This attempt is preserved. After resolving the error, change the scene seed to start a new cycle.</p>' : ''}</article>`;
}
async function loadCaptureCycles() {
  if (captureCycleLoading) return;
  captureCycleLoading = true;
  clearTimeout(captureCycleTimer);
  try {
    const response = await api('/api/collection/processing/jobs');
    const jobs = await Promise.all(response.jobs.map(job => ['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(job.state) ? job : api(`/api/collection/processing/jobs/${encodeURIComponent(job.id)}`)));
    for (const job of jobs) {
      if (job.state === 'SUCCEEDED' && !completedCaptureCycles.has(job.id)) {
        completedCaptureCycles.add(job.id);
        loadedTabs.delete('datasets');
      }
    }
    const latest = jobs.find(job => job.id === latestCaptureCycleId);
    if (latest) document.querySelector('#capture-cycle-message').textContent = `Cycle ${latest.id.slice(0, 8)}: ${latest.state}. The same recording, settings and pipeline version reopen this cycle without a duplicate.`;
    const container = document.querySelector('#capture-cycle-jobs');
    const expanded = new Set([...container.querySelectorAll('details[open][data-cycle-details]')].map(node => node.dataset.cycleDetails));
    container.innerHTML = jobs.map(job => renderCaptureCycle(job, expanded.has(job.id))).join('') || '<p class="collection-empty">No cycles yet. Import a recording, then run your first cycle above.</p>';
    if (jobs.some(job => !['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(job.state))) {
      captureCycleTimer = setTimeout(() => { if (!document.hidden && !document.querySelector('#collection').hidden) loadCaptureCycles(); }, 10000);
    }
  } catch (error) {
    document.querySelector('#capture-cycle-message').textContent = `Could not refresh processing jobs: ${error.message}`;
    captureCycleTimer = setTimeout(() => { if (!document.hidden && !document.querySelector('#collection').hidden) loadCaptureCycles(); }, 15000);
  } finally { captureCycleLoading = false; }
}
document.querySelector('#capture-cycle-form').addEventListener('submit', async event => {
  event.preventDefault();
  const button = document.querySelector('#capture-cycle-start');
  const message = document.querySelector('#capture-cycle-message');
  button.disabled = true; message.textContent = 'Preparing your cycle…';
  try {
    const job = await api('/api/collection/processing/jobs', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({
      capture_sha256: document.querySelector('#capture-cycle-recording').value,
      seed: Number(document.querySelector('#capture-cycle-seed').value), epochs: Number(document.querySelector('#capture-cycle-epochs').value),
      eval_episodes: Number(document.querySelector('#capture-cycle-episodes').value),
    })});
    latestCaptureCycleId = job.id;
    message.textContent = `Cycle ${job.id.slice(0, 8)}: ${job.state}. Repeating the same recording, settings and pipeline version reopens this cycle without submitting a duplicate.`;
    await loadCaptureCycles();
    document.querySelector('#capture-cycle-jobs').scrollIntoView({block: 'center'});
  } catch (error) { message.textContent = `Could not start the cycle: ${error.message}`; }
  finally { button.disabled = false; }
});
document.querySelector('#capture-cycle-jobs').addEventListener('click', async event => {
  const video = event.target.closest('[data-cycle-video]');
  if (video) {
    captureCyclePreviewSource = video.dataset.cycleVideo;
    document.querySelector('#capture-cycle-preview').hidden = false;
    document.querySelector('#capture-cycle-preview-title').textContent = video.dataset.videoTitle;
    document.querySelector('#capture-cycle-video').src = video.dataset.cycleVideo;
    document.querySelector('#capture-cycle-preview').scrollIntoView({block: 'center'});
    return;
  }
  const button = event.target.closest('[data-cycle-action]');
  if (!button) return;
  button.disabled = true;
  try {
    await api(`/api/collection/processing/jobs/${encodeURIComponent(button.dataset.cycleId)}/${button.dataset.cycleAction}`, {method: 'POST'});
    await loadCaptureCycles();
  } catch (error) { document.querySelector('#capture-cycle-message').textContent = error.message; button.disabled = false; }
});
document.querySelector('#capture-cycle-video').addEventListener('error', () => {
  document.querySelector('#capture-cycle-preview-title').textContent = 'Video could not load. Check the cluster connection and open the replay again.';
});
window.addEventListener('hashchange', () => { if (location.hash === '#collection') loadCaptureCycles(); });
document.addEventListener('visibilitychange', () => { if (!document.hidden && location.hash === '#collection') loadCaptureCycles(); });
if (location.hash === '#collection') loadCaptureCycles();

document.querySelector('#capture-cycle-setup').addEventListener('click', async event => {
  const button = event.currentTarget;
  const message = document.querySelector('#capture-cycle-setup-message');
  button.disabled = true; message.textContent = 'Verifying the pinned source, runtime packages and robot assets…';
  try {
    const result = await api('/api/collection/processing/setup', {method: 'POST'});
    message.textContent = result.detail;
  } catch (error) { message.textContent = `Setup could not complete: ${error.message}`; }
  finally { button.disabled = false; }
});

// Keep secondary controls compact without concealing invalid fields or changed settings.
document.querySelector('#capture-cycle-form').addEventListener('input', () => {
  const epochs = document.querySelector('#capture-cycle-epochs').value;
  const trials = document.querySelector('#capture-cycle-episodes').value;
  const seed = document.querySelector('#capture-cycle-seed').value;
  document.querySelector('#capture-cycle-settings-summary').textContent = `${epochs || '—'} epochs · ${trials || '—'} trial${trials === '1' ? '' : 's'} · seed ${seed || '—'}`;
});
document.querySelector('#capture-cycle-form').addEventListener('invalid', event => {
  const details = event.target.closest('details');
  if (details) details.open = true;
}, true);
document.querySelector('#capture-cycle-close-preview').addEventListener('click', () => {
  const video = document.querySelector('#capture-cycle-video');
  video.pause();
  video.removeAttribute('src');
  video.load();
  document.querySelector('#capture-cycle-preview').hidden = true;
  [...document.querySelectorAll('[data-cycle-video]')].find(button => button.dataset.cycleVideo === captureCyclePreviewSource)?.focus();
});
document.querySelector('#refresh-collection').addEventListener('click', loadCaptureCycles);
