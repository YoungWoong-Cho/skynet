/* The server owns job state and immutable request deduplication. */
let captureCycleTimer = null;
let captureCycleLoading = false;
let latestCaptureCycleId = null;
const completedCaptureCycles = new Set();
function updateCaptureCycleRecordings(captures) {
  const select = document.querySelector('#capture-cycle-recording');
  const selected = select.value;
  const compatible = captures.filter(c => c.provider === 'visionpro-local' && c.summary.tracked_hand_frames.right > 1);
  select.innerHTML = compatible.map(c => `<option value="${escapeHtml(c.sha256)}">${escapeHtml(c.summary.header.task)} · ${escapeHtml(formatDate(c.summary.header.created_at))} · ${escapeHtml(c.sha256.slice(0, 8))}</option>`).join('');
  if (compatible.some(c => c.sha256 === selected)) select.value = selected;
  document.querySelector('#capture-cycle-start').disabled = !compatible.length;
  if (!compatible.length) document.querySelector('#capture-cycle-message').textContent = 'Import a recording with right-hand tracking to begin.';
}
function captureArtifact(job, name) {
  return `/api/collection/processing/jobs/${encodeURIComponent(job.id)}/artifacts/${encodeURIComponent(name)}`;
}
function renderCaptureCycle(job) {
  const active = !['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(job.state);
  const stageLabels = {simulation: 'Simulation and dataset', training: 'Training', evaluation: 'Evaluation'};
  const stages = ['simulation', 'training', 'evaluation'].filter(name => job.stages?.[name]).map(name => {
    const stage = job.stages[name];
    let detail = stage.frames ? ` · ${stage.frames} frames` : stage.epoch ? ` · epoch ${stage.epoch}/${stage.total_epochs}` : '';
    if (stage.epochs) detail += ` · ${stage.epochs} epochs`;
    if (stage.detail) detail += ` · ${stage.detail}`;
    if (stage.task_success !== undefined) detail += ` · task ${stage.task_success ? 'succeeded' : 'not achieved'}`;
    if (stage.successes !== undefined) detail += ` · ${stage.successes}/${stage.episodes} successful trials`;
    return `<li>${escapeHtml(stageLabels[name] || name)}: ${escapeHtml(stage.status)}${escapeHtml(detail)}</li>`;
  }).join('');
  const artifacts = job.result?.artifacts || {};
  const links = Object.keys(artifacts).map(name => name.endsWith('.mp4')
    ? `<button type="button" class="button button-outline" data-cycle-video="${escapeHtml(captureArtifact(job, name))}" data-video-title="${escapeHtml(name)}">Watch ${name === 'replay.mp4' ? 'recording replay' : 'policy evaluation'}</button>`
    : `<a href="${captureArtifact(job, name)}" download>${escapeHtml(name)}</a>`).join(' · ');
  const reason = job.state === 'PENDING' && job.scheduler?.Reason ? queueReasonLabel({slurm_reason: job.scheduler.Reason, status: job.state}) : '';
  return `<article class="panel local-capture-panel"><h3>${escapeHtml(job.name)} · ${escapeHtml(job.state)}</h3>
    <p>Seed ${escapeHtml(job.config.seed)}${job.job_id ? ` · Cluster job ${escapeHtml(job.job_id)}` : ''}${reason ? ` · ${escapeHtml(reason)}` : ''}</p>
    ${job.preparation ? `<p>${escapeHtml(job.preparation)}</p>` : ''}
    ${job.error ? `<p role="alert">${escapeHtml(job.error)}</p>` : ''}
    ${job.refresh_error ? `<p role="alert">Status could not refresh: ${escapeHtml(job.refresh_error)}. Last known state is shown.</p>` : ''}
    <ul>${stages || '<li>Waiting for preparation and GPU availability.</li>'}</ul>
    ${job.result ? `<p>Dataset: ${escapeHtml(job.result.dataset.frames)} frames · ${job.result.dataset.capture_success ? 'successful task demonstration' : 'task not achieved in the source replay'}. Policy: state-based behavior cloning.</p><p>${links}</p><a href="#datasets">View derived dataset and original-recording lineage</a>` : ''}
    ${job.state === 'SUCCEEDED' ? `<p><button type="button" class="button button-accent" data-tensor-cycle="${escapeHtml(job.id)}" data-tensor-name="${escapeHtml(job.name)}" data-tensor-frames="${Number(job.result?.dataset?.frames) || 0}">Inspect model tensors</button></p>` : ''}
    <p>${job.root ? `<a href="/api/collection/processing/jobs/${encodeURIComponent(job.id)}/logs" target="_blank" rel="noopener">Read job log</a>` : ''}
    ${active && job.job_id ? `<button type="button" class="button button-outline" data-cycle-action="cancel" data-cycle-id="${escapeHtml(job.id)}" ${job.cancellation_requested ? 'disabled' : ''}>${job.cancellation_requested ? 'Cancellation requested' : 'Cancel cycle'}</button>` : ''}
    ${['FAILED', 'SUBMISSION_UNKNOWN'].includes(job.state) && !job.job_id ? `<button type="button" class="button button-outline" data-cycle-action="recover" data-cycle-id="${escapeHtml(job.id)}">Recover submission / retry setup</button>` : ''}</p>
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
    document.querySelector('#capture-cycle-jobs').innerHTML = jobs.map(renderCaptureCycle).join('') || '<p>No cycles yet. Choose a recording and run the full cycle above.</p>';
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
