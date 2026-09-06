/* The server owns job state and immutable request deduplication. */
let captureCycleTimer = null;
let captureCycleLoading = false;
let captureCycleSubmitting = false;
let captureCycleRecordings = [];
let latestCaptureCycleId = null;
let captureCyclePreviewSource = null;
const completedCaptureCycles = new Set();
function updateCaptureCycleRecordings(captures, preferredDigest = null) {
  captureCycleRecordings = captures;
  const select = document.querySelector('#capture-cycle-recording');
  const selected = preferredDigest || select.value;
  const compatible = captures.filter(captureCanReplay);
  select.innerHTML = '<option value="">Choose a recording…</option>' + captures.map(c => `<option value="${escapeHtml(c.sha256)}" ${captureCanReplay(c) ? '' : 'disabled'}>${escapeHtml(c.summary.header.task)} · ${escapeHtml(formatDate(c.summary.header.created_at))}${captureCanReplay(c) ? '' : ' · insufficient right-hand data'}</option>`).join('');
  select.value = compatible.some(c => c.sha256 === selected) ? selected : preferredDigest ? '' : compatible[0]?.sha256 || '';
  updateCaptureCycleSelection();
}
function updateCaptureCycleSelection() {
  const capture = captureCycleRecordings.find(c => c.sha256 === document.querySelector('#capture-cycle-recording').value);
  const summary = document.querySelector('#capture-cycle-selection');
  document.querySelector('#capture-cycle-start').disabled = captureCycleSubmitting || !capture || !captureCanReplay(capture) || !localCapturesAvailable;
  if (!capture) {
    summary.textContent = captureCycleRecordings.length ? 'Choose a recording with right-hand tracking. Unsupported recordings are listed but cannot be selected.' : 'Import a Vision Pro recording from the Recordings view to begin.';
    return;
  }
  const data = capture.summary;
  summary.innerHTML = `<strong>${Number(data.duration_seconds).toFixed(1)} seconds · ${escapeHtml(data.tracked_hand_frames.right)} right-hand updates</strong><span class="secondary">Recorded ${escapeHtml(formatDate(data.header.created_at))} · ID ${escapeHtml(capture.sha256.slice(0, 8))}</span>${data.warnings.length ? `<p class="collection-quality-warning">${data.warnings.map(escapeHtml).join(' ')} Right-hand replay is available; review the result before using it as a demonstration.</p>` : '<span class="secondary">Both hands and head were tracked.</span>'}`;
}

function captureArtifact(job, name) {
  return `/api/collection/processing/jobs/${encodeURIComponent(job.id)}/artifacts/${encodeURIComponent(name)}`;
}
function captureCycleStatus(state) {
  return {SUCCEEDED: 'Pipeline finished', FAILED: 'Failed', CANCELLED: 'Cancelled', RUNNING: 'Running', PENDING: 'Queued', PREPARING: 'Preparing', SUBMITTING: 'Submitting', SUBMISSION_UNKNOWN: 'Submission needs review'}[state] || state;
}
function renderCaptureCycle(job, expanded = false) {
  const active = !['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(job.state);
  const stages = ['simulation', 'training', 'evaluation'].map(name => {
    const stage = job.stages?.[name];
    if (!stage) return '<td><span class="secondary">Not started</span></td>';
    const details = [];
    if (stage.frames) details.push(`${stage.frames} frames`);
    if (stage.epoch) details.push(`Epoch ${stage.epoch}/${stage.total_epochs}`);
    if (stage.epochs) details.push(`${stage.epochs} epochs`);
    if (stage.task_success !== undefined) details.push(stage.task_success ? 'Task achieved' : 'Task not achieved');
    if (stage.successes !== undefined) details.push(`${stage.successes}/${stage.episodes} successful trials`);
    if (stage.detail) details.push(stage.detail);
    return `<td class="wrap-cell">${escapeHtml(details.join(' · ') || captureCycleStatus(stage.status))}${stage.status !== 'SUCCEEDED' && details.length ? `<span class="secondary">${escapeHtml(captureCycleStatus(stage.status))}</span>` : ''}</td>`;
  }).join('');
  const artifacts = job.result?.artifacts || {};
  const files = Object.keys(artifacts).filter(name => !name.endsWith('.mp4')).map(name => `<a href="${captureArtifact(job, name)}" download>${escapeHtml({'dataset-manifest.json': 'Dataset manifest', 'dataset.hdf5': 'Dataset (HDF5)', 'state-bc.pt': 'Policy checkpoint'}[name] || name)}</a>`).join('');
  const videos = Object.keys(artifacts).filter(name => name.endsWith('.mp4')).map(name => {
    const multiple = Object.keys(artifacts).filter(key => key.startsWith('evaluation-')).length > 1;
    const trial = Number(name.match(/evaluation-(\d+)/)?.[1]) + 1;
    const title = name === 'replay.mp4' ? 'Recording replay' : `Policy evaluation${multiple ? ` · trial ${trial}` : ''}`;
    const label = name === 'replay.mp4' ? 'Replay' : multiple ? `Trial ${trial} video` : 'Evaluation video';
    return `<button type="button" data-cycle-video="${escapeHtml(captureArtifact(job, name))}" data-video-title="${escapeHtml(title)}" data-video-context="${escapeHtml(job.name)} · ${escapeHtml(formatDate(job.created_at))} · Cycle ${escapeHtml(job.id.slice(0, 8))}" aria-label="Watch ${escapeHtml(title.toLowerCase())}">${escapeHtml(label)}</button>`;
  }).join('');
  const reason = job.state === 'PENDING' && job.scheduler?.Reason ? queueReasonLabel({slurm_reason: job.scheduler.Reason, status: job.state}) : '';
  const tone = ['FAILED', 'SUBMISSION_UNKNOWN'].includes(job.state) ? 'is-failed' : active ? 'is-running' : '';
  const notices = [
    {text: reason}, {text: job.preparation}, {text: job.error, error: true},
    {text: job.refresh_error ? `Status could not refresh: ${job.refresh_error}. Last known state is shown.` : '', error: true},
  ].filter(notice => notice.text);
  const detailId = `capture-cycle-detail-${job.id}`;
  return `<tr data-cycle-row="${escapeHtml(job.id)}">
    <td class="wrap-cell"><strong class="job-id">${escapeHtml(job.name)}</strong><span class="secondary">${escapeHtml(formatDate(job.created_at))} · ${escapeHtml(job.id.slice(0, 8))}</span></td>
    <td><span class="state-pill ${tone}">${escapeHtml(captureCycleStatus(job.state))}</span></td>
    ${stages}
    <td class="row-actions">${videos}<button type="button" class="disclosure-launcher" data-cycle-details-toggle="${escapeHtml(job.id)}" aria-controls="${escapeHtml(detailId)}" aria-expanded="${expanded}">${expanded ? 'Close' : 'Details'}</button>
      ${active && job.job_id ? `<button type="button" data-cycle-action="cancel" data-cycle-id="${escapeHtml(job.id)}" ${job.cancellation_requested ? 'disabled' : ''}>${job.cancellation_requested ? 'Cancellation requested' : 'Cancel cycle'}</button>` : ''}
      ${['FAILED', 'SUBMISSION_UNKNOWN'].includes(job.state) && !job.job_id ? `<button type="button" data-cycle-action="recover" data-cycle-id="${escapeHtml(job.id)}">Recover submission / retry setup</button>` : ''}
    </td></tr>
    ${notices.length ? `<tr class="collection-cycle-notices"><td colspan="6">${notices.map(notice => `<p class="${notice.error ? 'inline-alert' : 'secondary'}" role="${notice.error ? 'alert' : 'status'}">${escapeHtml(notice.text)}</p>`).join('')}</td></tr>` : ''}
    <tr id="${escapeHtml(detailId)}" class="row-disclosure-row" data-cycle-details="${escapeHtml(job.id)}" ${expanded ? '' : 'hidden'}><td colspan="6" class="row-disclosure-cell"><div class="row-disclosure">
      <div class="panel-heading row-disclosure-heading"><h2>Files and job details</h2><button type="button" class="text-button" data-cycle-details-close="${escapeHtml(job.id)}">Close</button></div>
      <div class="row-disclosure-body">
        <p class="secondary">Cycle ${escapeHtml(job.id)} · Seed ${escapeHtml(job.config.seed)}${job.job_id ? ` · Cluster job ${escapeHtml(job.job_id)}` : ''}</p>
        ${job.result ? `<p class="collection-section-help">Dataset: ${escapeHtml(job.result.dataset.frames)} frames. Source replay: ${job.result.dataset.capture_success ? 'task achieved' : 'task not achieved'}. Policy: state-based behavior cloning.</p><div class="collection-file-links">${files}<a href="#datasets">Open dataset registry</a></div>` : ''}
        ${job.root ? `<p><a href="/api/collection/processing/jobs/${encodeURIComponent(job.id)}/logs" target="_blank" rel="noopener">Read job log</a></p>` : ''}
        ${job.state === 'FAILED' && job.job_id ? '<p>This attempt is preserved. After resolving the error, change the scene seed to start a new cycle.</p>' : ''}
      </div>
    </div></td></tr>`;
}
async function loadCaptureCycles() {
  if (captureCycleLoading) return;
  captureCycleLoading = true;
  clearTimeout(captureCycleTimer);
  try {
    const response = await api('/api/collection/processing/jobs');
    const jobs = await Promise.all(response.jobs.map(job => ['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(job.state) ? job : api(`/api/collection/processing/jobs/${encodeURIComponent(job.id)}`)));
    document.querySelector('#collection-cycle-total').textContent = `${jobs.length}${jobs.some(job => !['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(job.state)) ? ' · active' : ''}`;
    document.querySelector('#capture-cycle-history-error').hidden = true;
    for (const job of jobs) {
      if (job.state === 'SUCCEEDED' && !completedCaptureCycles.has(job.id)) {
        completedCaptureCycles.add(job.id);
        loadedTabs.delete('datasets');
      }
    }
    const latest = jobs.find(job => job.id === latestCaptureCycleId);
    if (latest) document.querySelector('#capture-cycle-message').textContent = `Cycle ${latest.id.slice(0, 8)}: ${captureCycleStatus(latest.state)}. This cycle is shown below.`;
    const container = document.querySelector('#capture-cycle-jobs');
    const expanded = new Set([...container.querySelectorAll('[data-cycle-details]:not([hidden])')].map(node => node.dataset.cycleDetails));
    container.innerHTML = jobs.map(job => renderCaptureCycle(job, expanded.has(job.id))).join('') || '<tr class="empty-row"><td colspan="6">No cycles yet. Import a recording, then run your first cycle above.</td></tr>';
    if (jobs.some(job => !['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(job.state))) {
      captureCycleTimer = setTimeout(() => { if (!document.hidden && !document.querySelector('#collection').hidden) loadCaptureCycles(); }, 10000);
    }
  } catch (error) {
    const errorBox = document.querySelector('#capture-cycle-history-error');
    errorBox.textContent = `Cycle history could not refresh: ${error.message}. Refresh data to retry.`;
    errorBox.hidden = false;
    document.querySelector('#collection-cycle-total').textContent = 'Refresh needed';
    if (!document.querySelector('#capture-cycle-jobs [data-cycle-row]')) document.querySelector('#capture-cycle-jobs').innerHTML = '<tr class="empty-row"><td colspan="6">Cycle history is unavailable.</td></tr>';
    captureCycleTimer = setTimeout(() => { if (!document.hidden && !document.querySelector('#collection').hidden) loadCaptureCycles(); }, 15000);
  } finally { captureCycleLoading = false; }
}
document.querySelector('#capture-cycle-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (captureCycleSubmitting) return;
  captureCycleSubmitting = true;
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
  finally { captureCycleSubmitting = false; updateCaptureCycleSelection(); }
});
document.querySelector('#capture-cycle-jobs').addEventListener('click', async event => {
  const detailsButton = event.target.closest('[data-cycle-details-toggle], [data-cycle-details-close]');
  if (detailsButton) {
    const id = detailsButton.dataset.cycleDetailsToggle || detailsButton.dataset.cycleDetailsClose;
    const row = document.getElementById(`capture-cycle-detail-${id}`);
    const opening = Boolean(detailsButton.dataset.cycleDetailsToggle) && row.hidden;
    row.hidden = !opening;
    const launcher = document.querySelector(`[data-cycle-details-toggle="${CSS.escape(id)}"]`);
    launcher.setAttribute('aria-expanded', String(opening));
    launcher.textContent = opening ? 'Close' : 'Details';
    if (opening) {
      row.querySelector('[data-cycle-details-close]').focus({preventScroll: true});
      row.scrollIntoView({block: 'nearest'});
    } else launcher.focus({preventScroll: true});
    return;
  }
  const video = event.target.closest('[data-cycle-video]');
  if (video) {
    captureCyclePreviewSource = video.dataset.cycleVideo;
    const dialog = document.querySelector('#capture-cycle-preview');
    document.querySelector('#capture-cycle-preview-context').textContent = video.dataset.videoContext;
    document.querySelector('#capture-cycle-video-status').textContent = 'Loading video from the cluster…';
    document.querySelector('#capture-cycle-video-retry').hidden = true;
    document.querySelector('#capture-cycle-preview-title').textContent = video.dataset.videoTitle;
    if (!dialog.open) dialog.showModal();
    document.querySelector('#capture-cycle-video').src = video.dataset.cycleVideo;
    document.querySelector('#capture-cycle-close-preview').focus();
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
  document.querySelector('#capture-cycle-video-status').textContent = 'Video could not load. Check the cluster connection, then choose Retry video.';
  document.querySelector('#capture-cycle-video-retry').hidden = false;
});
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
function closeCaptureCycleVideo() {
  document.querySelector('#capture-cycle-preview').close();
}
document.querySelector('#capture-cycle-close-preview').addEventListener('click', closeCaptureCycleVideo);
document.querySelector('#capture-cycle-preview').addEventListener('close', () => {
  const video = document.querySelector('#capture-cycle-video');
  video.pause(); video.removeAttribute('src'); video.load();
  [...document.querySelectorAll('[data-cycle-video]')].find(button => button.dataset.cycleVideo === captureCyclePreviewSource)?.focus({preventScroll: true});
});
document.querySelector('#capture-cycle-video').addEventListener('loadedmetadata', event => {
  document.querySelector('#capture-cycle-video-status').textContent = `${event.target.duration.toFixed(1)} seconds · Use the play button to watch.`;
  document.querySelector('#capture-cycle-video-retry').hidden = true;
});
document.querySelector('#capture-cycle-video-retry').addEventListener('click', () => {
  document.querySelector('#capture-cycle-video-status').textContent = 'Retrying video from the cluster…';
  document.querySelector('#capture-cycle-video-retry').hidden = true;
  document.querySelector('#capture-cycle-video').src = captureCyclePreviewSource;
  document.querySelector('#capture-cycle-video').load();
});
document.querySelector('#capture-cycle-recording').addEventListener('change', updateCaptureCycleSelection);

document.querySelector('#capture-cycle-jobs').addEventListener('keydown', event => {
  if (event.key !== 'Escape') return;
  const details = event.target.closest('[data-cycle-details]');
  if (details && !details.hidden) {
    event.preventDefault();
    details.querySelector('[data-cycle-details-close]').click();
  }
});
