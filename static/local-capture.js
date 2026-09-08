/* Raw captures stay immutable; view selection and filters are local UI state. */
let localCaptureRows = [];
let localCaptureRequest = 0;
let localCaptureImporting = false;
let localCapturesAvailable = false;
function captureCanReplay(capture) {
  return capture.provider === 'visionpro-local' && capture.summary.tracked_hand_frames.right > 1;
}
function renderLocalCaptureRows() {
  const query = document.querySelector('#local-capture-search').value.trim().toLowerCase();
  const review = document.querySelector('#local-capture-review').checked;
  const rows = localCaptureRows.filter(capture => {
    const summary = capture.summary;
    return (!review || summary.warnings.length) && `${summary.header.task} ${formatDate(summary.header.created_at)} ${capture.sha256}`.toLowerCase().includes(query);
  });
  const reviewCount = localCaptureRows.filter(capture => capture.summary.warnings.length).length;
  document.querySelector('#local-capture-count').textContent = `${query || review ? `${rows.length} of ` : ''}${localCaptureRows.length} recordings${reviewCount ? ` · ${reviewCount} need review` : ''}`;
  document.querySelector('#collection-tracking-total').textContent = localCaptureRows.length;
  document.querySelector('#local-capture-body').innerHTML = rows.length ? rows.map(capture => {
    const summary = capture.summary;
    const hands = summary.tracked_hand_frames;
    const canReplay = captureCanReplay(capture);
    return `<tr data-capture-digest="${escapeHtml(capture.sha256)}">
      <td data-label="Recording" class="wrap-cell"><strong class="job-id">${escapeHtml(summary.header.task)}</strong><span class="secondary">${escapeHtml(formatDate(summary.header.created_at))}</span></td>
      <td data-label="Length">${Number(summary.duration_seconds).toFixed(1)} seconds<span class="secondary">${escapeHtml(summary.frames)} tracking update${summary.frames === 1 ? '' : 's'}</span></td>
      <td data-label="Tracking" class="wrap-cell"><span class="state-pill ${summary.warnings.length ? 'is-mixed' : 'is-idle'}">${summary.warnings.length ? 'Review tracking' : 'Both hands and head recorded'}</span>${summary.warnings.map(warning => `<p class="collection-quality-warning">${escapeHtml(warning)}</p>`).join('')}
        ${!canReplay ? '<p class="collection-quality-warning">DexVerse needs at least 2 right-hand updates.</p>' : ''}
        <details><summary>Tracking details</summary><p>Left hand: ${escapeHtml(hands.left)} updates<br>Right hand: ${escapeHtml(hands.right)} updates<br>Head tracked: ${escapeHtml(summary.head_tracked_frames)} updates</p><p class="secondary">Recording ID: ${escapeHtml(capture.sha256.slice(0, 12))}</p></details></td>
      <td data-label="Actions" class="row-actions"><div class="collection-record-actions"><button type="button" data-use-capture="${escapeHtml(capture.sha256)}" ${!canReplay || !localCapturesAvailable ? 'disabled' : ''}>Offline experiment</button><a class="row-action-link" href="/api/collection/local/captures/${encodeURIComponent(capture.sha256)}/download" download>Export original</a></div></td></tr>`;
  }).join('') : `<tr class="empty-row"><td colspan="4">${localCaptureRows.length ? 'No recordings match these filters. Clear the search or turn off “Needs review only”.' : 'No recordings yet. Open “How to record” to make one, or import an existing .jsonl file above.'}</td></tr>`;
}
async function loadLocalCollection(force = false, preferredDigest = null) {
  const request = ++localCaptureRequest;
  const errorBox = document.querySelector('#local-capture-error');
  const refresh = document.querySelector('#local-capture-refresh');
  refresh.disabled = true;
  errorBox.hidden = true;
  // Apply each response independently: an Xcode check cannot stall the library.
  const setup = api(`/api/collection/local/setup${force ? '?force=true' : ''}`).then(data => {
    if (request !== localCaptureRequest) return;
    document.querySelector('#local-capture-checks').innerHTML = data.checks.map(check => `<p><strong>${escapeHtml(check.name)} — ${escapeHtml(({ready: 'Ready', unsupported: 'Unsupported', needs_setup: 'Setup required', needs_user: 'Check on headset', needs_gpu: 'GPU host required'}[check.status] || check.status))}</strong><br>${escapeHtml(check.detail)}</p>`).join('');
    document.querySelector('#local-capture-storage').textContent = `Saved on this computer: ${data.storage_path}`;
  }).catch(error => {
    if (request !== localCaptureRequest) return;
    document.querySelector('#local-capture-checks').textContent = `Setup check failed: ${error.message}. You can still import and use saved recordings. Choose Check setup to retry.`;
  }).finally(() => { if (request === localCaptureRequest) refresh.disabled = false; });
  const captures = api('/api/collection/local/captures').then(data => {
    if (request !== localCaptureRequest) return;
    localCaptureRows = data.captures;
    localCapturesAvailable = true;
    if (typeof updateCaptureCycleRecordings === 'function') updateCaptureCycleRecordings(localCaptureRows, preferredDigest);
    renderLocalCaptureRows();
  }).catch(error => {
    if (request !== localCaptureRequest) return;
    localCapturesAvailable = false;
    renderLocalCaptureRows();
    document.querySelector('#capture-cycle-start').disabled = true;
    document.querySelector('#local-capture-count').textContent = 'Refresh failed';
    errorBox.textContent = `Recordings could not refresh: ${error.message}. Refresh data to retry; any listed recordings are from the last successful load.`;
    errorBox.hidden = false;
  });
  await Promise.all([setup, captures]);
}
function updateCaptureImportSelection() {
  const input = document.querySelector('#local-capture-file');
  const file = input.files[0];
  const status = document.querySelector('#local-capture-import-status');
  let error = '';
  if (file && !/\.jsonl$/i.test(file.name)) error = 'Choose a .jsonl tracking recording exported by Skynet Capture.';
  else if (file && (!file.size || file.size > 512 * 1024 * 1024)) error = file.size ? 'This file exceeds 512 MB. Record a shorter session.' : 'This file is empty. Choose a saved recording.';
  document.querySelector('#local-capture-upload').disabled = localCaptureImporting || !file || Boolean(error);
  status.dataset.state = error ? 'error' : 'info';
  status.textContent = error || (file ? `Ready to import ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB).` : 'Choose a .jsonl file to import. Your original will be preserved on this Mac.');
  return Boolean(file && !error);
}
document.querySelector('#local-capture-refresh').addEventListener('click', () => loadLocalCollection(true));
document.querySelector('#local-capture-search').addEventListener('input', renderLocalCaptureRows);
document.querySelector('#local-capture-review').addEventListener('change', renderLocalCaptureRows);
document.querySelector('#local-capture-body').addEventListener('click', event => {
  const button = event.target.closest('[data-use-capture]');
  if (button && !button.disabled) chooseCollectionRecording(button.dataset.useCapture);
});
document.querySelector('#local-capture-file').addEventListener('change', updateCaptureImportSelection);
document.querySelector('#local-capture-import').addEventListener('submit', async event => {
  event.preventDefault();
  if (localCaptureImporting || !updateCaptureImportSelection()) return;
  const input = document.querySelector('#local-capture-file');
  const status = document.querySelector('#local-capture-import-status');
  const button = document.querySelector('#local-capture-upload');
  const file = input.files[0];
  localCaptureImporting = true;
  button.disabled = true; input.disabled = true;
  status.textContent = 'Importing and checking tracking data…';
  try {
    const result = await api('/api/collection/local/captures/visionpro-local', {method: 'POST', headers: {'Content-Type': 'application/x-ndjson'}, body: file});
    status.dataset.state = 'success';
    status.textContent = `${result.imported ? 'Recording imported.' : 'This recording is already imported; no duplicate was created.'} ${captureCanReplay(result.capture) ? 'It is selected for your next DexVerse cycle.' : 'Stored as raw tracking. It needs more right-hand data for DexVerse.'}`;
    if (!localCaptureRows.some(capture => capture.sha256 === result.capture.sha256)) localCaptureRows.unshift(result.capture);
    localCapturesAvailable = true;
    updateCaptureCycleRecordings(localCaptureRows, result.capture.sha256);
    loadedTabs.delete('datasets');
    input.value = '';
    document.querySelector('#local-capture-search').value = '';
    document.querySelector('#local-capture-review').checked = false;
    await loadLocalCollection(false, result.capture.sha256);
    if (typeof loadCaptureCycles === 'function') await loadCaptureCycles();
  } catch (error) {
    status.dataset.state = 'error';
    status.textContent = `Import failed: ${error.message}`;
  } finally {
    localCaptureImporting = false;
    input.disabled = false;
    button.disabled = !input.files.length;
  }
});
if (!document.querySelector('#collection').hidden) loadLocalCollection();
