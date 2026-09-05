/* Native providers validate recordings on the server; the UI never invents sample data. */
async function loadLocalCollection(force = false) {
  const errorBox = document.querySelector('#local-capture-error');
  const refresh = document.querySelector('#local-capture-refresh');
  refresh.disabled = true;
  errorBox.hidden = true;
  try {
    const [setup, data] = await Promise.all([
      api(`/api/collection/local/setup${force ? '?force=true' : ''}`),
      api('/api/collection/local/captures'),
    ]);
    document.querySelector('#local-capture-checks').innerHTML = setup.checks.map(check =>
      `<p><strong>${escapeHtml(check.name)} — ${escapeHtml(({ready: 'Ready', unsupported: 'Unsupported', needs_setup: 'Setup required', needs_user: 'Check on headset', needs_gpu: 'GPU host required'}[check.status] || check.status))}</strong><br>${escapeHtml(check.detail)}</p>`
    ).join('');
    document.querySelector('#local-capture-storage').textContent = `Saved on this computer: ${setup.storage_path}`;
    document.querySelector('#local-capture-body').innerHTML = data.captures.length ? data.captures.map(capture => {
      const summary = capture.summary;
      const hands = summary.tracked_hand_frames;
      return `<tr><td>${escapeHtml(summary.header.task)}<span class="secondary">${escapeHtml(formatDate(summary.header.created_at))}</span></td>
        <td>${escapeHtml(summary.frames)} ${summary.frames === 1 ? 'frame' : 'frames'}<span class="secondary">${Number(summary.duration_seconds).toFixed(1)} seconds</span></td>
        <td>Left ${escapeHtml(hands.left)} · Right ${escapeHtml(hands.right)}<span class="secondary">Head tracked in ${escapeHtml(summary.head_tracked_frames)} ${summary.head_tracked_frames === 1 ? 'frame' : 'frames'}</span>${summary.warnings.map(warning => `<p>${escapeHtml(warning)}</p>`).join('')}</td>
        <td>This computer<span class="secondary">Raw tracking; conversion required for robot training</span></td>
        <td><a href="/api/collection/local/captures/${encodeURIComponent(capture.sha256)}/download" download>Export original</a><br><a href="#datasets">View dataset registry</a><span class="secondary">${escapeHtml(capture.sha256.slice(0, 12))}</span></td></tr>`;
    }).join('') : '<tr><td colspan="5">No recordings imported. Record on the headset, share the file to this Mac, then import it above.</td></tr>';
  } catch (error) {
    errorBox.textContent = `Collection setup could not refresh: ${error.message}. Displayed information may be out of date.`;
    errorBox.hidden = false;
  } finally { refresh.disabled = false; }
}
document.querySelector('#local-capture-refresh').addEventListener('click', () => loadLocalCollection(true));
document.querySelector('#local-capture-import').addEventListener('submit', async event => {
  event.preventDefault();
  const input = document.querySelector('#local-capture-file');
  const status = document.querySelector('#local-capture-import-status');
  const button = document.querySelector('#local-capture-upload');
  const file = input.files[0];
  if (!file) return;
  if (file.size > 512 * 1024 * 1024) { status.textContent = 'This file exceeds 512 MB. Record shorter sessions.'; return; }
  button.disabled = true; input.disabled = true;
  status.textContent = 'Importing and validating every frame…';
  try {
    const result = await api('/api/collection/local/captures/visionpro-local', {method: 'POST', headers: {'Content-Type': 'application/x-ndjson'}, body: file});
    status.textContent = result.imported ? `Imported ${result.capture.summary.frames} ${result.capture.summary.frames === 1 ? 'frame' : 'frames'}. Saved locally and registered as raw tracking; conversion is required before robot training.` : 'This exact recording is already imported. No duplicate was created.';
    loadedTabs.delete('datasets');
    input.value = '';
    await loadLocalCollection();
  } catch (error) { status.textContent = `Import failed: ${error.message}`; }
  finally { button.disabled = false; input.disabled = false; }
});
if (!document.querySelector('#collection').hidden) loadLocalCollection();
