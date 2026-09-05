/* A trace is immutable. Layer navigation uses only its already-loaded tensors. */
(() => {
  const element = id => document.getElementById(`tensor-${id}`);
  const state = {trace: null, step: 0, job: null, request: 0, controller: null};
  const number = value => Number(Number(value).toPrecision(6)).toString();
  const shape = values => escapeHtml(`[${values.join(', ')}]`);

  function tensorValues(tensor, heading) {
    const maximum = Math.max(...tensor.values.map(Math.abs), 1e-30);
    return `<section class="tensor-values"><h4>${escapeHtml(heading)} <code>${shape(tensor.shape)}</code></h4>
      <p>${escapeHtml(tensor.dtype)} · min ${number(tensor.stats.min)} · max ${number(tensor.stats.max)} · mean ${number(tensor.stats.mean)}</p>
      <div class="tensor-grid" role="list" tabindex="0" aria-label="${escapeHtml(heading)}: all ${tensor.values.length} values">${tensor.values.map((value, index) => `<div role="listitem" class="tensor-cell" style="background: ${value < 0 ? 'rgba(175,72,38,' : 'rgba(33,111,71,'}${(0.04 + Math.abs(value) / maximum * 0.19).toFixed(3)})" title="Feature ${index}: ${value}"><span>${index}</span><code>${number(value)}</code></div>`).join('')}</div>
      <p class="quiet-label">All ${tensor.values.length} values · row 0 (one sample), columns = features · scroll to inspect. Green ≥ 0, orange &lt; 0; intensity is scaled separately for each tensor.</p></section>`;
  }

  function render() {
    const trace = state.trace;
    if (!trace) return;
    const step = trace.steps[state.step];
    element('position').textContent = `Step ${state.step + 1} of ${trace.steps.length}`;
    element('current-label').textContent = `${step.title} · [${step.shape.join(', ')}]`;
    element('previous').disabled = state.step === 0;
    element('next').disabled = state.step === trace.steps.length - 1;
    // Preserve keyboard focus when navigating using the layer buttons.
    if (!element('steps').children.length) element('steps').innerHTML = trace.steps.map((item, index) => `<li><button type="button" data-tensor-step="${index}" aria-controls="tensor-step-detail">${index + 1}. ${escapeHtml(item.title)}<small>${shape(item.shape)}</small></button></li>`).join('');
    element('steps').querySelectorAll('button').forEach((button, index) => {
      if (index === state.step) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
    });
    const previous = trace.steps[state.step - 1];
    const parameters = step.parameters.map(parameter => `<li><strong>${escapeHtml(parameter.name)}</strong> <code>${shape(parameter.shape)}</code>${parameter.count ? ` · ${parameter.count.toLocaleString()} learned values` : ''}${parameter.values ? `<div class="tensor-parameter-values"><code>${parameter.values.map(number).join(', ')}</code></div>` : ''}</li>`).join('');
    const final = step.id === 'clamp' ? `<p><strong>${trace.changed_action_indices.length} of 28 actions were limited to their training range.</strong> ${trace.changed_action_indices.length ? `Changed feature indices: ${trace.changed_action_indices.join(', ')}.` : 'No values needed limiting for this frame.'}</p><details><summary>Compare with the demonstrated action for this dataset frame</summary><p>The demonstration is the training target, not the model prediction. This comparison does not measure task success. Mean absolute difference: ${number(trace.action_mae)} (mixed action units).</p><div class="table-scroll"><table><thead><tr><th>Action index</th><th>Model output</th><th>Demonstration</th></tr></thead><tbody>${step.values.map((value, index) => `<tr><td>${index}</td><td>${number(value)}</td><td>${number(trace.demonstrated_action[index])}</td></tr>`).join('')}</tbody></table></div></details>` : '';
    element('step-detail').innerHTML = `<h4 class="tensor-layer-title">${escapeHtml(step.title)}</h4><p>${escapeHtml(step.explanation)}</p><p class="tensor-formula"><code>${escapeHtml(step.formula)}</code> · ${previous ? `${shape(previous.shape)} → ` : ''}${shape(step.shape)}</p>
      <div class="tensor-pair">${previous ? tensorValues(previous, 'Input to this step') : ''}${tensorValues(step, previous ? 'Output of this step' : 'Input tensor')}</div>
      ${parameters ? `<details><summary>Parameters used in this step</summary><ul>${parameters}</ul></details>` : ''}${final}`;
    element('content').hidden = false;
  }

  async function load(frame) {
    if (!state.job) return;
    if (!Number.isInteger(frame) || frame < 0 || frame >= state.job.frames) {
      element('status').textContent = `Choose a whole-number frame from 0 to ${state.job.frames - 1}.`;
      return;
    }
    state.controller?.abort();
    state.controller = new AbortController();
    const request = ++state.request;
    state.trace = null; state.step = 0;
    element('content').hidden = true;
    element('steps').replaceChildren();
    element('frame').disabled = true; element('load').disabled = true;
    element('inspector').setAttribute('aria-busy', 'true');
    element('status').textContent = `Loading frame ${frame}: verifying the saved files and computing the model layers. The first load may take up to a minute; cached frames open faster.`;
    try {
      const trace = await api(`/api/collection/processing/jobs/${encodeURIComponent(state.job.id)}/tensor-trace?frame=${frame}`, {signal: state.controller.signal});
      if (request !== state.request) return;
      state.trace = trace;
      element('frame').value = String(trace.frame);
      element('provenance').textContent = `${trace.provenance} Dataset frame ${trace.frame} of ${trace.frame_count - 1} · source tracking index ${trace.source_index} · ${trace.parameter_count.toLocaleString()} learned parameters.`;
      element('checksums').textContent = `Checkpoint SHA-256: ${trace.checkpoint_sha256}\nDataset SHA-256: ${trace.dataset_sha256}`;
      element('status').textContent = `Frame ${trace.frame} loaded. Step buttons are ready; no cluster requests are needed to move between layers.`;
      render();
    } catch (error) {
      if (request !== state.request) return;
      element('status').textContent = `Could not load this frame. ${error.message} Use Load frame to try again.`;
    } finally {
      if (request === state.request) {
        element('frame').disabled = false; element('load').disabled = false;
        element('inspector').setAttribute('aria-busy', 'false');
      }
    }
  }

  document.getElementById('capture-cycle-jobs').addEventListener('click', event => {
    const button = event.target.closest('[data-tensor-cycle]');
    if (!button) return;
    state.job = {id: button.dataset.tensorCycle, name: button.dataset.tensorName, frames: Number(button.dataset.tensorFrames)};
    element('inspector').hidden = false;
    element('source').textContent = `${state.job.name} · Cycle ${state.job.id.slice(0, 8)} · State-based behavior cloning`;
    element('frame').max = String(state.job.frames - 1);
    element('frame').value = '0';
    element('frame-range').textContent = `${state.job.frames} frames available (0–${state.job.frames - 1}).`;
    element('title').focus({preventScroll: true});
    element('inspector').scrollIntoView({block: 'start'});
    load(0);
  });
  element('frame-form').addEventListener('submit', event => { event.preventDefault(); load(Number(element('frame').value)); });
  element('previous').addEventListener('click', () => { if (state.trace && state.step > 0) { state.step--; render(); } });
  element('next').addEventListener('click', () => { if (state.trace && state.step < state.trace.steps.length - 1) { state.step++; render(); } });
  element('steps').addEventListener('click', event => {
    const button = event.target.closest('[data-tensor-step]');
    if (button && state.trace) { state.step = Number(button.dataset.tensorStep); render(); }
  });
  element('close').addEventListener('click', () => {
    state.controller?.abort(); ++state.request;
    element('inspector').hidden = true;
    const trigger = [...document.querySelectorAll('[data-tensor-cycle]')].find(button => button.dataset.tensorCycle === state.job?.id);
    trigger?.focus();
  });
  element('download').addEventListener('click', () => {
    if (!state.trace) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(state.trace, null, 2)], {type: 'application/json'}));
    const anchor = document.createElement('a');
    anchor.href = url; anchor.download = `tensor-trace-${state.job.id.slice(0, 8)}-frame-${state.trace.frame}.json`;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
})();
