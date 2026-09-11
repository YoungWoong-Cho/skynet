/* Shared dataset preparation and management. All entry points use this controller. */
(() => {
  "use strict";
  const el = (id) => document.getElementById(id);
  const dialog = el("policy-export-dialog"),
    detail = el("prepared-dataset-dialog");
  if (!dialog || !detail) return;
  let catalogSequence = 0,
    catalogState = "loading";
  function catalogStatus(state, error) {
    catalogState = state;
    if (state !== "ready" && dialog.open) el("create-policy-export").disabled = true;
    document.dispatchEvent(new CustomEvent("dataset-preparation-status", {
      detail: {state, error},
    }));
  }
  const request = async (path = "", options = {}) => {
    const catalog = path === "" && (!options.method || options.method === "GET");
    const token = catalog ? ++catalogSequence : null;
    if (catalog) catalogStatus("loading");
    try {
      const result = await api("/api/data/exports" + path, options);
      if (catalog && token === catalogSequence) await updateSnapshot(result);
      if (catalog && token === catalogSequence) catalogStatus("ready");
      return result;
    } catch (e) {
      if (catalog && token === catalogSequence) catalogStatus("unavailable", e.message);
      throw e;
    }
  };
  const retryPreparation = document.createElement("button");
  retryPreparation.type = "button";
  retryPreparation.className = "button button-outline";
  retryPreparation.textContent = "Try again";
  retryPreparation.hidden = true;
  retryPreparation.dataset.preparationLoadRetry = "";
  el("policy-export-error").after(retryPreparation);
  const esc = escapeHtml;
  let snapshot = null,
    registrySignature = null,
    resourceId = null,
    selectedResource = null,
    sourceSessionId = null;
  let busy = false,
    timer = null,
    generation = 0,
    detailGeneration = 0,
    refreshPromise = null;
  const pendingDeletions = new Map(), deletionErrors = new Map();
  const deletionButtonDefaults = new WeakMap();
  let deletionQueue = Promise.resolve();
  const terminal = (job) => ["READY", "FAILED", "DELETE_FAILED"].includes(job.state);
  const recipe = (id) => snapshot?.policies.find((p) => p.id === id);
  function error(id, message) {
    el(id).textContent = message || "";
    el(id).hidden = !message;
  }
  function sourceNote() {
    const policy = recipe(el("policy-export-format").value);
    const source = snapshot?.sessions.find((s) => s.id === sourceSessionId);
    const hint = el("policy-export-format-help");
    hint.textContent = policy?.available && !policy.trainable ? policy.description : "";
    hint.hidden = !hint.textContent;
    const missingSplit =
      policy?.trainable &&
      (source?.episodes < 2 ||
        Number(el("preparation-validation").value) === 0);
    const missingImages = policy?.observations?.includes("rgb") && source?.images < source?.episodes;
    const message = !source
      ? "Loading recordings…"
      : !source.eligible
        ? source.reason || "This session is not ready for preparation."
        : !policy?.available
          ? policy?.description || "Choose an available policy."
          : missingImages ? "This format requires completed training images for every recording."
          : missingSplit
            ? "Training needs at least two recordings and a validation split."
            : "";
    error("policy-export-compatibility", message);
    el("create-policy-export").disabled =
      busy || catalogState !== "ready" ||
      !source?.eligible ||
      !source.episodes ||
      !policy?.available ||
      missingSplit || missingImages;
  }
  const stageLabels = {
    QUEUED: "Queued",
    STAGING: "Preparing submission to sky2",
    SUBMITTING: "Submitting CPU job to sky2",
    ARCHIVING: "Waiting for the recording archive on sky2",
    FETCHING: "Checking originals",
    CONVERTING: "Converting",
    VALIDATING: "Validating",
    TRANSFERRING: "Copying to cluster",
    VERIFYING_COPY: "Checking cluster copy",
    CHECKING_LOADER: "Checking policy data loader",
    READY: "Prepared",
  };
  function jobStatus(job) {
    if (job.state === "DELETE_FAILED") return `Deletion incomplete · ${job.error}`;
    if (job.state === "FAILED") return `Failed · ${job.error}`;
    const progress = job.progress
      ? ` · ${job.progress.episodes_done}/${job.progress.episodes_total} episodes`
      : "";
    if (!terminal(job) && job.execution === "cluster") {
      const location = [job.gateway || "sky2", job.cluster_partition,
        job.cluster_job_id ? `Slurm job ${job.cluster_job_id}` : "Not yet assigned a Slurm job",
        job.cluster_cpus ? `${job.cluster_cpus} CPUs` : null].filter(Boolean).join(" · ");
      return `${job.detail || stageLabels[job.stage] || job.state} · ${location}${progress}${job.error ? ` · ${job.error}` : ""}`;
    }
    return (
      (terminal(job) ? job.detail : stageLabels[job.stage] || job.detail) +
      progress
    );
  }
  function renderHistory() {
    const jobs = (snapshot?.exports || []).filter(
      (j) => !terminal(j) || ["FAILED", "DELETE_FAILED"].includes(j.state),
    );
    el("policy-export-history").hidden = !jobs.length;
    el("policy-export-jobs").innerHTML = jobs
      .map(
        (j) =>
          `<div class="policy-export-job"><div><strong>${esc(j.name)}</strong> · ${esc(recipe(j.format)?.name || j.format)}<p>${esc(jobStatus(j))}</p></div><div class="row-actions"><button type="button" data-prepared-resource="${esc(j.resource_id || "")}">View dataset</button>${j.state === "FAILED" ? `<button type="button" class="button button-outline" data-preparation-retry="${esc(j.id)}">Retry</button>` : ""}</div></div>`,
      )
      .join("");
    syncDeletionControls();
    document.dispatchEvent(
      new CustomEvent("dataset-preparation-changed", {
        detail: snapshot.exports,
      }),
    );
  }
  function refresh() {
    if (!refreshPromise)
      refreshPromise = refreshNow().finally(() => {
        refreshPromise = null;
      });
    return refreshPromise;
  }
  async function refreshAfterMutation() {
    // A poll already in flight can contain rows from before the deletion.
    // Wait for it, then issue a new read instead of reusing its stale result.
    if (refreshPromise) await refreshPromise;
    await refresh();
  }
  function showDeletionErrors() {
    if (!detail.open) return;
    error("prepared-dataset-error", [...deletionErrors.values()]
      .filter(item => item.resourceId === selectedResource?.id)
      .map(item => item.message).join("\n"));
  }
  function syncDeletionControls() {
    for (const container of [el("prepared-dataset-content"), el("policy-export-jobs")]) {
      for (const button of container.querySelectorAll("button")) {
        const data = button.dataset;
        const jobId = data.preparationDeleteFormat || data.preparationRetry
          || data.preparationTransfer || data.preparationRemove || data.preparationTrain;
        const owner = data.preparationDelete
          || snapshot?.exports.find(job => job.id === jobId)?.resource_id;
        const pending = [...pendingDeletions.values()].find(item =>
          item.resourceId === owner && (item.wholeDataset || data.preparationDelete || item.jobId === jobId));
        if (pending) {
          if (!deletionButtonDefaults.has(button))
            deletionButtonDefaults.set(button, {disabled: button.disabled, text: button.textContent});
          button.disabled = true;
          if (data.preparationDeleteFormat || (data.preparationDelete && pending.wholeDataset)) {
            button.textContent = pending.label;
            button.setAttribute("aria-busy", "true");
          }
        } else if (deletionButtonDefaults.has(button)) {
          const original = deletionButtonDefaults.get(button);
          button.disabled = original.disabled;
          button.textContent = original.text;
          button.removeAttribute("aria-busy");
          deletionButtonDefaults.delete(button);
        }
      }
    }
  }
  async function deletePrepared(data) {
    const wholeDataset = Boolean(data.preparationDelete);
    const jobId = data.preparationDeleteFormat;
    const job = snapshot?.exports.find(item => item.id === jobId);
    const owner = data.preparationDelete || job?.resource_id;
    if (!owner) return;
    const path = wholeDataset ? `/datasets/${encodeURIComponent(owner)}` : `/${encodeURIComponent(jobId)}`;
    if (pendingDeletions.has(path) || [...pendingDeletions.values()].some(item =>
      item.resourceId === owner && (wholeDataset || item.wholeDataset))) return;
    const pending = {resourceId: owner, jobId, wholeDataset, label: "Confirming…"};
    pendingDeletions.set(path, pending);
    syncDeletionControls();
    try {
      const target = wholeDataset ? "this dataset’s prepared versions" : `this prepared version (${recipe(job.format)?.name || job.format})`;
      if (!(await askUserDialog(`Delete ${target} from this computer and the training cluster, and remove the registry entries? Original collection recordings will be kept.`))) return;
      deletionErrors.delete(path);
      showDeletionErrors();
      pending.label = "Queued…";
      syncDeletionControls();
      const operation = deletionQueue.then(async () => {
        pending.label = "Deleting…";
        syncDeletionControls();
        try {
          await request(path, {method: "DELETE"});
          if (wholeDataset && detail.open && selectedResource?.id === owner) {
            SkynetDialog.close(detail);
            selectedResource = null;
          }
          await refreshAfterMutation();
          showToast("Prepared data deleted. Original recordings kept.");
        } catch (e) {
          const message = `${wholeDataset ? "Dataset" : recipe(job.format)?.name || job.format}: ${e.message}`;
          deletionErrors.set(path, {resourceId: owner, message});
          await refreshAfterMutation();
          if (detail.open && selectedResource?.id === owner) showDeletionErrors();
          else showToast(message);
        }
      });
      // A failed deletion must not discard the other confirmed requests.
      deletionQueue = operation.catch(() => {});
      await operation;
    } finally {
      pendingDeletions.delete(path);
      syncDeletionControls();
    }
  }
  async function updateSnapshot(value) {
    snapshot = value;
    renderHistory();
    const signature = JSON.stringify(snapshot.exports.map(j => [
      j.id, j.resource_id, j.version_id, j.bundle_id, j.state, j.locations,
    ]));
    // A dialog can observe completion before the poll does. Compare against the
    // registry's last update, rather than the previous dialog/poll snapshot.
    if (signature !== registrySignature) {
      await loadDataRegistry(true);
      registrySignature = signature;
    }
  }
  async function refreshNow() {
    clearTimeout(timer);
    try {
      await request();
      if (dialog.open && el("policy-export-format").options.length) {
        error("policy-export-error", null);
        retryPreparation.hidden = true;
        sourceNote();
      }
      if (detail.open && selectedResource) await renderDataset();
      if (snapshot.exports.some((j) => !terminal(j)))
        timer = setTimeout(refresh, 3000);
    } catch (e) {
      if (dialog.open) {
        error("policy-export-error", e.message);
        el("create-policy-export").disabled = true;
        retryPreparation.hidden = false;
        retryPreparation.onclick = () => el("policy-export-format").options.length
          ? refresh() : window.openPolicyExport(sourceSessionId);
      }
      else if (detail.open) datasetLoadError(e.message);
      else {
        el("policy-export-history").hidden = false;
        el("policy-export-jobs").textContent = e.message;
      }
    }
  }
  window.openPolicyExport = async (sessionId) => {
    const token = ++generation;
    SkynetDialog.close(detail);
    sourceSessionId = sessionId;
    resourceId = null;
    error("policy-export-error", null);
    retryPreparation.hidden = true;
    retryPreparation.onclick = () => window.openPolicyExport(sessionId);
    el("policy-export-format").replaceChildren();
    el("policy-export-name").value = "";
    el("policy-export-name").readOnly = false;
    el("policy-export-name-help").textContent = "Name the dataset created for these recordings.";
    el("preparation-validation").value = 20;
    el("preparation-seed").value = 42;
    el("create-policy-export").disabled = true;
    error("policy-export-compatibility", "Loading recordings…");
    SkynetDialog.open(dialog);
    try {
      if (!sessionId)
        throw new Error("Open preparation from a recording session.");
      await request();
      if (token !== generation || !dialog.open) return;
      const source = snapshot.sessions.find((s) => s.id === sessionId);
      if (!source)
        throw new Error("This recording session is no longer available.");
      resourceId = source.resource_id || null;
      const resource = resourceId
        ? (await api(`/api/data/resources/${encodeURIComponent(resourceId)}`))
            .resource
        : null;
      if (token !== generation || !dialog.open) return;
      for (const policy of snapshot.policies) {
        const format = policy.container && !policy.name.toLowerCase().includes(policy.container.toLowerCase())
          ? ` · ${policy.container}` : "";
        el("policy-export-format").add(
          new Option(
            `${policy.name}${format}${!policy.available ? " · unavailable" : !policy.trainable ? " · export only" : ""}`,
            policy.id,
          ),
        );
      }
      el("policy-export-name").value =
        resource?.metadata?.display_name || resource?.name || source.name;
      el("policy-export-name").readOnly = Boolean(resource);
      el("policy-export-name-help").textContent = resource
        ? "Adds a prepared version to this existing dataset; its name and earlier versions are retained."
        : "Name the dataset created for these recordings.";
      sourceNote();
      renderHistory();
    } catch (e) {
      if (token === generation && dialog.open) {
        error("policy-export-compatibility", null);
        error("policy-export-error", e.message);
        retryPreparation.hidden = false;
      }
    }
  };
  function locationHtml(locations) {
    return locations.filter(location => location.path).map(location => {
      const host = location.kind === "local" ? "This computer"
        : location.host === "skynet" ? "Training cluster"
          : location.host || (location.kind === "cluster" ? "Training cluster" : "Recorded location");
      return `<div><strong>${esc(location.label || host)}</strong><span class="secondary">${esc(location.path)}</span><button type="button" class="text-button" data-dataset-copy-path="${esc(location.path)}">Copy path</button></div>`;
    }).join("") || "—";
  }
  function preparedRow(job) {
    const policy = recipe(job.format);
    const copies = (job.locations || []).filter(l => l.status === "AVAILABLE");
    const local = copies.some(l => l.kind === "local");
    const cluster = copies.some(l => l.kind === "cluster");
    const failed = ["FAILED", "DELETE_FAILED"].includes(job.state);
    const state = failed ? "FAILED" : job.state === "READY" ? (job.training_ready ? "READY" : "PREPARED")
      : ["QUEUED", "PENDING"].includes(job.state) ? "QUEUED"
        : job.state === "STAGING" ? "PREPARING SUBMISSION"
          : job.state === "SUBMITTING" ? "SUBMITTING"
            : job.state === "SUBMISSION_UNKNOWN" ? "CHECKING SUBMISSION" : "RUNNING";
    const actions = [];
    if (job.version_id && (local || (cluster && job.remote_archive))) {
      actions.push(`<a class="row-action-link" href="/api/data/exports/${encodeURIComponent(job.id)}/dataset.zip" download>Download</a>`);
      actions.push(`<a class="row-action-link" href="/api/data/exports/${encodeURIComponent(job.id)}/manifest.json" download>Manifest</a>`);
    }
    if (job.state === "FAILED") {
      if (job.execution !== "cluster" || job.cluster_job_id)
        actions.push(`<a class="row-action-link" href="/api/data/exports/${encodeURIComponent(job.id)}/export.log" download>Preparation log</a>`);
      actions.push(`<button type="button" class="text-button" data-preparation-retry="${esc(job.id)}">Retry</button>`);
    }
    if (job.state === "READY" && !cluster) actions.push(`<button type="button" class="text-button" data-preparation-transfer="${esc(job.id)}">Copy to cluster</button>`);
    if (job.training_ready && job.bundle_id) actions.push(`<button type="button" class="text-button" data-preparation-train="${esc(job.id)}">Use in experiment</button>`);
    if (local && cluster && !(job.usage || []).length && job.state === "READY") actions.push(`<button type="button" class="text-button" data-preparation-remove="${esc(job.id)}">Remove local copy</button>`);
    if (terminal(job)) actions.push(`<button type="button" class="text-button" data-preparation-delete-format="${esc(job.id)}" ${(job.usage || []).length ? 'disabled title="Used by an experiment"' : ""}>Delete</button>`);
    const statusNote = failed || !terminal(job) ? jobStatus(job) : policy?.trainable === false ? "Export only" : "";
    const location = locationHtml(copies);
    return `<tr><td><strong>${esc(policy?.name || job.format)}</strong>${policy?.container ? `<span class="secondary">${esc(policy.container)}</span>` : ""}</td><td>${statusPill(state)}${statusNote ? `<span class="secondary">${esc(statusNote)}</span>` : ""}</td><td class="wrap-cell">${location}</td><td>${job.episodes || job.sources?.length || 0}${job.split ? `<span class="secondary">${job.split.train.length} train / ${job.split.validation.length} validation</span>` : ""}</td><td>${esc(formatDate(job.created_at))}</td><td><div class="row-actions">${actions.join("")}</div></td></tr>`;
  }
  function sourceVersionRow(version) {
    const locations = (version.locations || []).filter(l => l.status === "AVAILABLE");
    const paths = locations.length ? locations : version.path ? [{kind: version.metadata?.storage_location || "unknown", path: version.path}] : [];
    const episodes = version.metadata?.episodes ?? version.metadata?.num_episodes ?? "—";
    return `<tr><td><strong>${esc(version.format || "Original data")}</strong><span class="secondary">${esc(version.revision?.slice(0, 12) || "")}</span></td><td>${statusPill(dataVersionStatus(version))}</td><td class="wrap-cell">${locationHtml(paths)}</td><td>${esc(episodes)}</td><td>${esc(formatDate(version.created_at))}</td><td>—</td></tr>`;
  }
  function datasetLoadError(message) {
    error("prepared-dataset-error", message);
    const content = el("prepared-dataset-content");
    if (content.textContent === "Loading dataset…")
      content.innerHTML = "<p>Dataset details could not be loaded.</p>";
    if (!content.querySelector("[data-prepared-load-retry]"))
      content.insertAdjacentHTML("beforeend", `<button type="button" class="button button-outline" data-prepared-load-retry data-prepared-resource="${esc(selectedResource.id)}">Try again</button>`);
  }
  async function renderDataset(token = ++detailGeneration) {
    try {
      await renderDatasetCurrent(token);
      if (token === detailGeneration && detail.open) showDeletionErrors();
    } catch (e) {
      if (token !== detailGeneration || !detail.open) return;
      datasetLoadError(e.message);
    }
  }
  async function renderDatasetCurrent(token) {
    const payload = await api(`/api/data/resources/${encodeURIComponent(selectedResource.id)}`);
    if (token !== detailGeneration || !detail.open) return;
    selectedResource = payload.resource;
    const r = selectedResource;
    const jobs = snapshot.exports.filter(j => j.resource_id === r.id);
    const versions = r.versions || [];
    const managed = r.metadata?.managed_dataset || jobs.length > 0;
    const originals = versions.filter(v => v.format === "skynet.episodes/v1");
    el("prepared-dataset-title").textContent = r.metadata?.display_name || r.name;
    el("prepared-dataset-context").textContent = managed
      ? `${jobs.filter(j => j.version_id).length} published versions · ${jobs.length} preparation attempts`
       : `${versions.length} version${versions.length === 1 ? "" : "s"}`;
    const rows = managed ? jobs.map(preparedRow) : versions.map(sourceVersionRow);
    const sourceRows = originals.map(v => {
      const sessions = [...new Set((v.metadata?.sources || []).map(source => source.session_id))];
      const locations = sessions.flatMap(id => snapshot.sessions.find(session => session.id === id)?.locations || []);
      if (!locations.length && v.path) locations.push({kind: "local", path: v.path, label: "Recording manifest"});
      return `<tr><td>${esc(v.revision.slice(0, 12))}</td><td class="wrap-cell">${locationHtml(locations)}</td><td>${v.metadata.episodes}</td><td>${v.metadata.split?.train.length || 0} train / ${v.metadata.split?.validation.length || 0} validation</td></tr>`;
    }).join("");
    const usage = [...new Map(jobs.flatMap(j => j.usage || []).map(u => [`${u.experiment_id}:${u.revision_number}`, u])).values()];
    el("prepared-dataset-content").innerHTML = `<div class="table-frame"><div class="table-scroll"><table class="prepared-dataset-table"><thead><tr>${["Format", "Status", "Location", "Episodes", "Date", "Actions"].map(label => `<th scope="col">${label}</th>`).join("")}</tr></thead><tbody>${rows.join("") || `<tr><td colspan="6">${managed ? "No prepared formats yet." : "No versions yet."}</td></tr>`}</tbody></table></div></div>
      ${usage.length ? `<p>Used by: ${usage.map(u => `<button type="button" class="text-button" data-preparation-experiment="${esc(u.experiment_id)}" data-revision="${u.revision_number}">${esc(u.name)} · revision ${u.revision_number}</button>`).join(" ")}</p>` : ""}
      ${managed ? `<details class="collection-disclosure"><summary>Original recordings and revisions</summary><div class="table-scroll"><table><thead><tr><th>Revision</th><th>Location</th><th>Episodes</th><th>Split</th></tr></thead><tbody>${sourceRows || '<tr><td colspan="4">Original recordings are preserved.</td></tr>'}</tbody></table></div></details>
      <div class="form-actions"><button type="button" class="button button-outline" data-preparation-delete="${esc(r.id)}" ${jobs.some(j => !terminal(j)) || usage.length ? 'disabled title="Preparation is active or a version is used by an experiment"' : ""}>Delete dataset</button></div>` : ""}`;
    syncDeletionControls();
  }
  window.openPreparedDataset = async (id) => {
    const token = ++detailGeneration;
    SkynetDialog.close(dialog);
    error("prepared-dataset-error", null);
    selectedResource = { id };
    el("prepared-dataset-title").textContent = "Dataset";
    el("prepared-dataset-context").textContent = "";
    el("prepared-dataset-content").textContent = "Loading dataset…";
    SkynetDialog.open(detail);
    try {
      await request();
      if (token === detailGeneration && detail.open) await renderDataset(token);
    } catch (e) {
      if (token === detailGeneration && detail.open) datasetLoadError(e.message);
    }
  };
  async function action(event) {
    const button = event.target.closest("button");
    if (!button || button.disabled) return;
    const data = button.dataset;
    try {
      if (data.preparationDelete || data.preparationDeleteFormat)
        return await deletePrepared(data);
      button.disabled = true;
      if (data.datasetCopyPath) {
        await navigator.clipboard.writeText(data.datasetCopyPath);
        showToast("Path copied.");
        return;
      }
      if (data.preparedResource)
        return await window.openPreparedDataset(data.preparedResource);
      if (data.preparationExperiment) {
        SkynetDialog.close(detail);
        await activateTab("experiments", true, "submit");
        await loadExperimentConfiguration(
          data.preparationExperiment,
          Number(data.revision),
        );
        return;
      }
      const id =
        data.preparationRetry ||
        data.preparationTransfer ||
        data.preparationRemove ||
        data.preparationTrain;
      if (!id) return;
      if (data.preparationTrain) {
        const job = snapshot.exports.find((j) => j.id === id);
        SkynetDialog.close(detail);
        await window.usePreparedDataset(job);
        return;
      }
      if (data.preparationRemove) {
        if (
          !(await askUserDialog(
            "Remove this computer’s prepared copy? The verified cluster copy and original recordings are retained.",
          ))
        )
          return;
        await request(`/${encodeURIComponent(id)}/local-copy`, {
          method: "DELETE",
        });
      } else
        await request(`/${encodeURIComponent(id)}/retry`, {
          method: "POST",
          body: JSON.stringify({}),
        });
      await refresh();
    } catch (e) {
      if (detail.open) error("prepared-dataset-error", e.message);
      else showToast(e.message);
    } finally {
      button.disabled = false;
      syncDeletionControls();
    }
  }
  dialog.addEventListener("close", () => generation++);
  detail.addEventListener("close", () => detailGeneration++);
  el("policy-export-format").onchange = sourceNote;
  el("preparation-validation").oninput = sourceNote;
  el("refresh-policy-exports").onclick = refresh;
  el("refresh-data-registry").addEventListener("click", refresh);
  el("policy-export-jobs").onclick = action;
  el("prepared-dataset-content").onclick = action;
  el("policy-export-form").onsubmit = async (event) => {
    event.preventDefault();
    if (
      busy ||
      el("create-policy-export").disabled ||
      !el("policy-export-form").reportValidity()
    )
      return;
    busy = true;
    const token = generation;
    sourceNote();
    error("policy-export-error", null);
    try {
      const job = await request("", {
        method: "POST",
        body: JSON.stringify({
          session_id: sourceSessionId,
          format: el("policy-export-format").value,
          name: el("policy-export-name").value.trim(),
          resource_id: resourceId,
          gateway: el("gateway").value,
          validation_percent: Number(el("preparation-validation").value),
          seed: Number(el("preparation-seed").value),
        }),
      });
      if (token === generation && dialog.open) {
        SkynetDialog.close(dialog);
        await window.openPreparedDataset(job.resource_id);
      }
      await refresh();
    } catch (e) {
      if (token === generation && dialog.open) error("policy-export-error", e.message);
    } finally {
      busy = false;
      sourceNote();
    }
  };
  document.addEventListener("collection-recordings-changed", refresh);
  document.addEventListener("dataset-preparation-refresh-requested", refresh);
  refresh();
})();
