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
    if (state !== "ready" && dialog.open)
      el("create-policy-export").disabled = true;
    document.dispatchEvent(
      new CustomEvent("dataset-preparation-status", {
        detail: { state, error },
      }),
    );
  }
  const request = async (path = "", options = {}) => {
    const catalog =
      path === "" && (!options.method || options.method === "GET");
    const token = catalog ? ++catalogSequence : null;
    if (catalog) catalogStatus("loading");
    try {
      const result = await api("/api/data/exports" + path, options);
      if (catalog && token === catalogSequence) await updateSnapshot(result);
      if (catalog && token === catalogSequence) catalogStatus("ready");
      return result;
    } catch (e) {
      if (catalog && token === catalogSequence)
        catalogStatus("unavailable", e.message);
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
  const expandedResults = new Set();
  let datasetDetailsOpen = false;
  const terminal = (job) =>
    ["READY", "FAILED", "DELETE_FAILED"].includes(job.state);
  const recipe = (id) => snapshot?.policies.find((p) => p.id === id);
  const splitLabel = (split) =>
    !split?.validation?.length
      ? `${split?.train?.length || 0} train · no validation`
      : split?.mode === "single_episode_overfit"
        ? "1 train · same episode for validation (overfit)"
        : `${split?.train.length || 0} train / ${split?.validation.length || 0} validation`;
  function error(id, message) {
    el(id).textContent = message || "";
    el(id).hidden = !message;
  }
  function sourceNote() {
    const policy = recipe(el("policy-export-format").value);
    const source = snapshot?.sessions.find((s) => s.id === sourceSessionId);
    for (const id of ["preparation-validation", "preparation-seed"]) {
      el(id).disabled = source?.episodes === 1;
      el(id).closest(".field").hidden = source?.episodes === 1;
    }
    const hint = el("policy-export-format-help");
    hint.textContent =
      policy?.available && !policy.trainable ? policy.description : "";
    hint.hidden = !hint.textContent;
    const noValidation =
      policy?.trainable &&
      (source?.episodes < 2 ||
        Number(el("preparation-validation").value) === 0);
    const missingImages =
      policy?.observations?.includes("rgb") &&
      source?.images < source?.episodes;
    const message = !source
      ? "Loading recordings…"
      : !source.eligible
        ? source.reason || "This session is not ready for preparation."
        : !policy?.available
          ? policy?.description || "Choose an available policy."
          : missingImages
            ? "This format requires completed training images for every recording."
            : "";
    error(
      "policy-export-compatibility",
      message ||
        (noValidation
          ? "Warning: no validation recordings. Training will run without validation steps."
          : ""),
    );
    el("policy-export-compatibility").classList.toggle(
      "is-warning",
      !message && noValidation,
    );
    el("create-policy-export").disabled =
      busy ||
      catalogState !== "ready" ||
      !source?.eligible ||
      !source.episodes ||
      !policy?.available ||
      missingImages;
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
    if (job.state === "DELETE_FAILED")
      return `Deletion incomplete · ${job.error}`;
    if (job.state === "FAILED") return `Failed · ${job.error}`;
    const progress = job.progress
      ? ` · ${job.progress.episodes_done}/${job.progress.episodes_total} episodes`
      : "";
    if (!terminal(job) && job.execution === "cluster") {
      const location = [
        job.gateway || "sky2",
        job.cluster_partition,
        job.cluster_job_id
          ? `Slurm job ${job.cluster_job_id}`
          : "Not yet assigned a Slurm job",
        job.cluster_cpus ? `${job.cluster_cpus} CPUs` : null,
      ]
        .filter(Boolean)
        .join(" · ");
      return `${job.detail || stageLabels[job.stage] || job.state} · ${location}${progress}${job.error ? ` · ${job.error}` : ""}`;
    }
    return (
      (terminal(job) ? job.detail : stageLabels[job.stage] || job.detail) +
      progress
    );
  }
  function renderHistory() {
    el("policy-export-history").hidden = true;
    el("policy-export-jobs").replaceChildren();
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
  async function updateSnapshot(value) {
    snapshot = value;
    renderHistory();
    const signature = JSON.stringify(
      snapshot.exports.map((j) => [
        j.id,
        j.resource_id,
        j.version_id,
        j.bundle_id,
        j.state,
        j.locations,
      ]),
    );
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
        retryPreparation.onclick = () =>
          el("policy-export-format").options.length
            ? refresh()
            : window.openPolicyExport(sourceSessionId);
      } else if (detail.open) datasetLoadError(e.message);
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
    el("policy-export-name-help").textContent = "Name this dataset.";
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
        const format =
          policy.container &&
          !policy.name.toLowerCase().includes(policy.container.toLowerCase())
            ? ` · ${policy.container}`
            : "";
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
        ? "Adds a prepared result to this dataset."
        : "Name this dataset.";
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
    return (
      locations
        .filter((location) => location.path)
        .map((location) => {
          const host =
            location.kind === "local"
              ? "This computer"
              : location.host === "skynet"
                ? "Training cluster"
                : location.host ||
                  (location.kind === "cluster"
                    ? "Training cluster"
                    : "Registered path");
          return `<div><strong>${esc(location.label || host)}</strong><span class="secondary">${esc(location.path)}</span><button type="button" class="text-button" data-dataset-copy-path="${esc(location.path)}">Copy path</button></div>`;
        })
        .join("") || "—"
    );
  }
  function resultSettings(result) {
    const metadata = result.metadata || {};
    const split = result.split || metadata.split;
    const pairs = [
      ["Recipe", result.contract || metadata.contract],
      ["Split seed", split?.seed],
      [
        "Validation",
        split
          ? split.validation?.length
            ? `${split.validation.length} episode${split.validation.length === 1 ? "" : "s"}`
            : "None"
          : null,
      ],
      ["Converter", result.converter_sha256 || metadata.converter_sha256],
      ["Source revision", result.source_revision || result.revision],
    ].filter(([, value]) => value != null && value !== "");
    return pairs.length
      ? `<dl class="dataset-result-settings">${pairs.map(([label, value]) => `<dt>${esc(label)}</dt><dd>${esc(value)}</dd>`).join("")}</dl>`
      : '<p class="secondary">No conversion settings recorded.</p>';
  }
  function resultRow({
    id,
    versionId,
    title,
    container,
    status,
    note,
    episodes,
    split,
    createdAt,
    storage,
    settings,
    primary = [],
    secondary = [],
  }) {
    const detailId = `dataset-result-${encodeURIComponent(id)}`;
    const expanded = expandedResults.has(id);
    const badge =
      status && !["READY", "PUBLISHED", "SUCCEEDED"].includes(status)
        ? statusPill(status)
        : "";
    const dataset = selectedResource?.category === "dataset";
    const presetIds = [
      ...new Set(
        (selectedResource?.experiment_presets || [])
          .filter((link) => link.version_id === versionId)
          .map((link) => link.experiment_id),
      ),
    ];
    const presets = dataset
      ? `<td><button type="button" class="text-button" data-dataset-presets="${esc(selectedResource.id)}" data-dataset-label="${esc(selectedResource.metadata?.display_name || selectedResource.name)}" data-preset-ids="${esc(JSON.stringify(presetIds))}">${presetIds.length} preset${presetIds.length === 1 ? "" : "s"}</button></td>`
      : "";
    const source =
      dataset && resourceRecordingIds(selectedResource).length
        ? `<button type="button" data-dataset-recordings="${esc(selectedResource.id)}">View source recordings</button>`
        : "";
    return `<tr data-dataset-result="${esc(id)}">
      <td class="wrap-cell"><strong>${esc(title)}</strong>${container ? `<span class="secondary">${esc(container)}</span>` : ""}${badge}${note ? `<span class="secondary">${esc(note)}</span>` : ""}</td>
      <td class="wrap-cell">${esc(episodes)}${split ? `<span class="secondary">${esc(splitLabel(split))}</span>` : ""}</td>
      ${presets}<td>${esc(formatDate(createdAt))}</td>
      <td class="row-actions">${primary.join("")}${source}<button type="button" data-dataset-result-toggle="${esc(id)}" aria-expanded="${expanded}" aria-controls="${esc(detailId)}">Details</button></td>
    </tr><tr id="${esc(detailId)}" class="dataset-result-details"${expanded ? "" : " hidden"}><td colspan="${dataset ? 5 : 4}" class="wrap-cell">
      <div class="dataset-result-metadata"><section><h4>Storage</h4>${storage}</section><section><h4>Conversion settings</h4>${settings}</section></div>
      ${secondary.length ? `<div class="form-actions">${secondary.join("")}</div>` : ""}
    </td></tr>`;
  }
  function preparedRow(job) {
    const policy = recipe(job.format);
    const copies = (job.locations || []).filter(
      (l) => l.status === "AVAILABLE",
    );
    const local = copies.some((l) => l.kind === "local");
    const cluster = copies.some((l) => l.kind === "cluster");
    const failed = ["FAILED", "DELETE_FAILED"].includes(job.state);
    const state = failed
      ? "FAILED"
      : job.state === "READY"
        ? job.training_ready
          ? "READY"
          : "PREPARED"
        : ["QUEUED", "PENDING"].includes(job.state)
          ? "QUEUED"
          : job.state === "STAGING"
            ? "PREPARING SUBMISSION"
            : job.state === "SUBMITTING"
              ? "SUBMITTING"
              : job.state === "SUBMISSION_UNKNOWN"
                ? "CHECKING SUBMISSION"
                : "RUNNING";
    const primary = [],
      secondary = [];
    if (job.version_id && (local || (cluster && job.remote_archive))) {
      secondary.push(
        `<a class="button button-outline" href="/api/data/exports/${encodeURIComponent(job.id)}/dataset.zip" download><span aria-hidden="true">↓</span> dataset.zip</a>`,
      );
      secondary.push(
        `<a class="button button-outline" href="/api/data/exports/${encodeURIComponent(job.id)}/manifest.json" download><span aria-hidden="true">↓</span> manifest.json</a>`,
      );
    }
    if (job.state === "FAILED") {
      if (job.execution !== "cluster" || job.cluster_job_id)
        secondary.push(
          `<a class="button button-outline" href="/api/data/exports/${encodeURIComponent(job.id)}/export.log" download>Preparation log</a>`,
        );
      primary.push(
        `<button type="button" data-preparation-retry="${esc(job.id)}">Retry</button>`,
      );
    }
    if (job.state === "READY" && !cluster)
      primary.push(
        `<button type="button" data-preparation-transfer="${esc(job.id)}">Copy to cluster</button>`,
      );
    if (job.training_ready && job.version_id)
      primary.push(
        `<button type="button" data-preparation-train="${esc(job.id)}">Use in experiment</button>`,
      );
    const statusNote =
      failed || !terminal(job)
        ? jobStatus(job)
        : policy?.trainable === false
          ? "Export only"
          : "";
    return resultRow({
      id: `job-${job.id}`,
      versionId: job.version_id,
      title: policy?.name || job.format,
      container: policy?.container,
      status: state,
      note: statusNote,
      episodes: job.episodes ?? job.sources?.length ?? "—",
      split: job.split,
      createdAt: job.created_at,
      storage: locationHtml(copies),
      settings: resultSettings(job),
      primary,
      secondary,
    });
  }
  function sourceVersionRow(version, dataset) {
    const locations = (version.locations || []).filter(
      (l) => l.status === "AVAILABLE",
    );
    const paths = locations.length
      ? locations
      : version.path
        ? [
            {
              kind: version.metadata?.storage_location || "unknown",
              path: version.path,
            },
          ]
        : [];
    const episodes = Array.isArray(version.metadata?.episodes)
      ? version.metadata.episodes.length
      : (version.metadata?.num_episodes ?? version.metadata?.episodes ?? "—");
    return resultRow({
      id: `version-${version.id}`,
      versionId: version.id,
      title: version.format || "Data files",
      status: dataVersionStatus(version),
      episodes,
      split: version.metadata?.split,
      createdAt: version.created_at,
      storage: locationHtml(paths),
      settings: resultSettings(version),
      primary:
        dataset && version.format !== "skynet.episodes/v1"
          ? [
              `<button type="button" data-registered-train="${esc(version.id)}">Use in experiment</button>`,
            ]
          : [],
    });
  }

  function datasetLoadError(message) {
    error("prepared-dataset-error", message);
    const content = el("prepared-dataset-content");
    if (content.textContent === "Loading dataset…")
      content.innerHTML = "<p>Dataset details could not be loaded.</p>";
    if (!content.querySelector("[data-prepared-load-retry]"))
      content.insertAdjacentHTML(
        "beforeend",
        `<button type="button" class="button button-outline" data-prepared-load-retry data-prepared-resource="${esc(selectedResource.id)}">Try again</button>`,
      );
  }
  async function renderDataset(
    token = ++detailGeneration,
    refreshCatalog = false,
  ) {
    try {
      await renderDatasetCurrent(token, refreshCatalog);
    } catch (e) {
      if (token !== detailGeneration || !detail.open) return;
      datasetLoadError(e.message);
    }
  }
  async function renderDatasetCurrent(token, refreshCatalog) {
    const payload = await api(
      `/api/data/resources/${encodeURIComponent(selectedResource.id)}`,
    );
    if (token !== detailGeneration || !detail.open) return;
    selectedResource = payload.resource;
    let r = selectedResource;
    let activityError = "";
    if (
      r.category === "dataset" &&
      (refreshCatalog || catalogState !== "ready")
    ) {
      try {
        await request();
      } catch (e) {
        activityError = `Conversion history unavailable: ${e.message}`;
      }
      if (token !== detailGeneration || !detail.open) return;
      // The catalog read can publish a just-finished conversion. Read its
      // resource again so Formats and the detail rows use the same results.
      if (!activityError) {
        const current = await api(
          `/api/data/resources/${encodeURIComponent(r.id)}`,
        );
        if (token !== detailGeneration || !detail.open) return;
        selectedResource = r = current.resource;
      }
    }
    const jobs =
      r.category === "dataset"
        ? (snapshot?.exports || []).filter((j) => j.resource_id === r.id)
        : [];
    const versions = r.versions || [];
    el("prepared-dataset-title").textContent =
      r.metadata?.display_name || r.name;
    const results = versions.filter((v) => v.format !== "skynet.episodes/v1");
    el("prepared-dataset-context").textContent =
      `${results.length} result${results.length === 1 ? "" : "s"}`;
    const resultsById = new Map(results.map((v) => [v.id, v]));
    const shown = new Set();
    const rows = [];
    for (const job of jobs) {
      if (job.version_id && resultsById.has(job.version_id)) {
        if (shown.has(job.version_id)) continue;
        shown.add(job.version_id);
      }
      rows.push(preparedRow(job));
    }
    for (const version of results) {
      if (!shown.has(version.id))
        rows.push(sourceVersionRow(version, r.category === "dataset"));
    }
    const focusedResult = document.activeElement?.dataset?.datasetResultToggle;
    const metadataDisclosure = el("prepared-dataset-content").querySelector(
      "[data-dataset-metadata]",
    );
    if (metadataDisclosure) datasetDetailsOpen = metadataDisclosure.open;
    const title = r.metadata?.display_name || r.name;
    const description =
      r.description?.trim() && r.description.trim() !== title.trim()
        ? `<p>${esc(r.description)}</p>`
        : "";
    error("prepared-dataset-error", activityError);
    el("prepared-dataset-content").innerHTML =
      `${description}${activityError ? `<button type="button" class="button button-outline" data-prepared-load-retry data-prepared-resource="${esc(r.id)}">Try again</button>` : ""}
      <div class="table-scroll identity-table"><table data-dataset-results><thead><tr>${["Format", "Episodes", ...(r.category === "dataset" ? ["Experiments presets"] : []), "Created", "Actions"].map((label) => `<th scope="col">${label}</th>`).join("")}</tr></thead><tbody>${rows.join("") || `<tr><td colspan="${r.category === "dataset" ? 5 : 4}">No results yet.</td></tr>`}</tbody></table></div>
      <details class="collection-disclosure" data-dataset-metadata${datasetDetailsOpen ? " open" : ""}><summary>${r.category === "file" ? "File set details" : "Dataset details"}</summary>
        <div class="key-value-grid"><div class="key-value"><span>Type</span><strong>${esc(dataResourceTypeLabel(r))}</strong></div><div class="key-value"><span>Source</span><strong>${esc(r.provider || "—")}</strong></div><div class="key-value"><span>Conversion attempts</span><strong>${jobs.length}</strong></div></div>
        ${r.metadata?.test_fixture ? '<p class="secondary">Test fixture</p>' : ""}
      </details>`;
    el("prepared-dataset-actions").innerHTML =
      `<button type="button" class="button button-outline" data-data-history="${esc(r.id)}">Files and history</button>`;
    if (focusedResult) {
      [
        ...el("prepared-dataset-content").querySelectorAll(
          "[data-dataset-result-toggle]",
        ),
      ]
        .find((button) => button.dataset.datasetResultToggle === focusedResult)
        ?.focus({ preventScroll: true });
    }
  }

  window.openPreparedDataset = async (id) => {
    const token = ++detailGeneration;
    const retainDetails = detail.open && selectedResource?.id === id;
    if (!retainDetails) {
      expandedResults.clear();
      datasetDetailsOpen = false;
      selectedResource = { id };
      el("prepared-dataset-title").textContent = "Dataset";
      el("prepared-dataset-actions").replaceChildren();
      el("prepared-dataset-context").textContent = "";
      el("prepared-dataset-content").textContent = "Loading dataset…";
    }
    SkynetDialog.close(dialog);
    error("prepared-dataset-error", null);
    if (!retainDetails) SkynetDialog.open(detail);
    try {
      if (token === detailGeneration && detail.open)
        await renderDataset(token, true);
    } catch (e) {
      if (token === detailGeneration && detail.open)
        datasetLoadError(e.message);
    }
  };
  async function action(event) {
    const button = event.target.closest("button");
    if (!button || button.disabled) return;
    const data = button.dataset;
    if (data.datasetResultToggle) {
      const id = data.datasetResultToggle;
      const expanded = !expandedResults.has(id);
      if (expanded) expandedResults.add(id);
      else expandedResults.delete(id);
      button.setAttribute("aria-expanded", String(expanded));
      el(button.getAttribute("aria-controls")).hidden = !expanded;
      return;
    }
    try {
      if (
        data.deleteKind ||
        data.dataHistory ||
        data.recordingLink ||
        data.datasetRecordings ||
        data.datasetPresets
      )
        return;
      button.disabled = true;
      if (data.datasetCopyPath) {
        await navigator.clipboard.writeText(data.datasetCopyPath);
        showToast("Path copied.");
        return;
      }
      if (data.preparedResource)
        return await window.openPreparedDataset(data.preparedResource);
      if (data.registeredTrain) {
        SkynetDialog.close(detail);
        await window.useRegisteredDataset(data.registeredTrain);
        return;
      }
      const id =
        data.preparationRetry ||
        data.preparationTransfer ||
        data.preparationTrain;
      if (!id) return;
      if (data.preparationTrain) {
        const job = snapshot.exports.find((j) => j.id === id);
        SkynetDialog.close(detail);
        await window.usePreparedDataset(job);
        return;
      }
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
    }
  }
  window.refreshPreparedDatasets = async () => {
    await refreshAfterMutation();
    if (detail.open) {
      try {
        await renderDataset();
      } catch {
        SkynetDialog.close(detail);
      }
    }
  };
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
        await activateTab("data", true, "registry");
        await loadDataRegistry(true);
        const search = el("data-resource-search");
        delete search.dataset.recordingId;
        delete search.dataset.resourceIds;
        search.value = job.resource_id;
        refreshDataResourceTables();
        await window.openPreparedDataset(job.resource_id);
      }
      await refresh();
    } catch (e) {
      if (token === generation && dialog.open)
        error("policy-export-error", e.message);
    } finally {
      busy = false;
      sourceNote();
    }
  };
  document.addEventListener("collection-recordings-changed", refresh);
  document.addEventListener("dataset-preparation-refresh-requested", refresh);
  refresh();
})();
