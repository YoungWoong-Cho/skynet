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
    document.dispatchEvent(
      new CustomEvent("dataset-preparation-status", {
        detail: { state, error },
      }),
    );
  }
  const request = async (path = "/jobs", options = {}) => {
    const catalog =
      path === "/jobs" && (!options.method || options.method === "GET");
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
    preparation = null,
    preparationDirty = false,
    detailTarget = null,
    defaultName = "",
    sourceSessionId = null;
  let busy = false,
    timer = null,
    generation = 0,
    detailGeneration = 0,
    refreshPromise = null,
    pendingConfirmation = null;
  const terminal = (job) =>
    ["READY", "FAILED", "DELETE_FAILED"].includes(job.state);
  const adapterFor = (job) => job.adapter;
  const selectedAdapter = () => preparation?.adapters.find((p) => p.id === el("policy-export-adapter").value);
  const selectedPreset = () => {
    const adapter = selectedAdapter();
    return adapter?.data_presets?.find((preset) => preset.id === adapter.default_data_preset);
  };
  const inputModalities = [
    ["state", "State"], ["rgb", "RGB"], ["depth", "Depth"], ["point_cloud", "Point cloud"],
  ];
  function inputModalityItems(requirements) {
    const streams = requirements?.observation_requirements?.streams || [];
    // Rendering dependencies and action representations are not model inputs.
    const used = new Set([...(requirements?.observations || []), ...streams.map((stream) => stream.modality)]);
    return inputModalities.map(([id, label]) => {
      const active = used.has(id);
      const status = active ? "Used" : "Not used";
      return `<li data-input-modality="${id}" class="${active ? "is-used" : ""}" aria-label="${label}: ${status}">${label}</li>`;
    }).join("");
  }
  function renderInputModalities(preset) {
    el("policy-export-modalities-field").hidden = !preset;
    el("policy-export-modalities").innerHTML = inputModalityItems(preset);
  }
  function resultInputModalities(...declarations) {
    const requirements = declarations.find((item) =>
      Array.isArray(item?.observations) || Array.isArray(item?.observation_requirements?.streams));
    return requirements
      ? `<ul class="adapter-input-modalities" aria-label="Input modalities">${inputModalityItems(requirements)}</ul>`
      : '<span class="secondary">—</span>';
  }
  const splitLabel = (split) =>
    typeof split === "string" ? split : !split?.validation?.length
      ? `${split?.train?.length || 0} train · no validation`
      : split?.mode === "single_episode_overfit"
        ? "1 train · same episode for validation (overfit)"
        : `${split?.train.length || 0} train / ${split?.validation.length || 0} validation`;
  function error(id, message) {
    el(id).textContent = message || "";
    el(id).hidden = !message;
  }
  function sourceNote() {
    const adapter = selectedAdapter();
    const preset = selectedPreset();
    const policy = adapter && preset ? { ...adapter, ...preset } : adapter;
    const source = preparation?.session;
    for (const id of ["preparation-validation", "preparation-seed"]) {
      el(id).disabled = source?.episodes === 1;
      el(id).closest(".field").hidden = el(id).disabled;
    }
    renderInputModalities(preset);
    const noValidation =
      policy?.trainable &&
      (source?.episodes < 2 ||
        Number(el("preparation-validation").value) === 0);
    const insufficientEpisodes = source && policy?.minimum_episodes && source.episodes < policy.minimum_episodes;
    const needsValidation = preset?.validation_required && Number(el("preparation-validation").value) === 0;
    const message = !source
      ? "Loading recordings…"
      : !source.eligible
        ? source.reason || "This session is not ready for preparation."
        : !policy?.available || !preset
          ? policy?.reason || "Choose an available adapter."
          : insufficientEpisodes
            ? `This adapter requires at least ${policy.minimum_episodes} episodes.`
          : needsValidation
            ? "This adapter requires separate validation episodes."
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
      !preparation ||
      !source?.eligible ||
      !source.episodes ||
      !policy?.available ||
      !preset ||
      needsValidation ||
      insufficientEpisodes;
  }
  const stageLabels = {
    QUEUED: "Queued",
    STAGING: "Preparing submission",
    SUBMITTING: "Submitting CPU job",
    ARCHIVING: "Waiting for the recording archive",
    FETCHING: "Checking originals",
    OBSERVATIONS: "Preparing required observations",
    CONVERTING: "Converting",
    VALIDATING: "Validating",
    CHECKING_LOADER: "Checking adapter data loader",
    READY: "Prepared",
  };
  function jobStatus(job) {
    if (job.state === "DELETE_FAILED")
      return `Deletion incomplete · ${job.error}`;
    if (job.state === "FAILED") return `Failed · ${job.error}`;
    if (job.stage === "OBSERVATIONS") {
      const progress = job.observation_progress;
      return (job.detail || "Preparing required observations") +
        (progress && Number.isInteger(progress.ready) && Number.isInteger(progress.total)
          ? ` · ${progress.ready}/${progress.total} observations ready` : "");
    }
    const progress = job.progress
      ? ` · ${job.progress.episodes_done}/${job.progress.episodes_total} episodes`
      : "";
    if (!terminal(job) && job.execution === "cluster") {
      const location = [
        job.gateway,
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
    // Readiness and training setup come from the server's current adapter catalog.
    // A browser snapshot must never override a newly archived/restored adapter.
    snapshot = value;
    document.dispatchEvent(new CustomEvent("dataset-preparation-changed", {detail: value.exports || []}));
  }
  function populateAdapters(selected = null, version = null) {
    const control = el("policy-export-adapter");
    control.replaceChildren();
    control.add(new Option("Choose an adapter", ""));
    for (const adapter of preparation.adapters) {
      const option = new Option(`${adapter.name} · v${adapter.adapter_version_number}${adapter.available ? "" : " · unavailable"}`, adapter.id);
      option.disabled = !adapter.available;
      control.add(option);
    }
    const previous = preparation.adapters.find((adapter) => adapter.id === selected);
    control.value = selected === null
      ? preparation.adapters.find((adapter) => adapter.available)?.id || ""
      : previous?.adapter_version_id === version ? selected : "";
  }
  function visible() {
    return !window.SkynetRefresh?.stopped && document.visibilityState === "visible" &&
      (dialog.open || detail.open ||
        (typeof activeTab !== "undefined" && ["collection", "datasets"].includes(activeTab)));
  }
  async function refreshNow() {
    clearTimeout(timer);
    try {
      await request();
      if (dialog.open && preparationDirty) {
        preparationDirty = false;
        const token = generation;
        const current = await request(`/options/${encodeURIComponent(sourceSessionId)}`);
        if (token === generation && dialog.open) {
          const selected = el("policy-export-adapter").value;
          const version = selectedAdapter()?.adapter_version_id;
          preparation = current;
          populateAdapters(selected, version);
          sourceNote();
        }
      }
      if (detail.open && detailTarget && pendingConfirmation !== detailGeneration)
        await renderDetail();
      if (visible() && !window.SkynetRefresh?.connected && snapshot.exports.some((j) => !terminal(j)))
        timer = setTimeout(() => { if (visible() && !window.SkynetRefresh?.connected) void refresh(); }, 3000);
    } catch (e) {
      if (detail.open) detailError(e.message);
    }
  }
  window.openPolicyExport = async (sessionId) => {
    const token = ++generation;
    SkynetDialog.close(detail);
    sourceSessionId = sessionId;
    defaultName = "";
    preparation = null;
    error("policy-export-error", null);
    retryPreparation.hidden = true;
    retryPreparation.onclick = () => window.openPolicyExport(sessionId);
    el("policy-export-adapter").replaceChildren();
    renderInputModalities(null);
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
      const result = await request(`/options/${encodeURIComponent(sessionId)}`);
      if (token !== generation || !dialog.open) return;
      preparation = result;
      const { session: source } = preparation;
      if (!source)
        throw new Error("This recording session is no longer available.");
      preparationDirty = false;
      populateAdapters();
      updateDefaultName();
      sourceNote();
    } catch (e) {
      if (token === generation && dialog.open) {
        preparation = null;
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
      ["Adapter", adapterFor(result)?.name || metadata.adapter?.name],
      ["Data preset", result.adapter_data_preset || metadata.adapter_data_preset],
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
      ["Source revision", result.source_revision || metadata.source_revision],
    ].filter(([, value]) => value != null && value !== "");
    return pairs.length
      ? `<div class="key-value-grid">${keyValueHtml(pairs)}</div>`
      : '<p class="secondary">No conversion settings recorded.</p>';
  }
  function adapterLabel(dataset) {
    const adapter = dataset?.metadata?.adapter;
    return dataset?.adapter_name || (typeof adapter === "string" ? adapter : adapter?.name) || "—";
  }
  window.SkynetDatasetUI = {inputModalities: resultInputModalities, adapterLabel};

  function matchingJobs(scope = {}) {
    return (snapshot?.exports || []).filter((job) =>
      (!scope.jobId || job.id === scope.jobId) &&
      (!scope.versionId || job.version_id === scope.versionId) &&
      (!scope.resourceId || job.resource_id === scope.resourceId) &&
      (!scope.recordingId || job.session_id === scope.recordingId ||
        job.sources?.some((source) => source.session_id === scope.recordingId)))
      .sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || "")));
  }
  function conversionHistoryRow(job) {
    const actions = [];
    if (job.state === "FAILED") actions.push(`<button type="button" data-preparation-retry="${esc(job.id)}">Retry</button>`);
    if (job.version_id) actions.push(`<button type="button" data-prepared-dataset="${esc(job.version_id)}">View dataset</button>`);
    else if (["FAILED", "DELETE_FAILED"].includes(job.state))
      actions.push(`<button type="button" data-delete-kind="prepared" data-delete-id="${esc(job.id)}">Delete attempt</button>`);
    if (job.execution !== "cluster" || job.cluster_job_id)
      actions.push(`<a class="row-action-link" href="/api/data/exports/${encodeURIComponent(job.id)}/export.log" download="">Preparation log</a>`);
    return {id: job.id, cells: [
      `<span class="node-name">${esc(job.name || "Conversion")}</span>`,
      {className: "wrap-cell", html: esc(job.adapter?.name || "—")},
      statusPill(job.state),
      {className: "wrap-cell", html: esc(jobStatus(job) || "—")},
      esc(formatDate(job.created_at)),
      {className: "row-actions", html: actions.join("")},
    ]};
  }
  function renderConversionHistory(jobs) {
    const content = el("prepared-dataset-content");
    let table = content.querySelector(":scope > .table-scroll");
    if (!table) {
      table = document.createElement("div");
      content.replaceChildren(table);
    }
    window.SkynetJobHistory.render(table, {
      columns: ["Name", "Adapter", "State", "Progress", "Created at", "Actions"],
      rows: jobs.map(conversionHistoryRow),
      empty: "No conversion attempts.",
    });
  }
  function detailError(message) {
    error("prepared-dataset-error", message);
    if (!el("prepared-dataset-content").querySelector("[data-detail-retry]"))
      el("prepared-dataset-content").insertAdjacentHTML("beforeend",
        '<button type="button" class="button button-outline" data-detail-retry="">Try again</button>');
  }
  async function renderDetail(token = ++detailGeneration) {
    const target = detailTarget;
    if (!target) return;
    const current = () => token === detailGeneration && detail.open;
    try {
      if (target.kind === "job") {
        const job = matchingJobs({jobId: target.id})[0] || target.job;
        if (job?.state === "READY" && job.version_id) {
          detailTarget = {kind: "dataset", id: job.version_id};
          return renderDetail(token);
        }
        if (!current()) return;
        el("prepared-dataset-title").textContent = job?.name || "Dataset conversion";
        el("prepared-dataset-context").textContent = "Conversion";
        el("prepared-dataset-actions").replaceChildren();
        renderConversionHistory(job ? [job] : []);
      } else if (target.kind === "history") {
        if (!current()) return;
        const jobs = matchingJobs(target.scope);
        el("prepared-dataset-title").textContent = "Conversion history";
        el("prepared-dataset-context").textContent = `${jobs.length} attempt${jobs.length === 1 ? "" : "s"}`;
        el("prepared-dataset-actions").replaceChildren();
        renderConversionHistory(jobs);
      } else {
        const {dataset} = await api(`/api/data/datasets/${encodeURIComponent(target.id)}`);
        if (!current()) return;
        const jobs = matchingJobs({versionId: dataset.id});
        const job = jobs.find((item) => item.training_ready) || jobs[0];
        const metadata = dataset.metadata || {};
        const episodes = datasetEpisodeCount(dataset);
        const split = metadata.split || job?.split;
        el("prepared-dataset-title").textContent = dataset.display_name;
        el("prepared-dataset-context").textContent = `${episodes} episode${episodes === 1 ? "" : "s"}${dataset.archived_at ? " · Archived" : ""}`;
        const canUse = datasetCanTrain(dataset);
        el("prepared-dataset-actions").innerHTML = canUse
          ? `<button type="button" class="button button-accent" data-use-dataset="${esc(dataset.id)}">Use in experiment</button>` : "";
        el("prepared-dataset-content").innerHTML =
          `${dataset.description && dataset.description !== dataset.display_name ? `<p>${esc(dataset.description)}</p>` : ""}
          <div class="key-value-grid">
            ${keyValueHtml([["Adapter", adapterLabel(dataset)]])}
            <div class="key-value"><span>Input modalities</span>${resultInputModalities(metadata, job?.requirements)}</div>
            ${keyValueHtml([["Created at", formatDate(dataset.created_at)], ...(split ? [["Split", splitLabel(split)]] : [])])}
          </div>
          <div class="dataset-result-metadata"><section><h4>Storage</h4>${locationHtml(dataset.locations || [])}</section><section><h4>Conversion settings</h4>${resultSettings({...job, ...dataset})}</section></div>
          <div class="form-actions">
            ${job ? `<a class="button button-outline" href="/api/data/exports/${encodeURIComponent(job.id)}/manifest.json" download>Download manifest</a>` : ""}
            ${jobs.length ? `<button type="button" class="button button-outline" data-version-history="${esc(dataset.id)}">Conversion history</button>` : ""}
          </div>`;
        el("prepared-dataset-preview").hidden = false;
        window.SkynetEpisodeViewer?.openDataset?.("prepared-dataset-preview", dataset.id);
      }
      if (current()) error("prepared-dataset-error", null);
    } catch (e) {
      if (current()) detailError(e.message);
    }
  }
  function openDetail(target, options) {
    const same = detail.open && detailTarget?.kind === target.kind && detailTarget?.id === target.id;
    detailTarget = target;
    pendingConfirmation = null;
    const token = ++detailGeneration;
    SkynetDialog.close(dialog);
    error("prepared-dataset-error", null);
    if (!same) {
      window.SkynetEpisodeViewer?.closeDataset?.("prepared-dataset-preview");
      el("prepared-dataset-preview").hidden = true;
      el("prepared-dataset-title").textContent = target.kind === "history" ? "Conversion history" : "Dataset";
      el("prepared-dataset-context").textContent = "";
      el("prepared-dataset-actions").replaceChildren();
      el("prepared-dataset-content").textContent = "Loading…";
    }
    SkynetDialog.open(detail, options);
    return token;
  }
  window.openPreparedDataset = async (id) => {
    const token = openDetail({kind: "dataset", id});
    // Display a published result even if the conversion history service is down.
    await renderDetail(token);
  };
  window.openFileResource = (id) => {
    SkynetDialog.close(detail);
    openDataInspection(id);
  };
  window.openDatasetConversionHistory = async (scope = {}, options) => {
    const token = openDetail({kind: "history", scope}, options);
    try {
      await request();
      if (token === detailGeneration && detail.open) await renderDetail(token);
    } catch (e) {
      if (token === detailGeneration && detail.open) detailError(e.message);
    }
  };
  function showAcceptedPreparation(job) {
    const token = openDetail({kind: "job", id: job.id, job});
    pendingConfirmation = token;
    el("prepared-dataset-title").textContent = job.name || "Dataset conversion";
    el("prepared-dataset-context").textContent = job.state === "READY" ? "Dataset is ready" : "Conversion request accepted";
    renderConversionHistory([job]);
    return token;
  }
  async function finishAcceptedPreparation(token) {
    try {
      await refreshAfterMutation();
      if (token !== detailGeneration || !detail.open) return;
      pendingConfirmation = null;
      await renderDetail(token);
    } catch (e) {
      if (token === detailGeneration && detail.open) detailError(e.message);
    }
  }
  async function action(event) {
    const button = event.target.closest("button");
    if (!button || button.disabled) return;
    const data = button.dataset;
    if (data.dataHistory || data.deleteKind) return;
    button.disabled = true;
    try {
      if (data.datasetCopyPath) {
        await navigator.clipboard.writeText(data.datasetCopyPath);
        showToast("Path copied.");
      } else if (data.preparedDataset) {
        await window.openPreparedDataset(data.preparedDataset);
      } else if (data.useDataset) {
        SkynetDialog.close(detail);
        await window.useDataset(data.useDataset);
      } else if (data.versionHistory) {
        await window.openDatasetConversionHistory({versionId: data.versionHistory});
      } else if (Object.hasOwn(data, "detailRetry")) {
        await refreshAfterMutation();
        await renderDetail();
      } else if (data.preparationRetry) {
        await request(`/${encodeURIComponent(data.preparationRetry)}/retry`, {method: "POST", body: JSON.stringify({})});
        await refreshAfterMutation();
      }
    } catch (e) {
      if (detail.open) error("prepared-dataset-error", e.message);
      else showToast(e.message, true);
    } finally {
      button.disabled = false;
    }
  }
  window.refreshPreparedDatasets = async () => {
    await refreshAfterMutation();
    if (detail.open) {
      try {
        await renderDetail();
      } catch {
        SkynetDialog.close(detail);
      }
    }
  };
  dialog.addEventListener("close", () => generation++);
  detail.addEventListener("close", () => {
    detailGeneration++;
    window.SkynetEpisodeViewer?.closeDataset?.("prepared-dataset-preview");
  });
  function updateDefaultName() {
    const control = el("policy-export-name");
    if (!control.value || control.value === defaultName) {
      defaultName = [preparation?.session?.name, selectedAdapter()?.name].filter(Boolean).join(" · ").slice(0, 100);
      control.value = defaultName;
    }
  }
  el("policy-export-adapter").onchange = () => {
    updateDefaultName();
    sourceNote();
  };
  el("preparation-validation").oninput = sourceNote;
  el("refresh-data-registry").addEventListener("click", refresh);
  el("prepared-dataset-content").onclick = action;
  el("prepared-dataset-actions").onclick = action;
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
          adapter_id: selectedAdapter().adapter_id,
          adapter_version_id: selectedAdapter().adapter_version_id,
          adapter_data_preset: selectedPreset().id,
          name: el("policy-export-name").value.trim(),
          gateway: el("gateway").value,
          validation_percent: Number(el("preparation-validation").value),
          seed: Number(el("preparation-seed").value),
        }),
      });
      if (token === generation && dialog.open)
        void finishAcceptedPreparation(showAcceptedPreparation(job));
      else
        void refreshAfterMutation();
    } catch (e) {
      if (token === generation && dialog.open)
        error("policy-export-error", e.message);
    } finally {
      busy = false;
      sourceNote();
    }
  };
  window.SkynetRefresh?.register("exports", ["exports", "data", "adapters", "recordings"], visible, refresh,
    topics => {
      if (topics.includes("adapters")) catalogState = "loading";
      if (topics.some(topic => ["adapters", "recordings", "data"].includes(topic))) preparationDirty = true;
    });
  document.addEventListener("collection-recordings-changed", () => {
    if (visible()) void refresh();
  });
  document.addEventListener("dataset-preparation-refresh-requested", refresh);
  document.addEventListener("skynet-live-updates", () => {
    clearTimeout(timer);
    if (visible() && !window.SkynetRefresh?.connected) void refresh();
  });
  document.addEventListener("visibilitychange", () => {
    clearTimeout(timer);
    if (visible()) void refresh();
  });
  if (visible()) void refresh();
})();
