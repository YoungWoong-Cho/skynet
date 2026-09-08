/* Review and conversion share the same saved sessions. No implicit GPU work. */
(() => {
  const el = (id) => document.getElementById(id);
  const terminal = new Set(["CAPTURED", "STOPPED", "TIMED_OUT", "FAILED"]);
  const pending = new Set(["PREPARING", "QUEUED", "RUNNING", "DOWNLOADING"]);
  const dialog = el("conversion-dialog");
  let sessions = [],
    conversions = [],
    selectedSession,
    selectedJob,
    timer,
    token = 0,
    submitting = false;
  let lastRender = "";
  const label = (session) => session.profile.task_name || session.profile.task;
  const hand = (session) => session.profile.hand_name || session.profile.robot;
  const episodesLabel = (n) => `${n} episode${n === 1 ? "" : "s"}`;
  const latest = (session) =>
    conversions.find((j) => j.session_id === session.id);
  function text(parent, tag, value, className) {
    const item = document.createElement(tag);
    item.textContent = value;
    if (className) item.className = className;
    parent.append(item);
    return item;
  }
  function button(parent, title, action) {
    const b = text(parent, "button", title, "button button-outline");
    b.type = "button";
    b.onclick = action;
    return b;
  }
  async function request(path, options = {}) {
    const response = await fetch("/api/collection/live" + path, {
      ...options,
      headers: { "Content-Type": "application/json" },
      signal: AbortSignal.timeout(20000),
    });
    const result = await response.json();
    if (!response.ok)
      throw new Error(
        typeof result.detail === "string"
          ? result.detail
          : "Conversion request failed",
      );
    return result;
  }
  function error(message) {
    el("conversion-error").hidden = !message;
    el("conversion-error").textContent = message || "";
  }
  function stateLabel(job) {
    if (job?.connection_error) return "Connection unavailable";
    return (
      {
        READY: "Converted",
        FAILED: "Conversion failed",
        PREPARING: "Preparing…",
        QUEUED: "Queued",
        RUNNING: "Converting…",
        DOWNLOADING: "Registering…",
      }[job?.state] || "Not converted"
    );
  }
  function render() {
    const query = el("simulation-recordings-search").value.trim().toLowerCase();
    const saved = sessions.filter((s) => s.recordings?.length);
    const filtered = saved.filter((s) =>
      `${label(s)} ${hand(s)} ${s.id} ${new Date(s.created_at).toLocaleString()}`
        .toLowerCase()
        .includes(query),
    );
    const signature = JSON.stringify([
      filtered.map((s) => [
        s.id,
        s.state,
        s.recordings.length,
        s.recording_summary,
      ]),
      conversions,
      query,
    ]);
    if (signature === lastRender) return;
    lastRender = signature;
    const count = saved.reduce(
      (n, s) => n + (s.recording_summary?.episodes || s.recordings.length),
      0,
    );
    el("collection-library-total").textContent = count || "";
    el("simulation-recordings-count").textContent =
      `${episodesLabel(count)} · ${query ? filtered.length + " of " : ""}${saved.length} session${saved.length === 1 ? "" : "s"}`;
    const body = el("simulation-recordings-body");
    body.replaceChildren();
    if (!filtered.length) {
      const cell = text(
        text(body, "tr", "", "empty-row"),
        "td",
        query
          ? "No sessions match this filter."
          : "No recordings yet. Start a session in Teleoperate.",
      );
      cell.colSpan = 4;
    }
    for (const s of filtered) {
      const row = text(body, "tr", "");
      row.dataset.sessionId = s.id;
      const name = text(row, "td", "", "wrap-cell");
      text(name, "strong", label(s));
      text(name, "span", hand(s), "secondary");
      text(
        name,
        "span",
        `${new Date(s.created_at).toLocaleString()} · ${s.id.slice(0, 8)}`,
        "secondary",
      );
      const recordings = text(row, "td", "");
      const episodes = s.recording_summary?.episodes || s.recordings.length;
      text(recordings, "strong", episodesLabel(episodes));
      if (!terminal.has(s.state))
        text(recordings, "span", "Collection in progress", "secondary");
      const dataset = text(row, "td", "", "wrap-cell");
      const job = latest(s);
      text(
        dataset,
        "span",
        stateLabel(job),
        "state-pill" +
          (job?.state === "FAILED"
            ? " is-failed"
            : pending.has(job?.state)
              ? " is-running"
              : ""),
      );
      if (job?.state === "READY")
        text(
          dataset,
          "span",
          `${episodesLabel(job.metadata.episodes)} · ${job.name}`,
          "secondary",
        );
      const actions = text(row, "td", "", "row-actions");
      button(actions, "Review recordings", () => window.openLiveReview(s));
      button(
        actions,
        !job
          ? "Convert for training"
          : job.state === "READY"
            ? "Dataset"
            : job.state === "FAILED"
              ? "Retry conversion"
              : "View conversion",
        () => open(s, job),
      );
    }
  }
  window.renderSimulationRecordings = (value) => {
    sessions = value;
    el("simulation-recordings-error").hidden = true;
    render();
  };
  window.simulationRecordingsError = (message) => {
    el("simulation-recordings-error").hidden = false;
    el("simulation-recordings-error").textContent =
      "Could not refresh recordings: " + message;
    el("simulation-recordings-count").textContent = sessions.length
      ? "Showing last loaded recordings"
      : "Unavailable";
  };
  window.setCollectionConversions = (value) => {
    conversions = value;
    render();
    // This endpoint returns saved state immediately and refreshes the remote
    // worker in the background; polling never starts a new conversion.
    for (const j of value.filter((j) => pending.has(j.state))) {
      if (selectedJob?.id === j.id && dialog.open) continue;
      request("/conversions/" + j.id)
        .then(update)
        .catch(() => {});
    }
  };
  function update(job) {
    conversions = [job, ...conversions.filter((j) => j.id !== job.id)].sort(
      (a, b) => b.created_at.localeCompare(a.created_at),
    );
    render();
    if (selectedJob?.id === job.id && dialog.open) {
      selectedJob = job;
      renderJob();
    }
  }
  function selectedIndices() {
    return [
      ...el("conversion-recordings").querySelectorAll("input:checked"),
    ].map((i) => Number(i.value));
  }
  function selectionChanged() {
    const count = selectedIndices().length,
      total = selectedSession.recordings.length;
    el("conversion-selection-summary").textContent =
      `${count} of ${total} recording${total === 1 ? "" : "s"} selected`;
    const ongoing = !terminal.has(selectedSession.state);
    el("conversion-submit").disabled = submitting || ongoing || count === 0;
    el("conversion-status").textContent = ongoing
      ? "End collection before converting its saved recordings."
      : count === 0
        ? "Select at least one recording."
        : "";
  }
  function renderJob() {
    const job = selectedJob;
    el("conversion-form").hidden = !!job;
    el("conversion-progress").hidden = !job || !pending.has(job.state);
    el("conversion-result").hidden = job?.state !== "READY";
    el("conversion-retry").hidden = job?.state !== "FAILED";
    el("conversion-new").hidden = job?.state !== "READY";
    el("conversion-refresh").hidden = !job || job.state === "READY";
    el("conversion-logs").hidden = !job;
    if (!job) return;
    el("conversion-title").textContent = job.name;
    el("conversion-logs").href =
      "/api/collection/live/conversions/" + job.id + "/logs";
    el("conversion-status").textContent =
      job.state === "READY"
        ? ""
        : job.connection_error
          ? "Conversion status could not be refreshed."
          : job.state === "FAILED"
            ? "Conversion failed"
            : job.detail || stateLabel(job);
    error(job.connection_error || job.error);
    if (Number.isFinite(job.total) && job.total > 0) {
      el("conversion-progress").max = job.total;
      el("conversion-progress").value = job.completed || 0;
    } else el("conversion-progress").removeAttribute("value");
    if (job.state === "READY") {
      const m = job.metadata;
      el("conversion-summary").textContent =
        `${episodesLabel(m.episodes)} · ${m.steps} samples · HDF5`;
      el("conversion-download").href =
        `/api/collection/live/conversions/${job.id}/dataset.hdf5`;
      el("conversion-manifest").href =
        `/api/collection/live/conversions/${job.id}/manifest.json`;
      el("conversion-details").textContent = JSON.stringify(
        {
          hand: hand(selectedSession),
          task: label(selectedSession),
          observations: m.observation_shapes,
          action_dimensions: m.action_dim,
          stored_on: job.gateway,
          excluded_observations: m.excluded_observations,
        },
        null,
        2,
      );
      // Dataset conversion does not claim an incompatible model can train it.
      el("conversion-registry").onclick = () => {
        dialog.close();
        window.openConvertedDataset(job);
      };
    }
  }
  async function poll(ownToken) {
    clearTimeout(timer);
    if (ownToken !== token || !dialog.open || !selectedJob) return;
    const id = selectedJob.id;
    try {
      const job = await request("/conversions/" + id);
      if (ownToken !== token || !dialog.open) return;
      update(job);
    } catch (e) {
      if (ownToken === token && dialog.open) error(e.message);
    }
    if (ownToken === token && dialog.open && pending.has(selectedJob?.state))
      timer = setTimeout(() => poll(ownToken), 2000);
  }
  function open(session, job = latest(session)) {
    ++token;
    clearTimeout(timer);
    selectedSession = session;
    selectedJob = job;
    submitting = false;
    error("");
    el("conversion-title").textContent = "Convert for training";
    el("conversion-context").textContent =
      `${label(session)} · ${hand(session)} · ${session.recordings.length} recording${session.recordings.length === 1 ? "" : "s"}`;
    el("conversion-name").value =
      job?.name ||
      `${label(session)} · ${hand(session)} · ${session.id.slice(0, 8)}`;
    el("conversion-selection").open = false;
    el("conversion-recordings").replaceChildren();
    session.recordings.forEach((_, i) => {
      const label = text(
        el("conversion-recordings"),
        "label",
        "",
        "check-field",
      );
      const input = document.createElement("input");
      input.type = "checkbox";
      input.value = i;
      input.checked = !job || job.indices.includes(i);
      input.onchange = selectionChanged;
      label.append(input, document.createTextNode("Recording " + (i + 1)));
    });
    selectionChanged();
    renderJob();
    if (!dialog.open) dialog.showModal();
    if (job && pending.has(job.state)) poll(token);
  }
  window.openCollectionConversion = open;
  el("conversion-new").onclick = () => open(selectedSession, null);
  el("simulation-recordings-search").oninput = render;
  el("conversion-close").onclick = () => dialog.close();
  dialog.addEventListener("close", () => {
    ++token;
    clearTimeout(timer);
  });
  for (const [id, checked] of [
    ["conversion-select-all", true],
    ["conversion-select-none", false],
  ])
    el(id).onclick = () => {
      el("conversion-recordings")
        .querySelectorAll("input")
        .forEach((i) => {
          i.checked = checked;
        });
      selectionChanged();
    };
  el("conversion-retry").onclick = () => {
    selectedJob = null;
    el("conversion-title").textContent = "Retry conversion";
    error("");
    renderJob();
    selectionChanged();
    el("conversion-name").focus();
  };
  el("conversion-refresh").onclick = () => poll(token);
  el("conversion-form").onsubmit = async (event) => {
    event.preventDefault();
    if (submitting || el("conversion-submit").disabled || !selectedSession)
      return;
    if (!el("conversion-name").value.trim()) {
      error("Enter a dataset name.");
      el("conversion-name").focus();
      return;
    }
    submitting = true;
    const ownToken = token;
    selectionChanged();
    error("");
    el("conversion-status").textContent = "Submitting conversion…";
    try {
      const job = await request(
        "/sessions/" + selectedSession.id + "/conversions",
        {
          method: "POST",
          body: JSON.stringify({
            name: el("conversion-name").value.trim(),
            indices: selectedIndices(),
          }),
        },
      );
      update(job);
      if (ownToken !== token || !dialog.open) return;
      selectedJob = job;
      renderJob();
      poll(token);
    } catch (e) {
      if (ownToken === token && dialog.open) {
        error(e.message);
        el("conversion-status").textContent =
          "Conversion was not confirmed. Retry safely to recover the same request.";
      }
    } finally {
      if (ownToken === token) {
        submitting = false;
        el("conversion-submit").disabled =
          !terminal.has(selectedSession.state) || !selectedIndices().length;
      }
    }
  };
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) clearTimeout(timer);
    else if (dialog.open && pending.has(selectedJob?.state)) poll(token);
  });
})();
