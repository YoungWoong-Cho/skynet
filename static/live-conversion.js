/* Collection's recordings table. Preparation is owned by policy-exports.js. */
(() => {
  const el = (id) => document.getElementById(id);
  let sessions = [],
    preparations = [],
    preparationState = "loading",
    preparationKnown = false,
    preparationError = "",
    signature = "",
    sourceSignature = "";
  const label = (s) => s.profile.task_name || s.profile.task;
  const hand = (s) => s.profile.hand_name || s.profile.robot;
  const countLabel = (n) => `${n} episode${n === 1 ? "" : "s"}`;
  function text(parent, tag, value, className = "") {
    const child = document.createElement(tag);
    child.textContent = value;
    child.className = className;
    parent.append(child);
    return child;
  }
  function button(parent, label, action) {
    const child = text(parent, "button", label, "button button-outline");
    child.type = "button";
    child.onclick = action;
    return child;
  }
  function render() {
    const query = el("simulation-recordings-search").value.trim().toLowerCase();
    const saved = sessions.filter((s) => s.recordings?.length);
    const rows = saved.filter((s) =>
      `${label(s)} ${hand(s)} ${s.id} ${new Date(s.created_at).toLocaleString()}`
        .toLowerCase()
        .includes(query),
    );
    const next = JSON.stringify([rows, preparations, preparationState, preparationError, query]);
    if (next === signature) return;
    signature = next;
    const count = saved.reduce(
      (n, s) => n + (s.recording_summary?.episodes || s.recordings.length),
      0,
    );
    el("collection-library-total").textContent = count || "";
    el("simulation-recordings-count").textContent =
      `${countLabel(count)} · ${query ? rows.length + " of " : ""}${saved.length} session${saved.length === 1 ? "" : "s"}`;
    const body = el("simulation-recordings-body");
    body.replaceChildren();
    if (!rows.length)
      text(
        text(body, "tr", "", "empty-row"),
        "td",
        query
          ? "No sessions match your search."
          : "No recordings yet. Start a session in Collect.",
      ).colSpan = 4;
    for (const session of rows) {
      const row = text(body, "tr", "");
      row.dataset.sessionId = session.id;
      const name = text(row, "td", "", "wrap-cell");
      text(name, "strong", label(session));
      text(name, "span", hand(session), "secondary");
      text(
        name,
        "span",
        `${new Date(session.created_at).toLocaleString()} · ${session.id.slice(0, 8)}`,
        "secondary",
      );
      const archive = session.archive;
      const storage = archive?.state === "READY" ? "Stored on sky2"
        : ["VERIFIED", "CLEANUP_PENDING"].includes(archive?.state) ? "On sky2 · cleanup pending"
        : archive?.state === "COPYING" ? "Moving to sky2…"
        : archive?.state === "FAILED" ? "Transfer needs attention"
        : "On collection workstation";
      text(name, "span", storage, "secondary");
      if (archive?.error) text(name, "span", archive.error, "secondary");
      text(
        row,
        "td",
        countLabel(
          session.recording_summary?.episodes || session.recordings.length,
        ),
      );
      const jobs = preparations.filter((j) => {
        if (Object.hasOwn(j, "recording_session_id")) return j.recording_session_id === session.id;
        // Compatibility with an older API: only whole-session preparations can
        // supply this entry's dataset. Source membership alone is ambiguous.
        const selections = j.selections || [{ session_id: j.session_id }];
        return selections.length === 1 && selections[0].session_id === session.id
          && selections[0].indices == null && j.split?.mode !== "single_episode_overfit";
      });
      const ready = jobs.filter((j) => j.state === "READY");
      const formats = new Set(ready.map(j => j.format)).size;
      const active = jobs.find((j) => !["READY", "FAILED", "DELETE_FAILED"].includes(j.state));
      const cell = text(row, "td", "", "wrap-cell");
      const images = Object.keys(session.recording_images || {}).length;
      const knownLabel = active ? active.detail
        : ready.length ? `${formats} prepared format${formats === 1 ? "" : "s"}`
        : images === session.recordings.length ? "Images ready" : "Original recordings saved";
      const datasetLabel = preparationState === "unavailable" ? "Preparation status unavailable"
        : preparationState === "loading" && !preparationKnown ? "Checking preparation status…" : knownLabel;
      cell.innerHTML = statusPill(datasetLabel);
      if (preparationState === "unavailable") {
        cell.firstElementChild.className = "state-pill is-failed";
        text(cell, "span", preparationError || "Prepared formats could not be checked.", "secondary");
        if (jobs.length) text(cell, "span", "Last known: " + knownLabel, "secondary");
      }
      if (active && preparationState === "ready") cell.firstElementChild.className = `state-pill ${stateClass(active.stage || "PENDING")}`;
      if (jobs.some((j) => ["FAILED", "DELETE_FAILED"].includes(j.state)))
        cell.insertAdjacentHTML("beforeend", statusPill("Preparation needs attention"));
      const actions = text(row, "td", "", "row-actions");
      button(actions, "Review recordings", () =>
        window.openLiveReview(session),
      );
      button(actions, "Prepare for training", () =>
        window.openPolicyExport(session.id),
      );
      if (jobs[0]?.resource_id)
        button(actions, "View dataset", () =>
          window.openPreparedDataset(jobs[0].resource_id),
        );
      if (preparationState === "unavailable")
        button(actions, "Retry preparation status", () =>
          document.dispatchEvent(new CustomEvent("dataset-preparation-refresh-requested")),
        );
    }
  }
  window.renderSimulationRecordings = (value) => {
    sessions = value;
    el("simulation-recordings-error").hidden = true;
    render();
    const current = JSON.stringify(
      value.map((s) => [
        s.id,
        s.state,
        s.recordings,
        s.recording_checksums,
        s.recording_images,
      ]),
    );
    if (current !== sourceSignature) {
      sourceSignature = current;
      document.dispatchEvent(new CustomEvent("collection-recordings-changed"));
    }
  };
  window.simulationRecordingsError = (message) => {
    el("simulation-recordings-error").hidden = false;
    el("simulation-recordings-error").textContent =
      "Could not refresh recordings: " + message;
  };
  document.addEventListener("dataset-preparation-status", (event) => {
    preparationState = event.detail.state;
    if (preparationState === "ready") preparationKnown = true;
    preparationError = event.detail.error || "";
    render();
  });
  document.addEventListener("dataset-preparation-changed", (event) => {
    preparations = event.detail;
    render();
  });
  el("simulation-recordings-search").addEventListener("input", render);
})();
