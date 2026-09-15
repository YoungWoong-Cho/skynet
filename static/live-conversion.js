/* Collection's recordings table. Preparation is owned by policy-exports.js. */
(() => {
  const el = (id) => document.getElementById(id);
  let sessions = [],
    sessionsKnown = false,
    selectedIds = null,
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
      selectedIds
        ? selectedIds.has(s.id)
        : `${label(s)} ${hand(s)} ${s.id} ${new Date(s.created_at).toLocaleString()}`
            .toLowerCase()
            .includes(query),
    );
    const registrations = rows.map((session) =>
      recordingRegistrationSummary(session.id),
    );
    const next = JSON.stringify([rows, query, registrations, sessionsKnown]);
    if (next === signature) return;
    signature = next;
    const count = saved.reduce(
      (n, s) => n + (s.recording_summary?.episodes || s.recordings.length),
      0,
    );
    el("collection-library-total").textContent = count || "";
    const body = el("simulation-recordings-body");
    body.replaceChildren();
    if (!rows.length)
      text(
        text(body, "tr", "", "empty-row"),
        "td",
        !sessionsKnown
          ? "Loading recordings…"
          : query
            ? "No sessions match your search."
            : "No recordings yet. Start a session in Collect.",
      ).colSpan = 5;
    for (const session of rows) {
      const row = text(body, "tr", "");
      row.dataset.sessionId = session.id;
      const name = text(row, "td", "", "wrap-cell");
      text(name, "strong", label(session));
      text(name, "span", hand(session), "secondary");
      text(name, "span", session.id.slice(0, 8), "secondary");
      const archive = session.archive;
      const storage =
        archive?.state === "READY"
          ? ""
          : ["VERIFIED", "CLEANUP_PENDING"].includes(archive?.state)
            ? "On sky2 · cleanup pending"
            : archive?.state === "COPYING"
              ? "Moving to sky2…"
              : archive?.state === "FAILED"
                ? "Transfer needs attention"
                : "On collection workstation";
      if (storage) text(name, "span", storage, "secondary");
      if (archive?.error) text(name, "span", archive.error, "secondary");
      text(
        row,
        "td",
        countLabel(
          session.recording_summary?.episodes || session.recordings.length,
        ),
      );
      text(row, "td", formatDate(session.created_at));
      const registration = recordingRegistrationSummary(session.id);
      const registeredCell = text(row, "td", "");
      if (registration.state === "ready") {
        const registered = button(
          registeredCell,
          `${registration.count} dataset${registration.count === 1 ? "" : "s"}`,
          () => showRecordingResources(session.id),
        );
        registered.className = "text-button";
        registered.setAttribute(
          "aria-label",
          `${registration.count} dataset${registration.count === 1 ? "" : "s"} for recording ${session.id.slice(0, 8)}`,
        );
      } else {
        const pending = text(registeredCell, "span", "—", "secondary");
        pending.title =
          registration.state === "unavailable"
            ? "Datasets unavailable"
            : "Loading datasets…";
      }
      const actions = text(row, "td", "", "row-actions");
      button(actions, "View", () => window.openLiveReview(session));
      button(actions, "Convert", () => window.openPolicyExport(session.id));
      const remove = text(actions, "button", "Delete");
      remove.type = "button";
      remove.dataset.deleteKind = "recording";
      remove.dataset.deleteId = session.id;
    }
  }
  window.renderSimulationRecordings = (value) => {
    sessionsKnown = true;
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
  window.filterSimulationRecordings = (ids, name) => {
    selectedIds = new Set(ids);
    el("simulation-recordings-search").value = name;
    render();
  };
  document.addEventListener("recording-registry-changed", render);
  el("simulation-recordings-search").addEventListener("input", () => {
    selectedIds = null;
    render();
  });
  el("new-recording").addEventListener("click", () =>
    activateTab("data", true, "collect"),
  );
})();
