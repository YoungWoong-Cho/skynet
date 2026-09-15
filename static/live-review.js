(() => {
  const el = (id) => document.getElementById(id);
  let token = 0,
    base = "",
    data = null,
    episode = null,
    frame = 0,
    timer;
  const dialog = el("live-review-dialog");
  let reviewSession;
  function closeViewer() {
    window.SkynetEpisodeViewer?.close("live-episode-viewer");
  }
  const status = (message) => {
    el("live-review-status").textContent = message;
  };
  async function request(path, options = {}, timeout = 40000) {
    const response = await fetch(path, {
      ...options,
      signal: AbortSignal.timeout(timeout),
    });
    const result = await response.json();
    if (!response.ok)
      throw new Error(
        typeof result.detail === "string"
          ? result.detail
          : "Could not load the recording review",
      );
    return result;
  }
  function pause() {
    window.SkynetEpisodeViewer?.pause("live-episode-viewer");
  }
  function flat(value, prefix = "", result = {}) {
    for (const [key, item] of Object.entries(value)) {
      const path = prefix ? prefix + "." + key : key;
      if (Array.isArray(item)) result[path] = item.flat(Infinity);
      else if (item && typeof item === "object") flat(item, path, result);
    }
    return result;
  }
  function fieldLabel(path) {
    const parts = path.split(".");
    const property = parts.pop();
    const entity = (parts.pop() || "Scene").replaceAll("_", " ");
    const label =
      {
        root_pose: "position & orientation",
        root_velocity: "linear & angular velocity",
        joint_position: "joint positions",
        joint_velocity: "joint velocities",
      }[property] || property.replaceAll("_", " ");
    return entity.charAt(0).toUpperCase() + entity.slice(1) + " · " + label;
  }
  function componentLabel(field, index, count) {
    if (field.endsWith(".root_pose") && count === 7)
      return [
        "Position X (m)",
        "Position Y (m)",
        "Position Z (m)",
        "Quaternion W",
        "Quaternion X",
        "Quaternion Y",
        "Quaternion Z",
      ][index];
    if (field.endsWith(".root_velocity") && count === 6)
      return [
        "Linear X (m/s)",
        "Linear Y (m/s)",
        "Linear Z (m/s)",
        "Angular X (rad/s)",
        "Angular Y (rad/s)",
        "Angular Z (rad/s)",
      ][index];
    return "Component " + index;
  }
  function draw() {
    if (!episode) return;
    const current = episode.frames[frame];
    const field = el("live-review-field").value;
    const values =
      field === "action" ? current.action : flat(current.state)[field];
    const body = el("live-review-values");
    body.replaceChildren();
    if (!values) {
      const row = body.insertRow(),
        cell = row.insertCell();
      cell.colSpan = 2;
      cell.textContent =
        field === "action" && current.action_index === null
          ? "Initial state: no action has been applied yet. Choose Next frame to inspect action 0."
          : "This value is not present in the saved frame.";
    } else
      values.forEach((value, index) => {
        const row = body.insertRow();
        row.insertCell().textContent = componentLabel(
          field,
          index,
          values.length,
        );
        row.insertCell().textContent = Number(value).toPrecision(7);
      });
    el("live-review-value-note").textContent =
      field === "action"
        ? (current.action_index === null
            ? ""
            : `Action ${current.action_index} produced scene frame ${current.state_index}. `) +
          data.value_note
        : `Saved field: ${field}. Components keep the simulator's recorded order.`;
    el("live-review-state").textContent = JSON.stringify(
      current.state,
      null,
      2,
    );
  }
  function chooseEpisode() {
    pause();
    closeViewer();
    episode = data.episodes[Number(el("live-review-episode").value)];
    frame = 0;
    window.SkynetEpisodeViewer?.open(
      "live-episode-viewer",
      null,
      base,
      {
        collection: true,
        episode: Number(el("live-review-episode").value),
        robot: data.robot,
        sourceNames: data.hand_metadata?.source_names,
        timeline: episode.frames,
        onFrame(index) {
          frame = index;
          if (el("live-review-data").open) draw();
        },
      },
    );
    const fields = el("live-review-field");
    fields.replaceChildren(
      new Option("Action applied before this frame", "action"),
    );
    for (const name of Object.keys(flat(episode.frames[0].state)))
      fields.add(new Option(fieldLabel(name), name));
    fields.value = [...fields.options].some(
      (o) => o.value === "rigid_object.object.root_pose",
    )
      ? "rigid_object.object.root_pose"
      : "action";
    draw();
  }
  async function load(ownToken, start = false) {
    clearTimeout(timer);
    try {
      const result = await request(
        base + "/review",
        start ? { method: "POST" } : {},
      );
      if (ownToken !== token || !dialog.open) return;
      el("live-review-retry").hidden = true;
      if (result.state === "READY") {
        const source = result.recording_source;
        if (source?.path) {
          el("live-review-source-label").textContent =
            "Path" + (source.gateway ? " · " + source.gateway : "");
          el("live-review-source-path").textContent = source.path;
          el("live-review-source-copy").dataset.copyValue = source.path;
          el("live-review-source").hidden = false;
        }
        data = await request(base + "/review.json");
        if (ownToken !== token || !dialog.open) return;
        el("live-review-content").hidden = false;
        status(
          `${data.episodes.length} demonstration${data.episodes.length === 1 ? "" : "s"} · ${data.episodes.reduce((sum, ep) => sum + ep.duration_seconds, 0).toFixed(1)} s · ${data.simulation_hz || 60} Hz`,
        );
        const select = el("live-review-episode");
        select.parentElement.hidden = data.episodes.length === 1;
        select.replaceChildren();
        data.episodes.forEach((ep, index) =>
          select.add(
            new Option(
              `Demonstration ${index + 1} · ${ep.steps} steps · ${ep.duration_seconds.toFixed(2)} s`,
              index,
            ),
          ),
        );
        el("live-review-original").href = base + "/recording.pkl";
        el("live-review-summary").href = base + "/summary.json";
        el("live-review-json").href = base + "/review.json";
        chooseEpisode();
      } else if (result.state === "FAILED") throw new Error(result.error);
      else if (result.state === "NOT_DOWNLOADED") await load(ownToken, true);
      else {
        status("Loading recorded values…");
        timer = setTimeout(() => load(ownToken), 1500);
      }
    } catch (error) {
      if (ownToken !== token || !dialog.open) return;
      status("Review unavailable: " + error.message);
      el("live-review-retry").hidden = false;
    }
  }
  function selectRecording(index) {
    pause();
    closeViewer();
    const path = reviewSession.recordings?.[index];
    const remove = el("live-review-delete");
    remove.hidden = !path;
    if (path) remove.dataset.deleteId = `${reviewSession.id}:${btoa(encodeURIComponent(path).replace(/%([0-9A-F]{2})/g, (_, hex) => String.fromCharCode(parseInt(hex, 16)))).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "")}`;
    else delete remove.dataset.deleteId;
    el("live-review-data").open = false;
    clearTimeout(timer);
    const ownToken = ++token;
    base = `/api/collection/live/sessions/${reviewSession.id}/recordings/${index}`;
    data = episode = null;
    el("live-review-source").hidden = true;
    el("live-review-source-path").textContent = "";
    delete el("live-review-source-copy").dataset.copyValue;
    el("live-review-content").hidden = true;
    el("live-review-retry").hidden = true;
    status("Checking the saved recording…");
    // One idempotent request per user selection retains READY copies and
    // retries saved failures with current connection settings. Polls use GET.
    load(ownToken, true);
  }
  window.openLiveReview = (session, index = 0) => {
    reviewSession = session;
    const select = el("live-review-recording");
    const count = session.recordings?.length || 1;
    select.replaceChildren(
      ...Array.from(
        { length: count },
        (_, i) => new Option(`Recording ${(session.recording_slots?.[session.recordings?.[i]] ?? i) + 1}`, i),
      ),
    );
    select.value = index >= 0 && index < count ? index : 0;
    select.parentElement.hidden = count < 2;
    SkynetDialog.open(dialog);
    selectRecording(Number(select.value));
  };
  window.refreshLiveReviewAfterDeletion = async (identifier) => {
    if (!dialog.open || reviewSession?.id !== identifier) return;
    const selection = Number(el("live-review-recording").value);
    const session = await request(`/api/collection/live/sessions/${identifier}`);
    if (!dialog.open || reviewSession?.id !== identifier) return;
    if (!session.recordings?.length) SkynetDialog.close(dialog);
    else window.openLiveReview(session, Math.min(selection, session.recordings.length - 1));
  };
  el("live-review-delete").addEventListener("click", pause);
  el("live-review-recording").onchange = (event) =>
    selectRecording(Number(event.target.value));
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      pause();
    }
  });
  dialog.addEventListener("close", () => {
    ++token;
    clearTimeout(timer);
    pause();
    closeViewer();
  });
  el("live-review-retry").onclick = () => {
    status("Retrying download…");
    el("live-review-retry").hidden = true;
    load(token, true);
  };
  el("live-review-episode").onchange = chooseEpisode;
  el("live-review-field").onchange = draw;
  el("live-review-data").addEventListener("toggle", () => {
    if (el("live-review-data").open) draw();
  });
})();
