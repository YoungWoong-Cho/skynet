(() => {
  const el = (id) => document.getElementById(id);
  const terminal = new Set(["CAPTURED", "STOPPED", "TIMED_OUT", "FAILED"]);
  const sessionLabels = {
    PREPARING: "Checking server…",
    SUBMITTING: "Starting session…",
    SUBMISSION_UNKNOWN: "Checking startup…",
    PENDING: "Waiting for GPU…",
    STARTING_SERVER: "Starting stream…",
    STARTING_SIMULATION: "Loading scene…",
    AWAITING_HEADSET: "Scene ready",
    COLLECTING: "Session in progress",
    STOPPING: "Stopping…",
    RENDERING_IMAGES: "Preparing training images…",
  };
  const startingStates = new Set([
    "PREPARING",
    "SUBMITTING",
    "SUBMISSION_UNKNOWN",
    "PENDING",
    "STARTING_SERVER",
    "STARTING_SIMULATION",
  ]);
  const stageLabels = {
    server: "Connecting to server…",
    runtime: "Checking runtime…",
    hand: "Preparing hand files…",
    launch: "Starting session…",
    stream: "Starting stream…",
    simulation: "Loading hand and simulation…",
    ready: "Ready for Vision Pro",
  };
  let focusedSession = new URL(location.href).searchParams.get("live_session"),
    requestPending = false,
    requestFailure = null,
    target = null;
  let catalog = null,
    selectionWarning = null;
  let sessions = [],
    loading = false,
    submitting = false,
    revision = 0,
    timer;
  const visible = () =>
    !document.hidden &&
    !el("collection").hidden &&
    (!el("collection-view-live").hidden || !el("collection-view-recordings").hidden);
  function focusSession(id) {
    focusedSession = id;
    const url = new URL(location.href);
    url.searchParams.set("live_session", id);
    history.replaceState(null, "", url);
  }
  function error(message) {
    el("live-xr-error").textContent = message || "";
    el("live-xr-error").hidden = !message;
  }
  async function api(path = "", options = {}) {
    const r = await fetch("/api/collection/live" + path, {
      ...options,
      headers: { "Content-Type": "application/json" },
      signal: AbortSignal.timeout(55000),
    });
    const data = await r.json();
    if (!r.ok)
      throw new Error(
        typeof data.detail === "string"
          ? data.detail
          : "Live-session request failed",
      );
    return data;
  }
  function text(parent, tag, value, className) {
    const n = document.createElement(tag);
    n.textContent = value;
    if (className) n.className = className;
    parent.append(n);
    return n;
  }
  function progressTitle(session, pending, failedRequest) {
    if (pending) return "Submitting request…";
    if (failedRequest) return "Could not start";
    if (session.connection_check_failed) return "Status unavailable";
    if (session.state === "FAILED")
      return session.scene_ready_at ? "Session failed" : "Startup failed";
    const ended = {
      CAPTURED: "Recording saved · Session ended",
      TIMED_OUT: "Session timed out",
      STOPPED: "Session stopped",
    };
    if (ended[session.state]) return ended[session.state];
    if (session.state === "RENDERING_IMAGES") return session.detail || "Preparing training images…";
    if (session.stop_requested || session.state === "STOPPING")
      return "Stopping…";
    if (session.state === "PENDING") return "Waiting for GPU…";
    if (session.state === "COLLECTING")
      return session.detail || "Collecting episodes";
    return (
      stageLabels[session.startup_stage] ||
      sessionLabels[session.state] ||
      "Checking status…"
    );
  }
  function renderProgress(running) {
    const panel = el("live-xr-progress");
    const session =
      running || sessions.find((s) => s.id === focusedSession);
    const pending = requestPending && !running;
    const failedRequest = !running && requestFailure;
    const completed = session && ["CAPTURED", "STOPPED"].includes(session.state)
      && !session.error && !session.connection_check_failed;
    panel.hidden = !pending && !failedRequest && (!session || completed);
    if (panel.hidden) return;
    const shown = pending || failedRequest ? null : session;
    const profile = shown?.profile || {
      robot: el("live-xr-hand").value,
      task: el("live-xr-task").value,
    };
    const hand =
      profile.hand_name ||
      catalog?.hands.find((h) => h.key === profile.robot)?.name ||
      profile.robot ||
      "";
    const task =
      profile.task_name ||
      catalog?.tasks.find((t) => t.key === profile.task)?.name ||
      profile.task ||
      "";
    el("live-xr-progress-context").textContent = `${hand} · ${task}`;
    const state = shown?.state;
    const stage = shown?.failed_stage || shown?.startup_stage;
    const ended = terminal.has(state);
    el("live-xr-progress-title").textContent = progressTitle(
      shown,
      pending,
      failedRequest,
    );
    const failure =
      failedRequest ||
      (!pending && shown?.error) ||
      (state === "FAILED" ? shown.detail : "");
    el("live-xr-progress-error").textContent = failure || "";
    el("live-xr-progress-error").hidden = !failure;
    const steps = el("live-xr-progress-steps");
    steps.replaceChildren();
    // Old sessions did not record milestones; never invent completed/failed steps.
    steps.hidden =
      !!shown && (!shown.startup_stage || (ended && state !== "FAILED"));
    if (steps.hidden) return;
    const current = stage === "launch" ? "stream" : stage;
    for (const [key, label, timestamp, done] of [
      [
        "server",
        `Server · ${shown?.gateway || target?.host || "server"}`,
        "server_connected_at",
        "Connected",
      ],
      ["runtime", "Simulation runtime", "runtime_checked_at", "Checked"],
      ["hand", "Hand files", "hand_prepared_at", "Prepared"],
      ["stream", "Streaming", "stream_ready_at", "Ready"],
      ["simulation", "Hand & simulation", "scene_ready_at", "Loaded"],
    ]) {
      const reached = !!shown?.[timestamp];
      const failed =
        (state === "FAILED" || shown?.failed_stage) &&
        current === key &&
        !reached;
      let stepState = "pending",
        stepLabel = "Not started";
      if (shown?.connection_check_failed) {
        stepLabel = "Unknown";
      } else if (failed) {
        stepState = "failed";
        stepLabel = "Failed";
      } else if (reached) {
        stepState =
          ended || shown?.stop_requested || state === "STOPPING"
            ? "pending"
            : "done";
        stepLabel = stepState === "done" ? done : "Previously completed";
      } else if (current === key && !pending && !failedRequest) {
        if (shown.connection_check_failed) stepLabel = "Unknown";
        else if (ended || shown.stop_requested || state === "STOPPING")
          stepLabel = "Interrupted";
        else {
          stepState = "active";
          stepLabel = state === "PENDING" ? "Queued" : "In progress…";
        }
      }
      const row = text(steps, "li", "");
      row.dataset.state = stepState;
      text(row, "span", label);
      text(row, "strong", stepLabel);
    }
  }
  function apply(session) {
    revision += 1;
    sessions = sessions.some((s) => s.id === session.id)
      ? sessions.map((s) => (s.id === session.id ? session : s))
      : [session, ...sessions];
    render();
  }
  function render() {
    window.renderSimulationRecordings?.(sessions);
    const body = el("live-xr-sessions");
    body.replaceChildren();
    if (!sessions.length) {
      const cell = text(text(body, "tr", ""), "td", "No live sessions yet.");
      cell.colSpan = 4;
    }
    for (const session of sessions) {
      const row = text(body, "tr", ""),
        name = text(row, "td", ""),
        status = text(row, "td", ""),
        address = text(row, "td", ""),
        actions = text(row, "td", "");
      status.className = "wrap-cell";
      text(
        name,
        "strong",
        session.profile?.task_name ||
          catalog?.tasks.find((t) => t.key === session.profile?.task)?.name ||
          session.profile?.task ||
          session.id.slice(0, 8),
      );
      text(
        name,
        "div",
        session.profile?.hand_name ||
          catalog?.hands.find((h) => h.key === session.profile?.robot)?.name ||
          session.profile?.robot ||
          "",
        "secondary",
      );
      text(
        name,
        "div",
        session.id.slice(0, 8) +
          (session.created_at
            ? " · " + new Date(session.created_at).toLocaleString()
            : ""),
        "secondary",
      );
      const phaseLabel = {
        ready: "READY TO RECORD",
        recording: "RECORDING",
        saving: "SAVING",
        interrupted: "INTERRUPTED",
        ended: "ENDING",
        error: "FAILED",
      };
      text(
        status,
        "strong",
        session.state === "COLLECTING"
          ? phaseLabel[session.episode_phase] || "SESSION IN PROGRESS"
          : session.state.replaceAll("_", " "),
      );
      const detail =
        session.error ||
        (!terminal.has(session.state) || session.state === "FAILED"
          ? session.detail
          : "");
      if (detail) text(status, "div", detail, "secondary");
      if (!terminal.has(session.state)) {
        const check = text(status, "details", "", "live-xr-connection-check");
        text(check, "summary", "Connection check");
        const label = text(check, "label", "Headset result", "secondary");
        const select = document.createElement("select");
        select.setAttribute(
          "aria-label",
          "Headset result for " + session.id.slice(0, 8),
        );
        for (const [value, name] of Object.entries({
          NOT_TESTED: "Not tested",
          PORT_UNREACHABLE: "Streaming port unreachable",
          PORT_REACHABLE: "Streaming port reachable",
          SCENE_VISIBLE: "Scene visible in headset",
        }))
          select.add(new Option(name, value));
        select.value = session.headset_result || "NOT_TESTED";
        select.onchange = async () => {
          select.disabled = true;
          try {
            apply(
              await api("/sessions/" + session.id + "/headset-result", {
                method: "POST",
                body: JSON.stringify({ result: select.value }),
              }),
            );
            await load();
          } catch (e) {
            error(e.message);
            select.disabled = false;
          }
        };
        label.append(select);
      }
      if (session.headset_result === "PORT_UNREACHABLE")
        text(
          status,
          "div",
          "Headset cannot reach the server.",
          "collection-quality-warning",
        );
      if (
        session.server_ready &&
        !terminal.has(session.state) &&
        !session.stop_requested &&
        session.address
      ) {
        text(address, "code", session.address);
        text(address, "div", session.host || "", "secondary");
        const copy = text(address, "button", "Copy address", "text-button");
        copy.type = "button";
        copy.onclick = async () => {
          try {
            await navigator.clipboard.writeText(session.address);
            el("live-xr-message").textContent = "Copied " + session.address;
          } catch {
            error(
              "Clipboard unavailable. Enter the displayed address manually.",
            );
          }
        };
      } else text(address, "span", "—");
      if (session.recordings?.length) {
        const review = text(
          actions,
          "button",
          "Review recordings",
          "button button-outline",
        );
        review.type = "button";
        review.onclick = () => window.openLiveReview(session, 0);
      }
      const logs = text(actions, "a", "Logs ↗", "text-button");
      logs.href = "/api/collection/live/sessions/" + session.id + "/logs";
      logs.target = "_blank";
      logs.rel = "noreferrer";
      if (session.recordings?.length)
        text(
          status,
          "div",
          session.recording_summary
            ? `${session.recording_summary.episodes} demo${session.recording_summary.episodes === 1 ? "" : "s"} · ${session.recording_summary.steps} steps`
            : `${session.recordings.length} recording${session.recordings.length === 1 ? "" : "s"}`,
          "secondary",
        );
    }
    const running = sessions.find((s) => !terminal.has(s.state));
    const addressLine = el("live-xr-address");
    addressLine.hidden = !running?.server_ready || !running.address || running.stop_requested;
    addressLine.replaceChildren();
    if (!addressLine.hidden) {
      text(addressLine, "span", "Vision Pro server: ");
      text(addressLine, "strong", running.address);
      const copy = text(addressLine, "button", "Copy address", "text-button");
      copy.type = "button";
      copy.onclick = async () => {
        try { await navigator.clipboard.writeText(running.address); copy.textContent = "Copied"; }
        catch { error("Clipboard unavailable. Enter the displayed address in the headset."); }
      };
    }
    const active = !!running;
    if (catalog && running?.profile) {
      el("live-xr-hand").value = running.profile.robot;
      el("live-xr-task").value = running.profile.task;
      el("live-xr-task-instructions").textContent =
        running.profile.instructions ||
        catalog.tasks.find((t) => t.key === running.profile.task)
          ?.instructions ||
        "See the task instructions on the headset.";
      el("live-xr-selection-note").textContent = "";
    }
    if (!active && catalog) updateChoices(false);
    renderProgress(running);
    const stop = el("live-xr-stop");
    stop.disabled =
      !running?.job_id ||
      !!running.stop_requested ||
      running.state === "STOPPING" || running.state === "RENDERING_IMAGES";
    stop.textContent =
      running?.stop_requested || running?.state === "STOPPING"
        ? "Stopping…"
        : "Stop session";
    stop.onclick = async () => {
      if (!running?.job_id || stop.disabled) return;
      focusSession(running.id);
      stop.disabled = true;
      stop.textContent = "Stopping…";
      try {
        apply(
          await api("/sessions/" + running.id + "/stop", { method: "POST" }),
        );
        await load();
      } catch (e) {
        error(e.message);
        stop.disabled = false;
        stop.textContent = "Stop session";
      }
    };
    el("live-xr-start").disabled = active || submitting || !catalog;
    el("live-xr-hand").disabled = active || submitting || !catalog;
    el("live-xr-task").disabled = active || submitting || !catalog;
    if (el("live-xr-images")) el("live-xr-images").disabled = active || submitting || !catalog;
    el("live-xr-start").textContent = active
      ? running.state === "RENDERING_IMAGES" ? "Preparing training images…" : running.stop_requested
        ? "Stopping…"
        : (running.state === "PREPARING"
            ? stageLabels[running.startup_stage]
            : sessionLabels[running.state]) || "Session active"
      : submitting
        ? "Starting…"
        : "Start session";
  }
  function setupChoices(value) {
    catalog = value;
    const hand = el("live-xr-hand"),
      task = el("live-xr-task");
    hand.replaceChildren();
    task.replaceChildren();
    const unavailable = new Map();
    for (const item of catalog.hands) {
      const option = new Option(
        item.name + (item.available ? "" : " — unsupported"),
        item.key,
      );
      option.disabled = !item.available;
      hand.add(option);
      if (!item.available) {
        const names = unavailable.get(item.reason) || [];
        names.push(item.name);
        unavailable.set(item.reason, names);
      }
    }
    for (const [reason, names] of unavailable) {
      text(el("live-xr-unavailable"), "li", names.join(", ") + ": " + reason);
    }
    for (const item of catalog.tasks) task.add(new Option(item.name, item.key));
    const params = new URL(location.href).searchParams;
    const chosenHand = params.get("live_hand"),
      chosenTask = params.get("live_task");
    hand.value = catalog.hands.some((h) => h.key === chosenHand && h.available)
      ? chosenHand
      : catalog.default_robot;
    task.value = catalog.tasks.some((t) => t.key === chosenTask)
      ? chosenTask
      : catalog.default_task;
    const invalidLink =
      (chosenHand && hand.value !== chosenHand) ||
      (chosenTask && task.value !== chosenTask);
    if (invalidLink)
      selectionWarning =
        "Unsupported hand or task in link. Check the selection before starting.";
    updateChoices(false);
    hand.onchange = task.onchange = () => updateChoices(true);
  }
  function updateChoices(persist) {
    if (persist) selectionWarning = null;
    const robot = el("live-xr-hand").value,
      task = el("live-xr-task").value;
    const verified = sessions.some(s => s.profile?.robot === robot && s.profile?.task === task && s.recordings?.length) || catalog.verified_pairs.some(
      (pair) => pair.robot === robot && pair.task === task,
    );
    el("live-xr-selection-note").textContent =
      selectionWarning || (verified ? "" : "Not headset-tested");
    el("live-xr-task-instructions").textContent = catalog.tasks.find(
      (t) => t.key === task,
    ).instructions;
    if (persist) {
      const url = new URL(location.href);
      url.searchParams.set("live_hand", robot);
      url.searchParams.set("live_task", task);
      history.replaceState(null, "", url);
    }
  }
  async function load() {
    if (loading) return;
    loading = true;
    const startedAtRevision = revision;
    clearTimeout(timer);
    try {
      const result = await api();
      if (revision !== startedAtRevision) return;
      if (result.target) {
        target = result.target;
        el("live-xr-target").textContent =
          `${target.host} · ${target.duration_minutes} min limit`;
      }
      if (result.catalog && !catalog) setupChoices(result.catalog);
      // Older saved sessions predate the human-readable profile labels.
      // Resolve labels from the same catalog used by the hand/task controls.
      sessions = result.sessions.map(session => ({
        ...session,
        profile: {
          ...session.profile,
          task_name: session.profile.task_name || catalog?.tasks.find(task => task.key === session.profile.task)?.name,
          hand_name: session.profile.hand_name || catalog?.hands.find(hand => hand.key === session.profile.robot)?.name,
        },
      }));
      window.setConversionTarget?.(result.conversion_target);
      el("live-xr-consent-field").hidden = result.license.accepted;
      el("live-xr-consent").required = !result.license.accepted;
      render();
      for (const session of sessions.filter(
        (s) => !terminal.has(s.state) || (s.job_id && !s.scheduler_final),
      )) {
        const update = await api("/sessions/" + session.id);
        if (revision !== startedAtRevision) return;
        sessions = sessions.map((s) => (s.id === update.id ? update : s));
        render();
      }
      el("live-xr-message").textContent = "";
      error(null);
    } catch (e) {
      if (revision !== startedAtRevision) return;
      error(e.message);
      el("live-xr-message").textContent = "Status may be out of date.";
      window.simulationRecordingsError?.(e.message);
    } finally {
      loading = false;
      if (visible())
        timer = setTimeout(
          () => {
            if (visible()) load();
          },
          sessions.some((s) => startingStates.has(s.state)) || requestPending
            ? 1000
            : 10000,
        );
    }
  }
  el("live-xr-start-form").onsubmit = async (e) => {
    e.preventDefault();
    if (submitting || !catalog || sessions.some((s) => !terminal.has(s.state)))
      return;
    submitting = true;
    requestPending = true;
    requestFailure = null;
    render();
    el("live-xr-message").textContent = "";
    error(null);
    try {
      const session = await api("/sessions", {
        method: "POST",
        body: JSON.stringify({
          accepted_license: el("live-xr-consent").checked,
          task: el("live-xr-task").value,
          robot: el("live-xr-hand").value,
          image_capture: !!el("live-xr-images")?.checked,
        }),
      });
      requestPending = false;
      focusSession(session.id);
      apply(session);
      await load();
    } catch (e) {
      requestFailure = e.message;
      error(null);
      el("live-xr-message").textContent = "";
    } finally {
      requestPending = false;
      submitting = false;
      render();
    }
  };
  window.loadLiveXR = load;
  document.addEventListener("visibilitychange", () => {
    if (visible()) load();
    else clearTimeout(timer);
  });
  window.addEventListener("hashchange", () => {
    if (visible()) load();
    else clearTimeout(timer);
  });
  if (visible()) load();
})();
