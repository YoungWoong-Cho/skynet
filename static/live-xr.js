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
    STOPPING: "Stopping…",
  };
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
    !el("collection-view-live").hidden;
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
  function apply(session) {
    revision += 1;
    sessions = sessions.some((s) => s.id === session.id)
      ? sessions.map((s) => (s.id === session.id ? session : s))
      : [session, ...sessions];
    render();
  }
  function render() {
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
      text(status, "strong", session.state.replaceAll("_", " "));
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
      if (session.state === "CAPTURED") {
        (session.recordings || []).forEach((file, index) => {
          const review = text(
            actions,
            "button",
            session.recordings.length > 1
              ? `Review recording ${index + 1}`
              : "Review recording",
            "button button-outline",
          );
          review.type = "button";
          review.onclick = () => window.openLiveReview(session, index);
        });
      }
      const logs = text(actions, "a", "Logs ↗", "text-button");
      logs.href = "/api/collection/live/sessions/" + session.id + "/logs";
      logs.target = "_blank";
      logs.rel = "noreferrer";
      if (!terminal.has(session.state) && session.job_id) {
        const stop = text(
          actions,
          "button",
          session.stop_requested || session.state === "STOPPING"
            ? "Stopping…"
            : "Stop session",
          "button button-outline",
        );
        stop.type = "button";
        stop.disabled =
          !!session.stop_requested || session.state === "STOPPING";
        stop.onclick = async () => {
          stop.disabled = true;
          stop.textContent = "Stopping…";
          try {
            apply(
              await api("/sessions/" + session.id + "/stop", {
                method: "POST",
              }),
            );
            await load();
          } catch (e) {
            error(e.message);
            stop.disabled = false;
            stop.textContent = "Stop session";
          }
        };
      }
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
    el("live-xr-start").disabled = active || submitting || !catalog;
    el("live-xr-hand").disabled = active || submitting || !catalog;
    el("live-xr-task").disabled = active || submitting || !catalog;
    el("live-xr-start").textContent = active
      ? running.stop_requested
        ? "Stopping…"
        : sessionLabels[running.state] || "Session active"
      : submitting
        ? "Starting…"
        : "Start live session";
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
    const verified = catalog.verified_pairs.some(
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
        const target = result.target;
        el("live-xr-target").textContent =
          `${target.host} · ${target.duration_minutes} min limit`;
      }
      if (result.catalog && !catalog) setupChoices(result.catalog);
      sessions = result.sessions;
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
    } finally {
      loading = false;
      if (visible())
        timer = setTimeout(() => {
          if (visible()) load();
        }, 10000);
    }
  }
  el("live-xr-start-form").onsubmit = async (e) => {
    e.preventDefault();
    if (submitting || !catalog || sessions.some((s) => !terminal.has(s.state)))
      return;
    submitting = true;
    render();
    el("live-xr-message").textContent = "";
    error(null);
    try {
      apply(
        await api("/sessions", {
          method: "POST",
          body: JSON.stringify({
            accepted_license: el("live-xr-consent").checked,
            task: el("live-xr-task").value,
            robot: el("live-xr-hand").value,
          }),
        }),
      );
      await load();
    } catch (e) {
      error(e.message);
      el("live-xr-message").textContent = "";
    } finally {
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
