(() => {
  const el = (id) => document.getElementById(id);
  const terminal = new Set(["CAPTURED", "STOPPED", "TIMED_OUT", "FAILED"]);
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
      text(name, "strong", session.id.slice(0, 8));
      text(
        name,
        "div",
        session.job_id ? "GPU job " + session.job_id : "Preparing",
        "secondary",
      );
      text(status, "strong", session.state.replaceAll("_", " "));
      text(status, "div", session.error || session.detail || "", "secondary");
      if (!terminal.has(session.state)) {
        const label = text(
          status,
          "label",
          "Headset result (reported)",
          "secondary",
        );
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
      } else if (
        session.headset_result &&
        session.headset_result !== "NOT_TESTED"
      ) {
        text(
          status,
          "div",
          "Headset result: " +
            session.headset_result.replaceAll("_", " ").toLowerCase(),
          "secondary",
        );
      }
      if (session.headset_result === "PORT_UNREACHABLE")
        text(
          status,
          "div",
          "Network access is blocked or unavailable from the headset. Verify the permitted route before retrying.",
          "collection-quality-warning",
        );
      if (session.checked_at)
        text(
          status,
          "div",
          "Checked " + new Date(session.checked_at).toLocaleTimeString(),
          "secondary",
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
      } else
        text(
          address,
          "span",
          terminal.has(session.state) ? "Session ended" : "Not ready",
        );
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
          session.recordings.length +
            " native demonstration saved. Validation/conversion is still required.",
          "secondary",
        );
    }
    const active = sessions.some((s) => !terminal.has(s.state));
    el("live-xr-start").disabled = active || submitting;
    el("live-xr-start").textContent = submitting
      ? "Starting…"
      : active
        ? "Session in progress"
        : "Start live session";
  }
  async function load() {
    if (loading) return;
    loading = true;
    const startedAtRevision = revision;
    clearTimeout(timer);
    try {
      const result = await api();
      if (revision !== startedAtRevision) return;
      sessions = result.sessions;
      el("live-xr-consent-field").hidden = result.license.accepted;
      el("live-xr-consent-status").textContent = result.license.accepted
        ? "CloudXR license acceptance is saved for this installation."
        : "";
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
      el("live-xr-message").textContent = sessions.some(
        (s) => !terminal.has(s.state),
      )
        ? "Session status refreshes every 10 seconds while this view is open."
        : "No live session is running.";
      error(null);
    } catch (e) {
      if (revision !== startedAtRevision) return;
      error(e.message);
      el("live-xr-message").textContent =
        "Could not refresh session status. Displayed information may be out of date.";
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
    submitting = true;
    render();
    el("live-xr-message").textContent = "Submitting the live session…";
    error(null);
    try {
      apply(
        await api("/sessions", {
          method: "POST",
          body: JSON.stringify({
            accepted_license: el("live-xr-consent").checked,
          }),
        }),
      );
      await load();
    } catch (e) {
      error(e.message);
      el("live-xr-message").textContent = "Could not start the live session.";
    } finally {
      submitting = false;
      render();
    }
  };
  el("live-xr-refresh").onclick = load;
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
