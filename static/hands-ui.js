(() => {
  const el = (id) => document.getElementById(id);
  const version = new URL(document.currentScript.src).search;
  let catalog = [],
    selected = null,
    side = "right",
    metadata = null,
    values = {},
    viewer = null,
    request = 0,
    polling = null,
    poses = [];
  const error = (message) => {
    el("hands-error").textContent = message || "";
    el("hands-error").hidden = !message;
  };
  const status = (message) => {
    el("hand-status").textContent = message;
  };
  const endpoint = () => `/api/hands/${selected.key}/${side}`;
  async function api(url, options = {}) {
    const response = await fetch(url, {
      ...options,
      headers: { "Content-Type": "application/json" },
      signal: AbortSignal.timeout(45000),
    });
    const data = await response.json();
    if (!response.ok)
      throw new Error(
        typeof data.detail === "string"
          ? data.detail
          : "The request could not be completed",
      );
    return data;
  }
  function sourceLinks() {
    el("hand-source-link").href =
      `https://github.com/${selected.repository}/blob/${selected.revision}/${selected.sides[side]}`;
    el("hand-revision").textContent =
      `Source version ${selected.revision.slice(0, 10)}`;
    el("hand-model-notes").textContent = selected.notes;
    el("hand-license-link").hidden = true;
  }
  function renderCatalog() {
    el("hands-count").textContent = `${catalog.length} models`;
    const body = el("hands-catalog");
    body.replaceChildren();
    for (const hand of catalog) {
      const row = document.createElement("tr");
      row.classList.toggle("is-selected", hand.key === selected?.key);
      const info = document.createElement("td"),
        name = document.createElement("span"),
        detail = document.createElement("span");
      name.className = "hand-model-name";
      name.textContent = hand.name;
      detail.className = "secondary";
      const count = Object.values(hand.variants).filter(
        (v) => v.state === "READY",
      ).length;
      detail.textContent = count
        ? `${count} side${count === 1 ? "" : "s"} stored`
        : "Available to download";
      info.append(name, detail);
      const action = document.createElement("td"),
        button = document.createElement("button");
      button.className = "button button-outline";
      button.type = "button";
      button.textContent = hand.key === selected?.key ? "Selected" : "View";
      button.setAttribute("aria-label", `View ${hand.name}`);
      button.addEventListener("click", () => choose(hand.key));
      action.append(button);
      row.append(info, action);
      body.append(row);
    }
  }
  async function load(force = false) {
    try {
      if (!catalog.length || force) catalog = (await api("/api/hands")).hands;
      renderCatalog();
      if (!selected) {
        const params = new URLSearchParams(location.search);
        const key = catalog.some((h) => h.key === params.get("hand"))
          ? params.get("hand")
          : "shadow";
        await choose(key, params.get("side") || undefined);
      } else if (force) await choose(selected.key, side);
      else viewer?.render();
    } catch (e) {
      error(e.message);
    }
  }
  async function choose(key, chosenSide) {
    const token = ++request;
    clearTimeout(polling);
    error(null);
    metadata = null;
    values = {};
    viewer?.clear();
    el("hand-canvas").hidden = true;
    el("hand-view-help").hidden = true;
    el("hand-controls").hidden = true;
    el("hand-fit").disabled = true;
    el("hand-pose-message").textContent = "";
    el("hand-pose-name").value = "";
    selected = catalog.find((h) => h.key === key);
    if (!selected) return;
    side =
      chosenSide && selected.sides[chosenSide]
        ? chosenSide
        : selected.sides.right
          ? "right"
          : Object.keys(selected.sides)[0];
    const url = new URL(location.href);
    url.searchParams.set("hand", key);
    url.searchParams.set("side", side);
    history.replaceState(null, "", url);
    el("hand-title").textContent = selected.name;
    el("hand-source-label").textContent = selected.source_kind;
    const select = el("hand-side");
    select.replaceChildren();
    for (const s of Object.keys(selected.sides))
      select.add(new Option(s === "left" ? "Left hand" : "Right hand", s));
    select.value = side;
    select.disabled = select.length === 1;
    renderCatalog();
    sourceLinks();
    el("hand-install").hidden = true;
    const state = selected.variants[side];
    if (state.state === "UNSUPPORTED") {
      status("Unsupported source model");
      error(state.error);
      return;
    }
    if (state.state === "DOWNLOADING") {
      status(state.detail || "Downloading model…");
      polling = setTimeout(refreshDownload, 1500);
      return;
    }
    if (state.state !== "READY") {
      status(
        state.state === "FAILED"
          ? "Model download failed"
          : "Download the mesh model to view it here.",
      );
      el("hand-install").hidden = false;
      el("hand-install").textContent =
        state.state === "FAILED" ? "Retry download" : "Download model";
      if (state.error) error(state.error);
      return;
    }
    status("Loading 3D model…");
    try {
      const model = await api(endpoint() + "/model");
      if (token !== request) return;
      if (!viewer) {
        const module = await import("/static/hands-viewer.js" + version);
        if (token !== request) return;
        viewer = new module.HandsViewer(el("hand-canvas"));
      }
      el("hand-canvas").hidden = false;
      if (!(await viewer.load(model.urdf_url, model)) || token !== request)
        return;
      el("hand-view-help").hidden = false;
      metadata = model;
      values = Object.fromEntries(
        model.joints
          .filter((j) => !j.mimic)
          .map((j) => [j.name, Math.max(j.lower, Math.min(j.upper, 0))]),
      );
      el("hand-joint").replaceChildren();
      for (const j of model.joints)
        if (!j.mimic) el("hand-joint").add(new Option(j.name, j.name));
      el("hand-controls").hidden = false;
      el("hand-fit").disabled = false;
      el("hand-license-link").hidden = false;
      el("hand-license-link").href = model.license_url;
      const linked = model.joints.filter((j) => j.mimic).length;
      status(
        `${model.mesh_count} mesh assets · ${model.joints.length - linked} adjustable joints${linked ? ` · ${linked} linked joints` : ""}`,
      );
      updateJoint();
      await loadPoses(token);
    } catch (e) {
      if (token === request) {
        viewer?.clear();
        el("hand-canvas").hidden = true;
        el("hand-view-help").hidden = true;
        status("Unable to display this model");
        error(e.message);
      }
    }
  }
  async function refreshDownload() {
    if (el("hands").hidden) return;
    try {
      catalog = (await api("/api/hands")).hands;
      await choose(selected.key, side);
    } catch (e) {
      error(e.message);
      status("Could not check download. Use Refresh to retry.");
    }
  }
  function updateJoint() {
    const joint = metadata?.joints.find(
      (j) => j.name === el("hand-joint").value,
    );
    if (!joint) return;
    const factor = joint.type === "prismatic" ? 1000 : 180 / Math.PI;
    el("hand-value-label").textContent =
      joint.type === "prismatic" ? "Position (mm)" : "Angle (degrees)";
    for (const input of [el("hand-value"), el("hand-angle")]) {
      input.min = joint.lower * factor;
      input.max = joint.upper * factor;
      input.value = (values[joint.name] * factor).toFixed(2);
    }
    el("hand-joint-note").textContent =
      `Limits: ${(joint.lower * factor).toFixed(1)} to ${(joint.upper * factor).toFixed(1)} ${joint.type === "prismatic" ? "mm" : "degrees"}.${metadata.joints.some((j) => j.mimic) ? " Linked joints move automatically." : ""}`;
  }
  function adjust(event) {
    const joint = metadata?.joints.find(
      (j) => j.name === el("hand-joint").value,
    );
    if (!joint) return;
    const input = event.target;
    if (input.value === "") return;
    const number = Number(input.value);
    if (!Number.isFinite(number)) return;
    const factor = joint.type === "prismatic" ? 1000 : 180 / Math.PI;
    values[joint.name] = Math.max(
      joint.lower,
      Math.min(joint.upper, number / factor),
    );
    viewer.setJoints(values);
    if (input === el("hand-value") && event.type === "input")
      el("hand-angle").value = values[joint.name] * factor;
    else updateJoint();
  }
  async function loadPoses(token = request) {
    const result = await api(endpoint() + "/poses");
    if (token !== request) return;
    poses = result.poses;
    const select = el("hand-pose-list");
    select.replaceChildren(
      new Option(poses.length ? "Choose a saved pose…" : "No saved poses", ""),
    );
    for (const pose of poses) select.add(new Option(pose.name, pose.id));
    el("hand-pose-load").disabled = true;
  }
  el("hands-refresh").addEventListener("click", () => load(true));
  el("hand-side").addEventListener("change", (e) =>
    choose(selected.key, e.target.value),
  );
  el("hand-fit").addEventListener("click", () => viewer?.fit());
  el("hand-install").addEventListener("click", async () => {
    el("hand-install").disabled = true;
    error(null);
    try {
      await api(endpoint() + "/install", { method: "POST" });
      await refreshDownload();
    } catch (e) {
      error(e.message);
    } finally {
      el("hand-install").disabled = false;
    }
  });
  el("hand-joint").addEventListener("change", updateJoint);
  el("hand-angle").addEventListener("input", adjust);
  el("hand-value").addEventListener("change", adjust);
  el("hand-value").addEventListener("input", adjust);
  el("hand-reset").addEventListener("click", () => {
    if (!metadata) return;
    values = Object.fromEntries(
      metadata.joints
        .filter((j) => !j.mimic)
        .map((j) => [j.name, Math.max(j.lower, Math.min(j.upper, 0))]),
    );
    viewer.setJoints(values);
    updateJoint();
  });
  el("hand-pose-list").addEventListener("change", () => {
    el("hand-pose-load").disabled = !el("hand-pose-list").value;
  });
  el("hand-pose-load").addEventListener("click", () => {
    const pose = poses.find((p) => p.id === el("hand-pose-list").value);
    if (!pose) return;
    if (pose.revision !== metadata.revision) {
      error("This pose belongs to another model version.");
      return;
    }
    values = { ...pose.joints };
    viewer.setJoints(values);
    updateJoint();
    el("hand-pose-message").textContent = `Loaded “${pose.name}”.`;
  });
  el("hand-pose-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!metadata) return;
    const token = request,
      button = event.submitter;
    button.disabled = true;
    try {
      const pose = await api(endpoint() + "/poses", {
        method: "POST",
        body: JSON.stringify({
          name: el("hand-pose-name").value,
          revision: metadata.revision,
          joints: values,
        }),
      });
      if (token !== request) return;
      await loadPoses(token);
      el("hand-pose-message").textContent = `Saved “${pose.name}” locally.`;
      el("hand-pose-name").value = "";
    } catch (e) {
      if (token === request) error(e.message);
    } finally {
      button.disabled = false;
    }
  });
  el("hand-pose-export").addEventListener("click", () => {
    if (!metadata) return;
    const url = new URL(endpoint() + "/export", location.origin);
    url.searchParams.set("revision", metadata.revision);
    url.searchParams.set("joints", JSON.stringify(values));
    el("hand-export-download").href = url;
    el("hand-export-download").download = `${selected.key}-${side}-pose.json`;
    el("hand-export-json").value = JSON.stringify(
      {
        schema: "skynet.hand-pose/v1",
        key: selected.key,
        side,
        revision: metadata.revision,
        joints: values,
        units: "radians; prismatic joints in meters",
      },
      null,
      2,
    );
    el("hand-export-message").textContent =
      "If your browser restricts file downloads, use Copy JSON.";
    el("hand-export-dialog").showModal();
  });
  el("hand-export-close").addEventListener("click", () =>
    el("hand-export-dialog").close(),
  );
  el("hand-export-copy").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(el("hand-export-json").value);
      el("hand-export-message").textContent = "Copied pose JSON.";
    } catch {
      el("hand-export-json").focus();
      el("hand-export-json").select();
      el("hand-export-message").textContent =
        "Clipboard access is unavailable. The JSON is selected; copy it manually.";
    }
  });
  window.loadHands = load;
  if (location.hash === "#hands") load();
})();
