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
    poses = [],
    refreshing = false;
  const error = (message) => {
    el("hands-error").textContent = message || "";
    el("hands-error").hidden = !message;
  };
  const status = (message) => {
    el("hand-status").textContent = message;
  };
  const poseError = (message, { nameInvalid = false } = {}) => {
    el("hand-pose-error").textContent = message || "";
    el("hand-pose-error").hidden = !message;
    el("hand-pose-name").setAttribute(
      "aria-invalid",
      String(Boolean(message) && nameInvalid),
    );
  };
  const clearPoseFeedback = () => {
    error(null);
    poseError(null);
    el("hand-pose-message").textContent = "";
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
      button.disabled = hand.key === selected?.key;
      button.setAttribute(
        "aria-label",
        `${button.disabled ? "Selected" : "View"} ${hand.name}`,
      );
      button.addEventListener("click", () => choose(hand.key));
      action.append(button);
      row.append(info, action);
      body.append(row);
    }
  }
  async function load(force = false) {
    if (refreshing) return;
    refreshing = true;
    el("hands-refresh").disabled = true;
    const selectionToken = request;
    const preserved = metadata && {
      revision: metadata.revision,
      joints: { ...values },
      joint: el("hand-joint").value,
      poseName: el("hand-pose-name").value,
      poseId: el("hand-pose-list").value,
    };
    try {
      if (!catalog.length || force) {
        const nextCatalog = (await api("/api/hands")).hands;
        if (selectionToken !== request) return;
        catalog = nextCatalog;
      }
      if (selectionToken !== request) return;
      renderCatalog();
      if (!selected) {
        const params = new URLSearchParams(location.search);
        const key = catalog.some((h) => h.key === params.get("hand"))
          ? params.get("hand")
          : "shadow";
        await choose(key, params.get("side") || undefined);
      } else if (force) {
        const refreshed = catalog.find((hand) => hand.key === selected.key);
        if (
          metadata &&
          refreshed?.revision === metadata.revision &&
          refreshed.variants[side]?.state === "READY"
        ) {
          // Refresh catalog/poses without replacing an already loaded, pinned model.
          // This also preserves edits if refreshing the pose list fails.
          selected = refreshed;
          error(null);
          await loadPoses(selectionToken, preserved?.poseId);
          viewer?.render();
        } else await choose(selected.key, side, preserved);
      } else viewer?.render();
    } catch (e) {
      if (selectionToken === request) error(e.message);
    } finally {
      refreshing = false;
      el("hands-refresh").disabled = false;
    }
  }
  async function choose(key, chosenSide, preserved = null) {
    const token = ++request;
    clearTimeout(polling);
    clearPoseFeedback();
    metadata = null;
    poses = [];
    el("hand-pose-list").replaceChildren(
      new Option("Loading saved poses…", ""),
    );
    updatePoseActions();
    el("hand-use-simulation").hidden = true;
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
        const module = await import(
          document.querySelector('meta[name="hand-viewer-module"]')?.content ||
            "/static/hands-viewer.js" + version
        );
        if (token !== request) return;
        viewer = new module.HandsViewer(el("hand-canvas"));
      }
      el("hand-canvas").hidden = false;
      if (!(await viewer.load(model.urdf_url, model)) || token !== request)
        return;
      el("hand-view-help").hidden = false;
      metadata = model;
      if (model.simulation_robot) {
        const url = new URL(location.href);
        url.hash = "collection";
        url.searchParams.set("collection_view", "live");
        url.searchParams.set("live_hand", model.simulation_robot);
        el("hand-use-simulation").href = url.href;
        el("hand-use-simulation").hidden = false;
      }
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
      if (preserved?.revision === model.revision) {
        for (const joint of model.joints) {
          if (!joint.mimic && Number.isFinite(preserved.joints[joint.name]))
            values[joint.name] = Math.max(
              joint.lower,
              Math.min(joint.upper, preserved.joints[joint.name]),
            );
        }
        if (Object.hasOwn(values, preserved.joint))
          el("hand-joint").value = preserved.joint;
        el("hand-pose-name").value = preserved.poseName;
        viewer.setJoints(values);
      } else if (preserved) {
        error(
          "The model revision changed. The refreshed model starts with its neutral pose; saved poses are still available.",
        );
      }
      updateJoint();
    } catch (e) {
      if (token === request) {
        viewer?.clear();
        el("hand-canvas").hidden = true;
        el("hand-view-help").hidden = true;
        status("Unable to display this model");
        error(e.message);
      }
      return;
    }
    // Saved-pose availability does not determine whether the loaded model works.
    // Keep the canvas and joint controls usable when only this secondary read fails.
    try {
      await loadPoses(token, preserved?.poseId);
    } catch (e) {
      if (token === request) {
        el("hand-pose-list").replaceChildren(
          new Option("Saved poses unavailable — use Refresh", ""),
        );
        updatePoseActions();
        error(`Saved poses could not be loaded: ${e.message}`);
      }
    }
  }
  async function refreshDownload() {
    if (el("hands").hidden) return;
    const token = request;
    try {
      const nextCatalog = (await api("/api/hands")).hands;
      if (token !== request) return;
      catalog = nextCatalog;
      await choose(selected.key, side);
    } catch (e) {
      if (token !== request) return;
      error(e.message);
      status("Could not check download. Use Refresh to retry.");
    }
  }
  const displayJointValue = (value) => Number(value.toFixed(6));
  function updateJoint() {
    const joint = metadata?.joints.find(
      (j) => j.name === el("hand-joint").value,
    );
    if (!joint) return;
    const factor = joint.type === "prismatic" ? 1000 : 180 / Math.PI;
    el("hand-value-label").textContent =
      joint.type === "prismatic" ? "Position (mm)" : "Angle (degrees)";
    for (const input of [el("hand-value"), el("hand-angle")]) {
      input.min = displayJointValue(joint.lower * factor);
      input.max = displayJointValue(joint.upper * factor);
      input.step = "any";
      input.value = String(displayJointValue(values[joint.name] * factor));
    }
    el("hand-joint-note").textContent =
      `Limits: ${displayJointValue(joint.lower * factor)} to ${displayJointValue(joint.upper * factor)} ${joint.type === "prismatic" ? "mm" : "degrees"}.${metadata.joints.some((j) => j.mimic) ? " Linked joints move automatically." : ""}`;
  }
  function adjust(event) {
    const joint = metadata?.joints.find(
      (j) => j.name === el("hand-joint").value,
    );
    if (!joint) return;
    const input = event.target;
    if (input.value === "" || !Number.isFinite(Number(input.value))) {
      if (event.type !== "input") updateJoint();
      return;
    }
    const number = Number(input.value);
    if (!Number.isFinite(number)) return;
    const factor = joint.type === "prismatic" ? 1000 : 180 / Math.PI;
    values[joint.name] =
      number === Number(input.max)
        ? joint.upper
        : number === Number(input.min)
          ? joint.lower
          : Math.max(joint.lower, Math.min(joint.upper, number / factor));
    viewer.setJoints(values);
    if (input === el("hand-value") && event.type === "input")
      el("hand-angle").value = displayJointValue(values[joint.name] * factor);
    else updateJoint();
  }
  async function loadPoses(token = request, selectedId = "") {
    const result = await api(endpoint() + "/poses");
    if (token !== request) return;
    poses = result.poses;
    const select = el("hand-pose-list");
    select.replaceChildren(
      new Option(poses.length ? "Choose a saved pose…" : "No saved poses", ""),
    );
    for (const pose of poses) {
      const date = new Date(pose.created_at * 1000).toLocaleString();
      select.add(
        new Option(`${pose.name} · ${date} · ${pose.id.slice(0, 8)}`, pose.id),
      );
    }
    if (poses.some((pose) => pose.id === selectedId)) select.value = selectedId;
    updatePoseActions();
  }
  function updatePoseActions() {
    for (const id of ["hand-pose-load", "hand-pose-rename", "hand-pose-delete"])
      el(id).disabled = !el("hand-pose-list").value;
  }
  el("hand-pose-name").addEventListener("input", () => poseError(null));
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
  el("hand-value").addEventListener("blur", () => {
    if (
      el("hand-value").value === "" ||
      !Number.isFinite(Number(el("hand-value").value))
    )
      updateJoint();
  });
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
    updatePoseActions();
  });
  el("hand-pose-load").addEventListener("click", () => {
    const pose = poses.find((p) => p.id === el("hand-pose-list").value);
    if (!pose) return;
    clearPoseFeedback();
    if (pose.revision !== metadata.revision) {
      poseError("This pose belongs to another model version.");
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
    clearPoseFeedback();
    updateJoint();
    const name = el("hand-pose-name").value.trim();
    if (!name || name.length > 80) {
      poseError("Give the pose a name of 1–80 characters.", {
        nameInvalid: true,
      });
      el("hand-pose-name").focus();
      return;
    }
    const token = request,
      button =
        event.submitter ||
        el("hand-pose-form").querySelector('[type="submit"]');
    if (button.disabled) return;
    button.disabled = true;
    try {
      const pose = await api(endpoint() + "/poses", {
        method: "POST",
        body: JSON.stringify({
          name,
          revision: metadata.revision,
          joints: values,
        }),
      });
      if (token !== request) return;
      await loadPoses(token, pose.id);
      if (token !== request) return;
      el("hand-pose-message").textContent = `Saved “${pose.name}” locally.`;
      el("hand-pose-name").value = "";
    } catch (e) {
      if (token === request) {
        poseError(e.message);
        el("hand-pose-name").focus();
      }
    } finally {
      button.disabled = false;
    }
  });
  async function maintainPose(action) {
    const pose = poses.find((p) => p.id === el("hand-pose-list").value);
    if (!pose) return;
    const token = request;
    const url = `${endpoint()}/poses/${encodeURIComponent(pose.id)}`;
    clearPoseFeedback();
    const answer = await askUserDialog(
      action === "rename"
        ? "New pose name"
        : `Delete “${pose.name}”? This removes the saved local pose permanently.`,
      action === "rename" ? pose.name : null,
    );
    if (token !== request || answer === null || answer === false) return;
    const button = el(`hand-pose-${action}`);
    button.disabled = true;
    try {
      await api(
        url,
        action === "rename"
          ? { method: "PATCH", body: JSON.stringify({ name: answer.trim() }) }
          : { method: "DELETE" },
      );
      if (token !== request) return;
      await loadPoses(token, action === "rename" ? pose.id : "");
      if (token === request)
        el("hand-pose-message").textContent =
          action === "rename" ? "Pose renamed." : "Pose deleted.";
    } catch (e) {
      if (token === request) poseError(e.message);
    } finally {
      if (token === request) updatePoseActions();
    }
  }
  el("hand-pose-rename").addEventListener("click", () =>
    maintainPose("rename"),
  );
  el("hand-pose-delete").addEventListener("click", () =>
    maintainPose("delete"),
  );
  el("hand-pose-export").addEventListener("click", () => {
    if (!metadata) return;
    updateJoint();
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
    SkynetDialog.open(el("hand-export-dialog"));
  });
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
