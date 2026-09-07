(() => {
  const el = (id) => document.getElementById(id);
  let token = 0,
    base = "",
    data = null,
    episode = null,
    frame = 0,
    timer,
    animation,
    playing = false;
  const dialog = el("live-review-dialog");
  const video = el("live-review-video");
  let reviewSession;
  let videoToken = 0,
    videoTimer;
  function clearVideo() {
    ++videoToken;
    clearTimeout(videoTimer);
    video.pause();
    video.removeAttribute("src");
    video.load();
    video.hidden = true;
    el("live-review-video-download").hidden = true;
    el("live-review-video-retry").hidden = true;
  }
  async function loadVideo(ownToken, ownVideoToken, index, start = false) {
    clearTimeout(videoTimer);
    const current = () =>
      ownToken === token && ownVideoToken === videoToken && dialog.open;
    try {
      const result = await request(
        base + `/video?episode=${index}`,
        start ? { method: "POST" } : {},
      );
      if (!current()) return;
      el("live-review-video-retry").hidden = true;
      if (result.state === "READY") {
        const url =
          base +
          `/video.mp4?episode=${index}&v=${encodeURIComponent(result.sha256)}`;
        video.src = url;
        video.hidden = false;
        el("live-review-video-download").href = url;
        el("live-review-video-download").hidden = false;
        el("live-review-video-status").textContent =
          result.kind === "capture"
            ? "Collection video"
            : result.capture_error
              ? `Scene replay · video capture failed: ${result.capture_error}`
              : "Scene replay · rendered from recorded states";
      } else if (result.state === "FAILED") {
        throw new Error(result.error);
      } else if (result.state === "NOT_PREPARED") {
        await loadVideo(ownToken, ownVideoToken, index, true);
      } else {
        el("live-review-video-status").textContent =
          result.detail || "Preparing video…";
        videoTimer = setTimeout(
          () => loadVideo(ownToken, ownVideoToken, index),
          1500,
        );
      }
    } catch (error) {
      if (!current()) return;
      el("live-review-video-status").textContent =
        "Video unavailable: " + error.message;
      el("live-review-video-retry").hidden = false;
    }
  }
  video.addEventListener("error", () => {
    if (!video.hasAttribute("src") || !dialog.open) return;
    el("live-review-video-status").textContent =
      "The video could not be played. Retry to reload it.";
    el("live-review-video-retry").hidden = false;
  });
  video.addEventListener("timeupdate", () => {
    if (!episode || !el("live-review-data").open) return;
    let index = 0;
    while (
      index + 1 < episode.frames.length &&
      episode.frames[index + 1].time_seconds <= video.currentTime
    )
      index++;
    frame = index;
    draw();
  });
  el("live-review-video-retry").onclick = () => {
    clearVideo();
    el("live-review-video-status").textContent = "Retrying video…";
    loadVideo(token, videoToken, Number(el("live-review-episode").value), true);
  };
  const status = (message) => {
    el("live-review-status").textContent = message;
  };
  async function request(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      signal: AbortSignal.timeout(40000),
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
    playing = false;
    cancelAnimationFrame(animation);
    el("live-review-play").textContent = "Play values";
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
    el("live-review-frame").textContent =
      `Frame ${current.state_index} of ${episode.steps} · ${current.time_seconds.toFixed(3)} s`;
    el("live-review-seek").value = frame;
    el("live-review-prev").disabled = frame === 0;
    el("live-review-next").disabled = frame === episode.frames.length - 1;
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
    clearVideo();
    el("live-review-video-status").textContent = "Checking video…";
    loadVideo(token, videoToken, Number(el("live-review-episode").value));
    episode = data.episodes[Number(el("live-review-episode").value)];
    frame = 0;
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
    el("live-review-seek").max = episode.frames.length - 1;
    el("live-review-time-note").textContent =
      `${data.time_note} ${episode.sampled ? `Preview samples ${episode.frames.length} of ${episode.state_count} states; the original download contains every state.` : `All ${episode.state_count} saved scene states are available.`}`;
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
        data = await request(base + "/review.json");
        if (ownToken !== token || !dialog.open) return;
        el("live-review-content").hidden = false;
        el("live-review-context").textContent =
          `${data.task_name} · ${data.hand_name}`;
        status(
          `${data.episodes.length} demonstration${data.episodes.length === 1 ? "" : "s"} · ${data.episodes.reduce((sum, ep) => sum + ep.duration_seconds, 0).toFixed(1)} s`,
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
        status("Downloading recording…");
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
    clearVideo();
    el("live-review-data").open = false;
    clearTimeout(timer);
    const ownToken = ++token;
    base = `/api/collection/live/sessions/${reviewSession.id}/recordings/${index}`;
    data = episode = null;
    el("live-review-content").hidden = true;
    el("live-review-retry").hidden = true;
    el("live-review-context").textContent =
      `Session ${reviewSession.id.slice(0, 8)} · Recording ${index + 1}`;
    status("Checking the local recording copy…");
    load(ownToken);
  }
  window.openLiveReview = (session, index = 0) => {
    reviewSession = session;
    const select = el("live-review-recording");
    const count = session.recordings?.length || 1;
    select.replaceChildren(
      ...Array.from(
        { length: count },
        (_, i) => new Option(`Recording ${i + 1}`, i),
      ),
    );
    select.value = index >= 0 && index < count ? index : 0;
    select.parentElement.hidden = count < 2;
    if (!dialog.open) dialog.showModal();
    selectRecording(Number(select.value));
  };
  el("live-review-recording").onchange = (event) =>
    selectRecording(Number(event.target.value));
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      pause();
      video.pause();
    }
  });
  el("live-review-close").onclick = () => dialog.close();
  dialog.addEventListener("close", () => {
    ++token;
    clearTimeout(timer);
    pause();
    clearVideo();
  });
  el("live-review-retry").onclick = () => {
    status("Retrying download…");
    el("live-review-retry").hidden = true;
    load(token, true);
  };
  el("live-review-episode").onchange = chooseEpisode;
  el("live-review-field").onchange = draw;
  el("live-review-prev").onclick = () => {
    pause();
    frame = Math.max(0, frame - 1);
    draw();
  };
  el("live-review-next").onclick = () => {
    pause();
    frame = Math.min(episode.frames.length - 1, frame + 1);
    draw();
  };
  el("live-review-seek").oninput = (event) => {
    pause();
    frame = Number(event.target.value);
    draw();
  };
  el("live-review-play").onclick = () => {
    if (playing) {
      pause();
      return;
    }
    if (frame === episode.frames.length - 1) frame = 0;
    playing = true;
    el("live-review-play").textContent = "Pause";
    const started = performance.now(),
      offset = episode.frames[frame].time_seconds;
    const tick = (now) => {
      if (!playing) return;
      const seconds = offset + (now - started) / 1000;
      let next = frame;
      while (
        next + 1 < episode.frames.length &&
        episode.frames[next + 1].time_seconds <= seconds
      )
        next++;
      if (next !== frame) {
        frame = next;
        draw();
      }
      if (frame === episode.frames.length - 1) pause();
      else animation = requestAnimationFrame(tick);
    };
    animation = requestAnimationFrame(tick);
  };
})();
