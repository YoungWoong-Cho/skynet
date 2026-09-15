(() => {
  const version = new URL(document.currentScript.src).search;
  const instances = new Map();
  const colors = {
    actual: "#16835c",
    prediction: "#e68b22",
    demonstration: "#7860cf",
  };
  const node = (tag, text, className) => {
    const element = document.createElement(tag);
    if (text) element.textContent = text;
    if (className) element.className = className;
    return element;
  };

  // Camera metadata uses ROS optical coordinates (+x right, +y down, +z forward).
  function project(point, camera) {
    const q = camera.quaternion_world_ros,
      p = camera.position_world,
      k = camera.intrinsic_matrix;
    if (!q || !p || !k) return null;
    const norm = Math.hypot(...q);
    const [w, x, y, z] = q.map((value) => value / norm);
    const r = [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ];
    const delta = point.map((value, i) => value - p[i]);
    const local = [0, 1, 2].map((i) =>
      delta.reduce((sum, value, j) => sum + r[j][i] * value, 0),
    );
    if (local[2] <= 0) return null;
    return [
      (k[0][0] * local[0]) / local[2] + k[0][2],
      (k[1][1] * local[1]) / local[2] + k[1][2],
    ];
  }

  const icons = {
    loop: "M17 2l4 4-4 4M3 11V8a2 2 0 0 1 2-2h16M7 22l-4-4 4-4m14-1v3a2 2 0 0 1-2 2H3",
    previous: "M5 4v16M19 4 7 12l12 8Z",
    next: "M19 4v16M5 4l12 8-12 8Z",
    play: "M7 4l14 8-14 8Z",
    pause: "M7 4v16M17 4v16",
  };
  function playbackIcon(button, icon, label) {
    if (button.title === label) return;
    button.innerHTML = `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${icons[icon]}"/></svg>`;
    button.setAttribute("aria-label", label);
    button.title = label;
  }
  function iconButton(icon, label, action) {
    const button = node("button", null, "button button-outline");
    button.type = "button";
    playbackIcon(button, icon, label);
    button.onclick = action;
    return button;
  }

  class EpisodeViewer {
    constructor(host, video) {
      this.host = host;
      this.video = video;
      this.generation = 0;
      this.enabled = new Set(["actual", "prediction", "demonstration"]);
      this.view = "all";
      this.geometry = { hand: true, scene: true };
      host.classList.add("episode-viewer");
      this.main = node("div", null, "episode-viewer-main");
      this.tabs = node("div", null, "episode-viewer-tabs");
      this.tabs.setAttribute("aria-label", "Episode views");
      this.layers = node("div", null, "episode-viewer-layers");
      this.canvas = node("canvas");
      this.canvas.setAttribute(
        "aria-label",
        "Episode camera view with optional keypoints",
      );
      this.sceneHost = node("div", null, "episode-viewer-scene");
      this.sceneHost.hidden = true;
      this.sceneHelp = node(
        "p",
        "Recorded scene · Drag to rotate · Scroll to zoom",
        "secondary",
      );
      this.sceneHelp.hidden = true;
      this.stage = node("div", null, "episode-viewer-stage");
      this.stage.append(video, this.canvas, this.sceneHost);
      video.controls = false;
      this.legend = node("p", null, "secondary");
      this.retry = node("button", "Retry preview", "button button-outline");
      this.retry.type = "button";
      this.retry.hidden = true;
      this.retry.onclick = () => this.load(this.base, this.options);
      this.status = node("p", null, "secondary");
      this.status.setAttribute("role", "status");
      this.loop = iconButton("loop", "Loop playback", () => {
        this.looping = !this.looping;
        this.video.loop = this.looping;
        this.loop.setAttribute("aria-pressed", String(this.looping));
      });
      this.loop.setAttribute("aria-pressed", "false");
      this.previous = iconButton("previous", "Previous frame", () =>
        this.step(-1),
      );
      this.play = iconButton("play", "Play", () =>
        this.playing ? this.pausePlayback() : this.startPlayback(),
      );
      this.next = iconButton("next", "Next frame", () => this.step(1));
      this.seek = node("input");
      this.seek.type = "range";
      this.seek.min = "0";
      this.seek.step = "1";
      this.seek.value = "0";
      this.seek.setAttribute("aria-label", "Episode frame");
      this.seek.addEventListener("pointerdown", () => {
        this.clockTime = this.currentTime();
        this.scrubbing = true;
        this.resumeAfterSeek = this.playing;
        this.pausePlayback();
      });
      const finishSeek = () => {
        if (!this.scrubbing) return;
        if (this.video.readyState >= 1) this.video.currentTime = this.clockTime;
        this.scrubbing = false;
        if (this.resumeAfterSeek) this.startPlayback();
      };
      this.seek.addEventListener("pointerup", finishSeek);
      this.seek.addEventListener("pointercancel", finishSeek);
      this.seek.addEventListener("change", finishSeek);
      this.seek.oninput = () => {
        const index = Number(this.seek.value);
        this.pausePlayback();
        this.seekFrame(index);
      };
      this.time = node("output");
      const controls = node("div", null, "episode-viewer-playback");
      const buttons = node("div", null, "form-actions");
      buttons.append(this.loop, this.previous, this.play, this.next);
      controls.append(buttons, this.seek, this.time);
      this.aside = node("aside", null, "episode-viewer-hand");
      this.main.append(
        this.tabs,
        this.layers,
        this.legend,
        this.stage,
        this.sceneHelp,
        controls,
        this.status,
        this.retry,
      );
      host.append(this.main, this.aside);
      this.events = {};
      for (const event of [
        "timeupdate",
        "loadeddata",
        "seeked",
        "pause",
        "play",
        "ended",
      ]) {
        this.events[event] = () => {
          if (!video.requestVideoFrameCallback && !video.seeking)
            this.frameTime = video.currentTime;
          if (
            event === "loadeddata" &&
            this.active &&
            this.playing &&
            video.paused
          ) {
            // Continue the same timeline when a prepared video becomes available.
            video.currentTime = this.clockTime || 0;
            this.startPlayback();
          }
          if (event === "play") {
            this.playing = !video.paused;
            this.tick();
          }
          if (event === "pause" || event === "ended")
            this.playing = !video.paused;
          this.draw();
        };
        video.addEventListener(event, this.events[event]);
      }
      document.addEventListener("visibilitychange", () => {
        if (document.hidden && this.active) this.pausePlayback();
      });
      this.observer = new ResizeObserver(() => this.draw());
      this.observer.observe(this.stage);
    }

    message(text) {
      this.status.textContent = text || "";
    }
    timeline() {
      return this.options?.timeline?.length
        ? this.options.timeline
        : this.data?.frames || [];
    }
    frameTimeAt(frame) {
      return frame?.time_seconds ?? frame?.time ?? 0;
    }
    frameIndex(time) {
      const frames = this.timeline();
      let lo = 0,
        hi = frames.length - 1;
      while (lo < hi) {
        const mid = Math.ceil((lo + hi) / 2);
        if (this.frameTimeAt(frames[mid]) <= time + 1e-6) lo = mid;
        else hi = mid - 1;
      }
      return lo;
    }
    pausePlayback() {
      this.playRequest = (this.playRequest || 0) + 1;
      this.playing = false;
      this.video.pause();
      cancelAnimationFrame(this.animation);
      this.draw();
    }
    seekFrame(index) {
      const frames = this.timeline();
      index = Math.max(0, Math.min(frames.length - 1, index));
      this.clockTime = this.frameTimeAt(frames[index]);
      this.frameTime = this.clockTime;
      if (this.video.readyState >= 1) this.video.currentTime = this.clockTime;
      this.clockStart = performance.now();
      this.clockOffset = this.clockTime;
      this.draw();
    }
    step(direction) {
      this.pausePlayback();
      this.seekFrame(this.frameIndex(this.currentTime()) + direction);
    }
    currentTime() {
      if (this.scrubbing) return this.clockTime || 0;
      return this.video.readyState >= 2
        ? this.video.currentTime || 0
        : this.clockTime || 0;
    }
    async startPlayback() {
      if (!this.active) return;
      const generation = this.generation;
      const request = (this.playRequest = (this.playRequest || 0) + 1);
      const frames = this.timeline();
      if (
        this.video.ended ||
        (frames.length &&
          this.frameIndex(this.currentTime()) === frames.length - 1)
      )
        this.seekFrame(0);
      this.clockStart = performance.now();
      this.clockOffset = this.currentTime();
      this.playing = true;
      // Seeking back from the end can temporarily leave only metadata loaded.
      // play() waits for the seek/buffer; switching to the fallback clock here
      // would leave the video paused once its frames become available again.
      if (this.video.readyState >= 1) {
        try {
          await this.video.play();
        } catch (error) {
          if (generation !== this.generation || request !== this.playRequest)
            return;
          this.playing = false;
          this.message(error.message);
        }
      }
      if (generation !== this.generation || request !== this.playRequest)
        return;
      this.tick();
    }
    tick() {
      cancelAnimationFrame(this.animation);
      if (!this.active || !this.playing) return;
      if (this.video.readyState < 2) {
        this.clockTime =
          this.clockOffset + (performance.now() - this.clockStart) / 1000;
        const frames = this.timeline();
        const end = this.frameTimeAt(frames.at(-1));
        if (this.clockTime >= end) {
          if (this.looping) this.seekFrame(0);
          else {
            this.clockTime = end;
            this.playing = false;
          }
        }
      }
      this.draw();
      if (this.playing)
        this.animation = requestAnimationFrame(() => this.tick());
    }
    watchFrames() {
      if (!this.video.requestVideoFrameCallback) return;
      this.frameCallback = this.video.requestVideoFrameCallback(
        (_, metadata) => {
          if (!this.active) return;
          this.frameTime = metadata.mediaTime;
          this.draw();
          this.watchFrames();
        },
      );
    }

    async json(url, options = {}) {
      const response = await fetch(url, {
        ...options,
        signal: AbortSignal.timeout(45000),
      });
      const value = await response.json();
      if (!response.ok)
        throw new Error(
          typeof value.detail === "string"
            ? value.detail
            : "Episode data could not be loaded",
        );
      return value;
    }

    async load(
      base,
      {
        collection = false,
        episode = 0,
        robot = null,
        sourceNames = null,
        timeline = [],
        onFrame = null,
      } = {},
    ) {
      this.close();
      this.active = true;
      this.frameTime = null;
      this.retry.hidden = true;
      this.options = {
        collection,
        episode,
        robot,
        sourceNames,
        timeline,
        onFrame,
      };
      this.clockTime = 0;
      this.playRequest = 0;
      this.playing = false;
      this.scrubbing = false;
      this.lastNotifiedFrame = null;
      this.watchFrames();
      const generation = ++this.generation;
      this.base = base;
      this.collection = collection;
      this.episode = episode;
      this.canvas.width = this.canvas.width;
      this.data = null;
      this.hand = null;
      this.handPoseData = null;
      this.handPoseLayer = "actual";
      this.view = "all";
      this.legend.textContent = "";
      this.aside.replaceChildren(
        node("h4", "Hand"),
        node("p", "Loading hand information…", "secondary"),
      );
      this.renderControls();
      this.message("Loading episode views…");
      if (robot) this.loadHand(robot, generation);
      const query = collection ? `?episode=${episode}` : "";
      try {
        const result = await this.json(
          base + "/viewer" + query,
          collection ? { method: "POST" } : {},
        );
        if (generation !== this.generation) return;
        if (result.state === "PREPARING") {
          this.message(result.detail);
          this.pollTimer = setTimeout(() => this.poll(generation), 1200);
          return;
        }
        await this.accept(result, generation);
      } catch (error) {
        if (generation === this.generation) {
          this.message(error.message);
          this.retry.hidden = false;
        }
      }
    }

    async poll(generation) {
      try {
        const result = await this.json(
          this.base + `/viewer?episode=${this.episode}`,
        );
        if (generation !== this.generation) return;
        if (result.state === "PREPARING") {
          this.pollTimer = setTimeout(() => this.poll(generation), 1500);
          return;
        }
        await this.accept(result, generation);
      } catch (error) {
        if (generation === this.generation) {
          this.message(error.message);
          this.retry.hidden = false;
        }
      }
    }

    async accept(result, generation) {
      if (result.state !== "READY") {
        this.retry.hidden = result.state !== "FAILED";
        this.message(
          result.detail || "This result has no saved camera or keypoint data.",
        );
        if (result.robot) this.loadHand(result.robot, generation);
        return;
      }
      const data =
        result.viewer ||
        (await this.json(
          this.base + `/viewer/viewer.json?episode=${this.episode}`,
        ));
      if (generation !== this.generation) return;
      this.data = this.options.sourceNames
        ? { ...data, source_names: this.options.sourceNames }
        : data;
      this.renderControls();
      await this.loadHand(data.robot, generation);
      if (generation !== this.generation) return;
      this.legend.textContent = this.collection
        ? ""
        : "Prediction: model target · Actual: observed hand · Demonstration: original recording";
      if (data.demonstration)
        this.legend.textContent += ` · Source ${data.demonstration.session_id.slice(0, 8)}, recording ${data.demonstration.source_index + 1}. Aligned by elapsed time; hidden when the recording ends.`;
      if (this.collection) {
        this.video.src =
          this.base + `/viewer/views.mp4?episode=${this.episode}`;
        this.video.load();
      }
      this.message((data.warnings || []).join(" "));
      this.draw();
    }

    renderControls() {
      this.tabs.replaceChildren();
      this.layers.replaceChildren();
      const views = this.data?.views || [];
      const choices = [
        { id: "all", label: views.length ? "All cameras" : "Video" },
        ...views,
        { id: "interactive", label: "Interactive" },
      ];
      for (const choice of choices) {
        const button = node("button", choice.label, "button button-outline");
        button.type = "button";
        button.setAttribute("aria-pressed", String(this.view === choice.id));
        button.disabled =
          choice.id === "interactive" &&
          !this.data?.scene_objects?.length &&
          !this.data?.frames?.some(
            (frame) =>
              frame.actual ||
              frame.prediction ||
              frame.demonstration ||
              Object.keys(frame.hand_poses || {}).length,
          );
        button.onclick = async () => {
          this.view = choice.id;
          this.renderControls();
          if (choice.id === "interactive" && !this.scene) {
            const generation = this.generation;
            try {
              const { EpisodeScene } = await import(
                document.querySelector('meta[name="episode-scene-module"]')
                  ?.content || "./episode-scene.js" + version
              );
              if (generation === this.generation && this.active) {
                this.scene = new EpisodeScene(this.sceneHost);
                const warnings = this.scene.setObjects(
                  this.data?.scene_objects,
                );
                try {
                  await this.scene.load(this.data, this.hand);
                } catch (error) {
                  warnings.push("Hand model: " + error.message);
                }
                if (warnings.length) this.message(warnings.join(" · "));
                else if (this.data?.scene_appearance)
                  this.message(this.data.scene_appearance);
              }
            } catch (error) {
              if (generation === this.generation) {
                this.scene?.dispose();
                this.scene = null;
                this.message("Interactive view unavailable: " + error.message);
              }
            }
          }
          this.draw();
        };
        this.tabs.append(button);
      }
      const labels = this.collection
        ? { actual: "Keypoints" }
        : {
            prediction: "Prediction",
            actual: "Actual",
            demonstration: "Demonstration",
          };
      if (this.view === "interactive")
        for (const [key, label] of [
          ["hand", "Hand model"],
          ["scene", "Scene objects"],
        ]) {
          const wrapper = node("label", null, "check-field"),
            input = node("input");
          input.type = "checkbox";
          input.checked = this.geometry[key];
          input.disabled =
            key === "scene"
              ? !this.data?.scene_objects?.length
              : !this.data?.kinematics_urdf ||
                !this.data?.frames?.some(
                  (frame) => Object.keys(frame.hand_poses || {}).length,
                );
          input.checked = !input.disabled && this.geometry[key];
          if (input.disabled)
            wrapper.title =
              key === "scene"
                ? this.data?.frames?.some(
                    (frame) => Object.keys(frame.objects || {}).length,
                  )
                  ? "Object positions and rotations are saved, but their shapes and sizes were not saved."
                  : "Scene geometry was not saved for this episode."
                : "Joint poses or hand geometry were not saved for this episode.";
          if (!input.disabled && key === "hand") {
            const missing = Object.keys(labels).filter(
              (layer) =>
                this.data.frames.some((frame) => frame[layer]) &&
                !this.data.frames.some((frame) => frame.hand_poses?.[layer]),
            );
            if (missing.length)
              wrapper.title =
                missing.map((layer) => labels[layer]).join(" and ") +
                " have saved keypoints but no joint poses for a hand model.";
          }
          input.onchange = () => {
            this.geometry[key] = input.checked;
            this.draw();
          };
          wrapper.append(input, node("span", label));
          this.layers.append(wrapper);
        }
      for (const [key, label] of Object.entries(labels)) {
        const available = Boolean(
          this.data?.frames?.some(
            (frame) => frame[key] || frame.hand_poses?.[key],
          ),
        );
        const wrapper = node("label", null, "check-field");
        const input = node("input");
        input.type = "checkbox";
        input.checked = available && this.enabled.has(key);
        input.disabled = !available;
        input.onchange = () => {
          input.checked ? this.enabled.add(key) : this.enabled.delete(key);
          this.draw();
        };
        const text = node("span", label);
        text.style.color = colors[key];
        wrapper.append(input, text);
        if (!available)
          wrapper.title =
            "No saved " + label.toLowerCase() + " keypoints for this episode";
        this.layers.append(wrapper);
      }
    }

    async loadHand(robot, generation) {
      const key = `${generation}:${robot}`;
      if (this.handRequestKey === key) return this.handRequest;
      this.handRequestKey = key;
      this.handRequest = this.showHand(robot, generation);
      return this.handRequest;
    }
    async showHand(robot, generation) {
      try {
        const { hands } = await this.json("/api/hands");
        const hand =
          hands.find((h) =>
            robot?.startsWith("skynet_" + h.key.replaceAll("-", "_") + "_"),
          ) ||
          (robot?.startsWith("floating_shadow_")
            ? hands.find((h) => h.key === "shadow")
            : null);
        if (generation !== this.generation) return;
        this.hand = hand;
        if (!hand) {
          this.aside.replaceChildren(
            node("h4", "Hand"),
            node("p", "Hand model unavailable."),
          );
          return;
        }
        const side = robot.endsWith("_left") ? "left" : "right";
        const model = await this.json(
          `/api/hands/${encodeURIComponent(hand.key)}/${side}/model`,
        );
        const { HandsViewer } = await import(
          document.querySelector('meta[name="hand-viewer-module"]')?.content ||
            "./hands-viewer.js" + version
        );
        if (generation !== this.generation) return;
        this.handViewer?.dispose();
        const viewport = node("div", null, "episode-hand-viewport");
        const link = node("a", "View in Hands", "text-button");
        link.href = `/?hand=${encodeURIComponent(hand.key)}&side=${side}#hands`;
        this.handSide = side;
        const poseField = node("label", null, "field");
        this.handPoseSelect = node("select");
        this.handPoseSelect.setAttribute("aria-label", "Hand pose");
        this.handPoseSelect.onchange = () => {
          this.handPoseLayer = this.handPoseSelect.value;
          this.lastHandPose = Symbol();
          this.draw();
        };
        poseField.append(node("span", "Pose"), this.handPoseSelect);
        poseField.hidden = this.collection;
        this.handPoseStatus = node("p", null, "secondary");
        this.handPoseStatus.setAttribute("role", "status");
        this.aside.replaceChildren(
          node("h4", hand.name),
          node(
            "p",
            `${model.mesh_count} mesh assets · ${model.joints.filter((j) => !j.mimic).length} adjustable joints`,
            "secondary",
          ),
          poseField,
          viewport,
          this.handPoseStatus,
          link,
        );
        const viewer = (this.handViewer = new HandsViewer(viewport));
        await viewer.load(model.urdf_url, model);
        if (generation === this.generation) {
          this.handViewerReady = true;
          this.draw();
        }
      } catch (error) {
        if (generation === this.generation)
          this.aside.replaceChildren(
            node("h4", "Hand"),
            node("p", error.message, "secondary"),
          );
      }
    }

    updateHandPose(frame) {
      if (!this.handViewerReady || !this.data) return;
      if (this.handPoseData !== this.data) {
        this.handPoseData = this.data;
        this.lastHandPose = Symbol();
        const available = ["actual", "prediction", "demonstration"].filter(
          (key) => this.data.frames?.some((sample) => sample.hand_poses?.[key]),
        );
        if (!available.includes(this.handPoseLayer))
          this.handPoseLayer = available[0] || "actual";
        this.handPoseSelect.replaceChildren();
        for (const key of ["actual", "prediction", "demonstration"]) {
          const option = node("option", key[0].toUpperCase() + key.slice(1));
          option.value = key;
          option.disabled = !available.includes(key);
          this.handPoseSelect.append(option);
        }
        this.handPoseSelect.value = this.handPoseLayer;
        this.handPoseSelect.disabled = available.length < 2;
        this.handPlaybackError = null;
        try {
          if (!available.length)
            throw new Error("No saved joint poses for this episode.");
          const palm = this.hand.floating_hand?.palm
            ?.replaceAll("{side}", this.handSide)
            .replaceAll("{s}", this.handSide[0]);
          this.handViewer.configurePlayback({
            palm,
            jointNames: this.data.joint_names || [],
            robot: this.data.robot,
            side: this.handSide,
            sourceNames: this.data.source_names,
          });
        } catch (error) {
          this.handPlaybackError = error.message;
        }
        this.handPoseSelect.parentElement.hidden =
          this.collection || Boolean(this.handPlaybackError);
      }
      const pose = this.handPlaybackError
        ? null
        : frame?.hand_poses?.[this.handPoseLayer];
      if (pose === this.lastHandPose) return;
      this.lastHandPose = pose;
      const applied = this.handViewer.setFingerPose(pose);
      this.handPoseStatus.textContent = applied ? "" : "Default pose";
      this.handPoseStatus.hidden = applied;
    }

    draw() {
      if (!this.active) return;
      const time = this.currentTime();
      const timeline = this.timeline();
      const index = this.frameIndex(time);
      const current = timeline[index];
      playbackIcon(
        this.play,
        this.playing ? "pause" : "play",
        this.playing ? "Pause" : "Play",
      );
      this.seek.max = String(Math.max(0, timeline.length - 1));
      if (!this.scrubbing) this.seek.value = String(index);
      const stateIndex = current?.state_index ?? current?.index ?? index;
      const last = timeline.at(-1);
      this.time.textContent = current
        ? `Frame ${stateIndex} of ${last?.state_index ?? last?.index ?? timeline.length - 1} · ${this.frameTimeAt(current).toFixed(3)} s`
        : "No frames";
      this.play.disabled = !timeline.length && this.video.readyState < 2;
      this.seek.disabled = !timeline.length;
      this.previous.disabled = !timeline.length || index === 0;
      this.next.disabled = !timeline.length || index === timeline.length - 1;
      if (current && index !== this.lastNotifiedFrame) {
        this.lastNotifiedFrame = index;
        this.options?.onFrame?.(index);
      }
      const renderedTime =
        this.view === "interactive" || this.video.seeking || this.scrubbing
          ? time
          : (this.frameTime ?? time);
      const frames = this.data?.frames || [];
      let lo = 0,
        hi = frames.length - 1;
      while (lo < hi) {
        const mid = Math.ceil((lo + hi) / 2);
        if (frames[mid].time <= renderedTime + 1e-6) lo = mid;
        else hi = mid - 1;
      }
      const frame = frames[lo];
      this.updateHandPose(frame);
      this.sceneHelp.hidden = this.view !== "interactive";
      this.canvas.hidden = this.view === "interactive";
      this.sceneHost.hidden = this.view !== "interactive";
      if (this.view === "interactive") {
        this.scene?.update(
          frame,
          this.data?.edges || [],
          this.enabled,
          this.collection,
          this.geometry,
        );
        return;
      }
      if (this.video.readyState < 2 || this.frameTime === null) return;
      const views = this.data?.views || [];
      const selected = views.find((view) => view.id === this.view);
      const rect = selected?.rect || [
        0,
        0,
        this.video.videoWidth,
        this.video.videoHeight,
      ];
      if (!rect[2] || !rect[3]) return;
      if (this.canvas.width !== rect[2]) this.canvas.width = rect[2];
      if (this.canvas.height !== rect[3]) this.canvas.height = rect[3];
      const context = this.canvas.getContext("2d");
      context.drawImage(this.video, ...rect, 0, 0, rect[2], rect[3]);
      for (const view of selected ? [selected] : views) {
        context.save();
        const x = selected ? 0 : view.rect[0],
          y = selected ? 0 : view.rect[1];
        context.beginPath();
        context.rect(x, y, view.width, view.height);
        context.clip();
        for (const key of this.enabled) {
          const points = frame?.[key]?.map((point) => project(point, view));
          if (!points) continue;
          context.strokeStyle = context.fillStyle = colors[key];
          context.lineWidth = 1.5;
          for (const [a, b] of this.data.edges || [])
            if (points[a] && points[b]) {
              context.beginPath();
              context.moveTo(x + points[a][0], y + points[a][1]);
              context.lineTo(x + points[b][0], y + points[b][1]);
              context.stroke();
            }
          for (const p of points)
            if (p) {
              context.beginPath();
              context.arc(x + p[0], y + p[1], 2.5, 0, Math.PI * 2);
              context.fill();
            }
        }
        context.restore();
      }
    }

    close() {
      this.active = false;
      this.playing = false;
      this.video.pause();
      this.handViewer?.dispose();
      this.handViewer = null;
      this.handViewerReady = false;
      this.handPoseData = null;
      this.lastHandPose = Symbol();
      if (this.frameCallback !== undefined)
        this.video.cancelVideoFrameCallback?.(this.frameCallback);
      ++this.generation;
      clearTimeout(this.pollTimer);
      cancelAnimationFrame(this.animation);
      this.scene?.dispose();
      this.scene = null;
    }
  }

  window.SkynetEpisodeViewer = {
    open(hostId, videoId, base, options) {
      let viewer = instances.get(hostId);
      if (!viewer) {
        viewer = new EpisodeViewer(
          document.getElementById(hostId),
          document.getElementById(videoId),
        );
        instances.set(hostId, viewer);
      }
      viewer.load(base, options);
      return viewer;
    },
    close(hostId) {
      instances.get(hostId)?.close();
    },
  };
})();
