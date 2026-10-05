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
  async function requestJson(url, options = {}) {
    if (typeof window.api === "function")
      return window.api(url, {...options, timeoutMs: 45000});
    const timeout = AbortSignal.timeout(45000);
    const response = await fetch(url, {
      ...options,
      signal: options.signal ? AbortSignal.any([options.signal, timeout]) : timeout,
    });
    const value = await response.json();
    if (!response.ok) throw new Error(typeof value.detail === "string" ? value.detail : "Episode data could not be loaded");
    return value;
  }

  // Camera metadata uses ROS optical coordinates (+x right, +y down, +z forward).
  function project(point, camera) {
    if (camera.world_from_camera && camera.intrinsics) {
      const matrix = camera.world_from_camera, k = camera.intrinsics;
      const delta = point.map((value, index) => value - matrix[index][3]);
      const local = [0, 1, 2].map(index => delta.reduce((sum, value, axis) => sum + matrix[axis][index] * value, 0));
      return local[2] > 0 ? [k[0][0] * local[0] / local[2] + k[0][2], k[1][1] * local[1] / local[2] + k[1][2]] : null;
    }
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
    left: "m14 6-6 6 6 6",
    right: "m10 6 6 6-6 6",
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
  function playbackTime(seconds) {
    const centiseconds = Math.max(0, Math.round((seconds || 0) * 100));
    const minutes = Math.floor(centiseconds / 6000);
    const remainder = centiseconds % 6000;
    return `${String(minutes).padStart(2, "0")}:${String(Math.floor(remainder / 100)).padStart(2, "0")}.${String(remainder % 100).padStart(2, "0")}`;
  }

  function sequenceNavigation({host, position, previous, next, onSelect}) {
    if (!position) {
      position = node("span");
      previous = iconButton("left", "Previous recording", () => {});
      next = iconButton("right", "Next recording", () => {});
      previous.className = next.className = "";
      host.classList.add("live-review-recording-actions");
      host.append(position, previous, next);
    }
    let index = 0, count = 0;
    previous.onclick = () => { if (index > 0) onSelect(index - 1); };
    next.onclick = () => { if (index + 1 < count) onSelect(index + 1); };
    return {
      element: host,
      update(selected, total) {
        index = selected; count = total;
        position.textContent = `${index + 1}/${count}`;
        position.setAttribute("aria-label", `Recording ${index + 1} of ${count}`);
        previous.disabled = index === 0;
        next.disabled = index >= count - 1;
      },
    };
  }

  class EpisodeViewer {
    constructor(host, video) {
      this.host = host;
      this.video = video;
      this.generation = 0;
      this.enabled = new Set(["actual", "prediction", "demonstration"]);
      this.view = "all";
      this.geometry = { hand: true, scene: true };
      this.geometry.point_cloud = this.geometry.depth = true;
      this.showState = false;
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
      if (video) {
        this.stage.append(video);
        video.controls = false;
      }
      this.stage.append(this.canvas, this.sceneHost);
      this.buffering = node("div", "Loading inputs…", "episode-viewer-loading");
      this.buffering.hidden = true;
      this.buffering.setAttribute("role", "status");
      this.stateOverlay = node("pre", null, "episode-viewer-state-overlay");
      this.stateOverlay.hidden = true;
      this.stage.append(this.stateOverlay, this.buffering);
      this.legend = node("p", null, "secondary");
      this.retry = node("button", "Retry preview", "button button-outline");
      this.retry.type = "button";
      this.retry.hidden = true;
      this.retry.onclick = () => this.load(this.base, this.options);
      this.status = node("p", null, "secondary");
      this.status.setAttribute("role", "status");
      this.loop = iconButton("loop", "Loop playback", () => {
        this.looping = !this.looping;
        if (this.video) this.video.loop = this.looping;
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
        if (!this.collection && this.video?.readyState >= 1) this.video.currentTime = this.clockTime;
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
      this.rate = node("span", null, "episode-viewer-rate");
      this.frameLabel = node("span", null, "episode-viewer-frame");
      this.play.classList.add("episode-viewer-play");
      this.separator = node("span", null, "episode-viewer-playback-separator");
      this.separator.setAttribute("aria-hidden", "true");
      this.playbackRow = node("div", null, "episode-viewer-playback-row");
      this.playbackInfo = node("div", null, "episode-viewer-playback-info");
      const controls = this.controls = node("div", null, "episode-viewer-playback");
      const buttons = this.buttons = node("div", null, "form-actions");
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
      for (const event of video ? [
        "timeupdate",
        "loadeddata",
        "seeked",
        "pause",
        "play",
        "ended",
      ] : []) {
        this.events[event] = () => {
          if (this.collection) return;
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
    sceneMessage(text) {
      this.sceneStatus = text || "";
      if (this.options?.dataset) this.datasetStatus();
      else this.message(text);
    }
    datasetStatus(index = this.frameIndex(this.currentTime())) {
      if (!this.options?.dataset) return;
      const error = this.inputErrors?.get(Math.floor(index / this.inputChunkSize) * this.inputChunkSize);
      this.message(error ? "Saved inputs unavailable: " + error : this.imageError || (this.view === "interactive" ? this.sceneStatus : ""));
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
      this.video?.pause();
      cancelAnimationFrame(this.animation);
      this.draw();
    }
    seekFrame(index) {
      const frames = this.timeline();
      index = Math.max(0, Math.min(frames.length - 1, index));
      this.clockTime = this.frameTimeAt(frames[index]);
      this.frameTime = this.clockTime;
      if (!this.collection && this.video?.readyState >= 1) this.video.currentTime = this.clockTime;
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
      return !this.collection && this.video?.readyState >= 2
        ? this.video.currentTime || 0
        : this.clockTime || 0;
    }
    async startPlayback() {
      if (!this.active) return;
      const generation = this.generation;
      const request = (this.playRequest = (this.playRequest || 0) + 1);
      const frames = this.timeline();
      if (
        this.video?.ended ||
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
      if (!this.collection && this.video?.readyState >= 1) {
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
      if (this.collection || !(this.video?.readyState >= 2)) {
        const nextTime =
          this.clockOffset + (performance.now() - this.clockStart) / 1000;
        if (this.options.dataset && !this.savedFrames.has(this.frameIndex(nextTime))) {
          this.ensureInputs(this.frameIndex(nextTime));
          this.clockStart = performance.now();
          this.clockOffset = this.clockTime || 0;
        } else this.clockTime = nextTime;
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
      if (!this.video?.requestVideoFrameCallback) return;
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
      return requestJson(url, options);
    }

    async load(
      base,
      {
        collection = false,
        episode = 0,
        robot = null,
        sourceNames = null,
        timeline = [],
        simulationHz = null,
        toolbarActions = null,
        onLoadState = null,
        onFrame = null,
        dataset = null,
      } = {},
    ) {
      const previousDataset = this.options?.dataset;
      const previousView = this.view;
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
        simulationHz,
        toolbarActions,
        onLoadState,
        onFrame,
        dataset,
      };
      this.clockTime = 0;
      this.playRequest = 0;
      this.playing = false;
      this.scrubbing = false;
      this.lastNotifiedFrame = null;
      const generation = ++this.generation;
      this.base = base;
      this.collection = collection;
      this.savedFrames = new Map();
      this.inputBatches = new Map();
      this.inputRequests = new Map();
      this.imageCache = new Map();
      this.inputErrors = new Map();
      this.imageError = null;
      this.sceneStatus = "";
      this.inputChunkSize = Math.max(1, Math.min(120, Number(dataset?.chunkFrames) || 60));
      this.buffering.hidden = !dataset;
      this.stateOverlay.hidden = true;
      if (dataset) this.options.timeline = Array.from({length: dataset.steps}, (_, index) => ({index, time: index / dataset.hz}));
      this.controls.classList.toggle("episode-viewer-playback-recording", collection);
      if (collection) {
        this.buttons.replaceChildren(this.previous, this.play, this.next, this.separator, this.loop);
        this.playbackInfo.replaceChildren(this.rate, this.frameLabel, this.time);
        this.playbackRow.replaceChildren(this.buttons, this.playbackInfo);
        this.controls.replaceChildren(this.playbackRow, this.seek);
      } else {
        this.buttons.replaceChildren(this.loop, this.previous, this.play, this.next);
        this.controls.replaceChildren(this.buttons, this.seek, this.time);
      }
      if (!collection) this.watchFrames();
      if (this.video) this.video.hidden = collection;
      this.aside.hidden = collection;
      this.episode = episode;
      this.canvas.width = this.canvas.width;
      this.data = null;
      this.hand = null;
      this.handPoseData = null;
      this.handPoseLayer = "actual";
      this.view = dataset && previousDataset?.base === dataset.base
        ? previousView
        : dataset?.modalities.some(item => item.modality === "rgb") ? "images" : collection ? "interactive" : "all";
      this.legend.textContent = "";
      this.aside.replaceChildren(
        node("h4", "Hand"),
        node("p", "Loading hand information…", "secondary"),
      );
      this.renderControls();
      onLoadState?.("loading", "Loading scene…");
      this.message("Loading episode views…");
      if (dataset) this.draw();
      if (robot) this.loadHand(robot, generation);
      if (dataset && !base) {
        await this.accept({state: "READY", viewer: {kind: "collection", frames: [], views: []}}, generation);
        return;
      }
      const query = collection ? `?episode=${episode}` : "";
      try {
        const result = await this.json(
          base + "/viewer" + query,
          collection ? { method: "POST" } : {},
        );
        if (generation !== this.generation) return;
        if (result.state === "PREPARING") {
          this.sceneMessage(result.detail);
          this.pollTimer = setTimeout(() => this.poll(generation), 1200);
          return;
        }
        await this.accept(result, generation);
      } catch (error) {
        if (generation === this.generation) {
          if (dataset) {
            // Stored observations remain inspectable if their raw-source scene
            // is unavailable. Never generate substitute observations.
            this.sourceWarning = "Recorded 3D scene unavailable: " + error.message;
            await this.accept({state: "READY", viewer: {kind: "collection", frames: [], views: []}}, generation);
            return;
          }
          this.message(error.message);
          this.retry.hidden = false;
          this.options.onLoadState?.("error", error.message);
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
          if (this.options.dataset) {
            this.sourceWarning = "Recorded 3D scene unavailable: " + error.message;
            await this.accept({state: "READY", viewer: {kind: "collection", frames: [], views: []}}, generation);
            return;
          }
          this.message(error.message);
          this.retry.hidden = false;
          this.options.onLoadState?.("error", error.message);
        }
      }
    }

    async accept(result, generation) {
      if (result.state !== "READY") {
        if (this.options.dataset) {
          this.sourceWarning = result.detail || result.error || "Recorded 3D scene is unavailable.";
          return this.accept({state: "READY", viewer: {kind: "collection", frames: [], views: []}}, generation);
        }
        this.retry.hidden = result.state !== "FAILED";
        this.message(
          result.detail || "This result has no saved camera or keypoint data.",
        );
        if (result.robot) this.loadHand(result.robot, generation);
        this.options.onLoadState?.("error", result.detail || "Episode view unavailable.");
        return;
      }
      let data =
        result.viewer ||
        (await this.json(
          this.base + `/viewer/viewer.json?episode=${this.episode}`,
        ));
      if (generation !== this.generation) return;
      if (this.options.dataset?.sourceChecksum && data.source_sha256 !== this.options.dataset.sourceChecksum) {
        this.sourceWarning = "The original recording has changed. Only this dataset's saved inputs are shown.";
        data = {kind: "collection", frames: [], views: []};
      }
      this.data = this.options.sourceNames
        ? { ...data, source_names: this.options.sourceNames }
        : data;
      if (this.collection && data.hand_visual_available)
        this.data = { ...this.data, hand_visual_url: this.base + "/hand/simulation.urdf" };
      this.renderControls();
      await this.loadHand(data.robot, generation);
      if (generation !== this.generation) return;
      this.legend.textContent = this.collection
        ? ""
        : "Prediction: model target · Actual: observed hand · Demonstration: original recording";
      if (data.demonstration)
        this.legend.textContent += ` · Source ${data.demonstration.session_id.slice(0, 8)}, recording ${data.demonstration.source_index + 1}. Aligned by elapsed time; hidden when the recording ends.`;
      this.sceneMessage([...(data.warnings || []), this.sourceWarning].filter(Boolean).join(" "));
      if (this.collection) await this.loadScene();
      if (generation !== this.generation) return;
      this.draw();
      this.options.onLoadState?.("ready");
    }

    async loadScene() {
      if (this.scene || !this.data) return;
      const generation = this.generation;
      this.sceneHost.dataset.loading = "true";
      try {
        const { EpisodeScene } = await import(
          document.querySelector('meta[name="episode-scene-module"]')
            ?.content || "./episode-scene.js" + version
        );
        if (generation === this.generation && this.active) {
          this.scene = new EpisodeScene(this.sceneHost);
          const warnings = this.scene.setObjects(this.data?.scene_objects);
          try {
            const loading = this.scene.load(this.data, this.hand);
            // Geometry and keypoints are already available. Fit and draw them
            // while the immutable hand meshes load independently.
            this.draw();
            await loading;
          } catch (error) {
            warnings.push("Hand model: " + error.message);
          }
          if (generation !== this.generation || !this.active) return;
          if (warnings.length) this.sceneMessage([this.sceneStatus, ...warnings].filter(Boolean).join(" · "));
          this.draw();
        }
      } catch (error) {
        if (generation === this.generation) {
          this.scene?.dispose();
          this.scene = null;
          this.sceneMessage("Interactive view unavailable: " + error.message);
        }
      } finally {
        if (generation === this.generation) delete this.sceneHost.dataset.loading;
      }
    }

    ensureInputs(index) {
      const dataset = this.options?.dataset;
      if (!dataset || index < 0 || index >= dataset.steps || !this.active) return;
      const size = this.inputChunkSize;
      const start = Math.floor(index / size) * size;
      if (this.inputBatches.has(start)) {
        if (index - start >= Math.floor(size / 4) && start + size < dataset.steps)
          this.ensureInputs(start + size);
        return;
      }
      if (this.inputRequests.has(start) || this.inputErrors.has(start)) return;
      const generation = this.generation;
      const controller = new AbortController();
      this.inputRequests.set(start, {controller});
      const query = new URLSearchParams({start, count: size, gateway: document.getElementById("gateway")?.value || "auto"});
      this.json(`${dataset.base}/episodes/${dataset.episode}/frames?${query}`, {signal: controller.signal})
        .then(result => {
          if (generation !== this.generation || !this.active) return;
          this.inputBatches.set(start, result.frames.map(frame => frame.index));
          const used = new Set(dataset.modalities.map(item => item.modality));
          for (const frame of result.frames) this.savedFrames.set(frame.index, {
            ...frame,
            images: used.has("rgb") ? frame.images : [],
            state: used.has("state") ? frame.state : [],
            point_cloud: used.has("point_cloud") ? frame.point_cloud : [],
            depth: used.has("depth") ? frame.depth : [],
          });
          // A few neighboring chunks support seeking without retaining an
          // entire recording's RGB images in browser memory.
          while (this.inputBatches.size > 3) {
            const currentStart = Math.floor(this.frameIndex(this.currentTime()) / size) * size;
            const remove = [...this.inputBatches.keys()].filter(value => value !== currentStart && value !== start)[0];
            if (remove === undefined) break;
            for (const key of this.inputBatches.get(remove)) this.savedFrames.delete(key);
            this.inputBatches.delete(remove);
          }
          this.draw();
        })
        .catch(error => {
          if (generation !== this.generation || !this.active || controller.signal.aborted) return;
          this.inputErrors.set(start, error.message);
          this.message("Saved inputs unavailable: " + error.message);
          this.retry.hidden = false;
          this.pausePlayback();
        })
        .finally(() => {
          if (generation === this.generation) this.inputRequests.delete(start);
        });
    }

    drawInputs(observations, frame) {
      const images = observations?.images || [];
      const context = this.canvas.getContext("2d");
      if (!images.length) {
        context.clearRect(0, 0, this.canvas.width, this.canvas.height);
        return;
      }
      const width = images.reduce((sum, item) => sum + item.width, 0);
      const height = Math.max(...images.map(item => item.height));
      if (this.canvas.width !== width) this.canvas.width = width;
      if (this.canvas.height !== height) this.canvas.height = height;
      context.clearRect(0, 0, width, height);
      let offset = 0;
      let ready = true;
      for (const item of images) {
        const key = `${observations.index}:${item.id}`;
        let saved = this.imageCache.get(key);
        if (!saved) {
          const image = new Image();
          saved = {image, ready: false};
          this.imageCache.set(key, saved);
          const generation = this.generation;
          image.onload = () => {
            if (generation !== this.generation || !this.active) return;
            saved.ready = true;
            this.draw();
          };
          image.onerror = () => {
            if (generation === this.generation && this.active) {
              this.imageError = "A saved camera image could not be decoded.";
              this.datasetStatus();
            }
          };
          image.src = item.data_url;
          while (this.imageCache.size > 12) this.imageCache.delete(this.imageCache.keys().next().value);
        }
        if (saved.ready) {
          context.drawImage(saved.image, offset, 0, item.width, item.height);
          context.fillStyle = "rgba(0,0,0,.65)";
          context.fillRect(offset, 0, item.width, 24);
          context.fillStyle = "white";
          context.font = "12px sans-serif";
          context.fillText(item.label || item.id, offset + 8, 16);
          if (this.showState && observations.state?.length) {
            const points = frame?.actual?.map(point => project(point, item));
            context.save();
            context.beginPath();
            context.rect(offset, 0, item.width, item.height);
            context.clip();
            context.strokeStyle = context.fillStyle = colors.actual;
            context.lineWidth = 1.5;
            for (const [a, b] of this.data?.edges || [])
              if (points?.[a] && points?.[b]) {
                context.beginPath();
                context.moveTo(offset + points[a][0], points[a][1]);
                context.lineTo(offset + points[b][0], points[b][1]);
                context.stroke();
              }
            for (const point of points || []) if (point) {
              context.beginPath();
              context.arc(offset + point[0], point[1], 2.5, 0, Math.PI * 2);
              context.fill();
            }
            context.restore();
          }
        } else ready = false;
        offset += item.width;
      }
      this.buffering.hidden = ready;
      this.stateOverlay.textContent = (observations.state || []).map(item =>
        `${item.name}: ${item.values.flat(Infinity).map((value, index) => `${item.labels?.[index] ? item.labels[index] + "=" : ""}${Number(value).toFixed(3)}`).join("  ")}`,
      ).join("\n");
    }

    renderControls() {
      this.tabs.replaceChildren();
      const dataset = this.options?.dataset;
      this.tabs.hidden = this.collection && !dataset;
      this.layers.replaceChildren();
      const views = this.data?.views || [];
      const choices = dataset ? [{id: "images", label: "Images"}, {id: "interactive", label: "3D"}] : this.collection ? [] : [
        { id: "all", label: views.length ? "All cameras" : "Video" },
        ...views,
        { id: "interactive", label: "Interactive" },
      ];
      for (const choice of choices) {
        const button = node("button", choice.label, "button button-outline");
        button.type = "button";
        button.setAttribute("aria-pressed", String(this.view === choice.id));
        button.disabled = dataset ? choice.id === "images" && !dataset.modalities.some(item => item.modality === "rgb") :
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
          if (choice.id === "interactive") await this.loadScene();
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
      if (dataset && this.view === "images") {
        if (dataset.modalities.some(item => item.modality === "state")) {
          const wrapper = node("label", null, "check-field"), input = node("input");
          input.type = "checkbox";
          input.checked = this.showState;
          input.onchange = () => { this.showState = input.checked; this.draw(); };
          wrapper.append(input, node("span", "State"));
          this.layers.append(wrapper);
        }
        if (this.options.toolbarActions) this.layers.append(this.options.toolbarActions);
        return;
      }
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
                (this.collection && !this.data?.hand_visual_url) ||
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
      if (dataset && this.view === "interactive")
        for (const [key, label] of [["point_cloud", "Point cloud"], ["depth", "Depth"]]) {
          if (!dataset.modalities.some(item => item.modality === key)) continue;
          const wrapper = node("label", null, "check-field"), input = node("input");
          input.type = "checkbox";
          input.checked = this.geometry[key];
          input.onchange = () => { this.geometry[key] = input.checked; this.draw(); };
          wrapper.append(input, node("span", label));
          this.layers.append(wrapper);
        }
      if (this.collection && this.options?.toolbarActions)
        this.layers.append(this.options.toolbarActions);
    }

    async loadHand(robot, generation) {
      if (this.collection) {
        this.hand = null;
        this.aside.hidden = true;
        return;
      }
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
      this.datasetStatus(index);
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
      const lastIndex = last?.state_index ?? last?.index ?? timeline.length - 1;
      if (this.collection) {
        this.rate.textContent = this.options.simulationHz ? `${this.options.simulationHz} Hz` : "";
        this.rate.hidden = !this.options.simulationHz;
        this.frameLabel.textContent = current ? `Frame ${stateIndex} / ${lastIndex}` : "No frames";
        this.time.textContent = `${playbackTime(this.frameTimeAt(current))} / ${playbackTime(this.frameTimeAt(last))}`;
      } else {
        this.time.textContent = current
          ? `Frame ${stateIndex} of ${lastIndex} · ${this.frameTimeAt(current).toFixed(3)} s`
          : "No frames";
      }
      this.play.disabled = !timeline.length && !(this.video?.readyState >= 2);
      this.seek.disabled = !timeline.length;
      this.previous.disabled = !timeline.length || index === 0;
      this.next.disabled = !timeline.length || index === timeline.length - 1;
      if (current && index !== this.lastNotifiedFrame) {
        this.lastNotifiedFrame = index;
        this.options?.onFrame?.(index);
      }
      const renderedTime =
        this.view === "interactive" || this.video?.seeking || this.scrubbing
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
      const observations = this.options.dataset ? this.savedFrames.get(index) : null;
      if (this.options.dataset) this.ensureInputs(index);
      this.updateHandPose(frame);
      this.sceneHelp.hidden = this.view !== "interactive";
      this.canvas.hidden = this.view === "interactive";
      this.sceneHost.hidden = this.view !== "interactive";
      this.stateOverlay.hidden = !this.options.dataset || this.view !== "images" || !this.showState || !observations?.state?.length;
      if (this.options.dataset) this.buffering.hidden = Boolean(observations) || this.inputErrors.has(Math.floor(index / this.inputChunkSize) * this.inputChunkSize);
      if (this.view === "interactive") {
        this.scene?.update(
          frame,
          this.data?.edges || [],
          this.enabled,
          this.collection,
          this.geometry,
          observations,
        );
        return;
      }
      if (this.options.dataset) {
        this.drawInputs(observations, frame);
        return;
      }
      if (!(this.video?.readyState >= 2) || this.frameTime === null) return;
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
      this.video?.pause();
      this.handViewer?.dispose();
      this.handViewer = null;
      this.handViewerReady = false;
      this.handPoseData = null;
      this.lastHandPose = Symbol();
      if (this.frameCallback !== undefined)
        this.video?.cancelVideoFrameCallback?.(this.frameCallback);
      ++this.generation;
      clearTimeout(this.pollTimer);
      cancelAnimationFrame(this.animation);
      this.scene?.dispose();
      this.scene = null;
      for (const request of this.inputRequests?.values() || []) request.controller.abort();
      this.inputRequests?.clear();
      this.savedFrames?.clear();
      this.inputBatches?.clear();
      this.imageCache?.clear();
      this.sourceWarning = null;
      delete this.sceneHost.dataset.loading;
    }
  }

  class DatasetPreview {
    constructor(host) {
      this.host = host;
      this.generation = 0;
      this.message = node("p", null, "secondary");
      this.message.setAttribute("role", "status");
      this.viewerHost = node("div");
      this.viewerHost.id = host.id + "-viewer";
      this.actions = node("div");
      this.navigation = sequenceNavigation({host: this.actions, onSelect: index => this.select(index)});
      this.retry = node("button", "Retry preview", "button button-outline");
      this.retry.type = "button";
      this.retry.hidden = true;
      this.retry.onclick = () => this.open(this.id, true);
      host.append(node("h4", "Input preview"), this.message, this.viewerHost, this.retry);
    }
    async open(id, force = false) {
      if (this.id === id && this.active && !force) return;
      this.close();
      this.active = true;
      this.id = id;
      const generation = ++this.generation;
      this.base = `/api/data/datasets/${encodeURIComponent(id)}/preview`;
      this.message.textContent = "Loading saved inputs…";
      this.retry.hidden = true;
      this.viewerHost.hidden = true;
      try {
        const result = await requestJson(this.base);
        if (generation !== this.generation || !this.active) return;
        if (result.state !== "READY" || !result.episodes?.length) {
          this.message.textContent = result.detail || "No saved input preview is available for this dataset.";
          return;
        }
        this.data = result;
        this.message.textContent = "";
        this.select(0);
      } catch (error) {
        if (generation !== this.generation || !this.active) return;
        this.message.textContent = error.message;
        this.retry.hidden = false;
      }
    }
    select(index) {
      const episode = this.data?.episodes[index];
      if (!episode || !this.active) return;
      this.navigation.update(index, this.data.episodes.length);
      this.viewerHost.hidden = false;
      const source = episode.source_session_id && Number.isInteger(episode.source_recording_index)
        ? `/api/collection/live/sessions/${encodeURIComponent(episode.source_session_id)}/recordings/${episode.source_recording_index}` : null;
      window.SkynetEpisodeViewer.open(this.viewerHost.id, null, source, {
        collection: true,
        episode: episode.source_episode || 0,
        simulationHz: episode.hz,
        toolbarActions: this.actions,
        dataset: {base: this.base, episode: episode.index, steps: episode.steps, hz: episode.hz, modalities: this.data.modalities, chunkFrames: episode.chunk_frames, sourceChecksum: source ? episode.source_sha256 : null},
      });
    }
    close() {
      this.active = false;
      ++this.generation;
      this.viewerHost.hidden = true;
      window.SkynetEpisodeViewer.close(this.viewerHost.id);
    }
  }

  const datasets = new Map();
  window.SkynetEpisodeViewer = {
    sequenceNavigation,
    openDataset(hostId, datasetId) {
      let preview = datasets.get(hostId);
      if (!preview) {
        preview = new DatasetPreview(document.getElementById(hostId));
        datasets.set(hostId, preview);
      }
      preview.open(datasetId);
      return preview;
    },
    closeDataset(hostId) { datasets.get(hostId)?.close(); },
    open(hostId, videoId, base, options) {
      let viewer = instances.get(hostId);
      if (viewer && viewer.host !== document.getElementById(hostId)) {
        viewer.close();
        viewer.observer.disconnect();
        viewer = null;
      }
      if (!viewer) {
        viewer = new EpisodeViewer(
          document.getElementById(hostId),
          videoId ? document.getElementById(videoId) : null,
        );
        instances.set(hostId, viewer);
      }
      viewer.load(base, options);
      return viewer;
    },
    pause(hostId) {
      instances.get(hostId)?.pausePlayback();
    },
    close(hostId) {
      instances.get(hostId)?.close();
    },
  };
})();
