import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { JSDOM } from "jsdom";
const dom = new JSDOM(
  await readFile(new URL("../static/index.html", import.meta.url), "utf8"),
  {
    runScripts: "outside-only",
    pretendToBeVisual: true,
    url: "http://localhost:8080/#collection",
  },
);
const { window } = dom,
  requests = [];
let videoPauses = 0;
window.HTMLMediaElement.prototype.pause = function () {
  videoPauses++;
};
window.HTMLMediaElement.prototype.load = function () {};
const get = (id) => window.document.getElementById(id);
const watchdogs = new Map();
let nextWatchdog = 100000;
const realSetTimeout = window.setTimeout.bind(window);
const realClearTimeout = window.clearTimeout.bind(window);
window.setTimeout = (callback, delay, ...args) => {
  if (delay !== 30000) return realSetTimeout(callback, delay, ...args);
  const id = ++nextWatchdog;
  watchdogs.set(id, callback);
  return id;
};
window.clearTimeout = (id) => {
  watchdogs.delete(id);
  realClearTimeout(id);
};
const mediaEvent = (event, readyState = 2) => {
  Object.defineProperty(get("live-review-video"), "readyState", {
    value: readyState,
    configurable: true,
  });
  get("live-review-video").dispatchEvent(new window.Event(event));
};
const dialog = get("live-review-dialog");
dialog.showModal = () => {
  dialog.open = true;
};
dialog.close = () => {
  dialog.open = false;
  dialog.dispatchEvent(new window.Event("close"));
};
window.fetch = (path, options) =>
  new Promise((resolve) =>
    requests.push({
      path,
      options,
      resolve: (data) => resolve({ ok: true, json: async () => data }),
    }),
  );
const flush = () => new Promise((resolve) => setImmediate(resolve));
const data = {
  task_name: "Stick",
  hand_name: "Right",
  size_bytes: 100,
  time_note: "60 Hz",
  value_note: "Saved order",
  episodes: [
    {
      steps: 2,
      state_count: 3,
      duration_seconds: 2 / 60,
      frames: [0, 1, 2].map((i) => ({
        state_index: i,
        time_seconds: i / 60,
        action_index: i ? i - 1 : null,
        action: i ? [i * 10] : null,
        state: { robot: { joint_position: [[i]] } },
      })),
    },
  ],
};
try {
  window.eval(
    await readFile(
      new URL("../static/live-review.js", import.meta.url),
      "utf8",
    ),
  );
  window.openLiveReview({ id: "first" }, 0);
  requests[0].resolve({ state: "READY" });
  await flush();
  requests[1].resolve(data);
  await flush();
  assert.equal(requests[2].path.endsWith("/video?episode=0"), true);
  requests[2].resolve({
    state: "READY",
    kind: "replay",
    sha256: "first-video",
  });
  await flush();
  assert.equal(get("live-review-video").hidden, false);
  assert.match(
    get("live-review-video").src,
    /first\/recordings\/0\/video.mp4\?episode=0&v=first-video$/,
  );
  assert.equal(get("live-review-video-status").textContent, "Loading video…");
  assert.equal(watchdogs.size, 1);
  const initialWatchdog = [...watchdogs.values()][0];
  mediaEvent("stalled", 0);
  assert.equal(
    [...watchdogs.values()][0],
    initialWatchdog,
    "Repeated stalls must not postpone the timeout",
  );
  [...watchdogs.values()][0]();
  assert.match(
    get("live-review-video-status").textContent,
    /did not load within 30 seconds/,
  );
  assert.equal(get("live-review-video-retry").hidden, false);
  mediaEvent("loadeddata");
  assert.equal(watchdogs.size, 0);
  assert.equal(get("live-review-video-retry").hidden, true);
  assert.match(
    get("live-review-video-status").textContent,
    /rendered from recorded states/,
  );
  assert.equal(get("live-review-data").open, false);
  get("live-review-video").dispatchEvent(new window.Event("error"));
  assert.equal(get("live-review-video-retry").hidden, false);
  get("live-review-video-retry").click();
  assert.equal(requests[3].options.method, "POST");
  requests[3].resolve({
    state: "READY",
    kind: "capture",
    sha256: "captured-video",
  });
  await flush();
  mediaEvent("canplay");
  assert.equal(get("live-review-video-status").textContent, "Collection video");
  mediaEvent("waiting", 1);
  assert.equal(get("live-review-video-status").textContent, "Buffering video…");
  assert.equal(watchdogs.size, 1);
  mediaEvent("playing", 3);
  assert.equal(watchdogs.size, 0);
  mediaEvent("seeking", 1);
  assert.equal(watchdogs.size, 1);
  mediaEvent("seeked", 3);
  assert.equal(watchdogs.size, 0);
  assert.match(get("live-review-values").textContent, /Initial state/);
  assert.equal(get("live-review-prev").disabled, true);
  get("live-review-next").click();
  assert.match(get("live-review-frame").textContent, /Frame 1 of 2/);
  assert.match(
    get("live-review-value-note").textContent,
    /Action 0 produced scene frame 1/,
  );
  assert.match(get("live-review-values").textContent, /10\.00000/);
  get("live-review-next").click();
  assert.equal(get("live-review-next").disabled, true);
  get("live-review-prev").click();
  assert.match(get("live-review-frame").textContent, /Frame 1 of 2/);
  get("live-review-field").value = "robot.joint_position";
  get("live-review-field").dispatchEvent(new window.Event("change"));
  assert.match(get("live-review-values").textContent, /1\.000000/);
  mediaEvent("waiting", 1);
  const staleWatchdog = [...watchdogs.values()][0];
  get("live-review-close").click();
  assert.equal(watchdogs.size, 0);
  window.openLiveReview({ id: "stale" }, 0);
  get("live-review-close").click();
  window.openLiveReview({ id: "current" }, 0);
  staleWatchdog();
  assert.equal(
    get("live-review-video-retry").hidden,
    true,
    "A closed video's timer must not alter the new review",
  );
  assert.equal(get("live-review-video").hasAttribute("src"), false);
  assert.ok(videoPauses > 0);
  const beforeStale = requests.length;
  requests[4].resolve({ state: "READY" });
  await flush();
  assert.equal(
    requests.length,
    beforeStale,
    "Closed review must not fetch stale data",
  );
  requests[5].resolve({ state: "FAILED", error: "Host unreachable" });
  await flush();
  assert.match(get("live-review-status").textContent, /Host unreachable/);
  assert.equal(get("live-review-retry").hidden, false);
  get("live-review-retry").click();
  assert.equal(requests[6].options.method, "POST");
  window.openLiveReview({
    id: "multiple",
    recordings: ["first.pkl", "second.pkl"],
  });
  assert.equal(get("live-review-recording").parentElement.hidden, false);
  assert.equal(get("live-review-recording").options.length, 2);
  const staleRecording = requests.at(-1);
  get("live-review-recording").value = "1";
  get("live-review-recording").dispatchEvent(new window.Event("change"));
  const selectedRecording = requests.at(-1);
  assert.match(selectedRecording.path, /multiple\/recordings\/1\/review$/);
  const beforeSwitchResponse = requests.length;
  staleRecording.resolve({ state: "READY" });
  await flush();
  assert.equal(
    requests.length,
    beforeSwitchResponse,
    "Switching recording must ignore the previous download",
  );
  selectedRecording.resolve({ state: "READY" });
  await flush();
  requests.at(-1).resolve(data);
  await flush();
  assert.match(
    requests.at(-1).path,
    /multiple\/recordings\/1\/video\?episode=0$/,
  );
  requests
    .at(-1)
    .resolve({ state: "READY", kind: "replay", sha256: "second-video" });
  await flush();
  assert.match(get("live-review-video").src, /recordings\/1\/video.mp4/);
  console.log(
    "Review UI: frame boundaries, values, action alignment, close races and retry passed.",
  );
} finally {
  window.close();
}
