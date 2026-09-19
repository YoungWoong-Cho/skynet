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
const get = (id) => window.document.getElementById(id);
const recordingIdentity = (session, index) =>
  `${session.id}:${Buffer.from(session.recordings[index], "utf8").toString("base64url")}`;
function assertRecordingSelection(session, index) {
  assert.equal(get("live-review-recording-position").textContent, `${index + 1}/${session.recordings.length}`);
  assert.equal(get("live-review-previous-recording").disabled, index === 0);
  assert.equal(get("live-review-next-recording").disabled, index === session.recordings.length - 1);
  assert.equal(get("live-review-delete").disabled, false);
  assert.equal(get("live-review-delete").dataset.deleteKind, "recording-file");
  assert.equal(get("live-review-delete").dataset.deleteId, recordingIdentity(session, index), "Deletion identifies the saved path, never its mutable list index");
}
let nextPollId = 100000;
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
let viewerOptions, viewerVideoId, viewerPlaying = false;
const viewerPauses = [];
window.SkynetEpisodeViewer = {
  open: (...args) => {
    viewerOptions = args[3];
    viewerVideoId = args[1];
    const host = get(args[0]);
    if (!host.querySelector(".episode-viewer-stage")) {
      const layers = window.document.createElement("div");
      layers.className = "episode-viewer-layers";
      const layer = window.document.createElement("label");
      layer.textContent = "Hand model";
      layers.append(layer, viewerOptions.toolbarActions);
      const stage = window.document.createElement("div");
      stage.className = "episode-viewer-stage";
      const playback = window.document.createElement("div");
      playback.className = "episode-viewer-playback";
      host.append(layers, stage, playback);
    }
    viewerOptions.onLoadState?.("ready");
  },
  pause: (hostId) => { viewerPauses.push(hostId); viewerPlaying = false; },
  close() { viewerPlaying = false; },
};
function assertReviewState(state) {
  const busy = state !== "ready";
  assert.equal(get("live-review-content").hidden, false, "The modal content stays mounted and visible");
  assert.equal(get("live-review-content").dataset.reviewState, state);
  assert.equal(get("live-review-loading").parentElement.className, "episode-viewer-stage", "Loading feedback belongs to the scene, not the whole modal");
  assert.equal(get("live-review-loading").hidden, !busy);
  assert.equal(get("live-review-loading").getAttribute("aria-busy"), String(state === "loading"));
  assert.equal(get("live-review-source-copy").disabled, busy);
  assert.equal(get("live-review-episode").disabled, busy);
  assert.equal(get("live-review-values-content").inert, busy);
  assert.equal(get("live-episode-viewer").querySelector(".episode-viewer-playback").inert, busy);
  assert.equal(get("live-episode-viewer").querySelector(".episode-viewer-layers > label").inert, busy);
  assert.equal(get("live-review-recording-actions").hidden, false);
  for (let element = get("live-review-recording-actions"); element; element = element.parentElement)
    assert.notEqual(element.inert, true, "Recording navigation stays outside the inert loading regions");
}
const data = {
  task_name: "Stick",
  hand_name: "Right",
  size_bytes: 100,
  time_note: "60 Hz",
  value_note: "Saved order",
  hand_metadata: { source_names: {r_wrist: "h_r_wrist"} },
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
  window.eval(await readFile(new URL("../static/dialogs.js", import.meta.url), "utf8"));
  window.eval(
    await readFile(
      new URL("../static/live-review.js", import.meta.url),
      "utf8",
    ),
  );
  window.openLiveReview({ id: "first" }, 0);
  requests[0].resolve({ state: "READY", recording_source: {gateway: "sky2", path: "/cluster/original/episode-1.pkl"} });
  await flush();
  requests[1].resolve(data);
  await flush();
  assert.equal(requests.length, 2, "Opening a recording must not request video");
  assertReviewState("ready");
  assert.equal(get("live-review-video"), null, "Collection has no hidden media element");
  assert.equal(viewerVideoId, null, "Collection opens the state-only viewer");
  assert.equal(get("live-review-recording"), null, "The modal has no recording selector");
  assert.equal(get("live-review-recording-position").textContent, "1/1");
  assert.equal(get("live-review-previous-recording").disabled, true);
  assert.equal(get("live-review-next-recording").disabled, true);
  assert.deepEqual(
    [...get("live-review-recording-actions").children].map(element => element.id),
    ["live-review-recording-position", "live-review-previous-recording", "live-review-next-recording", "live-review-delete"],
    "Recording navigation sits between the position and deletion controls",
  );
  assert.equal(get("live-review-delete").disabled, true, "Deletion needs a known recording path");
  assert.equal(get("live-review-delete").hasAttribute("data-delete-id"), false);
  assert.equal(viewerOptions.toolbarActions, get("live-review-recording-actions"), "Recording position and deletion share the viewer layer toolbar");
  assert.equal(get("live-review-data").open, false);
  assert.equal(get("live-review-source").hidden, false);
  assert.equal(get("live-review-source-label"), null, "The code block needs no separate Path label");
  assert.equal(get("live-review-source-path").textContent, "/cluster/original/episode-1.pkl");
  assert.equal(get("live-review-source-copy").dataset.copyValue, "/cluster/original/episode-1.pkl");
  assert.equal(get("live-review-source").nextElementSibling, get("live-review-data"));
  assert.deepEqual(viewerOptions.sourceNames, data.hand_metadata.source_names);
  assert.match(get("live-review-values").textContent, /Initial state/);
  assert.equal(get('live-review-seek'), null, 'Recorded values have no independent seek bar');
  assert.equal(get('live-review-context'), null, 'The duplicate title subtitle was removed');
  assert.equal(get('live-review-status').textContent, "");
  assert.equal(get('live-review-status').hidden, true, "Loaded recordings have no summary above the viewer");
  assert.equal(viewerOptions.simulationHz, 60, "The recorded simulation rate belongs to the playback controls");
  assert.deepEqual(viewerOptions.timeline, data.episodes[0].frames);
  get('live-review-data').open = true;
  viewerOptions.onFrame(1);
  assert.match(get('live-review-value-note').textContent, /Action 0 produced scene frame 1/);
  assert.match(get('live-review-values').textContent, /10\.00000/);
  viewerOptions.onFrame(2);
  assert.match(get('live-review-values').textContent, /20\.00000/);
  viewerOptions.onFrame(1);
  get("live-review-field").value = "robot.joint_position";
  get("live-review-field").dispatchEvent(new window.Event("change"));
  assert.match(get("live-review-values").textContent, /1\.000000/);
  get("live-review-close").click();
  window.openLiveReview({ id: "stale" }, 0);
  const staleRequest = requests.at(-1);
  get("live-review-close").click();
  window.openLiveReview({ id: "current" }, 0);
  const currentRequest = requests.at(-1);
  const beforeStale = requests.length;
  staleRequest.resolve({state: "READY"});
  await flush();
  assert.equal(requests.length, beforeStale, "Closed review must not fetch stale data");
  currentRequest.resolve({state: "FAILED", error: "Host unreachable"});
  await flush();
  assertReviewState("error");
  assert.match(get("live-review-status").textContent, /Host unreachable/);
  assert.equal(get("live-review-status").hidden, false, "Removing the summary must not hide review failures");
  get("live-review-retry").click();
  assertReviewState("loading");
  assert.equal(requests.at(-1).options.method, "POST");
  const multiple = {
    id: "multiple",
    recordings: ["recordings/first.pkl", "recordings/second.pkl"],
  };
  window.openLiveReview(multiple);
  assertRecordingSelection(multiple, 0);
  const staleRecording = requests.at(-1);
  const beforeFirstBoundary = requests.length;
  get("live-review-previous-recording").click();
  assert.equal(requests.length, beforeFirstBoundary, "Previous is disabled at the first recording");
  viewerPlaying = true;
  get("live-review-next-recording").click();
  assert.equal(viewerPlaying, false, "Switching to the next recording pauses playback");
  assertRecordingSelection(multiple, 1);
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
  selectedRecording.resolve({ state: "READY", recording_source: {gateway: "rl2-bonjour", path: "/workstation/episode-2.pkl"} });
  await flush();
  requests.at(-1).resolve({...data, simulation_hz: 30});
  await flush();
  assert.equal(viewerOptions.simulationHz, 30, "The playback rate uses recording metadata when available");
  assert.equal(requests.some(r => /\/video(?:\?|\.|$)/.test(r.path)), false);
  assert.equal(get("live-review-source-path").textContent, "/workstation/episode-2.pkl");
  assert.equal(get("live-review-dialog").querySelector("video"), null);
  const beforeLastBoundary = requests.length;
  get("live-review-next-recording").click();
  assert.equal(requests.length, beforeLastBoundary, "Next is disabled at the last recording");
  get("live-review-data").open = true;
  const previousViewerOptions = viewerOptions;
  const sourceBeforeNavigation = get("live-review-source-path").textContent;
  const valuesBeforeNavigation = get("live-review-values").innerHTML;
  viewerPlaying = true;
  get("live-review-previous-recording").click();
  assertRecordingSelection(multiple, 0);
  assertReviewState("loading");
  assert.equal(get("live-review-data").open, true, "Navigating does not collapse expanded recorded values");
  assert.equal(get("live-review-source").hidden, false, "The path block retains its layout while loading");
  assert.equal(get("live-review-source-path").textContent, sourceBeforeNavigation, "Keep the masked path content to preserve wrapped height");
  assert.equal(get("live-review-values").innerHTML, valuesBeforeNavigation, "Keep the masked values layout during navigation");
  assert.equal(get("live-review-source-copy").dataset.copyValue, undefined, "The previous path is never copyable while the next recording loads");
  for (const id of ["live-review-original", "live-review-summary", "live-review-json"])
    assert.equal(get(id).hasAttribute("href"), false, "Old download targets are cleared during navigation");
  previousViewerOptions.onLoadState("ready");
  previousViewerOptions.onLoadState("error", "Stale scene failure");
  previousViewerOptions.onFrame(1);
  assertReviewState("loading");
  assert.doesNotMatch(get("live-review-status").textContent, /Stale scene failure/, "Callbacks from the previous viewer cannot change the current loading state");
  assert.equal(viewerPlaying, false, "Switching to the previous recording pauses playback");
  assert.match(requests.at(-1).path, /multiple\/recordings\/0\/review$/);
  requests.at(-1).resolve({state: "READY"});
  await flush();
  requests.at(-1).resolve(data);
  await flush();
  get("live-review-next-recording").click();
  assertRecordingSelection(multiple, 1);
  requests.at(-1).resolve({state: "READY"});
  await flush();
  requests.at(-1).resolve(data);
  await flush();
  viewerPlaying = true;
  const beforeDeletePreview = requests.length;
  let delegatedDelete = null;
  const observeDelete = (event) => {
    const button = event.target.closest('[data-delete-kind="recording-file"]');
    if (!button) return;
    delegatedDelete = { id: button.dataset.deleteId, playing: viewerPlaying };
  };
  window.document.addEventListener("click", observeDelete);
  get("live-review-delete").click();
  window.document.removeEventListener("click", observeDelete);
  assert.deepEqual(delegatedDelete, { id: recordingIdentity(multiple, 1), playing: false }, "Playback pauses before the shared deletion confirmation opens");
  assert.equal(requests.length, beforeDeletePreview, "The viewer delegates confirmation rather than deleting directly");
  get("live-review-close").click();

  // Resolve the old JSON after the new selection has loaded. Guarding only
  // DOM updates is insufficient: stale data must not overwrite shared state.
  const racing = { id: "racing-json", recordings: ["race/first.pkl", "race/second.pkl"] };
  window.openLiveReview(racing);
  requests.at(-1).resolve({state: "READY", recording_source: {path: "race/first.pkl"}});
  await flush();
  const staleJson = requests.at(-1);
  assert.match(staleJson.path, /recordings\/0\/review.json$/);
  get("live-review-next-recording").click();
  assertReviewState("loading");
  assertRecordingSelection(racing, 1);
  requests.at(-1).resolve({state: "READY", recording_source: {path: "race/second.pkl"}});
  await flush();
  const currentJson = requests.at(-1);
  assert.match(currentJson.path, /recordings\/1\/review.json$/);
  currentJson.resolve({...data, value_note: "Current recording values"});
  await flush();
  assertReviewState("ready");
  const currentOptions = viewerOptions;
  staleJson.resolve({...data, value_note: "Stale recording values"});
  await flush();
  get("live-review-data").open = true;
  currentOptions.onFrame(1);
  assert.match(get("live-review-value-note").textContent, /Current recording values/);
  assert.doesNotMatch(get("live-review-value-note").textContent, /Stale recording values/);
  assert.equal(viewerOptions, currentOptions, "Stale review JSON cannot reopen the previous viewer");
  assert.equal(get("live-review-source-path").textContent, "race/second.pkl");
  assert.match(get("live-review-original").href, /recordings\/1\/recording.pkl$/);
  currentOptions.onLoadState("loading", "Loading scene geometry…");
  assertReviewState("loading");
  assert.equal(get("live-review-data").open, true);
  currentOptions.onLoadState("error", "Scene geometry unavailable");
  assertReviewState("error");
  assert.equal(get("live-review-retry").hidden, false);
  assert.match(get("live-review-status").textContent, /Scene geometry unavailable/);
  assertRecordingSelection(racing, 1);
  get("live-review-retry").click();
  assertReviewState("loading");
  assert.equal(requests.at(-1).options.method, "POST");
  requests.at(-1).resolve({state: "READY", recording_source: {path: "race/second.pkl"}});
  await flush();
  requests.at(-1).resolve(data);
  await flush();
  assertReviewState("ready");
  assert.equal(get("live-review-data").open, true, "Retry also preserves expanded recorded values");
  get("live-review-close").click();
  const stateBeforeClosedCallback = get("live-review-content").dataset.reviewState;
  currentOptions.onLoadState("error", "Closed viewer failure");
  assert.equal(get("live-review-content").dataset.reviewState, stateBeforeClosedCallback, "Closed viewers ignore delayed state callbacks");

  // A cached connection failure must be retried once when the user opens that
  // recording; status polling must not repeatedly restart a failed download.
  const reviewPolls = new Map();
  const previousSetTimeout = window.setTimeout;
  const previousClearTimeout = window.clearTimeout;
  window.setTimeout = (callback, delay, ...args) => {
    if (delay !== 1500) return previousSetTimeout(callback, delay, ...args);
    const id = ++nextPollId;
    reviewPolls.set(id, callback);
    return id;
  };
  window.clearTimeout = (id) => {
    reviewPolls.delete(id);
    previousClearTimeout(id);
  };
  const pollReview = async () => {
    assert.equal(reviewPolls.size, 1);
    const [id, callback] = reviewPolls.entries().next().value;
    reviewPolls.delete(id);
    callback();
    await flush();
  };
  const cachedFailures = {
    id: "cached-host-failures",
    recordings: Array.from({length: 51}, (_, i) => `recording-${i}.pkl`),
  };
  const beforeRecovery = requests.length;
  window.openLiveReview(cachedFailures, 19);
  assertRecordingSelection(cachedFailures, 19);
  assert.equal(requests.length, beforeRecovery + 1);
  assert.match(requests.at(-1).path, /cached-host-failures\/recordings\/19\/review$/);
  assert.equal(requests.at(-1).options.method, "POST", "Opening a saved failure must request a fresh download using the current host settings");
  requests.at(-1).resolve({state: "DOWNLOADING"});
  await flush();
  await pollReview();
  assert.equal(requests.at(-1).options.method, undefined, "Polling observes the one download rather than restarting it");
  requests.at(-1).resolve({state: "READY"});
  await flush();
  requests.at(-1).resolve(data);
  await flush();
  assert.equal(get("live-review-content").hidden, false);
  assert.equal(get("live-review-retry").hidden, true);
  assert.equal(reviewPolls.size, 0);
  assert.ok(requests.slice(beforeRecovery).every(({path}) => path.includes("/recordings/19/")), "Opening one of 51 recordings must not retry the other cached failures");
  assert.equal(requests.slice(beforeRecovery).filter(({options}) => options.method === "POST").length, 1, "Review recovery must not prepare video or start other downloads");

  window.openLiveReview(cachedFailures, 20);
  assert.equal(requests.at(-1).options.method, "POST");
  requests.at(-1).resolve({state: "FAILED", error: "Current host is still unreachable"});
  await flush();
  const afterFreshFailure = requests.length;
  assert.equal(reviewPolls.size, 0, "A fresh failure must not schedule automatic retries");
  assert.equal(get("live-review-retry").hidden, false);
  assert.match(get("live-review-status").textContent, /Current host is still unreachable/);
  assert.equal(get("live-review-status").hidden, false);
  await flush();
  assert.equal(requests.length, afterFreshFailure);
  get("live-review-retry").click();
  assert.equal(requests.length, afterFreshFailure + 1);
  assert.equal(requests.at(-1).options.method, "POST", "Manual retry remains immediate");
  requests.at(-1).resolve({state: "FAILED", error: "Current host is still unreachable"});
  await flush();
  assert.equal(reviewPolls.size, 0);
  get("live-review-close").click();
  window.openLiveReview(cachedFailures, 20);
  assert.equal(requests.length, afterFreshFailure + 2);
  assert.equal(requests.at(-1).options.method, "POST", "A later user open gets one new recovery attempt");
  get("live-review-close").click();

  window.setTimeout = previousSetTimeout;
  window.clearTimeout = previousClearTimeout;
  const session = {id:"recording-delete-test", recordings:["recordings/one.pkl", "recordings/two.pkl", "recordings/three.pkl"]};
  window.openLiveReview(session, 1);
  assertRecordingSelection(session, 1);
  viewerPlaying = true;
  const pausesBeforeRefresh = viewerPauses.length;
  const refresh = window.refreshLiveReviewAfterDeletion(session.id);
  requests.at(-1).resolve({...session, recordings:[session.recordings[1],session.recordings[2]]});
  await refresh;
  assertRecordingSelection({...session, recordings:[session.recordings[1],session.recordings[2]]}, 0);
  assert.match(requests.at(-1).path, /recording-delete-test\/recordings\/0\/review$/, "Refresh follows the same file when its index shifts");
  assert.ok(viewerPauses.length > pausesBeforeRefresh);
  assert.equal(viewerPauses.at(-1), "live-episode-viewer");
  assert.equal(viewerPlaying, false, "Refreshing the selected recording pauses the actual 3D timeline");
  window.openLiveReview(session, 2);
  assertRecordingSelection(session, 2);
  const removedSelection = window.refreshLiveReviewAfterDeletion(session.id);
  requests.at(-1).resolve({...session, recordings:[session.recordings[0],session.recordings[1]]});
  await removedSelection;
  assertRecordingSelection({...session, recordings:[session.recordings[0],session.recordings[1]]}, 1);
  assert.match(requests.at(-1).path, /recording-delete-test\/recordings\/1\/review$/, "A removed selected file falls back to the nearest remaining index");
  const empty = window.refreshLiveReviewAfterDeletion(session.id);
  requests.at(-1).resolve({...session, recordings:[]});
  await empty;
  assert.equal(dialog.open, false, "deleting the last recording closes its empty viewer");
  window.openLiveReview({...session, recordings:[session.recordings[0]]});
  assertRecordingSelection({...session, recordings:[session.recordings[0]]}, 0);
  assert.match(requests.at(-1).path, /recording-delete-test\/recordings\/0\/review$/);
  get("live-review-close").click();
  const unicodeSession = {id:"6a257afb-642c-41dd-b77e-e7cc92ee1b28", recordings:['recordings/손/épisode "2".pkl']};
  window.openLiveReview(unicodeSession);
  assertRecordingSelection(unicodeSession, 0);
  const encodedPath = get("live-review-delete").dataset.deleteId.split(":")[1];
  assert.match(encodedPath, /^[A-Za-z0-9_-]+$/, "The path uses unpadded base64url");
  assert.equal(Buffer.from(encodedPath, "base64url").toString("utf8"), unicodeSession.recordings[0], "Unicode recording paths round-trip without changing the deletion target");
  window.openLiveReview({id:"unknown-recording-path"});
  assert.equal(get("live-review-recording-position").textContent, "1/1");
  assert.equal(get("live-review-previous-recording").disabled, true);
  assert.equal(get("live-review-next-recording").disabled, true);
  assert.equal(get("live-review-delete").disabled, true);
  assert.equal(get("live-review-delete").hasAttribute("data-delete-id"), false, "An unavailable path clears the prior recording's deletion target");
  get("live-review-close").click();
  console.log(
    "Review UI: stable navigation loading, stale response isolation, playback, recording position, deletion identities, confirmation delegation, deletion refresh, recovery and no video requests passed.",
  );
} finally {
  window.close();
}
