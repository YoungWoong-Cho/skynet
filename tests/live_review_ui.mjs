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
  open: (...args) => { viewerOptions = args[3]; viewerVideoId = args[1]; },
  pause: (hostId) => { viewerPauses.push(hostId); viewerPlaying = false; },
  close() { viewerPlaying = false; },
};
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
  assert.equal(get("live-review-video"), null, "Collection has no hidden media element");
  assert.equal(viewerVideoId, null, "Collection opens the state-only viewer");
  assert.equal(get("live-review-data").open, false);
  assert.equal(get("live-review-source").hidden, false);
  assert.equal(get("live-review-source-label").textContent, "Path · sky2");
  assert.equal(get("live-review-source-path").textContent, "/cluster/original/episode-1.pkl");
  assert.equal(get("live-review-source-copy").dataset.copyValue, "/cluster/original/episode-1.pkl");
  assert.equal(get("live-review-source").nextElementSibling, get("live-review-data"));
  assert.deepEqual(viewerOptions.sourceNames, data.hand_metadata.source_names);
  assert.match(get("live-review-values").textContent, /Initial state/);
  assert.equal(get('live-review-seek'), null, 'Recorded values have no independent seek bar');
  assert.equal(get('live-review-context'), null, 'The duplicate title subtitle was removed');
  assert.match(get('live-review-status').textContent, /60 Hz/);
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
  assert.match(get("live-review-status").textContent, /Host unreachable/);
  get("live-review-retry").click();
  assert.equal(requests.at(-1).options.method, "POST");
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
  selectedRecording.resolve({ state: "READY", recording_source: {gateway: "rl2-bonjour", path: "/workstation/episode-2.pkl"} });
  await flush();
  requests.at(-1).resolve(data);
  await flush();
  assert.equal(requests.some(r => /\/video(?:\?|\.|$)/.test(r.path)), false);
  assert.equal(get("live-review-source-path").textContent, "/workstation/episode-2.pkl");
  assert.equal(get("live-review-dialog").querySelector("video"), null);
  get("live-review-close").click();

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

  get("live-review-recording").value = "20";
  get("live-review-recording").dispatchEvent(new window.Event("change"));
  assert.equal(requests.at(-1).options.method, "POST");
  requests.at(-1).resolve({state: "FAILED", error: "Current host is still unreachable"});
  await flush();
  const afterFreshFailure = requests.length;
  assert.equal(reviewPolls.size, 0, "A fresh failure must not schedule automatic retries");
  assert.equal(get("live-review-retry").hidden, false);
  assert.match(get("live-review-status").textContent, /Current host is still unreachable/);
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
  const remove = get("live-review-delete");
  assert.equal(remove.hidden, false);
  assert.equal(remove.dataset.deleteKind, "recording-file", "reuse the existing delegated deletion control");
  viewerPlaying = true;
  const pausesBeforeDelete = viewerPauses.length;
  remove.click();
  assert.equal(viewerPauses.length, pausesBeforeDelete + 1);
  assert.equal(viewerPauses.at(-1), "live-episode-viewer");
  assert.equal(viewerPlaying, false, "Delete pauses the actual 3D timeline");
  const selectedIdentity = remove.dataset.deleteId;
  assert.equal(window.atob(selectedIdentity.split(":")[1]), session.recordings[1]);
  get("live-review-recording").value = "2";
  get("live-review-recording").dispatchEvent(new window.Event("change"));
  assert.notEqual(remove.dataset.deleteId, selectedIdentity);
  const refresh = window.refreshLiveReviewAfterDeletion(session.id);
  requests.at(-1).resolve({...session, recordings:[session.recordings[0],session.recordings[2]], recording_slots:{[session.recordings[0]]:0,[session.recordings[2]]:2}});
  await refresh;
  assert.equal(get("live-review-recording").options.length, 2);
  assert.equal(get("live-review-recording").selectedOptions[0].textContent, "Recording 3");
  assert.equal(window.atob(remove.dataset.deleteId.split(":")[1]), session.recordings[2], "deletion identity follows the file, not its shifting index");
  const empty = window.refreshLiveReviewAfterDeletion(session.id);
  requests.at(-1).resolve({...session, recordings:[]});
  await empty;
  assert.equal(dialog.open, false, "deleting the last recording closes its empty viewer");
  window.openLiveReview({...session, recordings:[session.recordings[0]]});
  assert.equal(remove.hidden, false, "a single recording still has Delete when the selector is hidden");
  get("live-review-close").click();
  console.log(
    "Review UI: frame boundaries, values, close races, selected recording recovery, no video requests, GET-only review polling and bounded retry passed.",
  );
} finally {
  window.close();
}
