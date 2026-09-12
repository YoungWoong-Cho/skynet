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
window.fetch = (path, options) =>
  new Promise((resolve) =>
    requests.push({
      path,
      options,
      resolve: (data, ok = true) => resolve({ ok, json: async () => data }),
    }),
  );
const flush = () => new Promise((resolve) => setImmediate(resolve));
const get = (id) => window.document.getElementById(id);
get("collection").hidden = false;
get("collection-view-live").hidden = false;
const catalog = {
  default_robot: "skynet_shadow_right",
  default_task: "Dexverse-PickUpStick-v0",
  hands: [
    { key: "skynet_shadow_right", name: "Right", available: true },
    { key: "skynet_shadow_left", name: "Left", available: true },
    {
      key: "skynet_allegro_v4_left",
      name: "Allegro left",
      available: false,
      reason: "Missing thumb mesh",
    },
    {
      key: "skynet_wuji_1_right",
      name: "WUJI right",
      available: true,
      imported: true,
    },
  ],
  tasks: [
    {
      key: "Dexverse-PickUpStick-v0",
      name: "Stick",
      instructions: "Lift vertically",
    },
    { key: "Dexverse-PickCube-v0", name: "Cube", instructions: "Lift cube" },
  ],
  verified_pairs: [
    { robot: "skynet_shadow_right", task: "Dexverse-PickUpStick-v0" },
  ],
  note: "Pinned release",
};
const job = {
  id: "test-session",
  state: "PENDING",
  startup_stage: "launch",
  server_connected_at: "2026-09-07T00:00:00Z",
  runtime_checked_at: "2026-09-07T00:00:01Z",
  hand_prepared_at: "2026-09-07T00:00:02Z",
  job_id: "123",
  server_ready: false,
  profile: { robot: "skynet_shadow_left", task: "Dexverse-PickCube-v0" },
};
try {
  window.eval(
    await readFile(new URL("../static/live-xr.js", import.meta.url), "utf8"),
  );
  get("live-xr-start-form").dispatchEvent(
    new window.Event("submit", { cancelable: true }),
  );
  assert.equal(requests.length, 1, "Cannot submit before choices load");
  assert.equal(get("live-xr-stop").disabled, true);
  assert.equal(
    get("live-xr-stop").parentElement,
    get("live-xr-start").parentElement,
  );
  requests[0].resolve({ sessions: [], license: { accepted: true }, catalog });
  await flush();
  assert.equal(
    get("live-xr-hand").querySelector('option[value="skynet_allegro_v4_left"]')
      .disabled,
    true,
  );
  get("live-xr-hand").value = "skynet_wuji_1_right";
  get("live-xr-hand").dispatchEvent(new window.Event("change"));
  assert.match(get("live-xr-selection-note").textContent, /Not headset-tested/);
  assert.equal(get("live-xr-start").disabled, false);
  get("live-xr-hand").value = "skynet_shadow_left";
  get("live-xr-task").value = "Dexverse-PickCube-v0";
  get("live-xr-task").dispatchEvent(new window.Event("change"));
  assert.match(get("live-xr-task-instructions").textContent, /Lift cube/);
  assert.match(get("live-xr-selection-note").textContent, /Not headset-tested/);
  assert.equal(
    new URL(window.location).searchParams.get("live_hand"),
    "skynet_shadow_left",
  );
  window.loadLiveXR(); // Pending refresh must not erase subsequent accepted submission.
  get("live-xr-start-form").dispatchEvent(
    new window.Event("submit", { cancelable: true }),
  );
  assert.equal(get("live-xr-start").textContent, "Starting…");
  assert.equal(
    JSON.parse(requests[2].options.body).robot,
    "skynet_shadow_left",
  );
  assert.equal(
    JSON.parse(requests[2].options.body).task,
    "Dexverse-PickCube-v0",
  );
  assert.equal(JSON.parse(requests[2].options.body).image_capture, true);
  requests[2].resolve(job);
  await flush();
  requests[1].resolve({ sessions: [], license: { accepted: true }, catalog });
  await flush();
  assert.match(get("live-xr-sessions").textContent, /test-ses/);
  assert.equal(get("live-xr-hand").disabled, true);
  assert.equal(get("live-xr-start").textContent, "Waiting for GPU…");
  assert.equal(get("live-xr-progress").hidden, false);
  const stages = () => [
    ...get("live-xr-progress-steps").querySelectorAll("li"),
  ];
  assert.equal(stages()[0].dataset.state, "done");
  assert.equal(stages()[2].dataset.state, "done");
  assert.match(stages()[3].textContent, /Queued/);
  assert.match(stages()[4].textContent, /Not started/);
  window.loadLiveXR();
  const stop = get("live-xr-stop");
  assert.equal(stop.disabled, false);
  assert.equal(
    get("live-xr-sessions").textContent.includes("Stop session"),
    false,
  );
  stop.click();
  assert.equal(
    new URL(window.location.href).searchParams.get("live_session"),
    job.id,
  );
  requests[4].resolve({ ...job, stop_requested: true });
  await flush();
  requests[3].resolve({
    sessions: [job],
    license: { accepted: true },
    catalog,
  });
  await flush();
  assert.equal(get("live-xr-stop").textContent, "Stopping…");
  assert.equal(get("live-xr-stop").disabled, true);
  assert.equal(get("live-xr-start").textContent, "Stopping…");
  assert.match(stages()[3].textContent, /Interrupted/);
  window.loadLiveXR();
  requests[5].resolve({
    sessions: [
      {
        ...job,
        state: "CAPTURED",
        scheduler_final: true,
        profile: { ...job.profile, execution: "workstation" },
        gateway: "test-workstation",
        recordings: ["recordings/live/demo.pkl", "recordings/live/demo-2.pkl"],
      },
    ],
    license: { accepted: true },
    catalog,
    target: {
      execution: "workstation",
      host: "test-workstation",
      duration_minutes: 30,
    },
  });
  await flush();
  assert.match(
    get("live-xr-target").textContent,
    /test-workstation · 30 min limit/,
  );
  assert.equal(get("live-xr-progress").hidden, true);
  let reviewed;
  window.openLiveReview = (...args) => {
    reviewed = args;
  };
  assert.equal(
    [...get("live-xr-sessions").querySelectorAll("button")].filter(
      (b) => b.textContent === "Review recordings",
    ).length,
    1,
  );
  [...get("live-xr-sessions").querySelectorAll("button")]
    .find((b) => b.textContent === "Review recordings")
    .click();
  assert.equal(reviewed[0].id, job.id);
  assert.equal(reviewed[1], 0);
  assert.doesNotMatch(
    get("live-xr-sessions").textContent,
    /Run training cycle/,
  );
  const failed = {
    ...job,
    id: "failed-attempt",
    job_id: null,
    scheduler_final: true,
    state: "FAILED",
    startup_stage: "server",
    failed_stage: "server",
    server_connected_at: null,
    runtime_checked_at: null,
    hand_prepared_at: null,
    error: "Connection timed out",
  };
  const overview = { sessions: [failed], license: { accepted: true }, catalog };
  window.loadLiveXR();
  requests.at(-1).resolve(overview);
  await flush();
  assert.equal(get("live-xr-start").textContent, "Start session");
  assert.equal(get("live-xr-progress").hidden, true,
    "An old failed session must not become the current startup status");
  assert.match(get("live-xr-sessions").textContent, /Connection timed out/,
    "The historical failure remains available in Session history");
  get("live-xr-start-form").dispatchEvent(
    new window.Event("submit", { cancelable: true }),
  );
  requests.at(-1).resolve(failed);
  await flush();
  requests.at(-1).resolve(overview);
  await flush();
  assert.equal(get("live-xr-progress").hidden, false,
    "A newly submitted session's failure must remain visible");
  assert.equal(get("live-xr-progress-title").textContent, "Startup failed");
  assert.equal(stages()[0].dataset.state, "failed");
  assert.ok(
    stages()
      .slice(1)
      .every((step) => step.textContent.endsWith("Not started")),
  );
  window.loadLiveXR();
  requests.at(-1).resolve(overview);
  await flush();
  assert.equal(get("live-xr-progress-error").hidden, false);
  assert.match(
    get("live-xr-progress-error").textContent,
    /Connection timed out/,
  );
  get("live-xr-start-form").dispatchEvent(
    new window.Event("submit", { cancelable: true }),
  );
  assert.equal(
    get("live-xr-progress-title").textContent,
    "Submitting request…",
  );
  requests.at(-1).resolve({ detail: "Hand files missing" }, false);
  await flush();
  window.loadLiveXR();
  requests.at(-1).resolve(overview);
  await flush();
  assert.equal(get("live-xr-progress-title").textContent, "Could not start");
  assert.match(get("live-xr-progress-error").textContent, /Hand files missing/);
  assert.ok(stages().every((step) => step.textContent.endsWith("Not started")));
  window.loadLiveXR();
  requests.at(-1).resolve({ sessions: [{ ...job, state: "RENDERING_IMAGES", stop_requested: true, server_ready: false, detail: "Preparing images for episode 1 of 2" }], license: { accepted: true }, catalog });
  await flush();
  assert.equal(get("live-xr-start").disabled, true);
  assert.equal(get("live-xr-stop").disabled, true);
  assert.equal(get("live-xr-start").textContent, "Preparing training images…");
  assert.equal(get("live-xr-progress-title").textContent, "Preparing images for episode 1 of 2");
  assert.equal(get("live-xr-address").hidden, true);
  console.log(
    "Live UI: selection, progress milestones, persistent failures, stale responses, and review passed.",
  );
} finally {
  window.close();
}
