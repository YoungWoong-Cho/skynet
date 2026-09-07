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
      resolve: (data) => resolve({ ok: true, json: async () => data }),
    }),
  );
const flush = () => new Promise((resolve) => setImmediate(resolve));
const get = (id) => window.document.getElementById(id);
get("collection").hidden = false;
get("collection-view-live").hidden = false;
const catalog = {
  default_robot: "floating_shadow_right",
  default_task: "Dexverse-PickUpStick-v0",
  hands: [
    { key: "floating_shadow_right", name: "Right", available: true },
    { key: "floating_shadow_left", name: "Left", available: true },
    {
      key: "wuji-1",
      name: "WUJI",
      available: false,
      reason: "Adapter missing",
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
    { robot: "floating_shadow_right", task: "Dexverse-PickUpStick-v0" },
  ],
  note: "Pinned release",
};
const job = {
  id: "test-session",
  state: "PENDING",
  job_id: "123",
  server_ready: false,
  profile: { robot: "floating_shadow_left", task: "Dexverse-PickCube-v0" },
};
try {
  window.eval(
    await readFile(new URL("../static/live-xr.js", import.meta.url), "utf8"),
  );
  get("live-xr-start-form").dispatchEvent(
    new window.Event("submit", { cancelable: true }),
  );
  assert.equal(requests.length, 1, "Cannot submit before choices load");
  requests[0].resolve({ sessions: [], license: { accepted: true }, catalog });
  await flush();
  assert.equal(
    get("live-xr-hand").querySelector('option[value="wuji-1"]').disabled,
    true,
  );
  get("live-xr-hand").value = "floating_shadow_left";
  get("live-xr-task").value = "Dexverse-PickCube-v0";
  get("live-xr-task").dispatchEvent(new window.Event("change"));
  assert.match(get("live-xr-task-instructions").textContent, /Lift cube/);
  assert.match(
    get("live-xr-selection-note").textContent,
    /not yet been tested/,
  );
  assert.equal(
    new URL(window.location).searchParams.get("live_hand"),
    "floating_shadow_left",
  );
  get("live-xr-refresh").click(); // Pending refresh must not erase subsequent accepted submission.
  get("live-xr-start-form").dispatchEvent(
    new window.Event("submit", { cancelable: true }),
  );
  assert.equal(get("live-xr-start").textContent, "Starting…");
  assert.equal(
    JSON.parse(requests[2].options.body).robot,
    "floating_shadow_left",
  );
  assert.equal(
    JSON.parse(requests[2].options.body).task,
    "Dexverse-PickCube-v0",
  );
  requests[2].resolve(job);
  await flush();
  requests[1].resolve({ sessions: [], license: { accepted: true }, catalog });
  await flush();
  assert.match(get("live-xr-sessions").textContent, /test-ses/);
  assert.equal(get("live-xr-hand").disabled, true);
  get("live-xr-refresh").click();
  const stop = [...get("live-xr-sessions").querySelectorAll("button")].find(
    (b) => b.textContent === "Stop session",
  );
  stop.click();
  requests[4].resolve({ ...job, stop_requested: true });
  await flush();
  requests[3].resolve({
    sessions: [job],
    license: { accepted: true },
    catalog,
  });
  await flush();
  assert.ok(
    [...get("live-xr-sessions").querySelectorAll("button")].find(
      (b) => b.textContent === "Stopping…",
    )?.disabled,
  );
  get("live-xr-refresh").click();
  requests[5].resolve({
    sessions: [
      {
        ...job,
        state: "CAPTURED",
        scheduler_final: true,
        profile: { ...job.profile, execution: "workstation" },
        gateway: "test-workstation",
        recordings: ["recordings/live/demo.pkl"],
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
    /directly on test-workstation/,
  );
  let reviewed;
  window.openLiveReview = (...args) => {
    reviewed = args;
  };
  [...get("live-xr-sessions").querySelectorAll("button")]
    .find((b) => b.textContent === "Review recording")
    .click();
  assert.equal(reviewed[0].id, job.id);
  assert.equal(reviewed[1], 0);
  assert.doesNotMatch(
    get("live-xr-sessions").textContent,
    /Run training cycle/,
  );
  console.log(
    "Live UI: selected values, unsupported hands, stale responses, and direct review passed.",
  );
} finally {
  window.close();
}
