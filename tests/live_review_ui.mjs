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
  get("live-review-close").click();
  window.openLiveReview({ id: "stale" }, 0);
  get("live-review-close").click();
  window.openLiveReview({ id: "current" }, 0);
  requests[2].resolve({ state: "READY" });
  await flush();
  assert.equal(requests.length, 4, "Closed review must not fetch stale data");
  requests[3].resolve({ state: "FAILED", error: "Host unreachable" });
  await flush();
  assert.match(get("live-review-status").textContent, /Host unreachable/);
  assert.equal(get("live-review-retry").hidden, false);
  get("live-review-retry").click();
  assert.equal(requests[4].options.method, "POST");
  console.log(
    "Review UI: frame boundaries, values, action alignment, close races and retry passed.",
  );
} finally {
  window.close();
}
