import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { JSDOM } from "jsdom";

const dom = new JSDOM(
  await readFile(new URL("../static/index.html", import.meta.url), "utf8"),
  {
    url: "http://localhost:8080/?collection_view=recordings#collection",
    runScripts: "outside-only",
    pretendToBeVisual: true,
  },
);
const { window: w } = dom;
const el = (id) => w.document.getElementById(id);
const requests = [];
w.HTMLDialogElement.prototype.showModal = function () {
  this.open = true;
};
w.HTMLDialogElement.prototype.close = function () {
  this.open = false;
  this.dispatchEvent(new w.Event("close"));
};
w.fetch = (path, options = {}) =>
  new Promise((resolve) =>
    requests.push({
      path,
      options,
      resolve: (data, ok = true) => resolve({ ok, json: async () => data }),
    }),
  );
const flush = () => new Promise((resolve) => setImmediate(resolve));
const submit = () =>
  el("conversion-form").dispatchEvent(
    new w.Event("submit", { cancelable: true }),
  );
const source = {
  id: "session-1",
  state: "CAPTURED",
  created_at: "2026-09-08T03:00:00Z",
  profile: {
    task: "cube",
    robot: "shadow",
    task_name: "Pick up cube",
    hand_name: "Shadow right",
  },
  recordings: ["a.pkl", "b.pkl"],
  recording_summary: { episodes: 2, steps: 30 },
};
const job = {
  id: "conversion-1",
  session_id: source.id,
  name: "Cube dataset",
  indices: [1],
  state: "QUEUED",
  created_at: "2026-09-08T04:00:00Z",
  gateway: "test-gpu",
};
let reviewed, registry;
w.openLiveReview = (s) => {
  reviewed = s;
};
w.openConvertedDataset = (j) => {
  registry = j;
};
w.eval(
  await readFile(
    new URL("../static/live-conversion.js", import.meta.url),
    "utf8",
  ),
);
w.renderSimulationRecordings([source]);
assert.match(el("simulation-recordings-body").textContent, /2 episodes/);
const actions = [
  ...el("simulation-recordings-body").querySelectorAll("button"),
];
actions[0].click();
assert.equal(reviewed.id, source.id);
actions[1].click();
assert.equal(el("conversion-dialog").open, true);
assert.match(el("conversion-selection-summary").textContent, /2 of 2/);
el("conversion-select-none").click();
assert.equal(el("conversion-submit").disabled, true);
assert.match(el("conversion-status").textContent, /at least one/);
el("conversion-select-all").click();
const checkboxes = el("conversion-recordings").querySelectorAll("input");
checkboxes[0].checked = false;
checkboxes[0].dispatchEvent(new w.Event("change"));
el("conversion-name").value = "   ";
submit();
assert.equal(requests.length, 0);
assert.match(el("conversion-error").textContent, /dataset name/);
el("conversion-name").value = "Cube dataset";
submit();
submit();
assert.equal(requests.length, 1, "Duplicate clicks must not submit twice");
assert.deepEqual(JSON.parse(requests[0].options.body), {
  name: "Cube dataset",
  indices: [1],
});
requests.shift().resolve(job);
await flush();
assert.equal(el("conversion-form").hidden, true);
assert.equal(el("conversion-progress").hidden, false);
requests
  .shift()
  .resolve({
    ...job,
    state: "RUNNING",
    detail: "Converting episode 1 of 1",
    completed: 0,
    total: 1,
  });
await flush();
assert.match(el("conversion-status").textContent, /episode 1/);
el("conversion-refresh").click();
requests
  .shift()
  .resolve({
    ...job,
    state: "RUNNING",
    connection_error: "Workstation unavailable",
  });
await flush();
assert.equal(el("conversion-error").hidden, false);
assert.match(el("conversion-error").textContent, /unavailable/);
el("conversion-refresh").click();
requests
  .shift()
  .resolve({ ...job, state: "FAILED", error: "Recording checksum changed" });
await flush();
assert.equal(el("conversion-retry").hidden, false);
el("conversion-retry").click();
assert.equal(el("conversion-form").hidden, false);
submit();
requests.shift().resolve({ ...job, id: "conversion-2" });
await flush();
const ready = {
  ...job,
  id: "conversion-2",
  state: "READY",
  resource_id: "dataset",
  metadata: {
    episodes: 1,
    steps: 15,
    action_dim: 28,
    observation_shapes: { "proprio/joint_pos": [28] },
  },
};
requests.shift().resolve(ready);
await flush();
assert.equal(el("conversion-result").hidden, false);
assert.ok(
  el("conversion-download").href.endsWith("/conversion-2/dataset.hdf5"),
);
assert.match(
  el("simulation-recordings-body").textContent,
  /1 episode · Cube dataset/,
);
el("conversion-registry").click();
assert.equal(registry.resource_id, "dataset");
assert.equal(el("conversion-dialog").open, false);
w.openCollectionConversion(source, ready);
el("conversion-new").click();
assert.equal(el("conversion-form").hidden, false);
assert.equal(
  el("conversion-recordings").querySelectorAll("input:checked").length,
  2,
);
el("conversion-close").click();
w.openCollectionConversion(
  { ...source, id: "active", state: "COLLECTING" },
  null,
);
assert.equal(el("conversion-submit").disabled, true);
assert.match(el("conversion-status").textContent, /End collection/);
el("conversion-close").click();
w.openCollectionConversion(source, null);
submit();
el("conversion-close").click();
w.openCollectionConversion({ ...source, id: "another" }, null);
requests.shift().resolve(job);
await flush();
assert.match(el("conversion-context").textContent, /Pick up cube/);
assert.equal(
  el("conversion-form").hidden,
  false,
  "A closed request must not replace the new dialog",
);
el("conversion-close").click();
el("simulation-recordings-search").value = "missing";
el("simulation-recordings-search").dispatchEvent(new w.Event("input"));
assert.match(el("simulation-recordings-body").textContent, /No sessions match/);
w.close();
console.log(
  "Conversion UI: selection, duplicate submit, progress, retry, download, registry handoff and stale-dialog requests passed.",
);
