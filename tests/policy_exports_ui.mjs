import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { JSDOM } from "jsdom";
const dom = new JSDOM(
  await readFile(new URL("../static/index.html", import.meta.url), "utf8"),
  {
    runScripts: "outside-only",
    url: "http://localhost:8080/#datasets",
    pretendToBeVisual: true,
  },
);
const w = dom.window,
  el = (id) => w.document.getElementById(id),
  calls = [];
w.HTMLDialogElement.prototype.showModal = function () {
  this.open = true;
};
w.HTMLDialogElement.prototype.close = function () {
  if (this.open) {
    this.open = false;
    this.dispatchEvent(new w.Event("close"));
  }
};
w.HTMLElement.prototype.scrollIntoView = () => {};
// Shared app helpers are loaded from their actual definitions.
const app = await readFile(
  new URL("../static/app.js", import.meta.url),
  "utf8",
);
for (const name of [
  "escapeHtml",
  "formatDataBytes",
  "formatDate",
  "stateClass",
  "statusPill",
  "dataVersionStatus",
  "datasetBindingValue",
  "adapterDataContracts",
  "experimentBundleCompatibility",
]) {
  const start = app.indexOf(`function ${name}(`),
    end = app.indexOf("\nfunction ", start + 1);
  let text = app.slice(start, end);
  // Only the first function; subsequent top-level declarations are not needed.
  const brace = text.indexOf("\n}\n");
  if (brace >= 0) text = text.slice(0, brace + 3);
  w.eval(text);
}
w.declaredAdapterInputFields = adapter => adapter.fields;
const nativeAdapter = {fields:[{data_binding:{role:'training_data',formats:['egoverse-episodes-zarr/v1'],contracts:['egoverse.native-pi0.5_bc_aria/v1'],value_path:'location.path'}}]};
const registered = {assignments:[{role:'training_data',version:{format:'egoverse-episodes-zarr/v1',manifest_sha256:'sha',metadata:{contract:'skynet.egoverse-rgb-joints/v1',validation:{status:'PASSED'}}},config:{location:{kind:'cluster',status:'AVAILABLE',manifest_sha256:'sha',path:'/prepared'}}}]};
assert.equal(w.experimentBundleCompatibility(registered,nativeAdapter).compatible,false,'matching file formats cannot substitute for matching model observations');
registered.assignments[0].version.metadata.contract='egoverse.native-pi0.5_bc_aria/v1';
assert.equal(w.experimentBundleCompatibility(registered,nativeAdapter).compatible,true,'verified native model data remains selectable');
registered.assignments[0].version.metadata.validation.status='FAILED';
assert.equal(w.experimentBundleCompatibility(registered,nativeAdapter).compatible,false,'failed validation cannot appear as compatible');
const registryRefreshes = [];
w.loadDataRegistry = async (force) => { registryRefreshes.push(force); };
w.askUserDialog = async () => true;
const options = {
  policies: [
    {
      id: "dp",
      name: "Diffusion Policy",
      available: true,
      trainable: true,
      container: "Zarr",
      description: "Recorded RGB and joints",
    },
    {
      id: "act",
      name: "ACT",
      available: true,
      trainable: false,
      container: "HDF5",
      description: "Training adapter not installed",
    },
    {
      id: "openpi",
      name: "openpi",
      available: false,
      description: "Requires a wrist camera",
    },
  ],
  formats: [],
  sessions: [
    {
      id: "old",
      name: "Old",
      created_at: "2026-09-01",
      eligible: false,
      reason: "Images unavailable",
      episodes: 2,
    },
    {
      id: "new",
      name: "New",
      created_at: "2026-09-08",
      eligible: true,
      episodes: 3,
      resource_id: "dataset",
    },
  ],
  exports: [],
};
const resource = {
  id: "dataset",
  name: "Hand demos",
  metadata: { managed_dataset: true, session_id: "new" },
  versions: [],
};
w.api = async (path, request = {}) => {
  calls.push([path, request]);
  if (request.method === "POST") return { id: "job", resource_id: "dataset" };
  if (path.startsWith("/api/data/resources/")) return { resource };
  return structuredClone(options);
};
const flush = async () => {
  for (let i = 0; i < 6; i++) await new Promise((r) => setImmediate(r));
};
try {
  w.eval(await readFile(new URL("../static/dialogs.js", import.meta.url), "utf8"));
  w.eval(
    await readFile(
      new URL("../static/policy-exports.js", import.meta.url),
      "utf8",
    ),
  );
  await flush();
  assert.equal(w.stateClass("SUCCEEDED"), "is-running");
  assert.equal(w.stateClass("ON CLUSTER"), "is-running");
  assert.equal(w.stateClass("LOCAL"), "is-local");
  assert.equal(w.stateClass("COPY UNAVAILABLE"), "is-failed");
  await w.openPolicyExport("old");
  assert.equal(el("create-policy-export").disabled, true);
  assert.match(
    el("policy-export-compatibility").textContent,
    /Images unavailable/,
  );
  await w.openPolicyExport("new", "dataset");
  assert.equal(el("create-policy-export").disabled, false);
  el("policy-export-format").value = "openpi";
  el("policy-export-format").dispatchEvent(new w.Event("change"));
  assert.equal(el("create-policy-export").disabled, true);
  assert.match(el("policy-export-compatibility").textContent, /wrist camera/);
  el("policy-export-format").value = "dp";
  el("policy-export-format").dispatchEvent(new w.Event("change"));
  assert.equal(el("policy-export-compatibility").hidden, true);
  for (const id of ["preparation-select-none", "preparation-episodes", "preparation-add-session", "preparation-revision", "policy-export-session", "policy-export-target", "show-policy-export", "prepared-dataset-prepare"])
    assert.equal(el(id), null, id);
  el("preparation-validation").value = 0;
  el("preparation-validation").dispatchEvent(new w.Event("input"));
  assert.equal(el("create-policy-export").disabled, true);
  el("preparation-validation").value = 20;
  el("preparation-validation").dispatchEvent(new w.Event("input"));
  assert.equal(el("create-policy-export").disabled, false);
  el("policy-export-name").value = "Hand demos";
  el("policy-export-form").dispatchEvent(
    new w.Event("submit", { cancelable: true }),
  );
  await flush();
  const post = calls.find(([, r]) => r.method === "POST");
  assert.deepEqual(JSON.parse(post[1].body), {
    session_id: "new",
    format: "dp",
    name: "Hand demos",
    resource_id: "dataset",
    gateway: "auto",
    validation_percent: 20,
    seed: 42,
  });
  assert.equal(el("policy-export-dialog").open, false);
  assert.equal(el("prepared-dataset-dialog").open, true);
  options.exports = [
    {
      id: "job",
      resource_id: "dataset",
      name: "<img src=x onerror=alert(1)>",
      session_id: "new",
      format: "dp",
      state: "FAILED",
      stage: "TRANSFERRING",
      error: "Gateway unavailable",
      version_id: "v",
      locations: [{ kind: "local", status: "AVAILABLE" }],
      split: { train: [0], validation: [1] },
      source_revision: "abc",
      sources: [{}, {}],
      usage: [],
    },
  ];
  await w.openPreparedDataset("dataset");
  assert.equal(
    el("prepared-dataset-content").querySelectorAll("img").length,
    0,
  );
  assert.match(
    el("prepared-dataset-content").textContent,
    /This computer/,
  );
  assert.equal(
    el("prepared-dataset-content").querySelector("[data-preparation-train]"),
    null,
  );
  assert.ok(el("prepared-dataset-content").querySelector("a[download]"));
  assert.deepEqual([...el("prepared-dataset-content").querySelectorAll(".prepared-dataset-table th")].map(n=>n.textContent), ["Format", "Status", "Location", "Episodes", "Date", "Actions"]);
  assert.ok(el("prepared-dataset-content").querySelector("[data-preparation-delete]"));
  el("prepared-dataset-content")
    .querySelector("[data-preparation-retry]")
    .click();
  await flush();
  assert.ok(calls.some(([p]) => p === "/api/data/exports/job/retry"));
  resource.versions = [
    {
      id: "source-two",
      format: "skynet.episodes/v1",
      revision: "abc123",
      metadata: {
        episodes: 2,
        sources: [
          { session_id: "new", index: 0 },
          { session_id: "new", index: 2 },
        ],
        split: { train: [1], validation: [0], seed: 7, validation_percent: 30 },
      },
    },
  ];
  options.exports[0].source_version_id = "source-two";
  await w.openPolicyExport("new");
  assert.equal(el("preparation-validation").value, "20");
  assert.equal(el("preparation-seed").value, "42");
  el("policy-export-form").dispatchEvent(new w.Event("submit", {cancelable: true}));
  await flush();
  const latest = calls.filter(([, r]) => r.method === "POST" && !r.body.includes("target")).at(-1);
  assert.equal(JSON.parse(latest[1].body).session_id, "new");
  assert.ok(!("selections" in JSON.parse(latest[1].body)), "older partial revisions cannot override all recordings");
  options.sessions[1].episodes = 1;
  await w.openPolicyExport("new");
  assert.equal(el("create-policy-export").disabled, true);
  el("close-policy-export").click();
  assert.equal(el("policy-export-dialog").open, false);
  assert.equal(el("policy-export-dialog").querySelector("details"), null);
  const beforeReady = registryRefreshes.length;
  options.exports[0].state = "READY";
  options.exports[0].bundle_id = "new-cluster-bundle";
  await w.openPreparedDataset("dataset");
  assert.equal(registryRefreshes.length, beforeReady + 1, "opening a completed conversion refreshes the registry even before polling");
  await w.openPreparedDataset("dataset");
  assert.equal(registryRefreshes.length, beforeReady + 1, "unchanged data does not reload the registry");
  options.exports = [];
  resource.metadata = {};
  resource.versions = [{id: "imported", format: "lerobot-v2.0", status: "READY", path: "/cluster/imported", revision: "abc123", metadata: {episodes: 42}}];
  await w.openPreparedDataset("dataset");
  assert.match(el("prepared-dataset-content").textContent, /lerobot-v2.0/);
  assert.match(el("prepared-dataset-content").textContent, /42/);
  assert.match(el("prepared-dataset-content").textContent, /cluster\/imported/);
  assert.ok(el("prepared-dataset-content").querySelector(".state-pill.is-running"));
  assert.equal(el("prepared-dataset-content").querySelector("[data-preparation-delete]"), null);
  console.log(
    "Preparation UI passed: policy requirements, all session recordings, split guard, submission, grouped management, failed-stage download and retry.",
  );
} finally {
  w.close();
}
