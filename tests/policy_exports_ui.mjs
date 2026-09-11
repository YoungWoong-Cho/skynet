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
const catalogEvents = [];
w.document.addEventListener("dataset-preparation-status", event => catalogEvents.push(event.detail));
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
const copied=[];
w.navigator.clipboard={writeText:async path=>copied.push(path)};
w.showToast=()=>{};
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
      locations: [{ kind: "local", status: "AVAILABLE", path: "/prepared/data <&>" }],
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
  el("prepared-dataset-content").querySelector("[data-dataset-copy-path]").click();
  await flush();
  assert.equal(copied.at(-1), "/prepared/data <&>");
  assert.equal(
    el("prepared-dataset-content").querySelector("[data-preparation-train]"),
    null,
  );
  assert.ok(el("prepared-dataset-content").querySelector("a[download]"));
  const savedLocations = options.exports[0].locations;
  options.exports[0].locations = [{kind: "cluster", host: "skynet", status: "AVAILABLE", path: "/cluster/prepared"}];
  options.exports[0].remote_archive = "/cluster/preparation/dataset.zip";
  await w.openPreparedDataset("dataset");
  assert.ok(el("prepared-dataset-content").querySelector('a[download][href$="/dataset.zip"]'), "A remote-only prepared dataset keeps its explicit streaming download");
  assert.ok(el("prepared-dataset-content").querySelector('a[download][href$="/manifest.json"]'));
  assert.doesNotMatch(el("prepared-dataset-content").textContent, /This computer/);
  options.exports[0].locations = savedLocations;
  delete options.exports[0].remote_archive;
  await w.openPreparedDataset("dataset");
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
  options.sessions[1].locations = [{kind:"remote", host:"sky2", path:"/recording/output/recordings/live"}];
  await w.openPreparedDataset("dataset");
  assert.match(el("prepared-dataset-content").textContent,/sky2/);
  const originalCopy = [...el("prepared-dataset-content").querySelectorAll("[data-dataset-copy-path]")].find(b=>b.dataset.datasetCopyPath.startsWith("/recording"));
  originalCopy.click();await flush();
  assert.equal(copied.at(-1), "/recording/output/recordings/live");
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
  options.sessions[1].episodes = 3;
  await w.openPolicyExport("new");
  assert.equal(el("policy-export-name").readOnly, true, "Preparing another version cannot rename the dataset");
  assert.match(el("policy-export-name-help").textContent, /earlier versions are retained/);
  const normalApi = w.api;
  let finishSubmission;
  w.api = (path, request = {}) => request.method === "POST"
    ? new Promise(resolve => { finishSubmission = resolve; }) : normalApi(path, request);
  el("policy-export-form").dispatchEvent(new w.Event("submit", {cancelable: true}));
  await flush();
  assert.equal(typeof finishSubmission, "function");
  el("close-policy-export").click();
  finishSubmission({id: "late-job", resource_id: "dataset"});
  await flush();
  assert.equal(el("policy-export-dialog").open, false);
  assert.equal(el("prepared-dataset-dialog").open, false, "A late submission response cannot reopen the dismissed workflow");
  w.api = normalApi;
  w.api = async (path, request = {}) => {
    if (path === "/api/data/resources/dataset") throw new Error("Dataset detail temporarily unavailable");
    return normalApi(path, request);
  };
  await w.openPreparedDataset("dataset");
  assert.equal(el("prepared-dataset-error").hidden, false, "The active resource-detail failure must not be mistaken for a stale response");
  assert.equal(catalogEvents.at(-1).state, "ready", "A resource-detail failure must not mark the independent preparation catalog unavailable");
  assert.match(el("prepared-dataset-error").textContent, /temporarily unavailable/);
  assert.doesNotMatch(el("prepared-dataset-content").textContent, /Loading dataset/);
  const retryDetail = el("prepared-dataset-content").querySelector("[data-prepared-resource]");
  assert.equal(retryDetail.textContent, "Try again", "Recovery must be available inside the modal");
  w.api = normalApi;
  retryDetail.click();
  await flush();
  assert.equal(el("prepared-dataset-error").hidden, true, "Successful retry clears the matching detail error");
  assert.match(el("prepared-dataset-content").textContent, /lerobot-v2.0/);
  w.api = async (path, request = {}) => {
    if (path === "/api/data/exports") throw new Error("Preparation catalog temporarily unavailable");
    return normalApi(path, request);
  };
  await w.openPreparedDataset("dataset");
  assert.match(el("prepared-dataset-error").textContent, /catalog temporarily unavailable/);
  assert.equal(catalogEvents.at(-1).state, "unavailable");
  assert.match(catalogEvents.at(-1).error, /catalog temporarily unavailable/);
  assert.ok(el("prepared-dataset-content").querySelector("[data-prepared-resource]"), "An initial catalog failure also has in-modal recovery");
  w.api = normalApi;
  el("prepared-dataset-content").querySelector("[data-prepared-resource]").click();
  await flush();
  assert.equal(el("prepared-dataset-error").hidden, true);
  assert.equal(catalogEvents.at(-1).state, "ready", "Retry broadcasts recovery to the recording table");
  w.api = async (path, request = {}) => {
    if (path === "/api/data/exports") throw new Error("Later catalog outage");
    return normalApi(path, request);
  };
  w.document.dispatchEvent(new w.CustomEvent("collection-recordings-changed"));
  await flush();
  assert.match(el("prepared-dataset-error").textContent, /Later catalog outage/);
  assert.match(el("prepared-dataset-content").textContent, /lerobot-v2.0/, "A later outage retains the already displayed dataset rows");
  assert.ok(el("prepared-dataset-content").querySelector("[data-prepared-resource]"), "Retained rows still provide in-modal retry");
  w.api = normalApi;
  el("prepared-dataset-content").querySelector("[data-prepared-resource]").click();
  await flush();
  assert.equal(el("prepared-dataset-error").hidden, true);
  w.api = async (path, request = {}) => {
    if (path === "/api/data/exports") throw new Error("Initial preparation catalog outage");
    return normalApi(path, request);
  };
  await w.openPolicyExport("new");
  const retryPreparation = el("policy-export-dialog").querySelector("[data-preparation-load-retry]");
  assert.equal(retryPreparation.hidden, false);
  assert.equal(el("create-policy-export").disabled, true);
  w.api = normalApi;
  retryPreparation.click();
  await flush();
  assert.equal(retryPreparation.hidden, true);
  assert.equal(el("policy-export-error").hidden, true);
  assert.equal(el("policy-export-name").value, "Hand demos");
  assert.equal(el("create-policy-export").disabled, false, "The same eligible source becomes preparable after catalog recovery");
  let failOldDetail;
  w.api = (path, request = {}) => path === "/api/data/resources/old-dataset"
    ? new Promise((_resolve, reject) => { failOldDetail = reject; }) : normalApi(path, request);
  const oldDetail = w.openPreparedDataset("old-dataset");
  await flush();
  assert.equal(typeof failOldDetail, "function");
  await w.openPreparedDataset("dataset");
  failOldDetail(new Error("Old dataset unavailable"));
  await oldDetail;
  assert.equal(el("prepared-dataset-error").hidden, true, "A late failure cannot contaminate the replacement dataset");
  assert.match(el("prepared-dataset-content").textContent, /lerobot-v2.0/);
  w.api = normalApi;
  let finishOldRegistryRefresh;
  const ordinaryRegistryRefresh = w.loadDataRegistry;
  w.loadDataRegistry = () => new Promise(resolve => { finishOldRegistryRefresh = resolve; });
  options.sessions[0].resource_id = "older-resource";
  options.exports = [{id: "force-registry-change", state: "READY", resource_id: "dataset"}];
  const oldPreparation = w.openPolicyExport("old");
  await flush();
  assert.equal(typeof finishOldRegistryRefresh, "function");
  w.loadDataRegistry = ordinaryRegistryRefresh;
  await w.openPolicyExport("new");
  assert.equal(el("create-policy-export").disabled, false);
  finishOldRegistryRefresh();
  await oldPreparation;
  el("policy-export-form").dispatchEvent(new w.Event("submit", {cancelable: true}));
  await flush();
  const currentPreparation = calls.filter(([path, request]) => path === "/api/data/exports" && request.method === "POST").at(-1);
  assert.equal(JSON.parse(currentPreparation[1].body).session_id, "new");
  assert.equal(JSON.parse(currentPreparation[1].body).resource_id, "dataset", "A late registry refresh for an earlier dialog cannot replace the current dataset binding");
  resource.metadata = {managed_dataset: true};
  for (const [state, expected, jobId] of [
    ["QUEUED", "QUEUED", null],
    ["STAGING", "PREPARING SUBMISSION", null],
    ["SUBMITTING", "SUBMITTING", null],
    ["SUBMISSION_UNKNOWN", "CHECKING SUBMISSION", null],
    ["PENDING", "QUEUED", "42"],
    ["RUNNING", "RUNNING", "42"],
  ]) {
    options.exports = [{id: "cluster-preparation", resource_id: "dataset", format: "dp", state,
      stage: state === "PENDING" ? "QUEUED" : state, execution: "cluster", gateway: "sky2",
      cluster_partition: "rl2-lab", cluster_cpus: 4, cluster_job_id: jobId}];
    await w.openPreparedDataset("dataset");
    const cell = el("prepared-dataset-content").querySelector("tbody tr td:nth-child(2)");
    assert.equal(cell.querySelector(".state-pill").textContent.trim(), expected);
    assert.match(cell.textContent, /sky2 · rl2-lab/);
    assert.match(cell.textContent, /4 CPUs/);
    assert.match(cell.textContent, jobId ? /Slurm job 42/ : /Not yet assigned a Slurm job/);
  }
  options.exports[0].state = "FAILED";
  options.exports[0].error = "Invalid run ID";
  options.exports[0].cluster_job_id = null;
  await w.openPreparedDataset("dataset");
  assert.equal(el("prepared-dataset-content").querySelector('a[href$="/export.log"]'), null,
    "A submission failure has no Slurm log to download");
  options.exports[0].cluster_job_id = "42";
  await w.openPreparedDataset("dataset");
  assert.ok(el("prepared-dataset-content").querySelector('a[href$="/export.log"]'));
  options.exports = [
    {id: "published-dp", resource_id: "dataset", format: "dp", state: "READY", version_id: "version-dp"},
    {id: "published-act", resource_id: "dataset", format: "act", state: "READY", version_id: "version-act"},
    {id: "failed-dp", resource_id: "dataset", format: "dp", state: "FAILED", error: "Conversion interrupted"},
  ];
  await w.openPreparedDataset("dataset");
  assert.equal(el("prepared-dataset-context").textContent, "2 published versions · 3 preparation attempts", "Failed unpublished preparations count as attempts, not published versions");
  assert.equal(el("prepared-dataset-content").querySelectorAll(".prepared-dataset-table tbody > tr").length, 3, "Each preparation attempt remains available in the detail rows");
  // Two confirmed deletions may overlap a slow cluster request and a stale poll.
  // Exercise real clicks with deferred responses; no server or real files are used.
  w.askUserDialog = async () => true;
  const deleteButton = id => el("prepared-dataset-content").querySelector(`[data-preparation-delete-format="${id}"]`);
  const deletes = [];
  let finishDelete, finishPoll;
  let holdPoll = false;
  w.api = (path, request = {}) => {
    if (request.method === "DELETE") {
      deletes.push(path);
      return new Promise(resolve => {
        finishDelete = () => {
          options.exports = options.exports.filter(job => path !== `/api/data/exports/${job.id}`);
          resolve({deleted: true});
        };
      });
    }
    if (path === "/api/data/exports" && holdPoll) {
      holdPoll = false;
      const stale = structuredClone(options);
      return new Promise(resolve => { finishPoll = () => resolve(stale); });
    }
    return normalApi(path, request);
  };
  deleteButton("published-dp").click();
  await flush();
  deleteButton("published-act").click();
  await flush();
  assert.deepEqual(deletes, ["/api/data/exports/published-dp"], "Rapid deletions wait their turn instead of starting overlapping operations");
  assert.equal(deleteButton("published-dp").textContent, "Deleting…");
  assert.equal(deleteButton("published-act").textContent, "Queued…");
  assert.equal(el("prepared-dataset-content").querySelector("[data-preparation-delete]").disabled, true, "Whole-dataset deletion cannot race queued format deletions");
  await w.openPreparedDataset("dataset");
  assert.equal(deleteButton("published-dp").disabled, true, "Reopening or refreshing retains pending state");
  deleteButton("published-dp").click();
  assert.equal(deletes.length, 1, "A fresh row cannot submit a duplicate delete");
  holdPoll = true;
  w.document.dispatchEvent(new w.CustomEvent("dataset-preparation-refresh-requested"));
  await flush();
  finishDelete();
  await flush();
  finishPoll();
  await flush();
  assert.equal(deleteButton("published-dp"), null, "A catalog read started before deletion cannot be the final refresh");
  assert.deepEqual(deletes, ["/api/data/exports/published-dp", "/api/data/exports/published-act"]);
  finishDelete();
  await flush();
  assert.equal(deleteButton("published-act"), null);
  assert.equal(el("prepared-dataset-context").textContent, "0 published versions · 1 preparation attempts");
  // An older dialog read must not overwrite a newer, post-deletion catalog.
  const olderDialogRead = structuredClone(options);
  olderDialogRead.exports.push({id: "already-deleted", resource_id: "dataset", format: "dp", state: "READY"});
  let finishOlderDialog;
  w.api = (path, request = {}) => path === "/api/data/exports"
    ? new Promise(resolve => { finishOlderDialog = () => resolve(olderDialogRead); })
    : normalApi(path, request);
  const olderDialog = w.openPreparedDataset("dataset");
  await flush();
  w.api = normalApi;
  w.document.dispatchEvent(new w.CustomEvent("dataset-preparation-refresh-requested"));
  await flush();
  finishOlderDialog();
  await olderDialog;
  assert.equal(deleteButton("already-deleted"), null, "A late pre-deletion response cannot bring back a deleted row");

  options.exports.push({id: "next-format", resource_id: "dataset", format: "act", state: "READY"});
  await w.openPreparedDataset("dataset");
  const queuedAfterFailure = [];
  let rejectDelete;
  w.api = (path, request = {}) => {
    if (request.method !== "DELETE") return normalApi(path, request);
    queuedAfterFailure.push(path);
    if (path.endsWith("/failed-dp")) return new Promise((_resolve, reject) => { rejectDelete = reject; });
    options.exports = options.exports.filter(job => path !== `/api/data/exports/${job.id}`);
    return Promise.resolve({deleted: true});
  };
  deleteButton("failed-dp").click();
  await flush();
  deleteButton("next-format").click();
  await flush();
  rejectDelete(new Error("Cluster temporarily unavailable"));
  await flush();
  assert.deepEqual(queuedAfterFailure, ["/api/data/exports/failed-dp", "/api/data/exports/next-format"], "Failure does not drop the next confirmed deletion");
  assert.equal(deleteButton("next-format"), null);
  assert.equal(deleteButton("failed-dp").disabled, false, "Failed deletions can be retried");
  assert.equal(deleteButton("failed-dp").textContent, "Delete");
  assert.match(el("prepared-dataset-error").textContent, /Cluster temporarily unavailable/, "A later successful deletion cannot erase the earlier failure");
  w.askUserDialog = async () => false;
  deleteButton("failed-dp").click();
  await flush();
  assert.equal(queuedAfterFailure.length, 2, "Cancelling never queues a deletion");
  assert.equal(deleteButton("failed-dp").disabled, false);
  w.askUserDialog = async () => true;
  w.api = (path, request = {}) => {
    if (request.method === "DELETE") {
      options.exports = [];
      return Promise.resolve({deleted: true});
    }
    return normalApi(path, request);
  };
  deleteButton("failed-dp").click();
  await flush();
  assert.equal(deleteButton("failed-dp"), null);
  assert.equal(el("prepared-dataset-error").hidden, true, "A successful retry clears its own error");

  let finishWholeDataset;
  w.api = (path, request = {}) => {
    if (request.method === "DELETE") return new Promise(resolve => { finishWholeDataset = resolve; });
    if (path === "/api/data/resources/other-dataset") return Promise.resolve({resource: {id: "other-dataset", name: "Other dataset", metadata: {}, versions: []}});
    return normalApi(path, request);
  };
  el("prepared-dataset-content").querySelector("[data-preparation-delete]").click();
  await flush();
  await w.openPreparedDataset("other-dataset");
  finishWholeDataset({deleted: true});
  await flush();
  assert.equal(el("prepared-dataset-dialog").open, true, "Finishing deletion in the background must not dismiss a different dataset");
  assert.equal(el("prepared-dataset-title").textContent, "Other dataset");
  w.api = normalApi;
  console.log(
    "Preparation UI passed: policy requirements, all session recordings, split guard, submission, grouped management, failed-stage download and retry.",
  );
} finally {
  w.close();
}
