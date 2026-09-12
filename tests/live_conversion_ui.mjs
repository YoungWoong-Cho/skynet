import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { JSDOM } from "jsdom";
const dom = new JSDOM(
  await readFile(new URL("../static/index.html", import.meta.url), "utf8"),
  {
    url: "http://localhost:8080/?collection_view=recordings#collection",
    runScripts: "outside-only",
  },
);
const w = dom.window,
  el = (id) => w.document.getElementById(id);
let reviewed,
  prepared,
  changes = 0;
w.openLiveReview = (s) => (reviewed = s.id);
w.openPolicyExport = (id) => (prepared = id);
w.recordingRegistrationSummary = () => ({state: "ready", count: 0});
w.showRecordingResources = () => {};
w.document.addEventListener("collection-recordings-changed", () => changes++);
const app = await readFile(new URL("../static/app.js", import.meta.url), "utf8");
for (const name of ["escapeHtml", "stateClass", "statusPill", "formatDate"]) {
  const start = app.indexOf(`function ${name}(`);
  const end = app.indexOf("\n}\n", start) + 3;
  w.eval(app.slice(start, end));
}
try {
  w.eval(
    await readFile(
      new URL("../static/live-conversion.js", import.meta.url),
      "utf8",
    ),
  );
  const s = {
    id: "session",
    state: "CAPTURED",
    created_at: "2026-09-08",
    profile: { task: "cube", robot: "Shadow" },
    recordings: ["a", "b"],
    recording_images: { a: {}, b: {} },
  };
  w.renderSimulationRecordings([s]);
  w.renderSimulationRecordings([s]);
  assert.equal(changes, 1);
  assert.match(el("simulation-recordings-body").textContent, /Checking preparation status/);
  w.document.dispatchEvent(new w.CustomEvent("dataset-preparation-status", {detail:{state:"unavailable", error:"Catalog timed out"}}));
  assert.match(el("simulation-recordings-body").textContent, /Preparation status unavailable/);
  assert.doesNotMatch(el("simulation-recordings-body").textContent, /Original recordings saved|Images ready/);
  let refreshRequests = 0;
  w.document.addEventListener("dataset-preparation-refresh-requested", () => refreshRequests++);
  [...el("simulation-recordings-body").querySelectorAll(".row-actions button")].find(b => b.textContent === "Retry preparation status").click();
  assert.equal(refreshRequests, 1);
  w.document.dispatchEvent(new w.CustomEvent("dataset-preparation-status", {detail:{state:"ready"}}));
  assert.match(el("simulation-recordings-body").textContent, /Images ready/);
  assert.ok(el("simulation-recordings-body").querySelector(".state-pill.is-running"));
  assert.doesNotMatch(el("simulation-recordings-body").textContent, /Earlier state HDF5/);
  let buttons = el("simulation-recordings-body").querySelectorAll(".row-actions button");
  buttons[0].click();
  buttons[1].click();
  assert.equal(reviewed, "session");
  assert.equal(prepared, "session");
  w.document.dispatchEvent(
    new w.CustomEvent("dataset-preparation-changed", {
      detail: [
        {
          id: "job",
          session_id: "session",
          state: "READY",
          resource_id: "data",
        },
      ],
    }),
  );
  assert.match(
    el("simulation-recordings-body").textContent,
    /1 prepared format/,
  );
  w.document.dispatchEvent(new w.CustomEvent("dataset-preparation-changed", {
    detail: [
      {id: "job-dp-1", session_id: "session", state: "READY", resource_id: "data", format: "dp"},
      {id: "job-dp-2", session_id: "session", state: "READY", resource_id: "data", format: "dp"},
      {id: "job-act", session_id: "session", state: "READY", resource_id: "data", format: "act"},
      {id: "job-failed", session_id: "session", state: "FAILED", resource_id: "data", format: "openpi"},
    ],
  }));
  assert.match(el("simulation-recordings-body").textContent, /2 prepared formats/);
  assert.doesNotMatch(el("simulation-recordings-body").textContent, /[34] prepared formats/);
  assert.match(el("simulation-recordings-body").textContent, /Preparation needs attention/);
  w.document.dispatchEvent(new w.CustomEvent("dataset-preparation-status", {detail:{state:"unavailable", error:"Host unavailable"}}));
  assert.match(el("simulation-recordings-body").textContent, /Last known: 2 prepared formats/);
  assert.ok(![...el("simulation-recordings-body").querySelectorAll(".row-actions button")].some(b => b.textContent === "View dataset"));
  w.document.dispatchEvent(new w.CustomEvent("dataset-preparation-status", {detail:{state:"ready"}}));
  assert.doesNotMatch(el("simulation-recordings-body").textContent, /Host unavailable|Last known/);
  assert.deepEqual([...el("simulation-recordings-body").querySelectorAll(".row-actions button")].map(b => b.textContent), ["View recordings", "Register"]);
  const subset = {...s, id: "subset", recordings: ["a"], recording_images: {a: {}}};
  w.renderSimulationRecordings([s, subset]);
  const mixed = [
    {id: "new-subset", session_id: "session", selections: [{session_id: "session", indices: [0]}], recording_session_id: "subset", state: "READY", resource_id: "subset-data", format: "egoverse"},
    {id: "original", session_id: "session", recording_session_id: "session", state: "READY", resource_id: "data", format: "act"},
  ];
  for (const detail of [mixed, [...mixed].reverse()]) {
    w.document.dispatchEvent(new w.CustomEvent("dataset-preparation-changed", {detail}));
    const rows = [...el("simulation-recordings-body").rows];
    assert.equal(rows[0].cells.length, 6);
    assert.equal(rows[1].cells.length, 6);
    assert.equal(rows[0].querySelectorAll(".row-actions button").length, 2);
    assert.equal(rows[1].querySelectorAll(".row-actions button").length, 2);
    assert.match(rows[0].textContent, /2 episodes/);
    assert.match(rows[1].textContent, /1 episode/);
  }
  w.document.dispatchEvent(new w.CustomEvent("dataset-preparation-changed", {detail: [
    {...mixed[0], recording_session_id: null}, mixed[1],
  ]}));
  assert.equal(el("simulation-recordings-body").rows[1].querySelectorAll(".row-actions button").length, 2, "unassigned subset cannot hijack another entry");
  assert.equal(el("conversion-dialog"), null);
  assert.doesNotMatch(
    el("simulation-recordings-body").textContent,
    /Convert for training/,
  );
  el("simulation-recordings-search").value = "missing";
  el("simulation-recordings-search").dispatchEvent(new w.Event("input"));
  assert.match(
    el("simulation-recordings-body").textContent,
    /No sessions match/,
  );
  console.log(
    "Recordings UI passed: unified preparation entry, source changes, dataset status, review and search.",
  );
} finally {
  w.close();
}
