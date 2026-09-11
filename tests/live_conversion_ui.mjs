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
  viewed,
  changes = 0;
w.openLiveReview = (s) => (reviewed = s.id);
w.openPolicyExport = (id) => (prepared = id);
w.openPreparedDataset = (id) => (viewed = id);
w.document.addEventListener("collection-recordings-changed", () => changes++);
const app = await readFile(new URL("../static/app.js", import.meta.url), "utf8");
for (const name of ["escapeHtml", "stateClass", "statusPill"]) {
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
  [...el("simulation-recordings-body").querySelectorAll("button")].find(b => b.textContent === "Retry preparation status").click();
  assert.equal(refreshRequests, 1);
  w.document.dispatchEvent(new w.CustomEvent("dataset-preparation-status", {detail:{state:"ready"}}));
  assert.match(el("simulation-recordings-body").textContent, /Images ready/);
  assert.ok(el("simulation-recordings-body").querySelector(".state-pill.is-running"));
  assert.doesNotMatch(el("simulation-recordings-body").textContent, /Earlier state HDF5/);
  let buttons = el("simulation-recordings-body").querySelectorAll("button");
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
  assert.ok([...el("simulation-recordings-body").querySelectorAll("button")].some(b => b.textContent === "View dataset"));
  w.document.dispatchEvent(new w.CustomEvent("dataset-preparation-status", {detail:{state:"ready"}}));
  assert.doesNotMatch(el("simulation-recordings-body").textContent, /Host unavailable|Last known/);
  buttons = el("simulation-recordings-body").querySelectorAll("button");
  buttons[2].click();
  assert.equal(viewed, "data");
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
