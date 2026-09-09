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
  assert.match(el("simulation-recordings-body").textContent, /Images ready/);
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
