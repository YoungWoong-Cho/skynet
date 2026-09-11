import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { JSDOM } from "jsdom";
const w = new JSDOM(
  await readFile(new URL("../static/index.html", import.meta.url), "utf8"),
  { runScripts: "outside-only", url: "http://skynet/#settings" },
).window;
const el = (id) => w.document.getElementById(id);
const flush = async () => {
  for (let i = 0; i < 4; i++) await new Promise((r) => setImmediate(r));
};
const calls = [];
let handler = async () => ({ work_root: "/team/alice" });
w.api = async (path, options = {}) => {
  calls.push({ path, ...options });
  return handler(path, options);
};
w.eval(
  await readFile(
    new URL("../static/workspace-storage.js", import.meta.url),
    "utf8",
  ),
);
try {
  const form = el("workspace-storage-form"),
    path = el("workspace-base-path");
  w.renderWorkspaceStorage({ work_root: "/shared/default" });
  assert.equal(path.value, "/shared/default");
  assert.equal(el("save-workspace-storage").disabled, true);
  path.value = "/team/alice";
  path.dispatchEvent(new w.Event("input"));
  assert.equal(el("init-workspace").disabled, true);
  await w.initializeWorkspaceStorage();
  assert.equal(calls.length, 0);
  w.renderWorkspaceStorage({ work_root: "/shared/new-default" });
  assert.equal(path.value, "/team/alice", "refresh retains the draft");
  let resolveSave;
  handler = () => new Promise((r) => (resolveSave = r));
  form.dispatchEvent(new w.Event("submit", { cancelable: true }));
  form.dispatchEvent(new w.Event("submit", { cancelable: true }));
  assert.equal(calls.length, 1);
  assert.equal(
    JSON.parse(calls[0].body).expected_work_root,
    "/shared/new-default",
  );
  resolveSave({ work_root: "/team/alice" });
  await flush();
  assert.equal(el("init-workspace").disabled, false);
  assert.equal(el("save-workspace-storage").disabled, true);
  handler = async () => {
    throw Error("Permission denied");
  };
  await w.initializeWorkspaceStorage();
  assert.equal(calls.at(-1).path.startsWith("/api/workspace/init?"), true);
  assert.match(el("workspace-storage-detail").textContent, /Permission denied/);
  assert.equal(el("init-workspace").disabled, false);
  assert.equal(path.value, "/team/alice");
  assert.equal(w.localStorage.length, 0);
  console.log(
    "Personal storage: load, drafts, stale save value, duplicate clicks, initialization and errors passed.",
  );
} finally {
  w.close();
}
