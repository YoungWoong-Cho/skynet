import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {JSDOM} from "jsdom";

const dom = new JSDOM(await readFile(new URL("../static/index.html", import.meta.url), "utf8"), {runScripts:"outside-only", url:"http://localhost/"});
const w = dom.window;
const el = id => w.document.getElementById(id);
const app = await readFile(new URL("../static/app.js", import.meta.url), "utf8");
const load = name => {
  const start = app.search(new RegExp(`(?:async )?function ${name}\\(`));
  assert.ok(start >= 0, name);
  w.eval(app.slice(start, app.indexOf("\n}\n", start) + 3));
};
w.elements = new Proxy({}, {get: (_, key) => el(key.replace(/[A-Z]/g, c => "-" + c.toLowerCase()))});
w.collectionAdapterRows = [];
w.collectionPrepareStates = new Set(["DRAFT", "READY"]);
w.collectionSubmittedStates = new Set(["CAPTURED", "COMPLETED"]);
w.collectionTerminalStates = new Set(["CAPTURED", "COMPLETED", "FAILED", "CANCELLED"]);
w.loadedTabs = new Set();
const notices = [];
w.showToast = message => notices.push(message);
w.updateLogView = (node, text) => { node.textContent = text; };
w.emptyRow = () => "";
w.compactJson = value => JSON.stringify(value);
w.shortId = value => value;
w.statusPill = value => value;
w.entityFrom = value => value.session;
w.currentRevealLauncher = () => null;
w.disclosureToken = () => null;
w.disclosureTokenIsCurrent = () => true;
w.revealPanel = panel => {panel.hidden = false;};
w.toggleDisclosure = () => {};
const order = [];
w.closeDisclosurePanel = panel => {order.push("close-form"); panel.hidden = true;};
w.loadCollection = async () => {
  order.push("refresh-list");
  el("collection-sessions-body").innerHTML = '<tr><td><button data-collection-session-action="view" data-id="created">View</button></td></tr>';
};
for (const name of ["escapeHtml", "formatDate", "parseDataJson", "collectionPretty", "collectionAdapterManifest", "collectionAdapterIdentity", "collectionSessionContext", "collectionSessionAdapterName", "collectionActionButton", "renderCollectionLifecycleActions", "renderCollectionSessionDetail", "collectionParseObject", "updateCollectionCapabilityEvidence", "viewCollectionSession", "createCollectionSession", "collectionTutorialResumeIndex", "tutorialAdapterSlug", "tutorialCollectionAdapterIdentity", "tutorialCollectionAdapterFormIdentity", "tutorialExperimentAdapterIdentity", "tutorialExperimentAdapterFormIdentity"]) load(name);
const session = {id:"created", name:"Saved session", status:"CAPTURED", adapter_snapshot:{manifest:{display_name:"Recorder"}}, storage_snapshot:{registration:null}, events:[{event_type:"STATUS_CHANGED", old_status:"RUNNING", new_status:"CAPTURED", details:{}, created_at:"2026-09-10"}]};
try {
  w.renderCollectionSessionDetail(session);
  assert.match(el("collection-events-body").textContent, /RUNNING → CAPTURED/);
  assert.equal(el("collection-complete-form").querySelector("button").textContent, "Complete without registration");
  for (const [id, maximum] of [["collection-gpu-count", 16], ["collection-cpu-count", 256], ["collection-memory-gb", 2048]]) {
    const input = el(id);
    const original = input.value;
    for (const invalid of ["", "0", "-1", "1.5", String(maximum + 1)]) {
      input.value = invalid;
      assert.equal(input.checkValidity(), false, `${id} must reject ${JSON.stringify(invalid)} before submission`);
    }
    for (const valid of ["1", String(maximum)]) {
      input.value = valid;
      assert.equal(input.checkValidity(), true, `${id} must accept its boundary ${valid}`);
    }
    input.value = original;
  }
  const form = el("collection-session-form");
  form.hidden = false;
  form.reportValidity = () => true;
  el("collection-session-name").value = "Unsaved draft";
  let finish, posts = 0;
  w.api = (path, request = {}) => request.method === "POST"
    ? (posts++, new Promise(resolve => {finish = resolve;})) : Promise.resolve({session});
  const submit = w.createCollectionSession({preventDefault(){}});
  await w.createCollectionSession({preventDefault(){}});
  assert.equal(posts, 1, `Repeated submit while saving must not create another immutable draft: ${notices.join("; ")}`);
  assert.equal(el("create-collection-session").disabled, true);
  finish({session});
  await submit;
  assert.deepEqual(order, ["close-form", "refresh-list"], "A row-owned form is closed before refreshing its owner table");
  assert.equal(form.hidden, true);
  assert.equal(el("collection-session-name").value, "");
  assert.equal(el("collection-session-detail").hidden, false);
  assert.equal(el("collection-session-detail-title").textContent, "Saved session");
  for (const stepId of ["launcher", "create", "read", "archive"]) {
    const saved = {stepId, token:"tutorial-same-token", bindings:{}, completedStepIds:["key", "launcher"], ownedRecords:[]};
    assert.equal(w.collectionTutorialResumeIndex(saved, 10), 0);
    assert.equal(saved.stepId, "open-adapter");
    assert.equal(saved.completedStepIds.length, 0, "Unsaved values must be validated again after resume");
    assert.equal(saved.token, "tutorial-same-token");
  }
  const legacyToken = "tutorial-20260910T041936Z-5bb489a96924";
  const collectionKey = w.tutorialCollectionAdapterIdentity({token:legacyToken})[0].value;
  assert.match(collectionKey, /^[a-z0-9][a-z0-9-]{1,63}$/, "Resumed legacy tutorial keys must satisfy the server contract");
  el("collection-adapter-key").value = `${legacyToken}-collection-adapter`;
  assert.equal(el("collection-adapter-key").checkValidity(), false, "Invalid legacy key must not pass the native field gate");
  el("collection-adapter-key").value = collectionKey;
  assert.equal(el("collection-adapter-key").checkValidity(), true);
  el("collection-adapter-metadata").value = JSON.stringify({tutorial_token:legacyToken});
  assert.equal(w.tutorialCollectionAdapterFormIdentity({token:legacyToken}), true);
  assert.equal(w.tutorialCollectionAdapterIdentity({token:legacyToken})[1].value, legacyToken, "Canonicalizing a slug must not rewrite the ownership token");
  const adapterSlug = w.tutorialExperimentAdapterIdentity({token:legacyToken})[0].value;
  assert.match(adapterSlug, /^[a-z0-9][a-z0-9-]*$/);
  el("adapter-editor-slug").value = adapterSlug;
  assert.equal(w.tutorialExperimentAdapterFormIdentity({token:legacyToken}), true);
  const bound = {stepId:"read", bindings:{collectionAdapterId:"bound-id"}, completedStepIds:["create"]};
  assert.equal(w.collectionTutorialResumeIndex(bound, 10), 10, "Verified saved identity keeps its recovery path");
  assert.deepEqual(bound.completedStepIds, ["create"]);
  console.log("Collection lifecycle UI: saved detail, duplicate-submit guard, form reset, completion mode, event transitions and unbound tutorial resume passed.");
} finally {w.close();}
