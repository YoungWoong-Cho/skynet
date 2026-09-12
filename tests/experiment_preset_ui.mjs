import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { JSDOM } from "jsdom";

const w = new JSDOM(
  await readFile(new URL("../static/index.html", import.meta.url), "utf8"),
  {
    runScripts: "outside-only",
    pretendToBeVisual: true,
    url: "http://localhost:8080/?experiment_view=presets#experiments",
  },
).window;
const observers = [],
  Observer = w.MutationObserver;
w.MutationObserver = class extends Observer {
  constructor(cb) {
    super(cb);
    observers.push(this);
  }
};
w.fetch = () => new Promise(() => {});
w.scrollTo = w.HTMLElement.prototype.scrollIntoView = () => {};
w.matchMedia = () => ({
  matches: false,
  addEventListener() {},
  removeEventListener() {},
});
w.HTMLDialogElement.prototype.showModal = function () {
  this.open = true;
};
w.HTMLDialogElement.prototype.close = function () {
  if (!this.open) return;
  this.open = false;
  this.dispatchEvent(new w.Event("close"));
};
const el = (id) => w.document.getElementById(id);
const flush = async () => {
  for (let i = 0; i < 5; i++) await new Promise((r) => setImmediate(r));
};
try {
  for (const file of [
    "dialogs.js",
    "workspace-navigation.js",
    "connection-settings.js",
    "app.js",
  ])
    w.eval(
      (await readFile(new URL("../static/" + file, import.meta.url), "utf8")) +
        (file === "app.js"
          ? "\nwindow.seedPresets = rows => {experimentRows = rows; renderExperiments();};"
          : ""),
    );
  el("email-workspace-content").hidden = false;
  const form = el("experiment-form"),
    home = el("experiment-composer").parentElement;
  const dialog = el("experiment-preset-dialog"),
    launch = el("new-experiment-preset");
  const search = el("experiment-search");
  const requests = [];
  w.api = async (path, options = {}) => {
    requests.push({ path, ...options });
    return {};
  };
  launch.click();
  assert.equal(dialog.open, true);
  assert.equal(
    dialog.querySelector("[data-dialog-body] #experiment-form"),
    form,
    "the original form is reused",
  );
  assert.equal(w.document.querySelectorAll("#experiment-form").length, 1);
  assert.equal(
    dialog.querySelectorAll(".dialog-heading").length,
    1,
    "common modal shell",
  );
  assert.equal(el("experiment-name").value, "", "New requires a new name");
  assert.equal(el("submit-experiment-button").hidden, true);
  assert.equal(el("save-experiment-button").textContent, "Create preset");
  assert.equal(el("save-experiment-button").type, "submit");
  el("experiment-name").value = "unsaved-preset";
  w.SkynetDialog.close(dialog);
  assert.equal(requests.length, 0, "Close does not create or submit anything");
  assert.equal(el("experiment-composer").parentElement, home);
  assert.equal(
    el("experiment-name").value,
    "unsaved-preset",
    "closing preserves the shared editor inputs",
  );
  assert.equal(el("save-experiment-button").type, "button");
  assert.equal(el("submit-experiment-button").hidden, false);
  assert.equal(w.document.activeElement, launch);

  // Runtime/adapter validation is exercised by training_contracts_ui. Here keep
  // real creation, modal lifecycle and request routing; stub only prerequisites.
  w.validateExperiment = () => Boolean(el("experiment-name").value);
  w.updateBatchCompatibility = () => ({ valid: true, errors: [] });
  w.experimentPreviewIsCurrent = () => false;
  w.experimentPayload = () => ({
    name: el("experiment-name").value,
    source: {
      repository: "https://example.com/repo",
      project_subdirectory: ".",
    },
  });
  w.matchingExperimentForPayload = () => null;
  w.loadExperiments = async () => {
    w.seedPresets([
      { id: "new-preset", name: "fresh-preset", dataset_ids: [] },
    ]);
  };
  launch.click();
  form.dispatchEvent(
    new w.Event("submit", { bubbles: true, cancelable: true }),
  );
  await flush();
  assert.equal(requests.length, 0, "validation still gates preset creation");
  el("experiment-name").value = "fresh-preset";
  search.value = "old dataset";
  search.dataset.datasetId = "old";
  search.dataset.presetIds = '["old"]';
  let finishSave;
  w.api = (path, options) => {
    requests.push({ path, ...options });
    return new Promise((resolve) => {
      finishSave = resolve;
    });
  };
  form.dispatchEvent(
    new w.Event("submit", { bubbles: true, cancelable: true }),
  );
  await flush();
  assert.equal(requests.length, 1);
  assert.equal(
    requests[0].path,
    "/api/experiments",
    "Enter creates a draft without Slurm submission",
  );
  assert.equal(JSON.parse(requests[0].body).name, "fresh-preset");
  assert.equal(el("save-experiment-button").disabled, true);
  assert.equal(dialog.querySelector("[data-dialog-close]").disabled, true);
  form.dispatchEvent(
    new w.Event("submit", { bubbles: true, cancelable: true }),
  );
  w.SkynetDialog.close(dialog);
  assert.equal(
    requests.length,
    1,
    "repeat Enter cannot create duplicate requests",
  );
  assert.equal(dialog.open, true, "saving cannot dismiss the dialog");
  finishSave({ experiment: { id: "new-preset" } });
  await flush();
  assert.equal(dialog.open, false);
  assert.equal(el("experiment-composer").parentElement, home);
  assert.equal(search.value, "");
  assert.equal(search.dataset.datasetId, undefined);
  assert.match(el("experiments-body").textContent, /fresh-preset/);
  assert.equal(
    requests.some(
      (r) => r.path.includes("/submit") || r.path.includes("/revisions"),
    ),
    false,
  );

  launch.click();
  el("experiment-name").value = "duplicate";
  w.matchingExperimentForPayload = () => ({
    id: "existing",
    latest_revision_number: 4,
  });
  await w.createExperiment(true);
  assert.equal(
    requests.length,
    1,
    "New cannot silently create a revision of an existing preset",
  );
  assert.match(
    dialog.querySelector(".dialog-notice").textContent,
    /already exists/,
  );
  w.matchingExperimentForPayload = () => null;
  w.api = async () => {
    throw Error("Save rejected");
  };
  await w.createExperiment(false);
  assert.equal(dialog.open, true, "failure keeps editable inputs in the modal");
  assert.equal(el("experiment-name").value, "duplicate");
  assert.equal(el("save-experiment-button").disabled, false);
  assert.match(
    dialog.querySelector(".dialog-notice").textContent,
    /Save rejected/,
  );
  w.document.dispatchEvent(
    new w.KeyboardEvent("keydown", {
      key: "Escape",
      bubbles: true,
      cancelable: true,
    }),
  );
  assert.equal(dialog.open, false);
  assert.equal(el("experiment-composer").parentElement, home);
  assert.equal(w.document.querySelectorAll("#experiment-form").length, 1);
  console.log(
    "Preset creation: shared editor, validation, save-only Enter, duplicate prevention, failure recovery, filtering and close passed.",
  );
} finally {
  for (const observer of observers) observer.disconnect();
  w.close();
}
