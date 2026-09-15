import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { JSDOM } from "jsdom";

const w = new JSDOM(
  await readFile(new URL("../static/index.html", import.meta.url), "utf8"),
  {
    runScripts: "outside-only",
    pretendToBeVisual: true,
    url: "http://localhost:8080/?data_view=registry#data",
  },
).window;
const observers = [],
  Observer = w.MutationObserver;
w.MutationObserver = class extends Observer {
  constructor(callback) {
    super(callback);
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
  if (this.open) {
    this.open = false;
    this.dispatchEvent(new w.Event("close"));
  }
};
const el = (id) => w.document.getElementById(id);
const flush = async () => {
  for (let i = 0; i < 5; i++)
    await new Promise((resolve) => setImmediate(resolve));
};

try {
  for (const file of [
    "dialogs.js",
    "workspace-navigation.js",
    "connection-settings.js",
    "app.js",
  ])
    w.eval(
      await readFile(new URL("../static/" + file, import.meta.url), "utf8"),
    );
  const dataset = {
    id: "dataset-id",
    category: "dataset",
    kind: "demonstrations",
    provider: "collection",
    namespace: "datasets",
    source_key: "recording-uuid",
    display_name: "Shadow cube demonstrations",
    description: "Live DexVerse · Shadow · right hand · Pick up cube",
    metadata: { session_id: "recording-uuid" },
    versions: [],
    updated_at: "2026-09-14T12:00:00Z",
  };
  const latest = {
    ...dataset,
    id: "other-dataset",
    source_key: "other-recording",
    metadata: { session_id: "other-recording" },
    display_name: "Latest dataset",
    updated_at: "2026-09-15T12:00:00Z",
  };
  const resources = [dataset, latest];
  const requests = [];
  w.api = async (path, request = {}) => {
    requests.push([path, request]);
    if (path.startsWith("/api/data/resources?"))
      return {
        resources: structuredClone(resources),
        resource_types: {
          dataset: { demonstrations: "Demonstrations" },
          file: { model: "Model" },
        },
      };
    if (path === "/api/data/resources/dataset-id") {
      if (request.method === "PATCH") {
        Object.assign(dataset, JSON.parse(request.body), {
          updated_at: "2026-09-16T12:00:00Z",
        });
      }
      return { resource: structuredClone(dataset) };
    }
    return { items: [] };
  };
  await w.activateTab("data", true, "registry");
  await w.loadDataRegistry(true);
  const rows = () => [
    ...el("data-resources-body").querySelectorAll("[data-resource-id]"),
  ];
  const row = () =>
    el("data-resources-body").querySelector('[data-resource-id="dataset-id"]');
  assert.deepEqual(
    rows().map((item) => item.dataset.resourceId),
    ["other-dataset", "dataset-id"],
    "Datasets start in Updated descending order",
  );
  assert.equal(
    row().querySelector(".node-name").textContent,
    dataset.display_name,
  );
  assert.equal(
    w.dataResourceIdentity(dataset),
    "collection:datasets/recording-uuid",
  );

  el("data-resource-search").value = "recording-uuid";
  el("data-resource-search").dispatchEvent(new w.Event("input"));
  assert.deepEqual(
    rows().map((item) => item.dataset.resourceId),
    ["dataset-id"],
    "Source keys remain searchable",
  );
  el("data-resource-search").value = "";
  el("data-resource-search").dispatchEvent(new w.Event("input"));

  row().querySelector('[data-resource-action="edit"]').click();
  await flush();
  assert.equal(el("data-resource-form-dialog").open, true);
  assert.equal(
    el("data-resource-name").value,
    row().querySelector(".node-name").textContent,
    "Edit and table use the same Name",
  );
  assert.equal(el("data-resource-name").disabled, false);
  assert.equal(
    el("data-resource-description").value,
    dataset.description,
    "Description remains distinct from Name",
  );
  assert.equal(el("data-resource-source-key").value, dataset.source_key);
  assert.equal(el("data-resource-source-key").readOnly, true);
  assert.equal(el("data-resource-visible-id").value, dataset.id);
  assert.equal(el("data-resource-visible-id").readOnly, true);
  assert.equal(el("data-resource-source-details").open, false);

  el("data-resource-name").value = "   ";
  const beforeEmpty = requests.length;
  await w.updateDataResource(dataset.id);
  assert.equal(
    requests.length,
    beforeEmpty,
    "Blank names are rejected before sending an update",
  );
  assert.equal(el("data-resource-form-dialog").open, true);
  el("data-resource-name").value = "  Renamed Shadow dataset  ";
  el("data-resource-description").value = "  A separate description  ";
  await w.updateDataResource(dataset.id);
  const patch = requests.find(([, request]) => request.method === "PATCH");
  assert.deepEqual(
    JSON.parse(patch[1].body),
    {
      display_name: "Renamed Shadow dataset",
      description: "A separate description",
    },
    "Rename sends no identity or metadata changes",
  );
  assert.equal(dataset.source_key, "recording-uuid");
  assert.equal(dataset.id, "dataset-id");
  assert.equal(dataset.metadata.session_id, "recording-uuid");
  assert.equal(
    row().querySelector(".node-name").textContent,
    "Renamed Shadow dataset",
    "Save refreshes the table immediately",
  );
  assert.equal(
    rows()[0].dataset.resourceId,
    dataset.id,
    "The renamed dataset moves to its new Updated position",
  );
  assert.equal(el("data-resource-form-dialog").open, false);
  await w.openDataResourceEditor(dataset.id);
  assert.equal(
    el("data-resource-name").value,
    "Renamed Shadow dataset",
    "Reopening reads the saved Name",
  );
  el("close-data-resource-form").click();

  el("show-data-resource-form").click();
  assert.equal(el("data-resource-source-details").open, true);
  assert.equal(el("data-resource-identity-field").hidden, true);
  assert.equal(el("data-resource-source-key").disabled, false);
  assert.equal(el("data-resource-source-key").readOnly, false);
  assert.equal(el("data-resource-name").required, false);
  el("data-resource-provider").value = "local";
  el("data-resource-namespace").value = "datasets";
  el("data-resource-source-key").value = "  external-source-key  ";
  let created;
  w.api = async (_path, request = {}) => {
    created = JSON.parse(request.body);
    throw new Error("Stop after capturing registration");
  };
  await w.createDataResource({ preventDefault() {} });
  assert.equal(created.source_key, "external-source-key");
  assert.equal(
    "display_name" in created,
    false,
    "A blank optional Name is omitted so the server chooses its documented default",
  );
  assert.equal(
    "name" in created,
    false,
    "Current resource requests do not use the old ambiguous field",
  );
  el("data-resource-name").value = "  External demo  ";
  await w.createDataResource({ preventDefault() {} });
  assert.equal(created.display_name, "External demo");

  // Old tutorial receipts must still verify the exact immutable source key.
  w.eval(`window.testLegacyResourceReceipt = () => {
    const saved = newTutorialSession("datasets");
    saved.ownedRecords = [{ kind: "data-resource", expectedIdentity: [{ paths: ["name"], value: "recording-uuid" }] }];
    localStorage.setItem(tutorialProgressKey("datasets"), JSON.stringify(saved));
    return loadTutorialSession("datasets").ownedRecords[0].expectedIdentity;
  };`);
  const rules = w.testLegacyResourceReceipt();
  assert.equal(rules[0].paths[0], "source_key");
  assert.equal(w.tutorialIdentityMatches(dataset, rules), true);
  assert.equal(
    w.tutorialIdentityMatches({ ...dataset, source_key: "different" }, rules),
    false,
  );
  console.log(
    "Resource names: table/edit consistency, immutable source identity, trimmed updates, immediate refresh/sort, creation defaults and tutorial receipt compatibility passed.",
  );
} finally {
  for (const observer of observers) observer.disconnect();
  await flush();
  w.close();
}
