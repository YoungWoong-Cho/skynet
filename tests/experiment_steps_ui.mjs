import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { pageWindow, byId, spyMutationObservers, stubBrowserApis, polyfillDialogs } from "./ui_harness.mjs";
const w = pageWindow({ url: "http://localhost:8080/#experiments" });
const observers = spyMutationObservers(w);
stubBrowserApis(w);
polyfillDialogs(w, { guarded: false, returnValue: false });
const el = byId(w);
const step = (name) =>
  w.document.querySelector(`[data-experiment-step="${name}"]`);
const plain = (x) => JSON.parse(JSON.stringify(x));
const tick = () => new Promise((resolve) => setTimeout(resolve, 220));
const change = (id, value, type = "input") => {
  el(id).value = value;
  el(id).dispatchEvent(new w.Event(type, { bubbles: true }));
};
const manifest = {
  schema_version: "skynet.adapter/v1",
  slug: "many-policy",
  display_name: "Many policy",
  repository: { url: "https://example.com/repo", project_subdirectory: "." },
  runtime: { allowed_backends: ["existing", "conda"] },
  capabilities: {
    minimum_gpus: 1,
    recommended_gpus: 2,
    maximum_gpus: 8,
    supports_resume: true,
  },
  defaults: {
    resources: { gpu_mode: "explicit", gpu_count: 2 },
    tracking: { enabled: false },
    evaluation: { enabled: false },
  },
  train: {
    supported_canonical_fields: [
      "train.learning_rate",
      "train.batch.value",
      "train.batch.declared_semantics",
      "train.batch.gradient_accumulation_steps",
    ],
    batch_compatibility: {
      schema_version: "skynet.batch-compatibility/v1",
      allowed_semantics: ["per_device", "global_effective"],
      multi_gpu_allowed_semantics: ["global_effective"],
      batch_size_divisible_by: "resolved_gpu_count",
      supports_gradient_accumulation: true,
    },
    input_fields: [
      {
        path: "native.config.mode",
        label: "Observation mode",
        kind: "string",
        choices: ["state", "rgb"],
        default: "state",
      },
      {
        path: "native.config.datasets",
        label: "Datasets",
        kind: "json",
        required: true,
        data_binding: {
          role: "training_data",
          cardinality: "many",
          value_path: "selection",
          formats: ["test/v1"],
          contracts: ["state/v1", "rgb/v1"],
          contract_selector: "native.config.mode",
          contract_choices: { state: ["state/v1"], rgb: ["rgb/v1"] },
        },
      },
    ],
  },
};
const dataset = (id, contract = "state/v1") => ({
  id,
  name: id,
  format: "test/v1",
  selection: { version_id: id, location_id: id + "-location" },
  assignments: [
    {
      role: "training_data",
      position: 0,
      resource: { name: id },
      version: {
        status: "READY",
        path: "/data/" + id,
        format: "test/v1",
        manifest_sha256: id.repeat(64).slice(0, 64),
        metadata: { registered_version_id: id, contract, validation: { status: "PASSED" } },
      },
      config: {
        location: {
          kind: "cluster",
          status: "AVAILABLE",
          path: "/data/" + id,
          manifest_sha256: id.repeat(64).slice(0, 64),
        },
      },
    },
  ],
});
try {
  for (const file of [
    "dialogs.js",
    "workspace-navigation.js",
    "connection-settings.js",
    "app.js",
  ]) {
    let source = await readFile(
      new URL("../static/" + file, import.meta.url),
      "utf8",
    );
    if (file === "app.js")
      source += `\nwindow.seedSteps=(manifest,datasets)=>{
   adapterRows=[{id:'adapter',name:'Policy',latest_version:{id:'adapter-v1',version_number:1,manifest}}];
   trainingDatasetRows=datasets; adaptersLoaded=dataBundlesLoaded=true;dataBundlesLoadPromise=null;
   populateExperimentAdapters();applySelectedAdapter({loadSource:false});
   elements.wandbEnabled.checked=elements.mlflowEnabled.checked=elements.evaluationEnabled.checked=false;
   updateExperimentFields();
  };
  window.installTestCode=()=>installPinnedSourceRevision({repository:'https://example.com/repo',revision:'a'.repeat(40),project_subdirectory:'.'});
  window.installTestRuntime=()=>{
    runtimeCandidates=[{value:'existing',label:'Existing',runnable:true,manual:true,requiresProfile:false,evidence:[]},{value:'conda',label:'Conda',runnable:true,manual:true,requiresProfile:true,evidence:[]}];
    runtimeProfiles=[];elements.experimentRuntime.innerHTML='<option value="existing">Existing</option><option value="conda">Conda</option>';elements.experimentRuntime.disabled=false;
    runtimeInspectionKey=repositoryInspectionKey(elements.experimentSource.value.trim(),elements.experimentRevision.value.trim(),elements.experimentWorkdir.value.trim()||'.',selectedAdapter());
    applyRuntimeSelection(false);
  };
  window.dataIds=()=>selectedExperimentDataBundle()?.selections.map(s=>s.version_id)||[];
  window.dataBinding=()=>resolveAdapterDataBinding(declaredAdapterInputFields().find(f=>f.data_binding));
  window.chooseDatasets=ids=>{setExperimentDatasetSelection(elements.experimentDataBundle,ids);elements.experimentDataBundle.dispatchEvent(new Event('change',{bubbles:true}));};
  window.dropDataset=id=>{trainingDatasetRows=trainingDatasetRows.filter(d=>d.id!==id);populateExperimentDataBundles();updateExperimentSubmitState();};
  window.swapManifest=manifest=>{loadedExperimentAdapterSnapshot=null;adapterRows[0].latest_version.manifest=manifest;populateExperimentAdapters();applySelectedAdapter({loadSource:false,preserveEdits:true});};
  window.previewCurrent=()=>experimentPreviewIsCurrent();
  window.hydrateTest=spec=>hydrateExperimentConfiguration(spec,experimentConfigurationLoadRequest);
  window.stopBackground=()=>{sourceMetadataResponses.clear();sourceMetadataRequests.clear();};
  `;
    w.eval(source);
  }
  assert.deepEqual(
    [...w.document.querySelectorAll("[data-experiment-step]")].map(
      (s) => s.dataset.experimentStep,
    ),
    [
      "code",
      "data",
      "runtime",
      "settings",
      "resources",
      "checkpoint",
      "tracking",
    ],
  );
  const largeDataset = dataset("a");
  const metadataMarker = "DATASET_METADATA_STAYS_OUT_OF_FORM";
  largeDataset.assignments[0].version.metadata.large_fixture_metadata = metadataMarker + "x".repeat(1024 * 1024);
  w.seedSteps(manifest, [largeDataset, dataset("b"), dataset("c", "rgb/v1")]);
  assert.equal(step("code").disabled, false);
  assert.equal(step("data").disabled, true);
  assert.equal(step("runtime").disabled, true);
  assert.equal(el("experiment-data-bundle").multiple, true);
  w.installTestCode();
  change("experiment-name", "test-training-run");
  w.updateExperimentSubmitState();
  assert.equal(step("data").disabled, false);
  assert.equal(
    step("runtime").disabled,
    true,
    "empty required dataset gates runtime",
  );
  w.chooseDatasets(["b", "a"]);
  assert.deepEqual(plain(w.dataIds()), ["b", "a"]);
  assert.deepEqual(
    plain(w.experimentPayload().data_selections).map((s) => [
      s.version_id,
      s.position,
    ]),
    [
      ["b", 0],
      ["a", 1],
    ],
  );
  assert.deepEqual(
    plain(w.dataBinding().value).map((s) => [s.version_id, s.position, s.path]),
    [
      ["b", 0, "/data/b"],
      ["a", 1, "/data/a"],
    ],
  );
  const derived = el("adapter-field-native-config-datasets");
  assert.equal(derived.type, "hidden", "the selected dataset binding has no duplicate editor");
  assert.equal(derived.value, "", "the validator control never stores metadata JSON");
  assert.ok(el("adapter-declared-fields-grid").innerHTML.length < 12000);
  assert.equal(el("adapter-declared-fields-grid").innerHTML.includes(metadataMarker), false);
  assert.equal(w.validateAdapterDeclaredFields({focus: false, notify: false}), true);
  assert.equal(JSON.stringify(w.experimentPayload()).includes(metadataMarker), false, "experiment preview sends dataset identities, not metadata copies");
  const apiBeforeSummary = w.api;
  let summaryBody;
  w.api = async (path, options) => {
    if (path === "/api/model-io/preview") summaryBody = options.body;
    return {entries: []};
  };
  w.refreshExperimentModelIO();
  await tick();
  assert.ok(summaryBody);
  assert.equal(summaryBody.includes(metadataMarker), false);
  assert.equal(JSON.parse(summaryBody).values["native.config.datasets"], undefined);
  assert.deepEqual(JSON.parse(summaryBody).data_selections.map(item => item.version_id), ["b", "a"]);
  w.api = apiBeforeSummary;
  w.populateExperimentDataBundles();
  assert.deepEqual(
    plain(w.dataIds()),
    ["b", "a"],
    "catalog render retains selection order",
  );
  assert.equal(step("runtime").disabled, false);
  assert.equal(step("settings").disabled, true);
  w.installTestRuntime();
  assert.equal(step("settings").disabled, false);
  assert.equal(w.validateExperiment({ notify: false }), true);
  change("hp-batch-size", "3");
  change("hp-batch-semantics", "global_effective", "change");
  assert.equal(
    step("resources").disabled,
    false,
    "GPU-dependent batch errors must leave Resources reachable",
  );
  assert.equal(step("checkpoint").disabled, true);
  change("experiment-gpu-count", "3");
  assert.equal(step("checkpoint").disabled, false);
  change("hp-learning-rate", "-0.1");
  assert.equal(
    step("resources").disabled,
    true,
    "invalid training settings gate resources",
  );
  change("hp-learning-rate", "0.002");
  assert.equal(step("resources").disabled, false);
  const mode = el("adapter-field-native-config-mode");
  mode.value = JSON.stringify("rgb");
  mode.dispatchEvent(new w.Event("change", { bubbles: true }));
  assert.equal(step("data").disabled, false);
  assert.equal(
    step("runtime").disabled,
    false,
    "data step accepts union of observation contracts",
  );
  assert.equal(
    step("settings").disabled,
    false,
    "mode selector remains reachable to repair contract mismatch",
  );
  assert.equal(step("resources").disabled, true);
  change("adapter-field-native-config-mode", JSON.stringify("state"), "change");
  assert.equal(step("resources").disabled, false);
  change("experiment-workdir", "changed");
  assert.equal(step("settings").disabled, true);
  assert.equal(el("hp-learning-rate").value, "0.002");
  assert.equal(el("experiment-gpu-count").value, "3");
  change("experiment-workdir", ".");
  assert.equal(
    step("settings").disabled,
    false,
    "returning to unchanged verified code restores validity",
  );
  change("checkpoint-mode", "weights", "change");
  assert.equal(step("tracking").disabled, true);
  change("checkpoint-path", "/models/start.ckpt");
  assert.equal(step("tracking").disabled, false);
  change("sweep-definition", "{");
  assert.equal(step("tracking").disabled, true);
  change("sweep-definition", "");
  assert.equal(step("tracking").disabled, false);
  // Restoring a preset keeps every exact dataset identity in its frozen order.
  const inspectOriginal = w.inspectRepositoryRuntime;
  w.inspectRepositoryRuntime = async () => w.installTestRuntime();
  const presetSpec = {
    name: "restored",
    source: {
      repository: "https://example.com/repo",
      revision: "a".repeat(40),
      project_subdirectory: ".",
      adapter: "many-policy",
      adapter_id: "adapter",
      adapter_version: 1,
      adapter_version_id: "adapter-v1",
      adapter_manifest: manifest,
    },
    runtime: { backend: "existing" },
    train: {
      learning_rate: 0.002,
      batch: { value: 3, declared_semantics: "global_effective" },
      checkpoint: { max_attempts: 3, save_before_timeout_seconds: 300 },
    },
    resources: {
      gateway: "sky2",
      queue_policy: "auto",
      gpu: { mode: "explicit", count: 3, gpu_type: "any" },
      nodes: 1,
      cpus_per_task: 12,
      memory_gb: 64,
      time_limit: "04:00:00",
    },
    native: {
      config: {
        mode: "state",
        initial_checkpoint: "/models/start.ckpt",
        initial_checkpoint_mode: "weights",
      },
    },
    tracking: { providers: [] },
    evaluation: [],
    data: {
      bundle: {
        assignments: [
          { ...dataset("b").assignments[0], position: 0 },
          { ...dataset("a").assignments[0], position: 1 },
        ],
      },
    },
  };
  assert.equal(await w.hydrateTest(presetSpec), true);
  assert.deepEqual(plain(w.dataIds()), ["b", "a"]);
  assert.equal(
    w
      .experimentPayload()
      .native_overrides.some((value) => value.startsWith("config.datasets=")),
    false,
    "bound snapshot is generated by the server",
  );
  w.inspectRepositoryRuntime = inspectOriginal;
  // Even an edit followed by an undo invalidates an already pending preview.
  let finishPreview;
  w.api = async (path) =>
    path === "/api/experiments/preview"
      ? await new Promise((resolve) => {
          finishPreview = resolve;
        })
      : { entries: [] };
  const preview = w.previewExperiment();
  assert.equal(typeof finishPreview, "function");
  change("hp-learning-rate", "0.003");
  change("hp-learning-rate", "0.002");
  finishPreview({
    scripts: ["#!/bin/bash\ntrue"],
    variant_count: 1,
    blockers: [],
    warnings: [],
    resolved_revision: "a".repeat(40),
  });
  await preview;
  assert.equal(
    w.previewCurrent(),
    false,
    "an obsolete response cannot re-enable submission after edit/undo",
  );
  assert.equal(el("experiment-preview-panel").hidden, true);
  w.dropDataset("b");
  assert.equal(
    el("experiment-data-bundle").selectedOptions[0].value,
    "b",
    "missing identity remains visible",
  );
  assert.equal(
    step("runtime").disabled,
    true,
    "missing dataset blocks later steps",
  );
  assert.equal(el("hp-learning-rate").value, "0.002");
  w.chooseDatasets(["a"]);
  assert.equal(step("runtime").disabled, false);
  const noData = plain(manifest);
  noData.train.input_fields = [];
  w.swapManifest(noData);
  assert.equal(step("data").disabled, false);
  assert.equal(
    step("runtime").disabled,
    false,
    "adapters without data bindings skip the data requirement",
  );
  assert.deepEqual(plain(w.dataIds()), []);
  assert.equal(
    el("hp-learning-rate").value,
    "0.002",
    "adapter change retains typed common values",
  );
  assert.equal(el("checkpoint-path").value, "/models/start.ckpt");
  // A runtime change must not discard an already chosen repository commit.
  const branchesOriginal = w.loadSourceBranches;
  let branchLoads = 0, inspections = 0;
  w.loadSourceBranches = async () => { branchLoads++; };
  w.inspectRepositoryRuntime = async () => { inspections++; };
  el("experiment-runtime").value = "";
  w.applySelectedAdapter({preserveEdits: true});
  assert.equal(branchLoads, 0);
  assert.equal(inspections, 1);
  assert.equal(el("experiment-revision").value, "a".repeat(40));
  w.loadSourceBranches = branchesOriginal;
  w.inspectRepositoryRuntime = inspectOriginal;
  // A manual runtime path survives invalidation and later inspection.
  w.installTestRuntime();
  change("experiment-runtime", "conda", "change");
  change("experiment-runtime-profile", "/env/custom");
  w.resetRuntimeInspection();
  assert.equal(el("experiment-runtime-profile").value, "/env/custom");
  // Runtime replies after an uncommitted workdir edit must not unlock later steps.
  w.stopBackground();
  let finishInspect;
  w.sourceMetadataRequest = () =>
    new Promise((resolve) => {
      finishInspect = resolve;
    });
  const inspection = w.inspectRepositoryRuntime(true);
  change("experiment-workdir", "new-dir");
  finishInspect({
    payload: {
      commit: "a".repeat(40),
      runtime_candidates: [
        { backend: "existing", runnable: true, confidence: "strong" },
      ],
    },
    fromCache: false,
  });
  await inspection;
  assert.equal(step("settings").disabled, true);
  assert.equal(el("experiment-workdir").value, "new-dir");
  assert.equal(el("hp-learning-rate").value, "0.002");
  assert.equal(el("experiment-runtime-profile").value, "/env/custom");
  w.sourceMetadataRequest = async () => ({
    payload: {
      commit: "a".repeat(40),
      runtime_candidates: [
        { backend: "conda", runnable: true, confidence: "manual" },
      ],
    },
    fromCache: false,
  });
  await w.inspectRepositoryRuntime(true);
  assert.equal(el("experiment-runtime").value, "conda");
  assert.equal(el("experiment-runtime-profile").value, "/env/custom");
  assert.equal(step("settings").disabled, false);
  // Submit and evaluation reuse one simultaneous dataset catalog request.
  let finishCatalog,
    catalogCalls = 0,
    evaluationUpdates = 0;
  w.populateEvaluationTargetDatasets = () => {
    evaluationUpdates++;
  };
  w.api = (path) =>
    path === "/api/data/selections"
      ? (catalogCalls++,
        new Promise((resolve) => {
          finishCatalog = resolve;
        }))
      : Promise.resolve({ entries: [] });
  const firstCatalog = w.loadDataBundles(true),
    secondCatalog = w.loadDataBundles(true);
  assert.equal(catalogCalls, 1);
  finishCatalog({ datasets: [dataset("a"), dataset("b")] });
  await Promise.all([firstCatalog, secondCatalog]);
  assert.equal(evaluationUpdates, 1);
  // A late use-prepared response must not replace a newer typed edit.
  let finishAdapter;
  w.activateTab = async () => {};
  w.loadTrainingInputs = async () => {};
  w.api = (path) =>
    path.startsWith("/api/adapters/")
      ? new Promise((resolve) => {
          finishAdapter = resolve;
        })
      : Promise.resolve({ entries: [] });
  const usePrepared = w.usePreparedDataset({
    name: "Old prepared",
    version_id: "a",
    adapter: {
      adapter_id: "adapter",
      adapter_version_id: "adapter-v1",
      adapter_version_number: 1,
      adapter_manifest_sha256: "f".repeat(64),
    },
    training_setup: { adapter: "many-policy" },
  });
  await tick();
  change("experiment-name", "keep-my-edit");
  finishAdapter({
    adapter: {
      id: "adapter",
      selected_version: {
        id: "adapter-v1",
        version_number: 1,
        manifest,
        manifest_sha256: "f".repeat(64),
      },
    },
  });
  await usePrepared;
  assert.equal(el("experiment-name").value, "keep-my-edit");
  change("gpu-mode", "manual", "change");
  change("experiment-gpu-count", "3");
  w.updateCpuResources();
  assert.equal(el("resource-cpus").value, "24");
  assert.equal(el("resource-cpus").readOnly, true);
  assert.equal(el("evaluation-resource-cpus").value, "8");
  assert.equal(el("evaluation-resource-cpus").readOnly, true);
  change("collection-gpu-count", "2");
  w.updateCpuResources();
  assert.equal(el("collection-cpu-count").value, "16");
  assert.equal(el("collection-cpu-count").readOnly, true);
  console.log(
    "Experiment steps: ordered gating, contract/GPU dependency cycles, many dataset order, composite bindings, optional datasets, stale inspection and preserved typed values passed.",
  );
} finally {
  for (const observer of observers) observer.disconnect();
  w.close();
}
