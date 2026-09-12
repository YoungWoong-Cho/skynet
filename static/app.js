const workspaceStorageKey = (key) =>
  window.SkynetWorkspace?.storageKey(key) || key;
const SSH_GATEWAY_PREFERENCE_KEY = workspaceStorageKey("skynet:ssh-gateway");

const elements = {
  gateway: document.querySelector("#gateway"),
  connectionDot: document.querySelector("#connection-dot"),
  statusLabel: document.querySelector("#status-label"),
  lastUpdated: document.querySelector("#last-updated"),
  errorBanner: document.querySelector("#error-banner"),
  refreshButton: document.querySelector("#refresh-button"),
  jobsBody: document.querySelector("#jobs-body"),
  usageBody: document.querySelector("#usage-body"),
  jobSummary: document.querySelector("#job-summary"),
  jobsPartitionFilterLabel: document.querySelector(
    "#jobs-partition-filter-label",
  ),
  jobsPartitionFilterOptions: document.querySelector(
    "#jobs-partition-filter-options",
  ),
  jobsAccountFilterLabel: document.querySelector("#jobs-account-filter-label"),
  jobsAccountFilterOptions: document.querySelector(
    "#jobs-account-filter-options",
  ),
  showDataResourceForm: document.querySelector("#show-data-resource-form"),
  showDataDerivationForm: document.querySelector("#show-data-derivation-form"),
  toast: document.querySelector("#toast"),
  tutorialLayer: document.querySelector("#tutorial-layer"),
  tutorialSpotlight: document.querySelector("#tutorial-spotlight"),
  tutorialDialog: document.querySelector("#tutorial-dialog"),
  tutorialPage: document.querySelector("#tutorial-page"),
  tutorialProgress: document.querySelector("#tutorial-progress"),
  tutorialTitle: document.querySelector("#tutorial-title"),
  tutorialInstruction: document.querySelector("#tutorial-instruction"),
  tutorialExit: document.querySelector("#tutorial-exit"),
  tutorialBack: document.querySelector("#tutorial-back"),
  tutorialNext: document.querySelector("#tutorial-next"),
  experimentsError: document.querySelector("#experiments-error"),
  refreshExperiments: document.querySelector("#refresh-experiments"),
  experimentForm: document.querySelector("#experiment-form"),
  newExperimentPreset: document.querySelector("#new-experiment-preset"),
  experimentPresetDialog: document.querySelector("#experiment-preset-dialog"),
  experimentName: document.querySelector("#experiment-name"),
  experimentAdapter: document.querySelector("#experiment-adapter"),
  experimentSource: document.querySelector("#experiment-source"),
  experimentBranch: document.querySelector("#experiment-branch"),
  experimentRevision: document.querySelector("#experiment-revision"),
  sourceRefRefresh: document.querySelector("#source-ref-refresh"),
  sourceRefStatus: document.querySelector("#source-ref-status"),
  sourceCommitMeta: document.querySelector("#source-commit-meta"),
  experimentWorkdir: document.querySelector("#experiment-workdir"),
  experimentRuntime: document.querySelector("#experiment-runtime"),
  experimentRuntimeProfileSelect: document.querySelector(
    "#experiment-runtime-profile-select",
  ),
  experimentRuntimeProfile: document.querySelector(
    "#experiment-runtime-profile",
  ),
  runtimeProfileHelp: document.querySelector("#runtime-profile-help"),
  hpLearningRate: document.querySelector("#hp-learning-rate"),
  hpLearningRateDefault: document.querySelector("#hp-learning-rate-default"),
  hpBatchSize: document.querySelector("#hp-batch-size"),
  hpBatchSizeDefault: document.querySelector("#hp-batch-size-default"),
  hpBatchSemantics: document.querySelector("#hp-batch-semantics"),
  hpBatchSemanticsDefault: document.querySelector(
    "#hp-batch-semantics-default",
  ),
  hpBatchCompatibility: document.querySelector("#hp-batch-compatibility"),
  hpGradAcc: document.querySelector("#hp-grad-acc"),
  hpGradAccDefault: document.querySelector("#hp-grad-acc-default"),
  hpNumWorkers: document.querySelector("#hp-num-workers"),
  hpNumWorkersDefault: document.querySelector("#hp-num-workers-default"),
  hpPrecision: document.querySelector("#hp-precision"),
  hpPrecisionDefault: document.querySelector("#hp-precision-default"),
  hpPrecisionStatus: document.querySelector("#hp-precision-status"),
  hpMaxSteps: document.querySelector("#hp-max-steps"),
  hpMaxStepsDefault: document.querySelector("#hp-max-steps-default"),
  nativeOverrides: document.querySelector("#native-overrides"),
  adapterDeclaredFields: document.querySelector("#adapter-declared-fields"),
  adapterDeclaredFieldsGrid: document.querySelector(
    "#adapter-declared-fields-grid",
  ),
  adapterDeclaredFieldsStatus: document.querySelector(
    "#adapter-declared-fields-status",
  ),
  adapterDeclaredFieldsError: document.querySelector(
    "#adapter-declared-fields-error",
  ),
  resourcePolicy: document.querySelector("#resource-policy"),
  gpuMode: document.querySelector("#gpu-mode"),
  experimentGpuCount: document.querySelector("#experiment-gpu-count"),
  experimentGpuType: document.querySelector("#experiment-gpu-type"),
  resourceNodes: document.querySelector("#resource-nodes"),
  resourceCpus: document.querySelector("#resource-cpus"),
  resourceMemory: document.querySelector("#resource-memory"),
  resourceTime: document.querySelector("#resource-time"),
  checkpointMode: document.querySelector("#checkpoint-mode"),
  checkpointPathField: document.querySelector("#checkpoint-path-field"),
  checkpointPath: document.querySelector("#checkpoint-path"),
  checkpointSaveSteps: document.querySelector("#checkpoint-save-steps"),
  checkpointSaveStepsDefault: document.querySelector(
    "#checkpoint-save-steps-default",
  ),
  checkpointMaxAttempts: document.querySelector("#checkpoint-max-attempts"),
  checkpointAutoResume: document.querySelector("#checkpoint-auto-resume"),
  sweepDefinition: document.querySelector("#sweep-definition"),
  wandbEnabled: document.querySelector("#wandb-enabled"),
  wandbProject: document.querySelector("#wandb-project"),
  wandbRunName: document.querySelector("#wandb-run-name"),
  wandbExperimentConnection: document.querySelector(
    "#wandb-experiment-connection",
  ),
  mlflowEnabled: document.querySelector("#mlflow-enabled"),
  mlflowUri: document.querySelector("#mlflow-uri"),
  mlflowExperiment: document.querySelector("#mlflow-experiment"),
  mlflowRunName: document.querySelector("#mlflow-run-name"),
  mlflowExperimentConnection: document.querySelector(
    "#mlflow-experiment-connection",
  ),
  trackingNamePreview: document.querySelector("#tracking-name-preview"),
  evaluationEnabled: document.querySelector("#evaluation-enabled"),
  experimentEvaluationSuites: document.querySelector(
    "#experiment-evaluation-suites",
  ),
  evaluationSuiteStatus: document.querySelector("#evaluation-suite-status"),
  adapterStatus: document.querySelector("#adapter-status"),
  adapterCapabilities: document.querySelector("#adapter-capabilities"),
  runtimeInspectionStatus: document.querySelector("#runtime-inspection-status"),
  runtimeEvidence: document.querySelector("#runtime-evidence"),
  experimentPreviewButton: document.querySelector("#experiment-preview-button"),
  saveExperimentButton: document.querySelector("#save-experiment-button"),
  submitExperimentButton: document.querySelector("#submit-experiment-button"),
  experimentRevisionIntent: document.querySelector(
    "#experiment-revision-intent",
  ),
  experimentPreviewEmpty: document.querySelector("#experiment-preview-empty"),
  experimentPreviewPanel: document.querySelector("#experiment-preview-panel"),
  experimentPreviewMeta: document.querySelector("#experiment-preview-meta"),
  experimentScriptPreview: document.querySelector("#experiment-script-preview"),
  experimentPreviewBlockers: document.querySelector(
    "#experiment-preview-blockers",
  ),
  experimentPreviewBlockerList: document.querySelector(
    "#experiment-preview-blocker-list",
  ),
  experimentsBody: document.querySelector("#experiments-body"),
  experimentCount: document.querySelector("#experiment-count"),
  experimentDetail: document.querySelector("#experiment-detail"),
  experimentDetailTitle: document.querySelector("#experiment-detail-title"),
  experimentDetailMeta: document.querySelector("#experiment-detail-meta"),
  experimentDetailTracking: document.querySelector(
    "#experiment-detail-tracking",
  ),
  variantsBody: document.querySelector("#variants-body"),
  runsError: document.querySelector("#runs-error"),
  refreshRuns: document.querySelector("#refresh-runs"),
  runSearch: document.querySelector("#run-search"),
  runStatusFilter: document.querySelector("#run-status-filter"),
  runCount: document.querySelector("#run-count"),
  runsBody: document.querySelector("#runs-body"),
  runDetail: document.querySelector("#run-detail"),
  runDetailTitle: document.querySelector("#run-detail-title"),
  runDetailActions: document.querySelector("#run-detail-actions"),
  runDetailMeta: document.querySelector("#run-detail-meta"),
  runDetailTracking: document.querySelector("#run-detail-tracking"),
  attemptsBody: document.querySelector("#attempts-body"),
  runAttemptDetail: document.querySelector("#run-attempt-detail"),
  runAttemptDetailTitle: document.querySelector("#run-attempt-detail-title"),
  runAttemptDetailMeta: document.querySelector("#run-attempt-detail-meta"),
  runAttemptDetailTracking: document.querySelector(
    "#run-attempt-detail-tracking",
  ),
  runAttemptHyperparameters: document.querySelector(
    "#run-attempt-hyperparameters",
  ),
  runAttemptAdapterSection: document.querySelector(
    "#run-attempt-adapter-section",
  ),
  runAttemptAdapterSettings: document.querySelector(
    "#run-attempt-adapter-settings",
  ),
  runAttemptStdoutStatus: document.querySelector("#run-attempt-stdout-status"),
  runAttemptStdoutLog: document.querySelector("#run-attempt-stdout-log"),
  runAttemptStderrStatus: document.querySelector("#run-attempt-stderr-status"),
  runAttemptStderrLog: document.querySelector("#run-attempt-stderr-log"),
  evaluationsError: document.querySelector("#evaluations-error"),
  refreshEvaluations: document.querySelector("#refresh-evaluations"),
  evaluationForm: document.querySelector("#evaluation-form"),
  evaluationRunId: document.querySelector("#evaluation-run-id"),
  evaluationRunValidation: document.querySelector("#evaluation-run-validation"),
  evaluationCheckpoint: document.querySelector("#evaluation-checkpoint"),
  evaluationCheckpointValidation: document.querySelector(
    "#evaluation-checkpoint-validation",
  ),
  evaluationSuite: document.querySelector("#evaluation-suite"),
  evaluationPlanStatus: document.querySelector("#evaluation-plan-status"),
  evaluationEnvironment: document.querySelector("#evaluation-environment"),
  evaluationEpisodes: document.querySelector("#evaluation-episodes"),
  evaluationTasksFilter: document.querySelector("#evaluation-tasks-filter"),
  evaluationTasksFilterLabel: document.querySelector(
    "#evaluation-tasks-filter-label",
  ),
  evaluationTasksOptions: document.querySelector("#evaluation-tasks-options"),
  evaluationTasksAll: document.querySelector("#evaluation-tasks-all"),
  evaluationTasksClear: document.querySelector("#evaluation-tasks-clear"),
  evaluationTasksStatus: document.querySelector("#evaluation-tasks-status"),
  evaluationSeeds: document.querySelector("#evaluation-seeds"),
  evaluationParallelism: document.querySelector("#evaluation-parallelism"),
  evaluationHeadless: document.querySelector("#evaluation-headless"),
  evaluationAutoResume: document.querySelector("#evaluation-auto-resume"),
  evaluationMaxAttempts: document.querySelector("#evaluation-max-attempts"),
  evaluationArgv: document.querySelector("#evaluation-argv"),
  evaluationResumeArgv: document.querySelector("#evaluation-resume-argv"),
  submitEvaluation: document.querySelector("#submit-evaluation"),
  evaluationsBody: document.querySelector("#evaluations-body"),
  evaluationCount: document.querySelector("#evaluation-count"),
  evaluationDetail: document.querySelector("#evaluation-detail"),
  evaluationDetailTitle: document.querySelector("#evaluation-detail-title"),
  evaluationDetailActions: document.querySelector("#evaluation-detail-actions"),
  evaluationDetailMeta: document.querySelector("#evaluation-detail-meta"),
  evaluationAttemptMeta: document.querySelector("#evaluation-attempt-meta"),
  evaluationStdoutStatus: document.querySelector("#evaluation-stdout-status"),
  evaluationStdoutLog: document.querySelector("#evaluation-stdout-log"),
  evaluationStderrStatus: document.querySelector("#evaluation-stderr-status"),
  evaluationStderrLog: document.querySelector("#evaluation-stderr-log"),
  evaluationResultJson: document.querySelector("#evaluation-result-json"),
  dataRegistryError: document.querySelector("#data-registry-error"),
  refreshDataRegistry: document.querySelector("#refresh-data-registry"),
  dataResourceCount: document.querySelector("#data-resource-count"),
  dataResourcesBody: document.querySelector("#data-resources-body"),
  dataImportCount: document.querySelector("#data-import-count"),
  dataImportsBody: document.querySelector("#data-imports-body"),
  dataVersionCount: document.querySelector("#data-version-count"),
  dataVersionsBody: document.querySelector("#data-versions-body"),
  dataDerivationCount: document.querySelector("#data-derivation-count"),
  dataDerivationsBody: document.querySelector("#data-derivations-body"),
  dataResourceForm: document.querySelector("#data-resource-form"),
  dataResourceProvider: document.querySelector("#data-resource-provider"),
  dataResourceNamespace: document.querySelector("#data-resource-namespace"),
  dataResourceName: document.querySelector("#data-resource-name"),
  dataResourceKind: document.querySelector("#data-resource-kind"),
  dataResourceDescription: document.querySelector("#data-resource-description"),
  createDataResource: document.querySelector("#create-data-resource"),
  dataVersionForm: document.querySelector("#data-version-form"),
  dataVersionResourceId: document.querySelector("#data-version-resource-id"),
  dataVersionResourceLabel: document.querySelector(
    "#data-version-resource-label",
  ),
  closeDataVersionForm: document.querySelector("#close-data-version-form"),
  dataVersionRevision: document.querySelector("#data-version-revision"),
  dataVersionFormat: document.querySelector("#data-version-format"),
  dataVersionPath: document.querySelector("#data-version-path"),
  dataVersionSourceUri: document.querySelector("#data-version-source-uri"),
  dataVersionManifest: document.querySelector("#data-version-manifest"),
  dataVersionStatus: document.querySelector("#data-version-status"),
  dataVersionSize: document.querySelector("#data-version-size"),
  createDataVersion: document.querySelector("#create-data-version"),
  dataImportForm: document.querySelector("#data-import-form"),
  dataImportResourceId: document.querySelector("#data-import-resource-id"),
  dataImportResourceLabel: document.querySelector(
    "#data-import-resource-label",
  ),
  dataImportRevision: document.querySelector("#data-import-revision"),
  dataImportSubset: document.querySelector("#data-import-subset"),
  dataImportFormat: document.querySelector("#data-import-format"),
  dataImportRole: document.querySelector("#data-import-role"),
  dataImportGateway: document.querySelector("#data-import-gateway"),
  dataImportQueue: document.querySelector("#data-import-queue"),
  submitDataImport: document.querySelector("#submit-data-import"),
  closeDataImportForm: document.querySelector("#close-data-import-form"),
  dataDerivationForm: document.querySelector("#data-derivation-form"),
  dataDerivationOutput: document.querySelector("#data-derivation-output"),
  dataDerivationInputs: document.querySelector("#data-derivation-inputs"),
  dataConverterRepository: document.querySelector("#data-converter-repository"),
  dataConverterCommit: document.querySelector("#data-converter-commit"),
  dataRuntimeLockSha: document.querySelector("#data-runtime-lock-sha"),
  dataConverterConfig: document.querySelector("#data-converter-config"),
  createDataDerivation: document.querySelector("#create-data-derivation"),
  experimentDataBundle: document.querySelector("#experiment-data-bundle"),
  experimentDataBundleStatus: document.querySelector(
    "#experiment-data-bundle-status",
  ),
  settingsError: document.querySelector("#settings-error"),
  refreshSettings: document.querySelector("#refresh-settings"),
  settingsBody: document.querySelector("#settings-body"),
  wandbConnectionForm: document.querySelector("#wandb-connection-form"),
  wandbApiKey: document.querySelector("#wandb-api-key"),
  wandbEntity: document.querySelector("#wandb-entity"),
  wandbConnectionStatus: document.querySelector("#wandb-connection-status"),
  wandbConnectionDetail: document.querySelector("#wandb-connection-detail"),
  disconnectWandb: document.querySelector("#disconnect-wandb"),
  mlflowConnectionForm: document.querySelector("#mlflow-connection-form"),
  mlflowConnectionUri: document.querySelector("#mlflow-connection-uri"),
  mlflowConnectionUsername: document.querySelector(
    "#mlflow-connection-username",
  ),
  mlflowConnectionPassword: document.querySelector(
    "#mlflow-connection-password",
  ),
  mlflowConnectionToken: document.querySelector("#mlflow-connection-token"),
  mlflowConnectionStatus: document.querySelector("#mlflow-connection-status"),
  mlflowConnectionDetail: document.querySelector("#mlflow-connection-detail"),
  disconnectMlflow: document.querySelector("#disconnect-mlflow"),
  adaptersError: document.querySelector("#adapters-error"),
  refreshAdapters: document.querySelector("#refresh-adapters"),
  addAdapter: document.querySelector("#add-adapter"),
  adapterCount: document.querySelector("#adapter-count"),
  adaptersBody: document.querySelector("#adapters-body"),
  adapterEditor: document.querySelector("#adapter-editor"),
  adapterEditorMode: document.querySelector("#adapter-editor-mode"),
  adapterEditorTitle: document.querySelector("#adapter-editor-title"),
  adapterEditorStatus: document.querySelector("#adapter-editor-status"),
  adapterEditorSlug: document.querySelector("#adapter-editor-slug"),
  adapterEditorName: document.querySelector("#adapter-editor-name"),
  adapterManifest: document.querySelector("#adapter-manifest"),
  adapterDescription: document.querySelector("#adapter-description"),
  adapterChangeNote: document.querySelector("#adapter-change-note"),
  adapterValidationRepository: document.querySelector(
    "#adapter-validation-repository",
  ),
  adapterValidationRevision: document.querySelector(
    "#adapter-validation-revision",
  ),
  adapterValidationReport: document.querySelector("#adapter-validation-report"),
  closeAdapterEditor: document.querySelector("#close-adapter-editor"),
  editAdapter: document.querySelector("#edit-adapter"),
  validateAdapter: document.querySelector("#validate-adapter"),
  saveAdapter: document.querySelector("#save-adapter"),
  adapterVersionSection: document.querySelector("#adapter-version-section"),
  adapterVersionsBody: document.querySelector("#adapter-versions-body"),
  collectionError: document.querySelector("#collection-error"),
  collectionImportDrawer: document.querySelector("#collection-import-drawer"),
  refreshCollection: document.querySelector("#refresh-collection"),
  addCollectionAdapter: document.querySelector("#add-collection-adapter"),
  collectionAdapterCount: document.querySelector("#collection-adapter-count"),
  collectionAdaptersBody: document.querySelector("#collection-adapters-body"),
  collectionAdapterForm: document.querySelector("#collection-adapter-form"),
  collectionAdapterEditorTitle: document.querySelector(
    "#collection-adapter-editor-title",
  ),
  collectionAdapterId: document.querySelector("#collection-adapter-id"),
  collectionAdapterKey: document.querySelector("#collection-adapter-key"),
  collectionAdapterName: document.querySelector("#collection-adapter-name"),
  collectionAdapterVersion: document.querySelector(
    "#collection-adapter-version",
  ),
  collectionAdapterRunnable: document.querySelector(
    "#collection-adapter-runnable",
  ),
  collectionAdapterDescription: document.querySelector(
    "#collection-adapter-description",
  ),
  collectionAdapterStreams: document.querySelector(
    "#collection-adapter-streams",
  ),
  collectionAdapterRequirements: document.querySelector(
    "#collection-adapter-requirements",
  ),
  collectionAdapterLauncher: document.querySelector(
    "#collection-adapter-launcher",
  ),
  collectionAdapterCapabilities: document.querySelector(
    "#collection-adapter-capabilities",
  ),
  collectionAdapterDefaults: document.querySelector(
    "#collection-adapter-defaults",
  ),
  collectionAdapterTodos: document.querySelector("#collection-adapter-todos"),
  collectionAdapterMetadata: document.querySelector(
    "#collection-adapter-metadata",
  ),
  closeCollectionAdapter: document.querySelector("#close-collection-adapter"),
  saveCollectionAdapter: document.querySelector("#save-collection-adapter"),
  collectionSessionForm: document.querySelector("#collection-session-form"),
  collectionSessionAdapter: document.querySelector(
    "#collection-session-adapter",
  ),
  collectionSessionName: document.querySelector("#collection-session-name"),
  collectionSessionOperator: document.querySelector(
    "#collection-session-operator",
  ),
  collectionSessionTask: document.querySelector("#collection-session-task"),
  collectionSessionEmbodiment: document.querySelector(
    "#collection-session-embodiment",
  ),
  collectionRuntimeProfile: document.querySelector(
    "#collection-runtime-profile",
  ),
  collectionOutputPath: document.querySelector("#collection-output-path"),
  collectionNativeFormat: document.querySelector("#collection-native-format"),
  collectionRegisterOutput: document.querySelector(
    "#collection-register-output",
  ),
  collectionRegisterProvider: document.querySelector(
    "#collection-register-provider",
  ),
  collectionRegisterNamespace: document.querySelector(
    "#collection-register-namespace",
  ),
  collectionRegisterName: document.querySelector("#collection-register-name"),
  collectionRegisterKind: document.querySelector("#collection-register-kind"),
  collectionStorageMetadata: document.querySelector(
    "#collection-storage-metadata",
  ),
  collectionCalibrationIdentity: document.querySelector(
    "#collection-calibration-identity",
  ),
  collectionCalibrationSha: document.querySelector(
    "#collection-calibration-sha",
  ),
  collectionCalibrationMetadata: document.querySelector(
    "#collection-calibration-metadata",
  ),
  collectionSoftware: document.querySelector("#collection-software"),
  collectionConfig: document.querySelector("#collection-config"),
  collectionCaptureSchemaName: document.querySelector(
    "#collection-capture-schema-name",
  ),
  collectionCaptureSchemaVersion: document.querySelector(
    "#collection-capture-schema-version",
  ),
  collectionClockSource: document.querySelector("#collection-clock-source"),
  collectionNominalRate: document.querySelector("#collection-nominal-rate"),
  collectionTimestampUnit: document.querySelector("#collection-timestamp-unit"),
  collectionAlignment: document.querySelector("#collection-alignment"),
  collectionTimestampsRecorded: document.querySelector(
    "#collection-timestamps-recorded",
  ),
  collectionCaptureMetadata: document.querySelector(
    "#collection-capture-metadata",
  ),
  collectionCapabilities: document.querySelector("#collection-capabilities"),
  collectionCapabilitiesHelp: document.querySelector(
    "#collection-capabilities-help",
  ),
  collectionGateway: document.querySelector("#collection-gateway"),
  collectionAccount: document.querySelector("#collection-account"),
  collectionPartition: document.querySelector("#collection-partition"),
  collectionNode: document.querySelector("#collection-node"),
  collectionGpuCount: document.querySelector("#collection-gpu-count"),
  collectionGpuType: document.querySelector("#collection-gpu-type"),
  collectionCpuCount: document.querySelector("#collection-cpu-count"),
  collectionMemoryGb: document.querySelector("#collection-memory-gb"),
  collectionTimeLimit: document.querySelector("#collection-time-limit"),
  createCollectionSession: document.querySelector("#create-collection-session"),
  collectionSessionCount: document.querySelector("#collection-session-count"),
  collectionSessionsBody: document.querySelector("#collection-sessions-body"),
  collectionSessionDetail: document.querySelector("#collection-session-detail"),
  collectionSessionDetailTitle: document.querySelector(
    "#collection-session-detail-title",
  ),
  collectionSessionActions: document.querySelector(
    "#collection-session-actions",
  ),
  collectionSessionMeta: document.querySelector("#collection-session-meta"),
  collectionRegistryLink: document.querySelector("#collection-registry-link"),
  collectionCompleteForm: document.querySelector("#collection-complete-form"),
  collectionCompleteManifest: document.querySelector(
    "#collection-complete-manifest",
  ),
  collectionCompleteRevision: document.querySelector(
    "#collection-complete-revision",
  ),
  collectionCompleteSize: document.querySelector("#collection-complete-size"),
  collectionLoadStdout: document.querySelector("#collection-load-stdout"),
  collectionLoadStderr: document.querySelector("#collection-load-stderr"),
  collectionLogStatus: document.querySelector("#collection-log-status"),
  collectionSessionLog: document.querySelector("#collection-session-log"),
  collectionSessionError: document.querySelector("#collection-session-error"),
  collectionSessionResult: document.querySelector("#collection-session-result"),
  collectionEventsBody: document.querySelector("#collection-events-body"),
};

let refreshTimer;
const trackingConnections = new Map();
let trackingConnectionsLoaded = false;

let renderedEvaluationTaskSuiteId = "";
let renderedEvaluationTaskPolicy = { mode: "subset", reason: "", count: 0 };
let evaluationTaskSelectionLocallyValid = true;
let evaluationCreateRequestedDisabled = true;
let evaluationCreateButton = null;
let evaluationCreateDisabledDescriptor = null;

function normalizedEvaluationTaskOptions(suite) {
  const config =
    suite?.config_json && typeof suite.config_json === "object"
      ? suite.config_json
      : {};
  const hasStructuredOptions =
    Array.isArray(suite?.task_options) || Array.isArray(config.task_options);
  const source = Array.isArray(suite?.task_options)
    ? suite.task_options
    : Array.isArray(config.task_options)
      ? config.task_options
      : Array.isArray(suite?.tasks)
        ? suite.tasks
        : Array.isArray(config.tasks)
          ? config.tasks
          : [];
  const seen = new Set();
  return source.flatMap((task) => {
    const structured = task && typeof task === "object" && !Array.isArray(task);
    const id = String(structured ? (task.id ?? "") : (task ?? "")).trim();
    if (!id || seen.has(id)) return [];
    seen.add(id);
    const metadata =
      structured &&
      task.metadata &&
      typeof task.metadata === "object" &&
      !Array.isArray(task.metadata)
        ? task.metadata
        : {};
    return [
      {
        id,
        label: String(structured ? task.label || id : id),
        description:
          structured && task.description ? String(task.description) : "",
        metadata,
        structured: hasStructuredOptions,
      },
    ];
  });
}

function evaluationTaskProfileText(option) {
  const profile =
    option?.metadata?.profile ??
    option?.metadata?.profile_id ??
    option?.metadata?.profile_name;
  if (profile === null || profile === undefined || profile === "") return "";
  if (typeof profile === "object") {
    try {
      return JSON.stringify(profile);
    } catch (_error) {
      return String(profile);
    }
  }
  return String(profile);
}

function evaluationTaskSelectionPolicy(suite, count) {
  const config =
    suite?.config_json && typeof suite.config_json === "object"
      ? suite.config_json
      : {};
  const declaration =
    suite?.task_selection && typeof suite.task_selection === "object"
      ? suite.task_selection
      : config.task_selection && typeof config.task_selection === "object"
        ? config.task_selection
        : {};
  const declaredMode =
    suite?.task_selection_mode ??
    config.task_selection_mode ??
    declaration.mode ??
    "subset";
  const mode = ["single", "subset", "all_only"].includes(String(declaredMode))
    ? String(declaredMode)
    : "invalid";
  const reason =
    suite?.task_selection_reason ??
    suite?.task_selection_mode_reason ??
    config.task_selection_reason ??
    config.task_selection_mode_reason ??
    declaration.reason ??
    "";
  return {
    mode,
    reason:
      mode === "invalid"
        ? `Unsupported task selection mode "${String(declaredMode)}". Expected single, subset, or all_only.`
        : String(reason || ""),
    count,
  };
}

function setEvaluationTaskCreateValidity(valid) {
  evaluationTaskSelectionLocallyValid = valid;
  if (evaluationCreateButton && evaluationCreateDisabledDescriptor?.set) {
    evaluationCreateDisabledDescriptor.set.call(
      evaluationCreateButton,
      evaluationCreateRequestedDisabled || !evaluationTaskSelectionLocallyValid,
    );
  }
}

function applyEvaluationTaskSelectionPolicy(changedInput = null) {
  const inputs = [...elements.evaluationTasksOptions.querySelectorAll("input")];
  const policy = renderedEvaluationTaskPolicy;
  const checked = inputs.filter((input) => input.checked);
  const invalidSingle =
    policy.mode === "single" && inputs.length > 1 && checked.length !== 1;
  const invalidAllOnly = policy.mode === "all_only" && checked.length > 0;
  const invalidPolicy = policy.mode === "invalid";
  const unavailableSelection =
    elements.evaluationSuite.dataset.taskSelectionError || "";
  const invalid =
    invalidSingle ||
    invalidAllOnly ||
    invalidPolicy ||
    Boolean(unavailableSelection);
  inputs.forEach((input, index) => {
    input.disabled = policy.mode === "all_only" && !invalidAllOnly;
    input.setCustomValidity(
      index === 0 && invalid
        ? unavailableSelection ||
            (invalidSingle
              ? policy.reason || "Choose exactly one task for this suite."
              : "") ||
            (invalidAllOnly
              ? "This suite only permits its complete default task set; clear individual selections."
              : "") ||
            policy.reason
        : "",
    );
  });
  elements.evaluationTasksFilter.classList.toggle(
    "is-disabled",
    !inputs.length || (policy.mode === "all_only" && !invalidAllOnly),
  );
  elements.evaluationTasksFilter.setAttribute(
    "aria-disabled",
    String(!inputs.length || (policy.mode === "all_only" && !invalidAllOnly)),
  );
  elements.evaluationTasksFilter.setAttribute("aria-invalid", String(invalid));
  elements.evaluationTasksAll.hidden = policy.mode !== "subset";
  const defaults =
    selectedEvaluationSuite()?.default_tasks ||
    selectedEvaluationSuite()?.config_json?.default_tasks ||
    [];
  elements.evaluationTasksClear.hidden =
    policy.mode === "single" && inputs.length > 1 && defaults.length !== 1;
  updateEvaluationTaskLabel();

  const modeStatus = invalidPolicy
    ? "The suite declares an unsupported task selection policy."
    : policy.mode === "single"
      ? inputs.length > 1
        ? invalidSingle
          ? "Choose exactly one task."
          : "One task selected."
        : "Choose the task or leave it blank to use the suite default."
      : policy.mode === "all_only"
        ? invalidAllOnly
          ? "Clear individual tasks; this suite only accepts its complete default task set."
          : "This suite always uses its complete default task set."
        : inputs.length
          ? "Choose tasks; leave blank for the default."
          : "No task catalog is declared; the suite or adapter default will be used.";
  elements.evaluationTasksStatus.textContent = [
    modeStatus,
    policy.reason,
    unavailableSelection,
  ]
    .filter(Boolean)
    .join(" ");
  setEvaluationTaskCreateValidity(!invalid);
}

function installEvaluationCreateValidityGuard() {
  evaluationCreateButton = elements.evaluationForm.querySelector(
    'button[type="submit"], input[type="submit"]',
  );
  const prototype =
    evaluationCreateButton instanceof HTMLButtonElement
      ? HTMLButtonElement.prototype
      : evaluationCreateButton instanceof HTMLInputElement
        ? HTMLInputElement.prototype
        : null;
  if (!prototype) return;
  evaluationCreateDisabledDescriptor = Object.getOwnPropertyDescriptor(
    prototype,
    "disabled",
  );
  if (
    !evaluationCreateDisabledDescriptor?.get ||
    !evaluationCreateDisabledDescriptor?.set
  )
    return;
  evaluationCreateRequestedDisabled =
    evaluationCreateDisabledDescriptor.get.call(evaluationCreateButton);
  Object.defineProperty(evaluationCreateButton, "disabled", {
    configurable: true,
    get() {
      return evaluationCreateDisabledDescriptor.get.call(this);
    },
    set(value) {
      evaluationCreateRequestedDisabled = Boolean(value);
      evaluationCreateDisabledDescriptor.set.call(
        this,
        evaluationCreateRequestedDisabled ||
          !evaluationTaskSelectionLocallyValid,
      );
    },
  });
}

function installStructuredEvaluationTaskOptions() {
  const basePopulateEvaluationTasks = populateEvaluationTasks;
  installEvaluationCreateValidityGuard();
  populateEvaluationTasks = (suite, options = {}) => {
    const renderOptions = Array.isArray(options)
      ? { preferredTasks: options }
      : options || {};
    const preserve = Boolean(renderOptions.preserve);
    const suiteId = String(suite?.id || suite?.suite_id || suite?.slug || "");
    const taskOptions = normalizedEvaluationTaskOptions(suite);
    renderedEvaluationTaskPolicy = evaluationTaskSelectionPolicy(
      suite,
      taskOptions.length,
    );
    const currentSelection =
      preserve && suiteId && suiteId === renderedEvaluationTaskSuiteId
        ? checkedEvaluationTaskIds()
        : [];
    const requestedSelection = Array.isArray(renderOptions.preferredTasks)
      ? renderOptions.preferredTasks.map(String)
      : [];
    const retainedSelection = requestedSelection.length
      ? requestedSelection
      : preserve
        ? currentSelection
        : suite?.default_tasks || suite?.config_json?.default_tasks || [];
    const availableIds = new Set(taskOptions.map((option) => option.id));
    const unavailable = retainedSelection.filter(
      (taskId) => !availableIds.has(taskId),
    );
    if (unavailable.length) {
      elements.evaluationSuite.dataset.taskSelectionError = `Requested task${unavailable.length === 1 ? "" : "s"} unavailable in this suite: ${unavailable.join(", ")}.`;
    } else {
      delete elements.evaluationSuite.dataset.taskSelectionError;
    }
    const normalizedSuite = suite
      ? {
          ...suite,
          task_options: undefined,
          tasks: taskOptions.map((option) => option.id),
          config_json: {
            ...(suite.config_json && typeof suite.config_json === "object"
              ? suite.config_json
              : {}),
            task_options: undefined,
            tasks: taskOptions.map((option) => option.id),
          },
        }
      : suite;

    basePopulateEvaluationTasks(normalizedSuite, { preserve: false });
    const retained = new Set(retainedSelection);
    const optionsById = new Map(
      taskOptions.map((option) => [option.id, option]),
    );
    elements.evaluationTasksOptions
      .querySelectorAll("input")
      .forEach((input) => {
        input.type =
          renderedEvaluationTaskPolicy.mode === "single" ? "radio" : "checkbox";
        input.name = "evaluation-task";
        input.checked = retained.has(input.value);
        const option = optionsById.get(input.value);
        const label = input.closest("label")?.querySelector("span");
        if (!option || !label) return;
        label.textContent = option.label;
        const profile = evaluationTaskProfileText(option);
        const helpText = [
          option.description,
          profile ? `Profile: ${profile}` : "",
        ]
          .filter(Boolean)
          .join(" · ");
        if (helpText) {
          const help = document.createElement("small");
          help.className = "secondary";
          help.textContent = helpText;
          help.title = helpText;
          label.append(document.createElement("br"), help);
        }
      });
    applyEvaluationTaskSelectionPolicy();
    renderedEvaluationTaskSuiteId = suiteId;
  };
  elements.evaluationTasksOptions.addEventListener("change", (event) => {
    const changedInput = event.target.closest("input");
    if (!changedInput) return;
    delete elements.evaluationSuite.dataset.taskSelectionError;
    applyEvaluationTaskSelectionPolicy(changedInput);
    scheduleEvaluationTargetValidation();
  });
  elements.evaluationTasksAll.addEventListener("click", () => {
    delete elements.evaluationSuite.dataset.taskSelectionError;
    applyEvaluationTaskSelectionPolicy();
    scheduleEvaluationTargetValidation();
  });
  elements.evaluationTasksClear.addEventListener("click", () => {
    delete elements.evaluationSuite.dataset.taskSelectionError;
    applyEvaluationTaskSelectionPolicy();
    scheduleEvaluationTargetValidation();
  });
}

let viewportFilterMenuFrame = 0;

function resetViewportFilterMenu(details) {
  const menu = details.querySelector(":scope > .filter-options");
  details.classList.remove("has-viewport-menu", "opens-upward");
  details
    .querySelector(":scope > summary")
    ?.setAttribute("aria-expanded", "false");
  if (!menu) return;
  for (const property of [
    "position",
    "left",
    "right",
    "top",
    "bottom",
    "width",
    "max-height",
  ]) {
    menu.style.removeProperty(property);
  }
}

function positionViewportFilterMenu(details) {
  if (!details.open) {
    resetViewportFilterMenu(details);
    return;
  }
  if (details.getAttribute("aria-disabled") === "true") {
    details.open = false;
    resetViewportFilterMenu(details);
    return;
  }
  const summary = details.querySelector(":scope > summary");
  const menu = details.querySelector(":scope > .filter-options");
  if (!summary || !menu) return;
  const anchor = summary.getBoundingClientRect();
  const viewportWidth = document.documentElement.clientWidth;
  const viewportHeight = document.documentElement.clientHeight;
  const margin = 8;
  const gap = 4;
  if (anchor.bottom <= margin || anchor.top >= viewportHeight - margin) {
    details.open = false;
    resetViewportFilterMenu(details);
    return;
  }

  const evaluationMenu = details.classList.contains("evaluation-task-select");
  const preferredWidth = evaluationMenu
    ? Math.max(anchor.width, 360)
    : Math.max(anchor.width, 230);
  const width = Math.max(
    1,
    Math.min(preferredWidth, viewportWidth - margin * 2),
  );
  const left = Math.min(
    viewportWidth - margin - width,
    Math.max(margin, evaluationMenu ? anchor.left : anchor.right - width),
  );
  const availableBelow = Math.max(
    0,
    viewportHeight - anchor.bottom - gap - margin,
  );
  const availableAbove = Math.max(0, anchor.top - gap - margin);
  const heightCap = Math.min(
    evaluationMenu ? 420 : 300,
    Math.floor(viewportHeight * 0.72),
  );
  const opensUpward =
    availableBelow < Math.min(heightCap, 220) &&
    availableAbove > availableBelow;
  const availableHeight = opensUpward ? availableAbove : availableBelow;
  const maxHeight = Math.max(72, Math.min(heightCap, availableHeight));

  details.classList.add("has-viewport-menu");
  details.classList.toggle("opens-upward", opensUpward);
  summary.setAttribute("aria-expanded", "true");
  menu.style.position = "fixed";
  menu.style.left = `${Math.round(left)}px`;
  menu.style.right = "auto";
  menu.style.width = `${Math.round(width)}px`;
  menu.style.maxHeight = `${Math.floor(maxHeight)}px`;
  if (opensUpward) {
    menu.style.top = "auto";
    menu.style.bottom = `${Math.round(viewportHeight - anchor.top + gap)}px`;
  } else {
    menu.style.top = `${Math.round(anchor.bottom + gap)}px`;
    menu.style.bottom = "auto";
  }
}

function scheduleViewportFilterMenus() {
  window.cancelAnimationFrame(viewportFilterMenuFrame);
  viewportFilterMenuFrame = window.requestAnimationFrame(() => {
    document
      .querySelectorAll("details.partition-filter[open]")
      .forEach(positionViewportFilterMenu);
  });
}

function installViewportFilterMenus() {
  const filters = [...document.querySelectorAll("details.partition-filter")];
  filters.forEach((details) => {
    const summary = details.querySelector(":scope > summary");
    summary?.setAttribute("aria-haspopup", "true");
    summary?.setAttribute("aria-expanded", String(details.open));
    details.addEventListener("toggle", () => {
      if (details.open) {
        filters.forEach((other) => {
          if (other !== details && other.open) other.open = false;
        });
        scheduleViewportFilterMenus();
      } else {
        resetViewportFilterMenu(details);
      }
    });
  });
  document.addEventListener("pointerdown", (event) => {
    filters.forEach((details) => {
      if (details.open && !details.contains(event.target)) details.open = false;
    });
  });
  document.addEventListener("keydown", (event) => {
    if (
      event.key !== "Escape" ||
      event.defaultPrevented ||
      SkynetDialog.fullscreenOwnsEscape()
    )
      return;
    if (document.querySelector("dialog[open]")) return;
    const open = filters.findLast((details) => details.open);
    if (!open) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    open.open = false;
    open.querySelector(":scope > summary")?.focus();
  });
  window.addEventListener("resize", scheduleViewportFilterMenus);
  document.addEventListener("scroll", scheduleViewportFilterMenus, true);
}

// A background refresh is one panel-level commit. It never moves focus or calls
// scrollIntoView, and identical rendered content performs no DOM writes.
const backgroundRefreshStates = new WeakMap();

function stableRefreshValue(value, seen = new WeakSet()) {
  if (value === null || typeof value !== "object") return value;
  if (seen.has(value)) return "[circular]";
  seen.add(value);
  const normalized = Array.isArray(value)
    ? value.map((item) => stableRefreshValue(item, seen))
    : Object.fromEntries(
        Object.keys(value)
          .sort()
          .map((key) => [key, stableRefreshValue(value[key], seen)]),
      );
  seen.delete(value);
  return normalized;
}

function refreshContentSignature(value) {
  return JSON.stringify(stableRefreshValue(value));
}

function setTextIfChanged(node, value) {
  const next = String(value ?? "");
  if (node.textContent === next) return false;
  node.textContent = next;
  return true;
}

function setHtmlIfChanged(node, value) {
  const next = String(value ?? "");
  if (node.innerHTML === next) return false;
  node.innerHTML = next;
  return true;
}

function captureBackgroundRefreshState(panel) {
  const scrollNodes = [
    panel,
    ...panel.querySelectorAll(
      ".table-scroll, .filter-options, textarea, pre, [data-preserve-scroll]",
    ),
  ]
    .filter((node, index, nodes) => nodes.indexOf(node) === index)
    .map((node) => ({ node, top: node.scrollTop, left: node.scrollLeft }));
  return {
    windowX: window.scrollX,
    windowY: window.scrollY,
    scrollNodes,
    details: [...panel.querySelectorAll("details")].map((node) => ({
      node,
      open: node.open,
    })),
  };
}

function restoreBackgroundRefreshState(snapshot) {
  snapshot.details.forEach(({ node, open }) => {
    if (node.isConnected && node.open !== open) node.open = open;
  });
  snapshot.scrollNodes.forEach(({ node, top, left }) => {
    if (!node.isConnected) return;
    if (node.scrollTop !== top) node.scrollTop = top;
    if (node.scrollLeft !== left) node.scrollLeft = left;
  });
  if (
    window.scrollX !== snapshot.windowX ||
    window.scrollY !== snapshot.windowY
  ) {
    window.scrollTo(snapshot.windowX, snapshot.windowY);
  }
}

function commitPanelRefresh(
  panel,
  channel,
  payload,
  mutation,
  { background = false } = {},
) {
  const boundary = panel instanceof Element ? panel : document.body;
  let state = backgroundRefreshStates.get(boundary);
  if (!state) {
    state = new Map();
    backgroundRefreshStates.set(boundary, state);
  }
  const signature = refreshContentSignature(payload);
  if (background && state.get(channel) === signature) return false;
  const snapshot = background ? captureBackgroundRefreshState(boundary) : null;
  mutation();
  state.set(channel, signature);
  if (snapshot) restoreBackgroundRefreshState(snapshot);
  return true;
}

function patchTableRow(row, cells) {
  while (row.cells.length < cells.length)
    row.append(document.createElement("td"));
  while (row.cells.length > cells.length) row.lastElementChild.remove();
  cells.forEach((descriptor, index) => {
    const cell = row.cells[index];
    if (descriptor.preserve && cell.childNodes.length) return;
    const className = descriptor.className || "";
    if (cell.className !== className) cell.className = className;
    const colSpan = Number(descriptor.colSpan || 1);
    if (cell.colSpan !== colSpan) cell.colSpan = colSpan;
    setHtmlIfChanged(cell, descriptor.html);
  });
}

function reconcileTableSequence(tbody, sequence) {
  let cursor = tbody.firstElementChild;
  sequence.forEach((node) => {
    if (node === cursor) {
      cursor = cursor.nextElementSibling;
      return;
    }
    tbody.insertBefore(node, cursor);
  });
}

// A slow response from an earlier GET must not overwrite a newer poll/manual
// refresh. Stale callers receive a clone of the newest response for that key.
function installLatestApiReadGuard() {
  if (window.fetch.__latestApiReadGuard) return;
  const nativeFetch = window.fetch.bind(window);
  const generations = new Map();
  const latest = new Map();
  const pending = new Map();
  const releasePending = (key) => {
    const remaining = (pending.get(key) || 1) - 1;
    if (remaining > 0) {
      pending.set(key, remaining);
      return;
    }
    pending.delete(key);
    latest.delete(key);
    generations.delete(key);
  };
  const guardedFetch = async (input, init = {}) => {
    const request =
      typeof Request !== "undefined" && input instanceof Request ? input : null;
    const method = String(
      init.method || request?.method || "GET",
    ).toUpperCase();
    const url = new URL(request?.url || String(input), window.location.href);
    if (
      method !== "GET" ||
      url.origin !== window.location.origin ||
      !url.pathname.startsWith("/api/")
    ) {
      return nativeFetch(input, init);
    }

    // Gateway selection is mutable UI state, so all cluster snapshots share a
    // key. Other resources retain their full path/query identity.
    const key =
      url.pathname === "/api/cluster"
        ? "GET:/api/cluster"
        : `GET:${url.pathname}${url.search}`;
    const requestIdentity = url.href;
    const inFlight = latest.get(key);
    pending.set(key, (pending.get(key) || 0) + 1);
    if (inFlight?.requestIdentity === requestIdentity) {
      try {
        const shared = await inFlight.responsePromise;
        return shared.replay.clone();
      } finally {
        releasePending(key);
      }
    }
    const generation = (generations.get(key) || 0) + 1;
    generations.set(key, generation);
    try {
      const responsePromise = nativeFetch(input, init).then((response) => ({
        response,
        replay: response.clone(),
      }));
      latest.set(key, { generation, requestIdentity, responsePromise });
      const result = await responsePromise;
      const current = latest.get(key);
      if (current && current.generation !== generation) {
        const newest = await current.responsePromise;
        return newest.replay.clone();
      }
      return result.response;
    } finally {
      releasePending(key);
    }
  };
  Object.defineProperty(guardedFetch, "__latestApiReadGuard", { value: true });
  window.fetch = guardedFetch;
}
let latestSnapshot;
let activeTab = "cluster";
let experimentRows = [];
let runRows = [];
let evaluationRows = [];
let adapterRows = [];
const adapterDeclaredValueState = new Map();
const adapterDeclaredScopeHashes = new Map();
let renderedAdapterDeclaredScope = null;
let dataResourceRows = [];
let dataResourceTypes = {};
let dataPreparationRows = [];
let dataResourceCatalogState = "loading";
let dataImportRows = [];
let selectedDataImportId = null;
const dataImportLogCache = new Map();
let dataVersionRows = [];
let dataDerivationRows = [];
let dataBundlesLoaded = false;
let collectionAdapterRows = [];
let collectionAdapterTemplates = [];
let collectionSessionRows = [];
let activeCollectionSession = null;
let evaluationSuites = [];
let evaluationCatalogRows = [];
let evaluationCatalogRequest = 0;
let evaluationSuitesScope = "global";
const evaluationSuitesCache = new Map();
const evaluationSuiteReasons = new Map();
const evaluationSuitesPromises = new Map();
const evaluationSuiteRequestGenerations = new Map();
let evaluationSuiteReloadTimer = null;
let pendingEvaluationSuiteId = "";
let evaluationTargetValidationTimer = null;
let evaluationTargetValidationRequest = 0;
let evaluationTargetValidationState = {
  pending: false,
  valid: false,
  planValid: false,
  signature: "",
};
let evaluationSubmitting = false;
let evaluationDetailPollTimer = null;
let activeEvaluationDetailId = null;
let evaluationListPollTimer = null;
let evaluationListPollInFlight = null;
let evaluationProgressRefreshPending = false;
const cancellationRequests = new Set();
let sourceBranches = [];
let sourceCommits = [];
let sourceBranchRequest = 0;
let sourceCommitRequest = 0;
let sourceBranchesKey = "";
let sourceCommitsKey = "";
let sourceUrlTimer;
let runtimeInspectionRequest = 0;
let runtimeInspectionKey = "";
let runtimeCandidates = [];
let runtimeProfiles = [];
let experimentBusy = false;
const submissionRecoveryRequests = new Set();
const runResumeRequests = new Set();
let loadedExperimentAdapterSnapshot = null;
let loadedExperimentCanonicalContext = null;
let experimentConfigurationLoadRequest = 0;
const sourceSelections = new Map();
const sourceSelectionWrites = new Map();
const sourceMetadataResponses = new Map();
const sourceMetadataRequests = new Map();
let adapterEditorState = {
  mode: "view",
  id: null,
  adapter: null,
  versions: [],
};
let pendingEvaluationSuiteDefaults = null;
let evaluationPrefillRequest = 0;
let pendingEvaluationEnvironmentHints = [];
const loadedTabs = new Set();

async function sourceMetadataRequest(path, { forceRefresh = false } = {}) {
  if (!forceRefresh && sourceMetadataResponses.has(path)) {
    return sourceMetadataResponses.get(path);
  }
  if (!forceRefresh && sourceMetadataRequests.has(path)) {
    return sourceMetadataRequests.get(path);
  }
  const requestPath = forceRefresh
    ? `${path}${path.includes("?") ? "&" : "?"}refresh=true`
    : path;
  const request = api(requestPath).then((payload) => ({
    payload,
    fromCache:
      payload?._cache?.backend === "database" && payload._cache.hit === true,
  }));
  if (!forceRefresh) sourceMetadataRequests.set(path, request);
  try {
    const result = await request;
    sourceMetadataResponses.set(path, result);
    return result;
  } finally {
    if (sourceMetadataRequests.get(path) === request)
      sourceMetadataRequests.delete(path);
  }
}

function savedSourceSelection(repository) {
  return sourceSelections.get(repository) || { branch: "", commits: {} };
}

function rememberSourceSelection(repository, saved) {
  if (!saved || typeof saved !== "object" || Array.isArray(saved)) {
    throw new Error("Database source selection must be an object.");
  }
  const commits = saved.commits;
  if (!commits || typeof commits !== "object" || Array.isArray(commits)) {
    throw new Error("Database source selection commits must be an object.");
  }
  const selection = {
    branch: String(saved.branch || ""),
    commits: { ...commits },
  };
  sourceSelections.set(repository, selection);
  return selection;
}

function saveSourceSelection(repository, branch, commit = "") {
  if (!repository || !branch) return;
  const existing = savedSourceSelection(repository);
  if (
    existing.branch === branch &&
    (!commit || existing.commits?.[branch] === commit)
  )
    return;
  const selection = { branch, commits: { ...(existing.commits || {}) } };
  if (commit) selection.commits[branch] = commit;
  sourceSelections.set(repository, selection);

  const query = new URLSearchParams({ repo_url: repository, branch });
  if (commit) query.set("commit", commit);
  const previous = sourceSelectionWrites.get(repository) || Promise.resolve();
  const write = previous
    .catch(() => undefined)
    .then(() =>
      api(`/api/source/selection?${query.toString()}`, { method: "PUT" }),
    )
    .then((result) => rememberSourceSelection(repository, result.selection));
  sourceSelectionWrites.set(repository, write);
  write
    .catch((error) => {
      setSourceRefStatus(
        `Could not save repository selection in the database: ${error.message}`,
        "error",
      );
    })
    .finally(() => {
      if (sourceSelectionWrites.get(repository) === write)
        sourceSelectionWrites.delete(repository);
    });
}

const queueFilters = {
  partition: {
    all: true,
    available: [],
    selected: new Set(),
    label: elements.jobsPartitionFilterLabel,
    options: elements.jobsPartitionFilterOptions,
    allLabel: "All partitions",
    noneLabel: "No partitions",
    countSuffix: "partitions",
  },
  account: {
    all: true,
    available: [],
    selected: new Set(),
    label: elements.jobsAccountFilterLabel,
    options: elements.jobsAccountFilterOptions,
    allLabel: "All accounts",
    noneLabel: "No accounts",
    countSuffix: "accounts",
  },
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function apiErrorMessage(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail))
    return detail.map(apiErrorMessage).filter(Boolean).join("; ");
  if (!detail || typeof detail !== "object")
    return detail == null ? "" : String(detail);
  const message =
    detail.message ||
    detail.msg ||
    detail.reason ||
    detail.error ||
    detail.detail;
  if (message) {
    const path = Array.isArray(detail.loc)
      ? detail.loc.filter((part) => part !== "body").join(".")
      : "";
    return `${path ? `${path}: ` : ""}${apiErrorMessage(message)}`;
  }
  return JSON.stringify(detail, null, 2);
}

async function apiRequest(path, options = {}) {
  // A disconnected read must not leave the page's loading controls stuck.
  // Mutations retain their response so a slow submission is not mistaken for failure.
  const readOnly = ["GET", "HEAD"].includes(
    String(options.method || "GET").toUpperCase(),
  );
  const { timeoutMs = readOnly ? 60000 : 0, ...request } = options;
  const controller = timeoutMs > 0 ? new AbortController() : null;
  const forwardAbort = () => controller?.abort(request.signal?.reason);
  if (request.signal?.aborted) forwardAbort();
  else request.signal?.addEventListener("abort", forwardAbort, { once: true });
  const signal = controller?.signal || request.signal;
  const timer = controller
    ? setTimeout(() => controller.abort(), timeoutMs)
    : null;
  try {
    const headers = new Headers(request.headers || {});
    if (request.body && !headers.has("Content-Type"))
      headers.set("Content-Type", "application/json");
    const response = await fetch(path, { ...request, headers, signal });
    const raw = await response.text();
    let payload = {};
    if (raw) {
      try {
        payload = JSON.parse(raw);
      } catch {
        payload = raw;
      }
    }
    if (!response.ok) {
      const detail =
        typeof payload === "object" && payload !== null
          ? payload.detail || payload.message
          : payload;
      const message = apiErrorMessage(detail);
      throw new Error(message || `Request failed (${response.status})`);
    }
    return payload;
  } catch (error) {
    if (controller?.signal.aborted && !request.signal?.aborted) {
      throw new Error(
        `Request timed out after ${Math.ceil(timeoutMs / 1000)} seconds. Check the connection and try again.`,
      );
    }
    throw error;
  } finally {
    if (timer !== null) clearTimeout(timer);
    request.signal?.removeEventListener("abort", forwardAbort);
  }
}

async function api(path, options = {}) {
  if (tutorialState?.applyingValue) {
    throw new Error("A tutorial value cannot initiate a network request.");
  }
  const claim = tutorialObserveApiStart(path, options);
  try {
    const payload = await apiRequest(path, options);
    tutorialObserveApiSuccess(claim, payload);
    return payload;
  } catch (error) {
    tutorialObserveApiFailure(claim, error);
    throw error;
  }
}

function listFrom(payload, keys) {
  if (Array.isArray(payload)) return payload;
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error(
      "List API response must be an array or an object containing a list.",
    );
  }
  for (const key of [...keys, "items"]) {
    if (Array.isArray(payload[key])) return payload[key];
  }
  throw new Error(
    `List API response did not contain any expected array: ${[...keys, "items"].join(", ")}.`,
  );
}

function entityFrom(payload, key) {
  if (payload && typeof payload === "object" && payload[key])
    return payload[key];
  throw new Error(`Entity API response did not contain "${key}".`);
}

function dismissToast(toast) {
  if (toast?.parentElement !== elements.toast) return;
  toast.remove();
  elements.toast.hidden = elements.toast.childElementCount === 0;
}

function clearNotificationScope(scope) {
  if (!scope) return;
  document.querySelectorAll("[data-notification-scope]").forEach((item) => {
    if (item.dataset.notificationScope !== scope) return;
    if (item.classList.contains("toast")) dismissToast(item);
    else if (item.classList.contains("notice-item"))
      dismissNoticeItem(item.parentElement, item);
  });
}

function showToast(message, isError = false, { scope = "" } = {}) {
  if (scope) {
    if (!isError) clearNotificationScope(scope);
    else
      elements.toast.querySelectorAll(".toast").forEach((item) => {
        if (item.dataset.notificationScope === scope) dismissToast(item);
      });
  }
  const modal = document.querySelector("dialog[data-panel-dialog][open]");
  if (modal && isError) {
    let notice = modal.querySelector(".dialog-notice");
    if (!notice) {
      notice = document.createElement("div");
      notice.className = "dialog-notice inline-alert";
      notice.setAttribute("role", "alert");
      modal.querySelector(".panel-heading").after(notice);
    }
    showNotice(notice, message, { scope });
    notice.scrollIntoView({ block: "nearest" });
    return;
  }
  if (!isError) {
    elements.toast
      .querySelectorAll(".toast:not(.is-error)")
      .forEach((toast) => toast.remove());
  }
  const toast = document.createElement("div");
  toast.className = `toast${isError ? " is-error" : ""}`;
  if (scope) toast.dataset.notificationScope = scope;
  toast.setAttribute("role", isError ? "alert" : "status");
  toast.setAttribute("aria-live", isError ? "assertive" : "polite");

  const content = document.createElement("span");
  content.className = "notification-message";
  content.textContent = message;
  toast.append(content);

  if (isError) {
    const close = document.createElement("button");
    close.type = "button";
    close.className = "notification-close";
    close.textContent = "Close";
    close.setAttribute("aria-label", "Close error message");
    close.addEventListener("click", () => dismissToast(toast));
    toast.append(close);
  }

  elements.toast.append(toast);
  elements.toast.hidden = false;
  if (!isError) window.setTimeout(() => dismissToast(toast), 5000);
}

function dismissNoticeItem(element, item) {
  if (item?.parentElement !== element) return;
  item.remove();
  if (element.querySelector(".notice-item")) return;
  element.hidden = true;
  delete element.dataset.persistentError;
}

function showNotice(element, message, { title = "", scope = "" } = {}) {
  if (scope)
    element.querySelectorAll(".notice-item").forEach((item) => {
      if (item.dataset.notificationScope === scope)
        dismissNoticeItem(element, item);
    });
  const normalizedMessage = String(message);
  const duplicate = [...element.querySelectorAll(".notice-item")].some(
    (item) =>
      item.dataset.message === normalizedMessage &&
      item.dataset.title === title &&
      (item.dataset.notificationScope || "") === scope,
  );
  if (duplicate) {
    element.hidden = false;
    return;
  }

  const item = document.createElement("div");
  item.className = "notice-item";
  item.dataset.message = normalizedMessage;
  item.dataset.title = title;
  if (scope) item.dataset.notificationScope = scope;
  if (title) {
    const heading = document.createElement("strong");
    heading.textContent = title;
    item.append(heading);
  }
  const content = document.createElement("span");
  content.className = "notification-message";
  content.textContent = normalizedMessage;
  item.append(content);
  const close = document.createElement("button");
  close.type = "button";
  close.className = "notification-close";
  close.textContent = "Close";
  close.setAttribute("aria-label", "Close error message");
  close.addEventListener("click", () => dismissNoticeItem(element, item));
  item.append(close);

  element.append(item);
  element.dataset.persistentError = "true";
  element.hidden = false;
}

function clearNotice(element) {
  element.hidden = true;
  element.replaceChildren();
  delete element.dataset.persistentError;
}

function emptyRow(columns, message) {
  return `<tr class="empty-row"><td colspan="${columns}">${escapeHtml(message)}</td></tr>`;
}

function formatDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function shortId(value, size = 12) {
  const text = String(value || "-");
  return text.length > size ? `${text.slice(0, size)}...` : text;
}

function stateClass(state) {
  const normalized = String(state || "")
    .toLowerCase()
    .replaceAll("_", " ");
  if (
    /fail|error|cancel|unavailable|missing|invalid|incomplete|not completed|attention|blocked/.test(
      normalized,
    )
  )
    return "is-failed";
  if (["local", "recorded", "original recordings saved"].includes(normalized))
    return "is-local";
  if (
    ["ready", "prepared", "on cluster", "available", "images ready"].includes(
      normalized,
    ) ||
    /prepared formats?$/.test(normalized)
  )
    return "is-running";
  if (
    /converting|fetching|validating|transferring|verifying|checking/.test(
      normalized,
    )
  )
    return "is-pending";
  if (normalized.includes("unconfirmed")) return "is-pending";
  if (normalized.includes("idle")) return "is-idle";
  if (normalized.includes("mixed")) return "is-mixed";
  if (normalized.includes("running")) return "is-running";
  if (
    normalized.includes("complete") ||
    normalized.includes("success") ||
    normalized.includes("succeed")
  )
    return "is-running";
  if (normalized.includes("pending") || normalized.includes("queue"))
    return "is-pending";
  if (normalized.includes("preempt") || normalized.includes("pause"))
    return "is-preempted";
  return "is-other";
}

function statusPill(state) {
  const label = state || "unknown";
  return `<span class="state-pill ${stateClass(label)}">${escapeHtml(label)}</span>`;
}

function compactJson(value, limit = 100) {
  if (value === null || value === undefined || value === "") return "-";
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return text.length > limit ? `${text.slice(0, limit)}...` : text;
}

function numberOrNull(element) {
  const label =
    element.labels?.[0]?.textContent?.trim() ||
    element.name ||
    element.id ||
    "Numeric value";
  if (element.validity?.badInput) {
    const message = `${label} must be a valid number.`;
    element.setCustomValidity(message);
    throw new Error(message);
  }
  if (!element.value.trim()) {
    element.setCustomValidity("");
    return null;
  }
  const value = Number(element.value);
  if (!Number.isFinite(value)) {
    const message = `${label} must be a finite number.`;
    element.setCustomValidity(message);
    throw new Error(message);
  }
  element.setCustomValidity("");
  return value;
}

function selectedValues(select) {
  return [...select.selectedOptions]
    .map((option) => option.value)
    .filter(Boolean);
}

function setConnection(state, label) {
  elements.connectionDot.className = `connection-dot is-${state}`;
  elements.statusLabel.textContent = label;
}

function itemPartitions(item, field) {
  const value = Array.isArray(item[field])
    ? item[field].join(",")
    : item[field];
  return String(value || "")
    .split(",")
    .map((partition) => partition.trim())
    .filter(Boolean);
}

function quotaCell(metric) {
  const usage = Number(metric?.usage || 0);
  const rawLimit = metric?.limit;
  const hasLimit =
    rawLimit !== null &&
    rawLimit !== undefined &&
    rawLimit !== "" &&
    Number.isFinite(Number(rawLimit));
  const limit = hasLimit ? Number(rawLimit) : null;
  const ratio =
    hasLimit && limit > 0
      ? Math.min(100, Math.round((usage / limit) * 100))
      : 0;
  const overLimit = hasLimit && usage > limit;
  const label = hasLimit ? `${usage} / ${limit}` : `${usage}`;
  return `
    <div class="quota-cell ${overLimit ? "is-over" : ""}">
      <span>${label}</span>
      ${hasLimit && limit > 0 ? `<div class="quota-track"><i style="width:${ratio}%"></i></div>` : ""}
    </div>`;
}

const allocationColors = [
  "#386cb0",
  "#1f8a5b",
  "#cb5a30",
  "#8065a5",
  "#c84662",
  "#277b88",
  "#80731f",
];

function userColor(username) {
  let hash = 0;
  for (const character of String(username || "other")) {
    hash = ((hash << 5) - hash + character.charCodeAt(0)) | 0;
  }
  return allocationColors[Math.abs(hash) % allocationColors.length];
}

function gpuAllocationCell(metric) {
  const usage = Number(metric?.usage || 0);
  const rawLimit = metric?.limit;
  const hasLimit =
    rawLimit !== null &&
    rawLimit !== undefined &&
    rawLimit !== "" &&
    Number.isFinite(Number(rawLimit));
  const limit = hasLimit ? Number(rawLimit) : null;
  const users = Array.isArray(metric?.users) ? metric.users : [];
  const assigned = users.reduce(
    (total, user) => total + Number(user.usage || 0),
    0,
  );
  const units = Math.max(1, usage, assigned, limit || 0);
  const other = Math.max(0, usage - assigned);
  const free = hasLimit ? Math.max(0, limit - Math.max(usage, assigned)) : 0;
  const overLimit = hasLimit && usage > limit;
  const label = hasLimit ? `${usage} / ${limit}` : `${usage}`;
  const segments = users.map((user) => {
    const count = Number(user.usage || 0);
    const width = (count / units) * 100;
    const name = String(user.name || "");
    const elapsed = user.elapsed || "unknown";
    return `<span class="allocation-segment" style="width:${width}%;background:${userColor(name)}" title="${escapeHtml(name)}: ${count} GPU${count === 1 ? "" : "s"} • elapsed ${escapeHtml(elapsed)}">${escapeHtml(name)}<b>${count}</b></span>`;
  });
  if (other) {
    segments.push(
      `<span class="allocation-segment is-other" style="width:${(other / units) * 100}%" title="Other or snapshot delta: ${other} GPUs">other<b>${other}</b></span>`,
    );
  }
  if (free) {
    segments.push(
      ...Array.from(
        { length: free },
        () =>
          `<span class="allocation-free-slot" style="width:${100 / units}%" title="Available GPU"></span>`,
      ),
    );
  }
  return `
    <div class="gpu-allocation ${overLimit ? "is-over" : ""}">
      <div class="allocation-meta"><span>${label}</span><span>${hasLimit ? (limit === 0 ? "No GPU quota" : `${Math.max(0, limit - usage)} free`) : "Limit not reported"}</span></div>
      <div class="allocation-bar" aria-label="${label} GPUs allocated">${segments.join("")}</div>
    </div>`;
}

function renderAccountUsage(rows) {
  if (!rows.length) {
    elements.usageBody.innerHTML = emptyRow(4, "Account usage is unavailable.");
    return;
  }
  elements.usageBody.innerHTML = rows
    .map(
      (row) => `
    <tr class="${row.account === "rl2-lab" ? "is-own-account" : ""}">
      <td><span class="node-name">${escapeHtml(row.account)}</span></td>
      <td>${gpuAllocationCell(row.l40s)}</td>
      <td>${gpuAllocationCell(row.a40)}</td>
      <td>${gpuAllocationCell(row.rtx_6000)}</td>
    </tr>`,
    )
    .join("");
}

function updateFilterLabel(key) {
  const filter = queueFilters[key];
  if (!filter) return;
  if (filter.selectionError) {
    filter.label.textContent = "Selection unavailable";
    filter.label.title = filter.selectionError;
  } else if (filter.all) filter.label.textContent = filter.allLabel;
  else if (!filter.selected.size) filter.label.textContent = filter.noneLabel;
  else if (filter.selected.size === 1)
    filter.label.textContent = [...filter.selected][0];
  else
    filter.label.textContent = `${filter.selected.size} ${filter.countSuffix}`;
  if (!filter.selectionError) filter.label.removeAttribute("title");
}

function renderFilterOptions(key) {
  const filter = queueFilters[key];
  if (!filter) return;
  filter.options.innerHTML = `
    <div class="filter-actions">
      <button type="button" data-action="all">All</button>
      <button type="button" data-action="none">None</button>
    </div>
    ${filter.available
      .map(
        (partition) => `
      <label class="filter-option">
        <input type="checkbox" value="${escapeHtml(partition)}" ${filter.all || filter.selected.has(partition) ? "checked" : ""}>
        <span>${escapeHtml(partition)}</span>
      </label>`,
      )
      .join("")}`;

  filter.options
    .querySelector('[data-action="all"]')
    .addEventListener("click", () => {
      filter.selectionError = "";
      filter.all = true;
      filter.selected.clear();
      renderFilterOptions(key);
      applyPartitionFilters();
    });
  filter.options
    .querySelector('[data-action="none"]')
    .addEventListener("click", () => {
      filter.selectionError = "";
      filter.all = false;
      filter.selected.clear();
      renderFilterOptions(key);
      applyPartitionFilters();
    });
  filter.options
    .querySelectorAll('input[type="checkbox"]')
    .forEach((checkbox) => {
      checkbox.addEventListener("change", () => {
        filter.selectionError = "";
        if (filter.all) {
          filter.selected = new Set(filter.available);
          filter.all = false;
        }
        if (checkbox.checked) filter.selected.add(checkbox.value);
        else filter.selected.delete(checkbox.value);
        if (filter.selected.size === filter.available.length) {
          filter.all = true;
          filter.selected.clear();
        }
        updateFilterLabel(key);
        applyPartitionFilters();
      });
    });
  updateFilterLabel(key);
}

function syncFilterOptions(key, items, field) {
  const filter = queueFilters[key];
  if (!filter) return;
  filter.available = [
    ...new Set(items.flatMap((item) => itemPartitions(item, field))),
  ].sort();
  if (!filter.all) {
    const unavailable = [...filter.selected].filter(
      (value) => !filter.available.includes(value),
    );
    filter.selectionError = unavailable.length
      ? `Previously selected ${key}${unavailable.length === 1 ? "" : "s"} unavailable in the refreshed queue: ${unavailable.join(", ")}. Choose All, None, or an available value.`
      : "";
    filter.selected = new Set(
      [...filter.selected].filter((value) => filter.available.includes(value)),
    );
  }
  renderFilterOptions(key);
}

function matchesFilter(key, item, field) {
  const filter = queueFilters[key];
  if (!filter) return true;
  return (
    filter.all ||
    itemPartitions(item, field).some((value) => filter.selected.has(value))
  );
}

function applyPartitionFilters() {
  if (!latestSnapshot) return;
  const pendingJobs = (latestSnapshot.jobs || []).filter(
    (job) => String(job.state || "").toUpperCase() === "PENDING",
  );
  const jobs = pendingJobs.filter(
    (job) =>
      matchesFilter("partition", job, "partition") &&
      matchesFilter("account", job, "account"),
  );
  renderJobs(jobs, pendingJobs.length);
}

let jobPage = 0;
let visibleQueueJobs = [];
let queueTotal = 0;
const JOB_PAGE_SIZE = 100;
function renderJobs(jobs, totalJobs = jobs.length) {
  visibleQueueJobs = jobs;
  queueTotal = totalJobs;
  jobPage = Math.min(
    jobPage,
    Math.max(0, Math.ceil(jobs.length / JOB_PAGE_SIZE) - 1),
  );
  const start = jobPage * JOB_PAGE_SIZE;
  const end = Math.min(start + JOB_PAGE_SIZE, jobs.length);
  document.querySelector("#jobs-previous").disabled = jobPage === 0;
  document.querySelector("#jobs-next").disabled = end >= jobs.length;
  document.querySelector("#jobs-page").textContent = jobs.length
    ? `${start + 1}–${end} of ${jobs.length} matching jobs`
    : "No matching jobs";
  if (!jobs.length) {
    const message = totalJobs
      ? "No pending jobs match the selected filters."
      : "No pending jobs in the Slurm queue.";
    elements.jobsBody.innerHTML = emptyRow(6, message);
    elements.jobSummary.textContent = totalJobs
      ? `Showing 0 of ${totalJobs} pending jobs.`
      : "No pending jobs in the Slurm queue.";
    return;
  }

  elements.jobSummary.textContent =
    jobs.length === totalJobs
      ? `${jobs.length} pending jobs in the queue.`
      : `Showing ${jobs.length} of ${totalJobs} pending jobs.`;

  elements.jobsBody.innerHTML = jobs
    .slice(start, end)
    .map((job) => {
      const nodeCount = Number.isFinite(Number(job.node_count))
        ? Number(job.node_count)
        : 0;
      const nodes = `${nodeCount} node${nodeCount === 1 ? "" : "s"}`;
      return `
      <tr>
        <td><span class="job-id">#${escapeHtml(job.id)}</span><span class="secondary">${escapeHtml(job.name)}</span></td>
        <td>${escapeHtml(job.user)}</td>
        <td>${statusPill(job.state)}</td>
        <td>${escapeHtml(nodes)}</td>
        <td>${escapeHtml(job.partition || "-")}</td>
        <td>${escapeHtml(job.account || "-")}</td>
      </tr>`;
    })
    .join("");
}

function renderSnapshot(snapshot, { background = false } = {}) {
  const pendingJobs = (snapshot.jobs || []).filter(
    (job) => String(job.state || "").toUpperCase() === "PENDING",
  );
  const panel =
    elements.jobsBody.closest('[data-tab-panel="cluster"]') ||
    elements.jobsBody.closest(".panel") ||
    elements.jobsBody;
  commitPanelRefresh(
    panel,
    "cluster-snapshot",
    {
      gateway: snapshot.gateway,
      account_usage: snapshot.account_usage || [],
      jobs: pendingJobs,
    },
    () => {
      latestSnapshot = { ...snapshot, jobs: pendingJobs };
      renderAccountUsage(snapshot.account_usage || []);
      syncFilterOptions("partition", pendingJobs, "partition");
      syncFilterOptions("account", pendingJobs, "account");
      applyPartitionFilters();
      clearNotice(elements.errorBanner);
      setConnection("online", `Connected through ${snapshot.gateway}`);
      setTextIfChanged(
        elements.lastUpdated,
        `Updated ${new Intl.DateTimeFormat(undefined, {
          hour: "numeric",
          minute: "2-digit",
          second: "2-digit",
        }).format(new Date())}`,
      );
    },
    { background },
  );
}

function clusterRefreshAllowed() {
  return document.visibilityState === "visible" && activeTab === "cluster";
}

function stopClusterAutoRefresh() {
  clearTimeout(refreshTimer);
  refreshTimer = null;
}

function scheduleClusterAutoRefresh() {
  stopClusterAutoRefresh();
  if (!clusterRefreshAllowed()) return;
  refreshTimer = window.setTimeout(
    () => refreshCluster({ background: true }),
    15000,
  );
}

async function refreshCluster({ force = false, background = false } = {}) {
  stopClusterAutoRefresh();
  if (!force && !clusterRefreshAllowed()) return;
  if (!background) {
    elements.refreshButton.disabled = true;
    setConnection("loading", "Contacting Slurm...");
  }
  try {
    const gateway = encodeURIComponent(elements.gateway.value);
    const snapshot = await api(`/api/cluster?gateway=${gateway}`);
    renderSnapshot(snapshot, { background });
  } catch (error) {
    const panel =
      elements.jobsBody.closest('[data-tab-panel="cluster"]') ||
      elements.jobsBody.closest(".panel") ||
      elements.jobsBody;
    commitPanelRefresh(
      panel,
      "cluster-snapshot",
      { error: error.message },
      () => {
        setConnection("offline", "Cluster connection failed");
        showNotice(elements.errorBanner, error.message, {
          title: "Cluster unavailable",
        });
      },
      { background },
    );
  } finally {
    if (!background) elements.refreshButton.disabled = false;
    scheduleClusterAutoRefresh();
  }
}

function setSelectOptions(select, rows, selectedValue) {
  const options = rows
    .map((row) => {
      const id = row.id || row.slug || row.name;
      const label = row.label || row.display_name || row.name || id;
      return `<option value="${escapeHtml(id)}">${escapeHtml(label)}</option>`;
    })
    .join("");
  select.innerHTML = options;
  if (
    selectedValue &&
    [...select.options].some((option) => option.value === selectedValue)
  ) {
    select.value = selectedValue;
  }
}

function adapterId(adapter) {
  return String(adapter?.id || adapter?.slug || adapter?.seed_key || "");
}

function adapterVersion(adapter) {
  return (
    adapter?.selected_version?.version_number ??
    adapter?.latest_version?.version_number ??
    adapter?.version_number ??
    adapter?.version ??
    "-"
  );
}

function adapterVersionId(adapter) {
  return String(
    adapter?.selected_version?.id ??
      adapter?.latest_version?.id ??
      adapter?.selected_version_id ??
      adapter?.latest_version_id ??
      adapter?.adapter_version_id ??
      adapter?.version_id ??
      "",
  ).trim();
}

function adapterOptionId(adapter) {
  const versionId = adapterVersionId(adapter);
  if (versionId) return `adapter-version:${versionId}`;
  return `adapter-version:${adapterId(adapter)}:${adapterVersion(adapter)}`;
}

function adapterManifest(adapter) {
  const raw =
    adapter?.selected_version?.manifest ??
    adapter?.latest_version?.manifest ??
    adapter?.manifest;
  if (raw === undefined || raw === null) {
    return { __parse_error: "Adapter manifest is missing." };
  }
  if (typeof raw === "string") {
    try {
      return JSON.parse(raw);
    } catch (error) {
      return {
        __parse_error: `Invalid adapter manifest JSON: ${error.message}`,
      };
    }
  }
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return { __parse_error: "Adapter manifest must be a JSON object." };
  }
  return raw;
}

function adapterArchived(adapter) {
  return Boolean(
    adapter?.archived ||
      adapter?.is_archived ||
      adapter?.status === "archived" ||
      adapter?.archived_at,
  );
}

function adapterAvailableForExperiment(adapter) {
  const manifest = adapterManifest(adapter);
  return (
    !adapterArchived(adapter) &&
    adapter?.manifest_valid !== false &&
    !manifest.__parse_error &&
    manifest.schema_version === "skynet.adapter/v1" &&
    typeof manifest.slug === "string" &&
    Boolean(manifest.slug.trim()) &&
    typeof manifest.display_name === "string" &&
    Boolean(manifest.display_name.trim())
  );
}

function firstValue(...values) {
  return values.find(
    (value) => value !== undefined && value !== null && value !== "",
  );
}

function repositoryDefaults(manifest, adapter = {}) {
  const repository = manifest.repository;
  const source = manifest.source;
  const repositoryObject =
    repository && typeof repository === "object" ? repository : {};
  const sourceObject = source && typeof source === "object" ? source : {};
  return {
    url:
      firstValue(
        typeof repository === "string" ? repository : null,
        repositoryObject.default_url,
        repositoryObject.url,
        repositoryObject.repository,
        sourceObject.repository,
        sourceObject.url,
        manifest.repository_url,
        manifest.default_source,
        manifest.default_repository,
        adapter.repository_url,
        adapter.repository,
      ) || "",
    workdir:
      firstValue(
        manifest.project_subdirectory,
        manifest.defaults?.project_subdirectory,
        manifest.defaults?.source?.project_subdirectory,
        manifest.defaults?.source?.workdir,
      ) || ".",
  };
}

function manifestDefault(manifest, section, key) {
  const configured = manifest.defaults?.[section]?.[key];
  if (configured !== undefined) return configured;
  const descriptor = manifest[section]?.[key];
  if (
    descriptor &&
    typeof descriptor === "object" &&
    !Array.isArray(descriptor) &&
    descriptor.default !== undefined
  ) {
    return descriptor.default;
  }
  return undefined;
}

function manifestCapabilityLabel(manifest) {
  const capabilities = manifest.capabilities;
  if (Array.isArray(capabilities)) return capabilities.join(", ");
  if (capabilities && typeof capabilities === "object") {
    const labels = [
      capabilities.supports_multi_gpu_single_node
        ? "single-node multi-GPU"
        : "single GPU",
      capabilities.supports_resume ? "resume" : "no resume",
      capabilities.supports_evaluation_resume ? "evaluation resume" : null,
      Array.isArray(capabilities.evaluation_adapters) &&
      capabilities.evaluation_adapters.length
        ? `eval: ${capabilities.evaluation_adapters.join(", ")}`
        : null,
    ].filter(Boolean);
    if (labels.length) return labels.join(" / ");
  }
  return (
    [manifest.train ? "train" : null, manifest.evaluation ? "evaluation" : null]
      .filter(Boolean)
      .join(", ") || "No capabilities declared"
  );
}

function adapterRuntimeDetection(manifest) {
  const detection =
    manifest.runtime_detection ??
    manifest.runtime?.detection ??
    manifest.runtime;
  if (typeof detection === "string") return detection;
  if (Array.isArray(detection)) return detection.join(", ");
  if (detection && typeof detection === "object") {
    const allowed = Array.isArray(detection.allowed_backends)
      ? detection.allowed_backends
      : detection.allowed_backends instanceof Set
        ? [...detection.allowed_backends]
        : [];
    if (allowed.length) {
      return `${detection.recommended_backend ? `recommended ${detection.recommended_backend}; ` : ""}allowed ${allowed.join(", ")}`;
    }
    return (
      firstValue(
        detection.mode,
        detection.strategy,
        detection.type,
        Object.keys(detection).join(", "),
      ) || "configured"
    );
  }
  return "not declared";
}

function selectedAdapter() {
  if (
    loadedExperimentAdapterSnapshot &&
    adapterOptionId(loadedExperimentAdapterSnapshot) ===
      elements.experimentAdapter.value
  )
    return loadedExperimentAdapterSnapshot;
  return (
    adapterRows.find(
      (adapter) =>
        adapterOptionId(adapter) === elements.experimentAdapter.value,
    ) || null
  );
}

const BATCH_SEMANTIC_LABELS = {
  per_device: "Per device",
  global_before_accumulation: "Global before accumulation",
  global_effective: "Global effective",
};

function selectedBatchCompatibility() {
  const manifest = selectedAdapter()
    ? adapterManifest(selectedAdapter())
    : null;
  const compatibility = manifest?.train?.batch_compatibility;
  if (
    !compatibility ||
    compatibility.schema_version !== "skynet.batch-compatibility/v1" ||
    !Array.isArray(compatibility.allowed_semantics)
  )
    return null;
  return compatibility;
}

function currentAdapterConditionValue(path) {
  const field = adapterInputFields().find(
    (candidate) => candidate.path === path,
  );
  if (field) {
    const control = document.getElementById(adapterFieldControlId(path));
    if (control) {
      const parsed = parseAdapterDeclaredValue(control, field);
      if (parsed.present && !parsed.error) return parsed.value;
    }
  }
  const commonValues = {
    "source.repository": elements.experimentSource.value.trim(),
    "source.revision": elements.experimentRevision.value.trim(),
    "source.project_subdirectory":
      elements.experimentWorkdir.value.trim() || ".",
    "train.learning_rate": numberOrNull(elements.hpLearningRate),
    "train.batch.declared_semantics":
      elements.hpBatchSemantics.value || "per_device",
    "train.batch.value": numberOrNull(elements.hpBatchSize),
    "train.batch.gradient_accumulation_steps":
      numberOrNull(elements.hpGradAcc) ?? 1,
    "train.num_workers_per_rank": numberOrNull(elements.hpNumWorkers),
    "train.max_steps": numberOrNull(elements.hpMaxSteps),
    "train.precision":
      elements.hpPrecision.value === "adapter-default"
        ? null
        : elements.hpPrecision.value,
    "resources.gpu.mode":
      elements.gpuMode.value === "manual" ? "explicit" : "auto",
    "resources.gpu.count":
      elements.gpuMode.value === "manual"
        ? numberOrNull(elements.experimentGpuCount)
        : null,
    "resources.gpu.gpu_type":
      elements.experimentGpuType.value === "auto"
        ? "any"
        : elements.experimentGpuType.value,
  };
  return commonValues[path];
}

function adapterConditionValuesEqual(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function matchingGpuRecommendation(manifest) {
  const recommendations = manifest?.defaults?.resources?.gpu_recommendations;
  if (!Array.isArray(recommendations)) return null;
  return (
    recommendations.find((recommendation) => {
      const conditions = recommendation?.enabled_when;
      return (
        conditions &&
        typeof conditions === "object" &&
        !Array.isArray(conditions) &&
        Object.entries(conditions).every(
          ([path, choices]) =>
            Array.isArray(choices) &&
            choices.some((choice) =>
              adapterConditionValuesEqual(
                currentAdapterConditionValue(path),
                choice,
              ),
            ),
        )
      );
    }) || null
  );
}

function resolvedExperimentGpuCount() {
  if (elements.gpuMode.value === "manual") {
    const count = Number(elements.experimentGpuCount.value);
    return Number.isInteger(count) && count > 0 ? count : null;
  }
  const adapter = selectedAdapter();
  const manifest = adapter ? adapterManifest(adapter) : null;
  if (!manifest || manifest.__parse_error) return null;
  const profile = manifest.defaults?.resources?.gpu_profile || "recommended";
  const capabilityField =
    {
      minimum: "minimum_gpus",
      recommended: "recommended_gpus",
      "maximum-throughput": "maximum_gpus",
    }[profile] || "recommended_gpus";
  const capabilities =
    matchingGpuRecommendation(manifest) || manifest.capabilities || {};
  const count = Number(capabilities[capabilityField]);
  return Number.isInteger(count) && count > 0 ? count : null;
}

function setBatchControlValidity(control, message) {
  control.setCustomValidity(message);
  if (message) control.setAttribute("aria-invalid", "true");
  else control.removeAttribute("aria-invalid");
}

function updateBatchCompatibility() {
  const compatibility = selectedBatchCompatibility();
  const batchRaw = elements.hpBatchSize.value.trim();
  const accumulationRaw = elements.hpGradAcc.value.trim();
  const batchExplicit = batchRaw !== "";
  const accumulationExplicit = accumulationRaw !== "";
  const batchSize = Number(batchRaw);
  const accumulation = Number(accumulationRaw);
  const gpuCount = resolvedExperimentGpuCount();
  const allowed = compatibility
    ? (gpuCount > 1 && Array.isArray(compatibility.multi_gpu_allowed_semantics)
        ? compatibility.multi_gpu_allowed_semantics
        : compatibility.allowed_semantics
      ).filter((semantic) => Object.hasOwn(BATCH_SEMANTIC_LABELS, semantic))
    : Object.keys(BATCH_SEMANTIC_LABELS);
  const allowedSet = new Set(allowed);
  const defaultSemantic = "per_device";

  if (
    compatibility &&
    batchExplicit &&
    !elements.hpBatchSemantics.value &&
    !allowedSet.has(defaultSemantic) &&
    allowed.length === 1
  ) {
    elements.hpBatchSemantics.value = allowed[0];
  }

  const selectedSemantic = elements.hpBatchSemantics.value;
  const resolvedSemantic = selectedSemantic || defaultSemantic;
  [...elements.hpBatchSemantics.options].forEach((option) => {
    if (option.value && !Object.hasOwn(BATCH_SEMANTIC_LABELS, option.value))
      return;
    option.disabled = Boolean(
      compatibility &&
        batchExplicit &&
        (option.value
          ? !allowedSet.has(option.value)
          : !allowedSet.has(defaultSemantic)),
    );
  });

  const batchErrors = [];
  const semanticErrors = [];
  const accumulationErrors = [];
  if (batchExplicit && (!Number.isInteger(batchSize) || batchSize < 1)) {
    batchErrors.push("Batch size must be a positive whole number.");
  }
  if (
    accumulationExplicit &&
    (!Number.isInteger(accumulation) || accumulation < 1)
  ) {
    accumulationErrors.push(
      "Gradient accumulation must be a positive whole number.",
    );
  }

  if (
    compatibility &&
    batchExplicit &&
    Number.isInteger(batchSize) &&
    batchSize > 0
  ) {
    const allowedLabels = allowed
      .map((semantic) => BATCH_SEMANTIC_LABELS[semantic])
      .join(" or ");
    if (
      !gpuCount &&
      compatibility.batch_size_divisible_by === "resolved_gpu_count"
    ) {
      batchErrors.push(
        "Enter a valid GPU allocation so batch divisibility can be checked.",
      );
    }
    if (!allowedSet.has(resolvedSemantic)) {
      semanticErrors.push(
        selectedSemantic
          ? `${BATCH_SEMANTIC_LABELS[selectedSemantic] || selectedSemantic} batch semantics are not supported with ${gpuCount || "the resolved number of"} GPUs; choose ${allowedLabels}.`
          : `Choose batch semantics for an explicit batch size: this adapter allows ${allowedLabels} with ${gpuCount || "the resolved number of"} GPUs.`,
      );
    }
    if (
      compatibility.batch_size_divisible_by === "resolved_gpu_count" &&
      gpuCount &&
      batchSize % gpuCount !== 0
    ) {
      const lower = Math.max(
        gpuCount,
        Math.floor(batchSize / gpuCount) * gpuCount,
      );
      const upper = Math.ceil(batchSize / gpuCount) * gpuCount;
      const suggestions =
        lower === upper ? String(lower) : `${lower} or ${upper}`;
      batchErrors.push(
        `Batch size ${batchSize} must be divisible by ${gpuCount} resolved GPUs; use ${suggestions}, or another multiple of ${gpuCount}.`,
      );
    }
  }
  if (
    compatibility &&
    accumulationExplicit &&
    Number.isInteger(accumulation) &&
    accumulation > 0 &&
    compatibility.supports_gradient_accumulation === false &&
    accumulation !== 1
  ) {
    accumulationErrors.push(
      "This adapter does not support gradient accumulation; leave it blank for the repository default or set it to 1.",
    );
  }

  setBatchControlValidity(elements.hpBatchSize, batchErrors.join(" "));
  setBatchControlValidity(elements.hpBatchSemantics, semanticErrors.join(" "));
  setBatchControlValidity(elements.hpGradAcc, accumulationErrors.join(" "));
  const errors = [...batchErrors, ...semanticErrors, ...accumulationErrors];
  elements.hpBatchCompatibility.classList.toggle(
    "is-invalid",
    errors.length > 0,
  );
  elements.hpBatchCompatibility.classList.toggle(
    "is-valid",
    errors.length === 0 && (batchExplicit || accumulationExplicit),
  );
  elements.hpBatchCompatibility.classList.toggle(
    "is-pending",
    errors.length === 0 && !batchExplicit && !accumulationExplicit,
  );
  if (errors.length) {
    elements.hpBatchCompatibility.textContent = errors.join(" ");
  } else if (!batchExplicit && !accumulationExplicit) {
    elements.hpBatchCompatibility.textContent =
      "Leave batch fields blank to preserve the displayed defaults.";
  } else if (compatibility && batchExplicit) {
    const semantics = selectedSemantic
      ? BATCH_SEMANTIC_LABELS[resolvedSemantic]
      : `repository-default ${BATCH_SEMANTIC_LABELS[resolvedSemantic]}`;
    const divisor =
      compatibility.batch_size_divisible_by === "resolved_gpu_count" && gpuCount
        ? ` Batch size ${batchSize} is divisible across ${gpuCount} resolved GPU${gpuCount === 1 ? "" : "s"}.`
        : "";
    const accumulationStatus =
      compatibility.supports_gradient_accumulation === false
        ? " Gradient accumulation is fixed at 1."
        : accumulationExplicit
          ? ` Gradient accumulation is ${accumulation}.`
          : "";
    elements.hpBatchCompatibility.textContent = `${semantics} semantics are valid.${divisor}${accumulationStatus}`;
  } else {
    elements.hpBatchCompatibility.textContent = batchExplicit
      ? "This adapter declares no additional batch compatibility constraints."
      : `Gradient accumulation ${accumulation} is valid; batch size remains at the displayed default.`;
  }
  return { valid: errors.length === 0, errors };
}

let adapterDefaultErrors = [];
const canonicalControlValues = new WeakMap();
const canonicalCheckboxValues = new WeakMap();

function setManifestDefault(element, value) {
  if (!canonicalControlValues.has(element))
    canonicalControlValues.set(element, element.value);
  element
    .querySelectorAll?.("option[data-invalid-adapter-default]")
    .forEach((option) => option.remove());
  element.setCustomValidity("");
  delete element.dataset.adapterDefaultError;
  if (value === undefined || value === null || value === "") {
    element.value = canonicalControlValues.get(element);
    return;
  }
  const normalized = String(value);
  if (
    element instanceof HTMLSelectElement &&
    ![...element.options].some((option) => option.value === normalized)
  ) {
    const label =
      element.labels?.[0]?.textContent?.trim() ||
      element.name ||
      element.id ||
      "field";
    const message = `Adapter default "${normalized}" is unavailable for ${label}.`;
    element.dataset.adapterDefaultError = message;
    element.setCustomValidity(message);
    element.insertAdjacentHTML(
      "afterbegin",
      `<option value="${escapeHtml(normalized)}" data-invalid-adapter-default disabled>${escapeHtml(message)}</option>`,
    );
    element.value = normalized;
    adapterDefaultErrors.push(message);
    return;
  }
  element.value = normalized;
}

function setManifestCheckbox(element, value) {
  if (!canonicalCheckboxValues.has(element))
    canonicalCheckboxValues.set(element, element.checked);
  element.checked =
    typeof value === "boolean" ? value : canonicalCheckboxValues.get(element);
}

const canonicalPrecisionLabels = Object.freeze({
  bf16: "BF16",
  fp16: "FP16",
  fp32: "FP32",
});
let activePrecisionSupport = new Set();
let activePrecisionBindingDeclared = false;

function supportedCanonicalFields(manifest) {
  const declared = Array.isArray(manifest?.train?.supported_canonical_fields)
    ? manifest.train.supported_canonical_fields
    : [];
  const bindings = manifest?.train?.parameter_flags;
  const mapped =
    bindings && typeof bindings === "object" && !Array.isArray(bindings)
      ? Object.keys(bindings)
      : [];
  return new Set([...declared, ...mapped]);
}

function canonicalFieldIsSupported(path, supported) {
  return [...supported].some(
    (item) => path === item || path.startsWith(`${item}.`),
  );
}

function configureCanonicalHyperparameterFields(manifest) {
  const supported = supportedCanonicalFields(manifest);
  [
    [elements.hpLearningRate, "train.learning_rate"],
    [elements.hpBatchSize, "train.batch.value"],
    [elements.hpGradAcc, "train.batch.gradient_accumulation_steps"],
    [elements.hpNumWorkers, "train.num_workers_per_rank"],
    [elements.hpMaxSteps, "train.max_steps"],
    [elements.checkpointSaveSteps, "train.checkpoint.save_every_steps"],
  ].forEach(([control, path]) => {
    const enabled = canonicalFieldIsSupported(path, supported);
    control.disabled = !enabled;
    control.title = enabled
      ? ""
      : `The selected adapter does not map ${path}. Add a versioned adapter mapping to enable it.`;
  });
}

function precisionBinding(manifest) {
  const bindings = manifest?.train?.parameter_flags;
  if (!bindings || typeof bindings !== "object" || Array.isArray(bindings))
    return null;
  const binding = bindings["train.precision"];
  return binding && typeof binding === "object" && !Array.isArray(binding)
    ? binding
    : null;
}

function precisionSupportFromManifest(manifest) {
  const binding = precisionBinding(manifest);
  if (!binding) {
    const declared = supportedCanonicalFields(manifest);
    return canonicalFieldIsSupported("train.precision", declared)
      ? {
          bindingDeclared: true,
          supported: new Set(Object.keys(canonicalPrecisionLabels)),
        }
      : { bindingDeclared: false, supported: new Set() };
  }
  const valueMap = binding.value_map;
  if (
    !valueMap ||
    typeof valueMap !== "object" ||
    Array.isArray(valueMap) ||
    Object.keys(valueMap).length === 0
  ) {
    return {
      bindingDeclared: true,
      supported: new Set(Object.keys(canonicalPrecisionLabels)),
    };
  }
  return {
    bindingDeclared: true,
    supported: new Set(
      Object.keys(valueMap).filter(
        (value) => value in canonicalPrecisionLabels,
      ),
    ),
  };
}

function precisionSupportLabel() {
  const labels = [...activePrecisionSupport]
    .map((value) => canonicalPrecisionLabels[value])
    .filter(Boolean);
  return labels.length ? labels.join(", ") : "none";
}

function updatePrecisionStatus(resetValue = "") {
  if (resetValue) {
    const label = canonicalPrecisionLabels[resetValue] || resetValue;
    elements.hpPrecisionStatus.textContent = `${label} is not supported by this adapter. Choose Default or a supported override: ${precisionSupportLabel()}.`;
    return;
  }
  if (!activePrecisionBindingDeclared) {
    elements.hpPrecisionStatus.textContent =
      "This adapter does not declare canonical precision overrides; leave Default selected to preserve the displayed value.";
    return;
  }
  elements.hpPrecisionStatus.textContent = `Supported overrides: ${precisionSupportLabel()}. Leave Default selected to preserve the displayed value.`;
}

function enforceSupportedPrecision() {
  const selected = elements.hpPrecision.value;
  if (selected !== "adapter-default" && !activePrecisionSupport.has(selected)) {
    const message = `${canonicalPrecisionLabels[selected] || selected} is not supported by the selected adapter.`;
    elements.hpPrecision.setCustomValidity(message);
    elements.hpPrecision.setAttribute("aria-invalid", "true");
    updatePrecisionStatus(selected);
    return selected;
  }
  elements.hpPrecision.setCustomValidity("");
  elements.hpPrecision.removeAttribute("aria-invalid");
  updatePrecisionStatus();
  return selected;
}

function configurePrecisionField(manifest) {
  const previous = elements.hpPrecision.value || "adapter-default";
  const support = precisionSupportFromManifest(manifest);
  activePrecisionSupport = support.supported;
  activePrecisionBindingDeclared = support.bindingDeclared;

  [...elements.hpPrecision.options].forEach((option) => {
    if (option.value === "adapter-default") {
      option.disabled = false;
      option.textContent = "Default: not declared";
      return;
    }
    const label = canonicalPrecisionLabels[option.value] || option.value;
    option.disabled = !activePrecisionSupport.has(option.value);
    option.textContent = option.disabled
      ? `${label} (unsupported by adapter)`
      : label;
  });

  const declaredDefault = manifestDefault(
    manifest,
    "hyperparameters",
    "precision",
  );
  elements.hpPrecision.value = previous;
  enforceSupportedPrecision();
  if (declaredDefault && !activePrecisionSupport.has(String(declaredDefault))) {
    const message = `Adapter default precision "${declaredDefault}" is unsupported by its declared precision mapping.`;
    elements.hpPrecision.dataset.adapterDefaultError = message;
    elements.hpPrecision.setCustomValidity(message);
    adapterDefaultErrors.push(message);
  }
}

elements.hpPrecision.addEventListener("change", enforceSupportedPrecision);

function nativeDefaultsText(value) {
  if (Array.isArray(value)) return value.map(String).join("\n");
  if (value && typeof value === "object") {
    return Object.entries(value)
      .map(([key, nested]) => `${key}=${JSON.stringify(nested)}`)
      .join("\n");
  }
  return typeof value === "string" ? value : "";
}

let repositoryInputOptionsState = null;

function declaredAdapterInputFields(adapter = selectedAdapter()) {
  const fields = adapter ? adapterManifest(adapter)?.train?.input_fields : null;
  return Array.isArray(fields)
    ? fields.filter((field) => field && typeof field.path === "string")
    : [];
}

function adapterInputOptionsScopeKey(
  adapter,
  repository,
  commit,
  projectSubdirectory,
) {
  const scope = adapterDeclaredScope(adapter);
  if (!scope) return "";
  return [
    repository,
    commit,
    projectSubdirectory,
    scope.stableId,
    scope.versionId,
    scope.manifestHash,
  ].join("\u001f");
}

function normalizeRepositoryInputOptions(payload) {
  const rawOptions = payload?.input_options;
  const normalized = new Map();
  if (
    !rawOptions ||
    typeof rawOptions !== "object" ||
    Array.isArray(rawOptions)
  )
    return normalized;
  Object.entries(rawOptions).forEach(([path, raw]) => {
    if (!path || !raw || typeof raw !== "object" || Array.isArray(raw)) return;
    normalized.set(path, {
      choices: Array.isArray(raw.choices) ? [...raw.choices] : [],
      metadata: raw.metadata,
      complete: raw.complete === true,
      source: raw.source || "",
      warnings: Array.isArray(raw.warnings)
        ? raw.warnings.map((warning) => String(warning)).filter(Boolean)
        : raw.warning
          ? [String(raw.warning)]
          : [],
    });
  });
  return normalized;
}

function activeRepositoryInputOptions(adapter = selectedAdapter()) {
  const state = repositoryInputOptionsState;
  const scope = adapterDeclaredScope(adapter);
  if (!state || !scope) return null;
  const repository = elements.experimentSource.value.trim();
  const revision = elements.experimentRevision.value.trim();
  const projectSubdirectory = elements.experimentWorkdir.value.trim() || ".";
  if (
    state.repository !== repository ||
    ![state.requestedRevision, state.resolvedCommit].includes(revision) ||
    state.projectSubdirectory !== projectSubdirectory ||
    state.adapterStableId !== scope.stableId ||
    state.adapterVersionId !== scope.versionId ||
    state.manifestHash !== scope.manifestHash
  )
    return null;
  return state.options;
}

const COMMON_HYPERPARAMETER_DEFAULT_FIELDS = Object.freeze([
  {
    path: "train.learning_rate",
    control: "hpLearningRate",
    status: "hpLearningRateDefault",
    manifestKeys: ["learning_rate"],
  },
  {
    path: "train.batch.value",
    control: "hpBatchSize",
    status: "hpBatchSizeDefault",
    manifestKeys: ["batch_size"],
  },
  {
    path: "train.batch.declared_semantics",
    control: "hpBatchSemantics",
    status: "hpBatchSemanticsDefault",
    manifestKeys: ["batch_semantics"],
    defaultOption: "",
  },
  {
    path: "train.batch.gradient_accumulation_steps",
    control: "hpGradAcc",
    status: "hpGradAccDefault",
    manifestKeys: ["gradient_accumulation_steps", "gradient_accumulation"],
  },
  {
    path: "train.num_workers_per_rank",
    control: "hpNumWorkers",
    status: "hpNumWorkersDefault",
    manifestKeys: ["num_workers_per_rank", "num_workers"],
  },
  {
    path: "train.precision",
    control: "hpPrecision",
    status: "hpPrecisionDefault",
    manifestKeys: ["precision"],
    defaultOption: "adapter-default",
  },
  {
    path: "train.max_steps",
    control: "hpMaxSteps",
    status: "hpMaxStepsDefault",
    manifestKeys: ["max_steps"],
  },
  {
    path: "train.checkpoint.save_every_steps",
    control: "checkpointSaveSteps",
    status: "checkpointSaveStepsDefault",
    manifestKeys: ["save_every_steps", "save_steps"],
    manifestSection: "checkpoint",
  },
]);

function displayCommonHyperparameterDefault(value) {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean")
    return String(value);
  return compactJson(value, 160);
}

function selectedRepositoryHyperparameterDefaults(adapter = selectedAdapter()) {
  const defaults = new Map();
  const options = activeRepositoryInputOptions(adapter);
  if (!options) return defaults;
  adapterInputFields(adapter).forEach((field) => {
    const option = options.get(field.path);
    const metadata = option?.metadata;
    if (
      !option?.complete ||
      !metadata ||
      typeof metadata !== "object" ||
      Array.isArray(metadata)
    )
      return;
    const control = document.getElementById(adapterFieldControlId(field.path));
    if (!control) return;
    const parsed = parseAdapterDeclaredValue(control, field);
    if (!parsed.present || parsed.error) return;
    const selected = metadata[String(parsed.value)];
    const values = selected?.values;
    const evidence = selected?.evidence;
    if (!values || typeof values !== "object" || Array.isArray(values)) return;
    Object.entries(values).forEach(([path, value]) => {
      if (value === undefined || value === null || value === "") return;
      defaults.set(path, {
        value,
        source: option.source,
        evidence:
          evidence && typeof evidence === "object" && !Array.isArray(evidence)
            ? evidence[path]
            : null,
        inputPath: field.path,
        choice: parsed.value,
        origin: "repository",
      });
    });
  });
  return defaults;
}

function manifestHyperparameterDefault(
  manifest,
  keys,
  section = "hyperparameters",
) {
  for (const key of keys) {
    const value = manifestDefault(manifest, section, key);
    if (value !== undefined && value !== null && value !== "") return value;
  }
  return undefined;
}

function resolvedCommonHyperparameterDefault(
  path,
  manifestKeys,
  adapter,
  repositoryDefaults,
  manifestSection = "hyperparameters",
) {
  const command = adapterManifest(adapter)?.train || {};
  const control = document.getElementById(
    adapterFieldControlId("native.config.training_preset"),
  );
  const preset =
    (command.presets || []).find(
      (p) => JSON.stringify(p.id) === control?.value,
    ) ||
    (!control?.value
      ? (command.presets || []).find((p) => p.id === command.default_preset)
      : null);
  if (
    preset?.values &&
    Object.prototype.hasOwnProperty.call(preset.values, path)
  )
    return { value: preset.values[path], origin: "preset", preset, adapter };
  const repositoryDefault = repositoryDefaults.get(path);
  if (repositoryDefault) return repositoryDefault;
  if (!adapter) return null;
  const value = manifestHyperparameterDefault(
    adapterManifest(adapter),
    manifestKeys,
    manifestSection,
  );
  if (value === undefined) return null;
  return { value, origin: "adapter", adapter };
}

function repositoryDefaultProvenance(record) {
  const evidence =
    record.evidence &&
    typeof record.evidence === "object" &&
    !Array.isArray(record.evidence)
      ? record.evidence
      : {};
  const source =
    record.source &&
    typeof record.source === "object" &&
    !Array.isArray(record.source)
      ? record.source
      : {};
  const file = firstValue(evidence.file, evidence.path, evidence.source_file);
  const location = file
    ? `${file}${Number.isInteger(evidence.line) ? `:${evidence.line}` : ""}`
    : "repository inspection metadata";
  const sourcePath = evidence.source_path ? ` (${evidence.source_path})` : "";
  const commit = source.commit
    ? ` at commit ${shortId(source.commit, 12)}`
    : "";
  const selectedBy = record.inputPath
    ? `, selected by ${record.inputPath}=${displayCommonHyperparameterDefault(record.choice)}`
    : "";
  return `Repository provenance: ${location}${sourcePath}${commit}${selectedBy}.`;
}

function adapterDefaultProvenance(record) {
  if (record.origin === "preset")
    return `Preset: ${record.preset.name} (${record.preset.id}).`;
  const adapter = record.adapter || {};
  const version = adapterVersion(adapter);
  const manifestHash = firstValue(
    adapter.selected_version?.manifest_sha256,
    adapter.latest_version?.manifest_sha256,
    adapter.manifest_sha256,
  );
  return `Adapter provenance: manifest v${version}${manifestHash ? ` (${shortId(manifestHash, 12)})` : ""}.`;
}

function commonHyperparameterDefaultText(record) {
  if (!record) return "Default: not declared.";
  return `Default: ${displayCommonHyperparameterDefault(record.value)}.`;
}

function renderCommonHyperparameterDefaults(adapter = selectedAdapter()) {
  const repositoryDefaults = selectedRepositoryHyperparameterDefaults(adapter);
  COMMON_HYPERPARAMETER_DEFAULT_FIELDS.forEach((definition) => {
    const control = elements[definition.control];
    const status = elements[definition.status];
    const record = resolvedCommonHyperparameterDefault(
      definition.path,
      definition.manifestKeys,
      adapter,
      repositoryDefaults,
      definition.manifestSection,
    );
    let sourceRecord = record;
    let statusText = commonHyperparameterDefaultText(record);
    let placeholderText = record
      ? `Default: ${displayCommonHyperparameterDefault(record.value)}`
      : "Default: not declared";
    if (definition.path === "train.max_steps" && !record) {
      const maxEpochs = resolvedCommonHyperparameterDefault(
        "train.max_epochs",
        ["max_epochs"],
        adapter,
        repositoryDefaults,
      );
      if (maxEpochs) {
        sourceRecord = maxEpochs;
        const provenance =
          maxEpochs.origin === "repository"
            ? repositoryDefaultProvenance(maxEpochs)
            : adapterDefaultProvenance(maxEpochs);
        placeholderText = `No max-steps default; ${displayCommonHyperparameterDefault(maxEpochs.value)} epochs`;
        statusText = `No max-steps default; ${maxEpochs.origin === "repository" ? "repository" : "adapter manifest"} uses ${displayCommonHyperparameterDefault(maxEpochs.value)} epochs. ${provenance}`;
      }
    }
    if (definition.defaultOption !== undefined) {
      const option = [...control.options].find(
        (candidate) => candidate.value === definition.defaultOption,
      );
      if (option) option.textContent = placeholderText;
    } else {
      control.placeholder = placeholderText;
    }
    status.textContent = statusText;
    status.title = sourceRecord
      ? sourceRecord.origin === "repository"
        ? repositoryDefaultProvenance(sourceRecord)
        : adapterDefaultProvenance(sourceRecord)
      : "";
    status.dataset.defaultSource = sourceRecord?.origin || "not-declared";
  });
}

function resetCommonHyperparameterOverrides() {
  [
    elements.hpLearningRate,
    elements.hpBatchSize,
    elements.hpGradAcc,
    elements.hpNumWorkers,
    elements.hpMaxSteps,
    elements.checkpointSaveSteps,
  ].forEach((control) => {
    control.value = "";
    control.setCustomValidity("");
    control.removeAttribute("aria-invalid");
  });
  elements.hpBatchSemantics.value = "";
  elements.hpPrecision.value = "adapter-default";
}

function clearRepositoryInputOptions({ render = false } = {}) {
  repositoryInputOptionsState = null;
  if (render && elements.adapterDeclaredFieldsGrid)
    renderAdapterDeclaredFields();
}

function adapterInputFields(adapter = selectedAdapter()) {
  const fields = declaredAdapterInputFields(adapter).map((field) =>
    field.path === "native.config.training_preset" &&
    adapterManifest(adapter)?.train?.presets?.length
      ? {
          ...field,
          choices: [...new Set([...(field.choices || []), "custom"])],
        }
      : field,
  );
  const options = activeRepositoryInputOptions(adapter);
  return fields.map((field) => {
    const discovered = options?.get(field.path) || null;
    const binding = resolveAdapterDataBinding(field);
    if (!field.choice_source && !discovered && !field.data_binding)
      return field;
    const discovery = field.choice_source
      ? discovered || {
          choices: [],
          complete: false,
          source: "",
          warnings: options
            ? [
                "Repository inspection returned no option result for this field.",
              ]
            : [
                "Select and inspect an exact commit to load its allowed values.",
              ],
        }
      : null;
    return {
      ...field,
      ...(discovery?.complete ? { choices: [...discovery.choices] } : {}),
      _inputOptionDiscovery: discovery,
      _dataBindingResolution: binding,
    };
  });
}

let trainingDatasetRows = [];

function experimentInputSlots(adapter = selectedAdapter()) {
  const slots = new Map();
  for (const field of declaredAdapterInputFields(adapter)) {
    const binding = field.data_binding;
    if (!binding) continue;
    const key = JSON.stringify([binding.role, Number(binding.position || 0)]);
    if (!slots.has(key))
      slots.set(key, {
        key,
        role: binding.role,
        position: Number(binding.position || 0),
        bindings: [],
      });
    slots.get(key).bindings.push(binding);
  }
  return [...slots.values()].sort(
    (a, b) =>
      (a.role === "training_data" ? -1 : 1) -
        (b.role === "training_data" ? -1 : 1) || a.position - b.position,
  );
}

function selectedExperimentDataBundle() {
  const selected = [];
  experimentInputSlots().forEach((slot, index) => {
    const control = index
      ? document.getElementById("experiment-data-input-" + index)
      : elements.experimentDataBundle;
    const dataset = trainingDatasetRows.find(
      (row) => row.id === control?.value,
    );
    if (dataset) selected.push({ dataset, slot });
  });
  if (!selected.length) return null;
  return {
    ...selected[0].dataset,
    selections: selected.map(({ dataset, slot }) => ({
      ...dataset.selection,
      role: slot.role,
      position: slot.position,
    })),
    assignments: selected.flatMap(({ dataset, slot }) =>
      dataset.assignments.map((a) => ({
        ...a,
        role: slot.role,
        position: slot.position,
      })),
    ),
  };
}

function datasetBindingValue(assignment, binding) {
  return {
    "version.path": assignment.version?.path,
    mount_path: assignment.mount_path,
    "location.path": assignment.config?.location?.path,
    "version.manifest_sha256": assignment.version?.manifest_sha256,
  }[binding.value_path || "version.path"];
}

function adapterDataContracts(binding, adapter = selectedAdapter()) {
  let contracts = binding.contracts?.length
    ? binding.contracts
    : binding.contract
      ? [binding.contract]
      : [];
  if (binding.contract_selector && binding.contract_choices) {
    const control = document.getElementById(
      adapterFieldControlId(binding.contract_selector),
    );
    const field = declaredAdapterInputFields(adapter).find(
      (f) => f.path === binding.contract_selector,
    );
    const selected =
      control && field
        ? parseAdapterDeclaredValue(control, field).value
        : field?.default;
    contracts = binding.contract_choices[selected] || contracts;
  }
  return contracts;
}

function resolveAdapterDataBinding(field) {
  const binding = field?.data_binding;
  if (!binding) return null;
  const bundle = selectedExperimentDataBundle();
  if (!bundle) {
    return {
      value: null,
      state: "empty",
      message: "Choose a prepared dataset.",
    };
  }
  const matches = (Array.isArray(bundle.assignments) ? bundle.assignments : [])
    .filter(
      (assignment) =>
        String(assignment?.role || "") === String(binding.role || ""),
    )
    .sort(
      (left, right) => Number(left.position || 0) - Number(right.position || 0),
    );
  const assignment = matches.find(
    (item) => Number(item.position || 0) === Number(binding.position || 0),
  );
  if (!assignment) {
    return {
      value: null,
      state: "error",
      message: `Selected dataset has no ${binding.role} role at position ${Number(binding.position || 0)}.`,
    };
  }
  const format = String(assignment.version?.format || "");
  const formats = Array.isArray(binding.formats)
    ? binding.formats.map(String)
    : [];
  if (
    formats.length &&
    !formats.some(
      (candidate) => candidate.toLowerCase() === format.toLowerCase(),
    )
  ) {
    return {
      value: null,
      state: "error",
      message: `Selected ${binding.role} format “${format || "undeclared"}” is incompatible; this adapter accepts ${formats.join(", ")}.`,
    };
  }
  const contracts = adapterDataContracts(binding);
  const metadata = assignment.version?.metadata || {};
  if (
    contracts.length &&
    (!contracts.includes(metadata.contract) ||
      metadata.validation?.status !== "PASSED")
  ) {
    return {
      value: null,
      state: "error",
      message:
        "Dataset does not satisfy the selected observation requirements.",
    };
  }
  const value = datasetBindingValue(assignment, binding);
  if (!value) {
    return {
      value: null,
      state: "error",
      message: `Selected dataset does not declare ${binding.value_path || "version.path"}.`,
    };
  }
  return {
    value: String(value),
    state: "complete",
    message: "From the selected dataset.",
  };
}

function adapterManifestFingerprint(manifest) {
  const source = JSON.stringify(manifest || {});
  let hash = 2166136261;
  for (let index = 0; index < source.length; index += 1) {
    hash ^= source.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return `local-${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

function adapterDeclaredScope(adapter) {
  if (!adapter) return null;
  const version = adapter.selected_version || adapter.latest_version || {};
  const stableId = adapterId(adapter);
  const versionId = String(
    firstValue(
      version.id,
      adapter.selected_version_id,
      adapter.latest_version_id,
      adapterVersion(adapter),
      "unversioned",
    ),
  );
  const manifestHash = String(
    firstValue(
      version.manifest_sha256,
      adapter.manifest_sha256,
      adapterManifestFingerprint(adapterManifest(adapter)),
    ),
  );
  const base = `${stableId}\u001f${versionId}`;
  return {
    stableId,
    versionId,
    manifestHash,
    base,
    key: `${base}\u001f${manifestHash}`,
  };
}

function reconcileAdapterDeclaredScope(scope) {
  if (!scope) return;
  const previousHash = adapterDeclaredScopeHashes.get(scope.base);
  if (previousHash && previousHash !== scope.manifestHash) {
    for (const key of adapterDeclaredValueState.keys()) {
      if (key.startsWith(`${scope.base}\u001f`))
        adapterDeclaredValueState.delete(key);
    }
  }
  adapterDeclaredScopeHashes.set(scope.base, scope.manifestHash);
}

function adapterDeclaredScopeValues(
  scope = renderedAdapterDeclaredScope,
  create = false,
) {
  if (!scope) return null;
  let values = adapterDeclaredValueState.get(scope.key);
  if (!values && create) {
    values = new Map();
    adapterDeclaredValueState.set(scope.key, values);
  }
  return values || null;
}

function adapterFieldControlId(path) {
  return `adapter-field-${path.replace(/[^a-z0-9_-]+/gi, "-")}`;
}

function adapterFieldRawValue(control) {
  return control.type === "checkbox" ? Boolean(control.checked) : control.value;
}

function setAdapterFieldControlValue(control, field, value) {
  if (control.type === "checkbox") {
    control.checked = Boolean(value);
  } else if (
    field.choices?.length &&
    control.dataset.adapterInputAllowCustom !== "true"
  ) {
    control.value =
      value === null || value === undefined ? "" : JSON.stringify(value);
  } else if (value === null || value === undefined) {
    control.value = "";
  } else if (field.kind === "json") {
    control.value = JSON.stringify(value, null, 2);
  } else if (field.kind === "string_list") {
    control.value = Array.isArray(value) ? value.join("\n") : String(value);
  } else {
    control.value = String(value);
  }
}

function captureAdapterDeclaredValues() {
  const values = adapterDeclaredScopeValues(renderedAdapterDeclaredScope, true);
  if (!values) return;
  elements.adapterDeclaredFieldsGrid
    .querySelectorAll("[data-adapter-input-path]")
    .forEach((control) => {
      const path = control.dataset.adapterInputPath;
      if (control.dataset.adapterInputSensitive === "true") {
        values.delete(path);
        return;
      }
      const previous = values.get(path);
      values.set(path, {
        kind: control.dataset.adapterInputKind,
        raw: adapterFieldRawValue(control),
        touched: Boolean(previous?.touched),
      });
    });
}

function applyTrainingPreset(identifier) {
  const command = adapterManifest(selectedAdapter())?.train || {};
  const preset = (command.presets || []).find((item) => item.id === identifier);
  if (!preset) {
    if (identifier === "custom") {
      captureAdapterDeclaredValues();
      renderCommonHyperparameterDefaults();
      invalidateExperimentPreview();
    }
    return;
  }
  const fields = adapterInputFields();
  const values = adapterDeclaredScopeValues(renderedAdapterDeclaredScope, true);
  for (const [path, value] of Object.entries({
    ...preset.values,
    "native.config.training_preset": identifier,
  })) {
    const field = fields.find((item) => item.path === path);
    const control = field
      ? document.getElementById(adapterFieldControlId(path))
      : null;
    if (control) {
      setAdapterFieldControlValue(control, field, value);
      values?.set(path, {
        kind: field.kind,
        raw: adapterFieldRawValue(control),
        touched: true,
      });
    } else {
      const common = COMMON_HYPERPARAMETER_DEFAULT_FIELDS.find(
        (item) => item.path === path,
      );
      if (common) elements[common.control].value = String(value);
    }
  }
  renderAdapterDeclaredFields();
  validateAdapterDeclaredFields({ focus: false, notify: false });
  invalidateExperimentPreview();
}

function updateTrainingPresetLabel() {
  const fields = adapterInputFields();
  const presetField = fields.find(
    (f) => f.path === "native.config.training_preset",
  );
  const control =
    presetField &&
    document.getElementById(adapterFieldControlId(presetField.path));
  const preset = (
    adapterManifest(selectedAdapter())?.train?.presets || []
  ).find((p) => JSON.stringify(p.id) === control?.value);
  if (!preset) return;
  const commonControls = [];
  const matches = Object.entries(preset.values)
    .map(([path, expected]) => {
      const field = fields.find((f) => f.path === path);
      const input =
        field && document.getElementById(adapterFieldControlId(path));
      if (input) {
        const parsed = parseAdapterDeclaredValue(input, field);
        return (
          !parsed.error &&
          JSON.stringify(parsed.present ? parsed.value : field.default) ===
            JSON.stringify(expected)
        );
      }
      const definition = COMMON_HYPERPARAMETER_DEFAULT_FIELDS.find(
        (f) => f.path === path,
      );
      if (!definition) return true;
      const inputCommon = elements[definition.control];
      commonControls.push([inputCommon, expected]);
      const raw = inputCommon.value;
      const value =
        raw === "" || raw === definition.defaultOption
          ? expected
          : typeof expected === "number"
            ? Number(raw)
            : raw;
      return JSON.stringify(value) === JSON.stringify(expected);
    })
    .every(Boolean);
  if (matches) return;
  // Preserve inherited values before switching off preset defaults.
  for (const [input, value] of commonControls) {
    if (!input.value || input.value === "adapter-default")
      input.value = String(value);
  }
  setAdapterFieldControlValue(control, presetField, "custom");
  adapterDeclaredScopeValues(renderedAdapterDeclaredScope, true)?.set(
    presetField.path,
    { kind: presetField.kind, raw: control.value, touched: true },
  );
  renderCommonHyperparameterDefaults();
}

function renderAdapterDeclaredFields(adapter = selectedAdapter()) {
  captureAdapterDeclaredValues();
  const scope = adapterDeclaredScope(adapter);
  reconcileAdapterDeclaredScope(scope);
  renderedAdapterDeclaredScope = scope;
  const fields = adapterInputFields(adapter);
  const openGroups = new Set(
    [
      ...elements.adapterDeclaredFieldsGrid.querySelectorAll("details[open]"),
    ].map((item) => item.dataset.adapterFieldGroup),
  );
  elements.adapterDeclaredFieldsGrid.replaceChildren();
  elements.adapterDeclaredFieldsError.hidden = true;
  elements.adapterDeclaredFieldsError.textContent = "";
  elements.nativeOverrides.setCustomValidity("");

  if (!adapter) {
    elements.adapterDeclaredFieldsStatus.textContent = "Select an adapter.";
    renderCommonHyperparameterDefaults(null);
    renderModelIO(document.getElementById("experiment-model-io"), null);
    return;
  }
  if (!fields.length) {
    elements.adapterDeclaredFieldsStatus.textContent =
      "No typed inputs declared.";
    renderCommonHyperparameterDefaults(adapter);
    refreshExperimentModelIO();
    return;
  }

  elements.adapterDeclaredFieldsStatus.textContent = `${fields.length} declared field${fields.length === 1 ? "" : "s"}`;
  const groups = new Map();
  const fieldGroup = (name, label) => {
    if (!groups.has(name)) {
      const details = document.createElement("details");
      details.className = "advanced-details";
      details.dataset.adapterFieldGroup = name;
      details.open = openGroups.has(name);
      const summary = document.createElement("summary");
      summary.textContent = label;
      const body = document.createElement("div");
      body.className = "adapter-declared-fields-grid";
      details.append(summary, body);
      groups.set(name, { details, body });
    }
    return groups.get(name);
  };
  fields.forEach((field) => {
    const wrapper = document.createElement("div");
    wrapper.className = "field";
    const label = document.createElement("label");
    const id = adapterFieldControlId(field.path);
    label.htmlFor = id;
    label.textContent = `${field.label}${field.required ? " (required)" : ""}`;
    wrapper.append(label);

    const choiceDiscovery = field._inputOptionDiscovery || null;
    let control;
    let choiceList = null;
    if (field.sensitive) {
      control = document.createElement("input");
      control.type = "password";
    } else if (field.choice_source?.allow_custom === true) {
      control = document.createElement("input");
      control.type = ["integer", "number"].includes(field.kind)
        ? "number"
        : "text";
      if (control.type === "number")
        control.step = field.kind === "integer" ? "1" : "any";
      control.placeholder = field.choices?.length
        ? "Enter a value or choose a repository suggestion"
        : "Enter a custom value";
      choiceList = document.createElement("datalist");
      choiceList.id = `${id}-choices`;
      (field.choices || []).forEach((choice) => {
        const option = document.createElement("option");
        option.value =
          typeof choice === "string" ? choice : JSON.stringify(choice);
        choiceList.append(option);
      });
      control.setAttribute("list", choiceList.id);
    } else if (field.choices?.length || field.choice_source) {
      control = document.createElement("select");
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = field.choices?.length
        ? "Choose a value"
        : choiceDiscovery?.complete
          ? "No values were discovered"
          : "Options unavailable until repository inspection succeeds";
      control.append(placeholder);
      (field.choices || []).forEach((choice) => {
        const option = document.createElement("option");
        option.value = JSON.stringify(choice);
        option.textContent =
          field.path === "native.config.training_preset"
            ? choice === "custom"
              ? "Custom"
              : (adapterManifest(adapter)?.train?.presets || []).find(
                  (p) => p.id === choice,
                )?.name || choice
            : typeof choice === "string"
              ? choice
              : JSON.stringify(choice);
        control.append(option);
      });
      if (!field.choices?.length && field.choice_source)
        control.disabled = true;
    } else if (field.kind === "json" || field.kind === "string_list") {
      control = document.createElement("textarea");
      control.rows = field.kind === "json" ? 4 : 3;
      control.placeholder =
        field.kind === "json" ? "Valid JSON" : "One value per line";
    } else {
      control = document.createElement("input");
      if (field.kind === "boolean") {
        control.type = "checkbox";
      } else if (field.kind === "integer" || field.kind === "number") {
        control.type = "number";
        control.step = field.kind === "integer" ? "1" : "any";
      } else {
        control.type = field.sensitive ? "password" : "text";
      }
    }
    if (field.minimum != null) control.min = field.minimum;
    if (field.maximum != null) control.max = field.maximum;
    control.id = id;
    control.dataset.adapterInputPath = field.path;
    control.dataset.adapterInputKind = field.kind;
    control.dataset.adapterInputSensitive = String(Boolean(field.sensitive));
    control.dataset.adapterInputAllowCustom = String(
      field.choice_source?.allow_custom === true,
    );
    control.autocomplete = field.sensitive ? "new-password" : "off";
    if (field.required && control.type !== "checkbox") control.required = true;
    control.setAttribute("aria-describedby", `${id}-help`);

    const saved = field.sensitive
      ? null
      : adapterDeclaredScopeValues(scope)?.get(field.path);
    if (field._dataBindingResolution?.state === "complete") {
      setAdapterFieldControlValue(
        control,
        field,
        field._dataBindingResolution.value,
      );
      control.readOnly = true;
      wrapper.classList.add("derived-input");
    } else if (saved?.kind === field.kind && saved.touched) {
      if (control.type === "checkbox") control.checked = Boolean(saved.raw);
      else if (control instanceof HTMLSelectElement) {
        const savedValue = String(saved.raw ?? "");
        control.value = [...control.options].some(
          (option) => option.value === savedValue,
        )
          ? savedValue
          : "";
      } else control.value = String(saved.raw ?? "");
    } else if (!field.sensitive && field._dataBindingResolution?.value) {
      setAdapterFieldControlValue(
        control,
        field,
        field._dataBindingResolution.value,
      );
    } else if (
      !field.sensitive &&
      field.default !== null &&
      field.default !== undefined
    ) {
      setAdapterFieldControlValue(control, field, field.default);
    }
    wrapper.append(control);
    if (choiceList) wrapper.append(choiceList);

    const help = document.createElement("small");
    help.id = `${id}-help`;
    const currentPreset = (adapterManifest(adapter)?.train?.presets || []).find(
      (p) => JSON.stringify(p.id) === control.value,
    );
    const fieldHelp =
      field.path === "native.config.training_preset"
        ? "Choose a preset or customize its values."
        : currentPreset?.description || field.help || "";
    const baseHelp = field.sensitive
      ? `${fieldHelp} Masked on this screen only. The submitted value is included in the experiment metadata and reproducibility specification; do not enter credentials.`
      : fieldHelp;
    if (field.choice_source) {
      const source = choiceDiscovery?.source
        ? typeof choiceDiscovery.source === "string"
          ? choiceDiscovery.source
          : choiceDiscovery.source.entrypoint ||
            choiceDiscovery.source.kind ||
            "pinned repository metadata"
        : "";
      const warnings = choiceDiscovery?.warnings?.join(" ") || "";
      const allowCustom = field.choice_source.allow_custom === true;
      const status = choiceDiscovery?.complete
        ? field.choices?.length
          ? `${allowCustom ? "Suggested" : "Allowed"} values were discovered${source ? ` from ${source}` : ""} at the selected commit.${allowCustom ? " Custom values are allowed." : ""}`
          : `Repository inspection completed${source ? ` from ${source}` : ""}, but returned no ${allowCustom ? "suggested" : "allowed"} values.${allowCustom ? " Enter a custom value." : " Arbitrary text is disabled."}`
        : `${allowCustom ? "Suggested-value" : "Allowed-value"} discovery is incomplete${source ? ` for ${source}` : ""}.${allowCustom ? " Custom values remain editable." : " Arbitrary text is disabled."}`;
      help.textContent = [baseHelp, status, warnings].filter(Boolean).join(" ");
      help.dataset.discoveryState =
        choiceDiscovery?.complete && field.choices?.length
          ? "complete"
          : "error";
    } else {
      help.textContent =
        field._dataBindingResolution?.state === "complete"
          ? field._dataBindingResolution.message
          : [baseHelp, field._dataBindingResolution?.message]
              .filter(Boolean)
              .join(" ");
      if (field._dataBindingResolution)
        help.dataset.discoveryState = field._dataBindingResolution.state;
    }
    wrapper.append(help);
    const group =
      field._dataBindingResolution?.state === "complete"
        ? fieldGroup("dataset", "Selected dataset details")
        : field.kind === "json" || /(?:command|resume_args)$/.test(field.path)
          ? fieldGroup("advanced", "Advanced policy settings")
          : null;
    if (group) {
      group.body.append(wrapper);
      control.addEventListener("invalid", () => {
        group.details.open = true;
      });
      if (field.required && !control.value) group.details.open = true;
    } else elements.adapterDeclaredFieldsGrid.append(wrapper);
  });
  for (const { details } of groups.values())
    elements.adapterDeclaredFieldsGrid.append(details);
  renderCommonHyperparameterDefaults(adapter);
  refreshExperimentModelIO();
}

function parseAdapterDeclaredValue(control, field) {
  if (field.kind === "boolean" && control.type === "checkbox")
    return { present: true, value: Boolean(control.checked) };
  const raw = control.value.trim();
  if (!raw) return { present: false, value: null };
  try {
    let value;
    if (
      field.choices?.length &&
      control.dataset.adapterInputAllowCustom !== "true"
    )
      value = JSON.parse(raw);
    else if (field.kind === "boolean") {
      if (!["true", "false"].includes(raw.toLowerCase()))
        throw new Error("must be true or false");
      value = raw.toLowerCase() === "true";
    } else if (field.kind === "integer" || field.kind === "number")
      value = Number(raw);
    else if (field.kind === "json") value = JSON.parse(raw);
    else if (field.kind === "string_list")
      value = raw
        .split(/\r?\n/)
        .map((item) => item.trim())
        .filter(Boolean);
    else value = raw;
    if (field.kind === "integer" && !Number.isInteger(value))
      throw new Error("must be an integer");
    if (field.kind === "number" && !Number.isFinite(value))
      throw new Error("must be a finite number");
    if (field.minimum != null && value < field.minimum)
      throw new Error(`minimum is ${field.minimum}`);
    if (field.maximum != null && value > field.maximum)
      throw new Error(`maximum is ${field.maximum}`);
    if (field.maximum_path) {
      const bound = document.getElementById(
        adapterFieldControlId(field.maximum_path),
      );
      const limit = Number(bound?.value);
      if (bound?.value && Number.isFinite(limit) && value > limit)
        throw new Error(`cannot exceed ${limit}`);
    }
    if (field.kind === "string_list" && !value.length)
      return { present: false, value: null };
    return { present: true, value };
  } catch (error) {
    return { present: true, error: error.message || "invalid value" };
  }
}

function adapterOverrideDescriptor(path) {
  if (path === "native.argv") return { key: "argv", namespace: "reserved" };
  if (path === "native.resume_argv")
    return { key: "resume_argv", namespace: "reserved" };
  if (path.startsWith("native.config.")) {
    const suffix = path.slice("native.config.".length);
    const segments = suffix.split(".");
    if (
      !suffix ||
      segments.some(
        (segment) =>
          !/^[a-z][a-z0-9_]*$/.test(segment) ||
          ["__proto__", "prototype", "constructor"].includes(
            segment.toLowerCase(),
          ),
      )
    ) {
      return { error: `Unsupported canonical path “${path}”.` };
    }
    return { key: `config.${suffix}`, namespace: "config" };
  }
  if (path.startsWith("native.overrides.")) {
    const suffix = path.slice("native.overrides.".length);
    const segments = suffix.split(".");
    if (
      !suffix ||
      segments.some(
        (segment) =>
          !/^[A-Za-z_][A-Za-z0-9_-]*$/.test(segment) ||
          ["__proto__", "prototype", "constructor"].includes(
            segment.toLowerCase(),
          ),
      )
    ) {
      return { error: `Unsupported canonical path “${path}”.` };
    }
    return { key: suffix, namespace: "override" };
  }
  return { error: `Unsupported canonical path “${path}”.` };
}

function adapterOverrideKeysCollide(left, right) {
  if (left.key === right.key) return true;
  if (left.namespace !== "config" || right.namespace !== "config") return false;
  return (
    left.key.startsWith(`${right.key}.`) || right.key.startsWith(`${left.key}.`)
  );
}

function collectAdapterDeclaredOverrides() {
  const declaredLines = [];
  const declaredDescriptors = [];
  const errors = [];
  let firstInvalid = null;
  adapterInputFields().forEach((field) => {
    const control = document.getElementById(adapterFieldControlId(field.path));
    if (!control) {
      if (field.required)
        errors.push(`${field.label} is required but is not available.`);
      return;
    }
    control.setCustomValidity("");
    if (
      field.sensitive &&
      (field.choices?.length ||
        (field.default !== null && field.default !== undefined) ||
        (field.tutorial_value !== null && field.tutorial_value !== undefined))
    ) {
      const message = `${field.label}: masked fields cannot declare choices, defaults, or tutorial values.`;
      control.setCustomValidity(message);
      errors.push(message);
      firstInvalid ||= control;
      return;
    }
    const descriptor = adapterOverrideDescriptor(field.path);
    if (descriptor.error) {
      control.setCustomValidity(descriptor.error);
      errors.push(descriptor.error);
      firstInvalid ||= control;
      return;
    }
    const parsed = parseAdapterDeclaredValue(control, field);
    if (parsed.error) {
      const message = `${field.label}: ${parsed.error}`;
      control.setCustomValidity(message);
      errors.push(message);
      firstInvalid ||= control;
      return;
    }
    if (!parsed.present) {
      if (field.required) {
        const message = `${field.label} is required.`;
        control.setCustomValidity(message);
        errors.push(message);
        firstInvalid ||= control;
      }
      return;
    }
    const collision = declaredDescriptors.find((existing) =>
      adapterOverrideKeysCollide(existing, descriptor),
    );
    if (collision) {
      const message = `${field.label}: canonical key “${descriptor.key}” conflicts with “${collision.key}”.`;
      control.setCustomValidity(message);
      errors.push(message);
      firstInvalid ||= control;
      return;
    }
    declaredDescriptors.push(descriptor);
    declaredLines.push(`${descriptor.key}=${JSON.stringify(parsed.value)}`);
  });

  const advancedLines = elements.nativeOverrides.value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  const advancedDescriptors = [];
  advancedLines.forEach((line) => {
    const separator = line.indexOf("=");
    const key = (separator < 0 ? line : line.slice(0, separator)).trim();
    const descriptor = {
      key,
      namespace: key.startsWith("config.") ? "config" : "advanced",
    };
    const declaredCollision = declaredDescriptors.find((existing) =>
      adapterOverrideKeysCollide(existing, descriptor),
    );
    const advancedCollision = advancedDescriptors.find((existing) =>
      adapterOverrideKeysCollide(existing, descriptor),
    );
    if (!key)
      errors.push("Advanced native values must use a nonempty key before '='.");
    if (declaredCollision)
      errors.push(
        `Advanced native key “${key}” conflicts with declared field “${declaredCollision.key}”.`,
      );
    if (advancedCollision)
      errors.push(
        `Advanced native key “${key}” conflicts with earlier key “${advancedCollision.key}”.`,
      );
    advancedDescriptors.push(descriptor);
  });
  const advancedError =
    errors.find((message) => message.startsWith("Advanced")) || "";
  elements.nativeOverrides.setCustomValidity(advancedError);
  if (!firstInvalid && advancedError) firstInvalid = elements.nativeOverrides;
  elements.adapterDeclaredFieldsError.hidden = errors.length === 0;
  elements.adapterDeclaredFieldsError.textContent = errors.join(" ");
  return { declaredLines, advancedLines, errors, firstInvalid };
}

function validateAdapterDeclaredFields({ focus = true, notify = true } = {}) {
  const result = collectAdapterDeclaredOverrides();
  if (!result.errors.length) return true;
  if (focus) result.firstInvalid?.focus({ preventScroll: true });
  if (notify) showToast(result.errors[0], true);
  return false;
}

function tutorialAdapterDeclaredFieldsValid() {
  return validateAdapterDeclaredFields({ focus: false, notify: false });
}

function tutorialApplyAdapterDeclaredValues() {
  let firstChanged = null;
  let firstMissing = null;
  adapterInputFields().forEach((field) => {
    const control = document.getElementById(adapterFieldControlId(field.path));
    if (!control) return;
    if (
      !field.sensitive &&
      field.tutorial_value !== null &&
      field.tutorial_value !== undefined
    ) {
      setAdapterFieldControlValue(control, field, field.tutorial_value);
      adapterDeclaredScopeValues(renderedAdapterDeclaredScope, true)?.set(
        field.path,
        {
          kind: field.kind,
          raw: adapterFieldRawValue(control),
          touched: true,
        },
      );
      control.dispatchEvent(new Event("input", { bubbles: true }));
      control.dispatchEvent(new Event("change", { bubbles: true }));
      firstChanged ||= control;
    } else if (
      field.required &&
      !parseAdapterDeclaredValue(control, field).present
    ) {
      firstMissing ||= control;
    }
  });
  if (firstMissing)
    tutorialSetError(
      "This adapter does not declare a tutorial value for every required field. Enter or choose the missing value yourself.",
    );
  return firstChanged || firstMissing || elements.adapterDeclaredFields;
}

function experimentPreviewShapeErrors(payload) {
  const errors = [];
  if (!payload || typeof payload !== "object" || Array.isArray(payload))
    return ["Response is not an object."];
  if (!Array.isArray(payload.blockers))
    errors.push("blockers must be an array");
  else if (
    payload.blockers.some(
      (blocker) =>
        !blocker ||
        typeof blocker.variant !== "string" ||
        !blocker.variant.trim() ||
        !Array.isArray(blocker.reasons) ||
        blocker.reasons.some(
          (reason) => typeof reason !== "string" || !reason.trim(),
        ),
    )
  ) {
    errors.push("each blocker must have a variant and nonempty reason strings");
  }
  if (
    !Array.isArray(payload.scripts) ||
    payload.scripts.some(
      (script) => typeof script !== "string" || !script.trim(),
    )
  )
    errors.push("scripts must be a nonempty-string array");
  if (typeof payload.script !== "string" || !payload.script.trim())
    errors.push("script must be a nonempty string");
  if (!Number.isInteger(payload.variant_count) || payload.variant_count <= 0)
    errors.push("variant_count must be a positive integer");
  if (
    Array.isArray(payload.scripts) &&
    Number.isInteger(payload.variant_count) &&
    payload.scripts.length !== payload.variant_count
  )
    errors.push("variant_count must equal scripts.length");
  if (
    Array.isArray(payload.scripts) &&
    typeof payload.script === "string" &&
    payload.script !== payload.scripts.join("\n\n")
  )
    errors.push("script must be the canonical joined scripts value");
  if (
    !Array.isArray(payload.warnings) ||
    payload.warnings.some((warning) => typeof warning !== "string")
  )
    errors.push("warnings must be a string array");
  if (
    typeof payload.resolved_revision !== "string" ||
    !payload.resolved_revision.trim()
  )
    errors.push("resolved_revision must be a nonempty string");
  return errors;
}

function experimentPreviewScriptIsRunnable(script) {
  return (
    typeof script === "string" &&
    Boolean(script.trim()) &&
    !script.includes("# BLOCKED")
  );
}

function applyEvaluationSuiteDefaults(suiteIds) {
  if (!Array.isArray(suiteIds)) return;
  pendingEvaluationSuiteDefaults = suiteIds.map(String);
  if (!evaluationSuites.length) return;
  const available = new Set(
    [...elements.experimentEvaluationSuites.options].map(
      (option) => option.value,
    ),
  );
  const unavailable = pendingEvaluationSuiteDefaults.filter(
    (suiteId) => !available.has(suiteId),
  );
  const message = unavailable.length
    ? `Adapter default evaluation suite${unavailable.length === 1 ? " is" : "s are"} unavailable: ${unavailable.join(", ")}.`
    : "";
  elements.experimentEvaluationSuites.setCustomValidity(message);
  if (message) {
    elements.experimentEvaluationSuites.dataset.adapterDefaultError = message;
    if (!adapterDefaultErrors.includes(message))
      adapterDefaultErrors.push(message);
    showNotice(
      elements.experimentsError,
      `Adapter manifest defaults are invalid: ${adapterDefaultErrors.join(" ")}`,
    );
  } else {
    delete elements.experimentEvaluationSuites.dataset.adapterDefaultError;
  }
  [...elements.experimentEvaluationSuites.options].forEach((option) => {
    option.selected = pendingEvaluationSuiteDefaults.includes(option.value);
  });
}

function applySelectedAdapter({ loadSource = true } = {}) {
  const previousSource = elements.experimentSource.value.trim();
  const previousWorkdir = elements.experimentWorkdir.value.trim();
  adapterDefaultErrors = [];
  document
    .querySelectorAll("[data-adapter-default-error]")
    .forEach((control) => {
      control.setCustomValidity?.("");
      delete control.dataset.adapterDefaultError;
    });
  const adapter = selectedAdapter();
  if (!adapter) {
    renderAdapterDeclaredFields(null);
    resetCommonHyperparameterOverrides();
    renderCommonHyperparameterDefaults(null);
    elements.adapterCapabilities.textContent = "Select an active adapter.";
    elements.experimentSource.value = "";
    elements.experimentWorkdir.value = "";
    scheduleSourceBranchLoad();
    return;
  }
  const manifest = adapterManifest(adapter);
  if (manifest.__parse_error) {
    showNotice(elements.experimentsError, manifest.__parse_error);
    renderAdapterDeclaredFields(null);
    resetRuntimeInspection(manifest.__parse_error, "error");
    return;
  }
  populateExperimentDataBundles();
  resetCommonHyperparameterOverrides();
  renderAdapterDeclaredFields(adapter);
  const defaults = repositoryDefaults(manifest, adapter);
  elements.adapterCapabilities.textContent = `v${adapterVersion(adapter)} / ${manifestCapabilityLabel(manifest)}`;
  elements.experimentSource.value = defaults.url;
  elements.experimentWorkdir.value = defaults.workdir;

  configurePrecisionField(manifest);
  renderCommonHyperparameterDefaults(adapter);
  setManifestDefault(
    elements.resourceCpus,
    manifestDefault(manifest, "resources", "cpus_per_task"),
  );
  setManifestDefault(
    elements.resourceMemory,
    manifestDefault(manifest, "resources", "memory_gb"),
  );
  setManifestDefault(
    elements.resourceTime,
    manifestDefault(manifest, "resources", "time_limit"),
  );
  setManifestDefault(
    elements.resourcePolicy,
    manifestDefault(manifest, "resources", "queue_policy"),
  );
  setManifestDefault(
    elements.gpuMode,
    manifestDefault(manifest, "resources", "gpu_mode"),
  );
  setManifestDefault(
    elements.experimentGpuCount,
    firstValue(
      manifestDefault(manifest, "resources", "gpu_count"),
      manifestDefault(manifest, "resources", "gpus_per_node"),
    ),
  );
  setManifestDefault(
    elements.experimentGpuType,
    manifestDefault(manifest, "resources", "gpu_type"),
  );
  setManifestDefault(
    elements.resourceNodes,
    manifestDefault(manifest, "resources", "nodes"),
  );
  setManifestDefault(
    elements.checkpointMode,
    manifestDefault(manifest, "checkpoint", "mode"),
  );
  setManifestDefault(
    elements.checkpointPath,
    manifestDefault(manifest, "checkpoint", "path"),
  );
  setManifestDefault(
    elements.checkpointMaxAttempts,
    manifestDefault(manifest, "checkpoint", "max_attempts"),
  );
  document.querySelector("#checkpoint-warning-seconds").value =
    manifestDefault(manifest, "checkpoint", "save_before_timeout_seconds") ??
    300;
  configureCanonicalHyperparameterFields(manifest);
  setManifestCheckbox(
    elements.checkpointAutoResume,
    manifestDefault(manifest, "checkpoint", "auto_resume"),
  );
  setManifestCheckbox(
    elements.mlflowEnabled,
    manifestDefault(manifest, "tracking", "enabled"),
  );
  setManifestDefault(
    elements.mlflowExperiment,
    firstValue(
      manifestDefault(manifest, "tracking", "mlflow_experiment"),
      manifestDefault(manifest, "tracking", "experiment"),
    ),
  );
  setManifestDefault(
    elements.mlflowRunName,
    manifestDefault(manifest, "tracking", "run_name_template"),
  );
  setManifestCheckbox(
    elements.evaluationEnabled,
    manifestDefault(manifest, "evaluation", "enabled"),
  );
  applyEvaluationSuiteDefaults(
    manifestDefault(manifest, "evaluation", "suite_ids"),
  );
  const nativeDefaults = firstValue(
    manifest.defaults?.native_overrides,
    manifest.defaults?.native?.overrides,
  );
  setManifestDefault(
    elements.nativeOverrides,
    nativeDefaults === undefined ? null : nativeDefaultsText(nativeDefaults),
  );
  const sweepDefault = manifestDefault(manifest, "sweep", "definition");
  setManifestDefault(
    elements.sweepDefinition,
    sweepDefault === undefined
      ? null
      : typeof sweepDefault === "string"
        ? sweepDefault
        : JSON.stringify(sweepDefault, null, 2),
  );
  const capabilities = manifest.capabilities || {};
  if (Number.isFinite(Number(capabilities.minimum_gpus))) {
    elements.experimentGpuCount.min = String(capabilities.minimum_gpus);
  }
  if (Number.isFinite(Number(capabilities.maximum_gpus))) {
    elements.experimentGpuCount.max = String(capabilities.maximum_gpus);
  }
  if (Number.isFinite(Number(capabilities.recommended_gpus))) {
    elements.experimentGpuCount.value = String(capabilities.recommended_gpus);
  }
  if (capabilities.supports_resume === false)
    elements.checkpointAutoResume.checked = false;
  updateExperimentFields();
  if (adapterDefaultErrors.length) {
    showNotice(
      elements.experimentsError,
      `Adapter manifest defaults are invalid: ${adapterDefaultErrors.join(" ")}`,
    );
  } else {
    clearNotice(elements.experimentsError);
  }
  const keepSource =
    previousSource === defaults.url &&
    previousWorkdir === defaults.workdir &&
    /^[a-f0-9]{40}$/i.test(elements.experimentRevision.value) &&
    (manifest.runtime?.allowed_backends || []).includes(
      elements.experimentRuntime.value,
    );
  if (loadSource) {
    if (keepSource) inspectRepositoryRuntime();
    else loadSourceBranches();
  }
}

function renderAdapters() {
  if (!adapterRows.length) {
    elements.adaptersBody.innerHTML = emptyRow(
      7,
      "No adapters are registered.",
    );
    elements.adapterCount.textContent = "0 adapters";
    return;
  }
  const query = document
    .getElementById("adapter-search")
    .value.trim()
    .toLowerCase();
  const rows = adapterRows.filter(
    (adapter) =>
      (document.getElementById("adapter-show-archived").checked ||
        !adapterArchived(adapter)) &&
      [
        adapter.name,
        adapter.slug,
        adapter.id,
        adapterRuntimeDetection(adapterManifest(adapter)),
      ]
        .join(" ")
        .toLowerCase()
        .includes(query),
  );
  elements.adapterCount.textContent = `${rows.length} of ${adapterRows.length} adapters`;
  elements.adaptersBody.innerHTML = rows
    .map((adapter) => {
      const id = adapterId(adapter);
      const manifest = adapterManifest(adapter);
      const source =
        manifest.__parse_error ||
        repositoryDefaults(manifest, adapter).url ||
        "-";
      const archived = adapterArchived(adapter);
      const version = adapterVersion(adapter);
      const hash =
        adapter.selected_version?.manifest_sha256 ||
        adapter.latest_version?.manifest_sha256 ||
        adapter.manifest_sha256 ||
        "";
      return `
      <tr>
        <td><span class="node-name">${escapeHtml(adapter.name || adapter.label || id)}</span><span class="secondary">${escapeHtml(adapter.seed_key || adapter.slug || id)}</span></td>
        <td><span class="job-id">v${escapeHtml(version)}</span></td>
        <td>${statusPill(manifest.__parse_error ? "invalid" : archived ? "archived" : adapter.status || "active")}</td>
        <td class="wrap-cell">${escapeHtml(source)}</td>
        <td>${escapeHtml(adapterRuntimeDetection(manifest))}</td>
        <td>${escapeHtml(formatDate(adapter.updated_at || adapter.latest_version?.created_at || adapter.created_at))}</td>
        <td class="row-actions adapter-row-actions">
          <button type="button" data-adapter-action="view" data-id="${escapeHtml(id)}">View</button>
          ${archived || adapter.editable === false ? "" : `<button type="button" data-adapter-action="edit" data-id="${escapeHtml(id)}">Edit</button>`}
          ${archived ? "" : `<button type="button" data-adapter-action="clone" data-id="${escapeHtml(id)}">Duplicate</button>`}
          ${
            adapter.editable === false
              ? ""
              : `<button type="button" data-adapter-action="${archived ? "restore" : "archive"}" data-id="${escapeHtml(id)}">${archived ? "Restore" : "Archive"}</button>
          <button type="button" data-delete-kind="adapter" data-delete-id="${escapeHtml(id)}">Delete</button>`
          }
        </td>
      </tr>`;
    })
    .join("");
}

function populateExperimentAdapters(preserveId = "") {
  const active = adapterRows.filter(adapterAvailableForExperiment);
  const pinned = loadedExperimentAdapterSnapshot;
  const versions = new Map();
  active.forEach((adapter) => versions.set(adapterOptionId(adapter), adapter));
  if (pinned && !versions.has(adapterOptionId(pinned))) {
    versions.set(adapterOptionId(pinned), pinned);
  }
  const available = [...versions.values()];
  if (!available.length) {
    elements.experimentAdapter.innerHTML =
      '<option value="">No active adapters</option>';
    elements.experimentAdapter.disabled = true;
    elements.adapterStatus.textContent = "No active adapters registered";
    elements.adapterCapabilities.textContent =
      "Create or restore an adapter in the Adapter Registry.";
    return;
  }
  elements.experimentAdapter.innerHTML = available
    .map((adapter) => {
      const id = adapterOptionId(adapter);
      const pinnedLabel = adapter._pinnedExperiment ? " / pinned revision" : "";
      return `<option value="${escapeHtml(id)}">${escapeHtml(adapter.name || adapter.label || id)} / v${escapeHtml(adapterVersion(adapter))}${pinnedLabel}</option>`;
    })
    .join("");
  elements.experimentAdapter.disabled = false;
  if (
    preserveId &&
    available.some((adapter) => adapterOptionId(adapter) === preserveId)
  ) {
    elements.experimentAdapter.value = preserveId;
  } else if (preserveId) {
    const message = `Previously selected adapter "${preserveId}" is unavailable. Choose an active adapter.`;
    elements.experimentAdapter.insertAdjacentHTML(
      "afterbegin",
      `<option value="${escapeHtml(preserveId)}" disabled>${escapeHtml(message)}</option>`,
    );
    elements.experimentAdapter.value = preserveId;
    elements.experimentAdapter.setCustomValidity(message);
    elements.experimentAdapter.setAttribute("aria-invalid", "true");
    elements.adapterStatus.textContent = message;
    return;
  }
  elements.experimentAdapter.setCustomValidity("");
  elements.experimentAdapter.removeAttribute("aria-invalid");
  const pinnedIsHistorical =
    pinned &&
    !active.some(
      (adapter) => adapterOptionId(adapter) === adapterOptionId(pinned),
    );
  const pinnedIsSelected =
    pinnedIsHistorical &&
    elements.experimentAdapter.value === adapterOptionId(pinned);
  elements.adapterStatus.textContent = pinnedIsSelected
    ? `Pinned adapter v${adapterVersion(pinned)} loaded from the experiment revision.`
    : `${active.length} active adapter${active.length === 1 ? "" : "s"}`;
}

async function loadAdapters(force = false) {
  if (adapterRows.length && !force) {
    populateExperimentAdapters(elements.experimentAdapter.value);
    renderAdapters();
    return;
  }
  const selected = elements.experimentAdapter.value;
  elements.experimentAdapter.disabled = true;
  elements.adapterStatus.textContent = "Loading adapters...";
  if (elements.refreshAdapters) elements.refreshAdapters.disabled = true;
  clearNotice(elements.adaptersError);
  try {
    const payload = await api("/api/adapters?include_archived=true");
    adapterRows = listFrom(payload, ["adapters"]);
    populateExperimentAdapters(selected);
    if (
      !selected ||
      selected !== elements.experimentAdapter.value ||
      !selectedAdapter()
    ) {
      applySelectedAdapter();
    }
  } catch (error) {
    adapterRows = [];
    elements.experimentAdapter.innerHTML =
      '<option value="">Adapter API unavailable</option>';
    elements.experimentAdapter.disabled = true;
    elements.adapterStatus.textContent = `Adapter API unavailable: ${error.message}`;
    showNotice(
      elements.adaptersError,
      `Adapter registry unavailable: ${error.message}`,
    );
  } finally {
    if (elements.refreshAdapters) elements.refreshAdapters.disabled = false;
  }
  renderAdapters();
}

function formatManifest(manifest) {
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    throw new Error("Adapter manifest must be a JSON object.");
  }
  return JSON.stringify(manifest, null, 2);
}

function editorManifest() {
  let manifest;
  try {
    manifest = JSON.parse(elements.adapterManifest.value);
  } catch (error) {
    throw new Error(`Manifest is not valid JSON: ${error.message}`);
  }
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    throw new Error("Manifest must be a JSON object.");
  }
  return manifest;
}

function normalizedEditorManifest() {
  const manifest = editorManifest();
  const identitySlug = elements.adapterEditorSlug.value.trim().toLowerCase();
  manifest.slug =
    adapterEditorState.mode === "create"
      ? identitySlug
      : String(
          adapterManifest(adapterEditorState.adapter).slug || identitySlug,
        );
  manifest.display_name = elements.adapterEditorName.value.trim();
  if (manifest.capabilities && typeof manifest.capabilities === "object") {
    manifest.capabilities.name = manifest.slug;
  }
  return manifest;
}

function renderAdapterVersions(versions) {
  elements.adapterVersionSection.hidden = !versions.length;
  elements.adapterVersionsBody.innerHTML = versions.length
    ? versions
        .map(
          (version) => `
      <tr>
        <td><span class="job-id">v${escapeHtml(version.version_number ?? version.version ?? "-")}</span></td>
        <td><code>${escapeHtml(shortId(version.manifest_sha256 || version.sha256 || "-", 16))}</code></td>
        <td class="wrap-cell">${escapeHtml(version.change_note || version.note || "-")}</td>
        <td>${escapeHtml(formatDate(version.created_at))}</td>
      </tr>`,
        )
        .join("")
    : emptyRow(4, "No version history was returned.");
}

function setAdapterEditorMode(mode) {
  adapterEditorState.mode = mode;
  const editing = mode === "create" || mode === "edit";
  const creating = mode === "create";
  elements.adapterEditorMode.textContent = creating
    ? "NEW ADAPTER"
    : editing
      ? "NEW IMMUTABLE VERSION"
      : "ADAPTER DETAIL";
  elements.adapterEditorSlug.disabled = !creating;
  elements.adapterEditorName.disabled = !editing;
  elements.adapterManifest.readOnly = !editing;
  elements.adapterDescription.disabled = !editing;
  elements.adapterChangeNote.disabled = !editing;
  elements.editAdapter.hidden =
    editing ||
    adapterArchived(adapterEditorState.adapter) ||
    adapterEditorState.adapter?.editable === false;
  elements.saveAdapter.hidden = !editing;
  elements.validateAdapter.disabled = false;
  elements.saveAdapter.textContent = creating
    ? "Create adapter"
    : "Create version";
  elements.adapterEditorStatus.textContent = creating
    ? "Unsaved"
    : `v${adapterVersion(adapterEditorState.adapter)}${adapterArchived(adapterEditorState.adapter) ? " / archived" : ""}`;
  elements.adapterValidationReport.hidden = true;
  elements.adapterValidationReport.textContent = "";
  refreshAdapterModelIO();
}

const revealedPanelLaunchers = new WeakMap();
const revealedPanelGenerations = new WeakMap();
const closedPanelGenerations = new WeakMap();

function currentRevealLauncher() {
  return document.activeElement instanceof HTMLElement
    ? document.activeElement
    : null;
}

const rowDisclosureHomes = new WeakMap();

function rowDisclosureHome(panel) {
  if (!panel) return null;
  let home = rowDisclosureHomes.get(panel);
  if (home) return home;
  const anchor = document.createComment(`home:${panel.id || "row-disclosure"}`);
  panel.before(anchor);
  home = { anchor, parent: anchor.parentNode };
  rowDisclosureHomes.set(panel, home);
  return home;
}

function unmountRowDisclosure(panel) {
  if (!panel) return false;
  const companion = panel.closest("tr.row-disclosure-companion");
  const detached = panel.classList.contains("row-disclosure-detached");
  if (
    !detached &&
    (!companion ||
      companion.dataset.disclosurePanel !== (panel.id || "row-disclosure"))
  )
    return false;
  const home = rowDisclosureHomes.get(panel);
  if (home?.anchor?.parentNode) home.anchor.after(panel);
  else if (home?.parent?.isConnected) home.parent.append(panel);
  companion?.remove();
  panel.classList.remove("row-disclosure-detached");
  return true;
}

function mountRowDisclosure(panel, launcher) {
  if (!panel || !(launcher instanceof HTMLElement)) return false;
  const selectedRow = launcher.closest("tr");
  const tableBody = selectedRow?.parentElement;
  if (!selectedRow || tableBody?.tagName !== "TBODY") return false;
  rowDisclosureHome(panel);
  const scroller = selectedRow.closest(".table-scroll, .table-frame");
  if (scroller) {
    // Detail content must not inherit the wide table's horizontal scroll offset,
    // and replacing tbody rows must never remove an active form or detail panel.
    if (panel.previousElementSibling !== scroller) {
      unmountRowDisclosure(panel);
      scroller.after(panel);
    }
    panel.classList.add("row-disclosure-detached");
    return true;
  }
  const currentRow = panel.closest("tr.row-disclosure-companion");
  if (currentRow?.previousElementSibling === selectedRow) return true;
  unmountRowDisclosure(panel);
  const detailRow = document.createElement("tr");
  detailRow.className =
    "run-detail-row row-disclosure-row row-disclosure-companion";
  detailRow.dataset.disclosurePanel = panel.id || "row-disclosure";
  const detailCell = document.createElement("td");
  detailCell.className = "run-detail-cell row-disclosure-cell";
  detailCell.colSpan = Math.max(1, selectedRow.cells.length);
  detailCell.style.padding = "0";
  detailCell.style.border = "0";
  detailRow.append(detailCell);
  selectedRow.after(detailRow);
  detailCell.append(panel);
  return true;
}

function remountActiveRunAttemptDisclosure() {
  if (!activeRunAttemptDisclosure || !elements.runAttemptDetail) return;
  const attemptKey = String(activeRunAttemptDisclosure.attemptKey || "");
  const launcher = [
    ...elements.attemptsBody.querySelectorAll('[data-attempt-action="view"]'),
  ].find(
    (candidate) =>
      String(
        candidate.dataset.attemptKey ||
          candidate.closest("[data-attempt-key]")?.dataset.attemptKey ||
          "",
      ) === attemptKey,
  );
  if (!launcher) {
    closeRunAttemptDisclosure({ restoreFocus: false });
    return;
  }
  if (activeRunAttemptDisclosure.launcher !== launcher) {
    setRunAttemptLauncherActive(activeRunAttemptDisclosure.launcher, false);
    activeRunAttemptDisclosure.launcher = launcher;
    setRunAttemptLauncherActive(launcher, true);
  }
  revealedPanelLaunchers.set(elements.runAttemptDetail, launcher);
}

function revealPanel(
  panel,
  { focusTarget = null, launcher = null, scroll = true } = {},
) {
  if (!disclosureRevealIsCurrent(panel, launcher)) return;
  if (!panel) return;
  const generation = (revealedPanelGenerations.get(panel) || 0) + 1;
  revealedPanelGenerations.set(panel, generation);
  const actualLauncher =
    launcher instanceof HTMLElement ? launcher : currentRevealLauncher();
  if (
    actualLauncher &&
    actualLauncher !== panel &&
    !panel.contains(actualLauncher)
  )
    revealedPanelLaunchers.set(panel, actualLauncher);
  const modal = panel.closest("dialog[data-panel-dialog]");
  if (
    !modal &&
    ((activeDisclosure?.panel === panel && activeDisclosure.rowOwned) ||
      panel === elements.runAttemptDetail)
  ) {
    mountRowDisclosure(panel, actualLauncher);
  }
  if (activeDisclosure?.panel === panel) activeDisclosure.revealed = true;
  panel.hidden = false;
  if (modal) SkynetDialog.open(modal, { launcher: actualLauncher });
  window.requestAnimationFrame(() => {
    if (panel.hidden || revealedPanelGenerations.get(panel) !== generation)
      return;
    const tutorialOwnsViewport = document.body.classList.contains(
      "has-active-tutorial",
    );
    if (tutorialOwnsViewport) return;
    // Row details now live after the whole table. Even callers that previously
    // relied on inline placement must reveal the heading when opening them.
    const detachedRowDetail = panel.classList.contains(
      "row-disclosure-detached",
    );
    if ((scroll || detachedRowDetail) && !modal) {
      const reducedMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)",
      ).matches;
      panel.scrollIntoView({
        behavior: reducedMotion ? "auto" : "smooth",
        block: detachedRowDetail ? "start" : "nearest",
        inline: "nearest",
      });
    }
    const target = focusTarget || panel;
    if (!target.matches("a[href], button, input, select, textarea, [tabindex]"))
      target.tabIndex = -1;
    target.focus?.({ preventScroll: true });
  });
}

function hideRevealedPanel(
  panel,
  fallbackLauncher = null,
  { restoreFocus = true, fromDisclosure = false } = {},
) {
  if (!panel) return;
  closedPanelGenerations.set(
    panel,
    (closedPanelGenerations.get(panel) || 0) + 1,
  );
  if (!fromDisclosure && activeDisclosure?.panel === panel) {
    closeActiveDisclosure({ restoreFocus });
    return;
  }
  revealedPanelGenerations.set(
    panel,
    (revealedPanelGenerations.get(panel) || 0) + 1,
  );
  panel.hidden = true;
  const modal = panel.closest("dialog[data-panel-dialog]");
  if (modal) SkynetDialog.close(modal);
  const detachedRowDetail = panel.classList.contains("row-disclosure-detached");
  unmountRowDisclosure(panel);
  if (panel === elements.evaluationDetail) stopEvaluationDetailPolling();
  const recordedLauncher = revealedPanelLaunchers.get(panel);
  const launcher = recordedLauncher?.isConnected
    ? recordedLauncher
    : fallbackLauncher;
  revealedPanelLaunchers.delete(panel);
  if (!restoreFocus) return;
  const closedGeneration = revealedPanelGenerations.get(panel);
  window.requestAnimationFrame(() => {
    if (
      !panel.hidden ||
      revealedPanelGenerations.get(panel) !== closedGeneration
    )
      return;
    if (launcher?.isConnected) {
      if (detachedRowDetail)
        launcher.scrollIntoView({
          behavior: "auto",
          block: "center",
          inline: "nearest",
        });
      launcher.focus({ preventScroll: true });
    }
  });
}

let activeDisclosure = null;
let disclosureSequence = 0;
const disclosureLauncherSnapshots = new WeakMap();
let activeRunAttemptDisclosure = null;
let runAttemptDisclosureSequence = 0;
let activeRunDetailId = null;
const runDetailAttemptRecords = new Map();

function disclosureRevealIsCurrent(panel, launcher = null) {
  if (!panel || panel.dataset.disclosureManaged !== "true" || !launcher)
    return true;
  return Boolean(
    activeDisclosure &&
      activeDisclosure.panel === panel &&
      String(activeDisclosure.token) ===
        String(panel.dataset.disclosureToken || ""),
  );
}

function disclosureToken(panel, launcher = null) {
  if (
    !activeDisclosure ||
    activeDisclosure.panel !== panel ||
    (launcher && activeDisclosure.launcher !== launcher)
  )
    return { closedGeneration: closedPanelGenerations.get(panel) || 0 };
  return activeDisclosure.token;
}

function disclosureTokenIsCurrent(panel, token) {
  if (token && typeof token === "object")
    return token.closedGeneration === (closedPanelGenerations.get(panel) || 0);
  return (
    token == null ||
    Boolean(
      activeDisclosure &&
        activeDisclosure.panel === panel &&
        activeDisclosure.token === token,
    )
  );
}

function setDisclosureLauncherActive(launcher, panel, token) {
  if (!launcher || panel?.closest("dialog[data-panel-dialog]")) return;
  launcher.classList.add("disclosure-launcher");
  if (!disclosureLauncherSnapshots.has(launcher)) {
    disclosureLauncherSnapshots.set(launcher, {
      html: launcher.innerHTML,
      className: launcher.className,
    });
  }
  launcher.setAttribute("aria-controls", panel.id);
  launcher.setAttribute("aria-expanded", "true");
  launcher.dataset.disclosureToken = String(token);
  launcher.classList.add("is-active");
  launcher.textContent = "Close";
  queueMicrotask(() => {
    if (
      activeDisclosure?.launcher !== launcher ||
      activeDisclosure.token !== token
    )
      return;
    launcher.classList.add("disclosure-launcher", "is-active");
    launcher.setAttribute("aria-expanded", "true");
    launcher.textContent = "Close";
  });
}

function restoreDisclosureLauncher(launcher, panel = null) {
  if (!launcher || panel?.closest("dialog[data-panel-dialog]")) return;
  const snapshot = disclosureLauncherSnapshots.get(launcher);
  if (snapshot) {
    launcher.innerHTML = snapshot.html;
    launcher.className = snapshot.className;
    disclosureLauncherSnapshots.delete(launcher);
  } else {
    launcher.classList.remove("is-active");
  }
  launcher.classList.add("disclosure-launcher");
  if (panel?.id) launcher.setAttribute("aria-controls", panel.id);
  launcher.setAttribute("aria-expanded", "false");
  delete launcher.dataset.disclosureToken;
}

function closeActiveDisclosure({ restoreFocus = true } = {}) {
  if (!activeDisclosure) return false;
  const closing = activeDisclosure;
  activeDisclosure = null;
  disclosureSequence += 1;
  if (closing.panel === elements.runDetail) resetRunAttemptContext();
  if (closing.panel === elements.evaluationDetail)
    stopEvaluationDetailPolling();
  restoreDisclosureLauncher(closing.launcher, closing.panel);
  delete closing.panel.dataset.disclosureToken;
  hideRevealedPanel(closing.panel, closing.launcher, {
    restoreFocus,
    fromDisclosure: true,
  });
  return true;
}

function closeDisclosurePanel(panel, fallbackLauncher = null) {
  if (activeDisclosure?.panel === panel) return closeActiveDisclosure();
  hideRevealedPanel(panel, fallbackLauncher);
  return true;
}

function toggleDisclosure(
  panel,
  revealKey,
  launcher,
  { rowOwned = false } = {},
) {
  if (panel?.closest("dialog[data-panel-dialog]")) rowOwned = false;
  if (!panel || !launcher || !revealKey) return { opened: true, token: null };
  const sameDisclosure = Boolean(
    activeDisclosure &&
      activeDisclosure.panel === panel &&
      activeDisclosure.revealKey === revealKey &&
      activeDisclosure.launcher === launcher,
  );
  if (sameDisclosure) {
    closeActiveDisclosure();
    return { opened: false, token: null };
  }
  if (activeDisclosure) closeActiveDisclosure({ restoreFocus: false });
  const token = ++disclosureSequence;
  panel.dataset.disclosureManaged = "true";
  panel.dataset.disclosureKey = revealKey;
  panel.dataset.disclosureToken = String(token);
  activeDisclosure = {
    panel,
    revealKey,
    launcher,
    token,
    rowOwned,
    revealed: false,
  };
  if (rowOwned && !mountRowDisclosure(panel, launcher)) {
    activeDisclosure = null;
    delete panel.dataset.disclosureToken;
    return { opened: false, token: null };
  }
  setDisclosureLauncherActive(launcher, panel, token);
  return { opened: true, token };
}

function disclosureRowId(launcher) {
  return String(
    launcher?.dataset.id ||
      launcher?.dataset.runId ||
      launcher?.dataset.experimentId ||
      launcher?.dataset.evaluationId ||
      launcher?.closest("[data-id]")?.dataset.id ||
      "unknown",
  );
}

function disclosureIntentForLauncher(launcher) {
  if (!launcher) return null;
  const fixed = {
    "show-collection-import": [
      elements.collectionImportDrawer,
      "collection-import:new",
    ],
    "show-collection-session-form": [
      elements.collectionSessionForm,
      "collection-session:new",
    ],
    "show-data-resource-form": [elements.dataResourceForm, "data-resource:new"],
    "show-data-derivation-form": [
      elements.dataDerivationForm,
      "data-derivation:new",
    ],
    "add-collection-adapter": [
      elements.collectionAdapterForm,
      "collection-adapter:new",
    ],
    "add-adapter": [elements.adapterEditor, "adapter:new"],
  }[launcher.id];
  if (fixed) return { panel: fixed[0], revealKey: fixed[1] };

  const id = disclosureRowId(launcher);
  if (
    launcher.closest("#experiments-body") &&
    launcher.dataset.experimentAction === "view"
  ) {
    return {
      panel: elements.experimentDetail,
      revealKey: `experiment:${id}`,
      rowOwned: true,
    };
  }
  if (launcher.closest("#runs-body") && launcher.dataset.runAction === "view") {
    return {
      panel: elements.runDetail,
      revealKey: `run:${id}`,
      rowOwned: true,
    };
  }
  if (
    launcher.closest("#evaluations-body") &&
    launcher.dataset.evaluationAction === "view"
  ) {
    return {
      panel: elements.evaluationDetail,
      revealKey: `evaluation:${id}`,
      rowOwned: true,
    };
  }
  if (
    launcher.closest("#collection-sessions-body") &&
    launcher.dataset.collectionSessionAction === "view"
  ) {
    return {
      panel: elements.collectionSessionDetail,
      revealKey: `collection-session:${id}`,
      rowOwned: true,
    };
  }

  if (launcher.closest("#data-resources-body")) {
    const action = launcher.dataset.resourceAction;
    if (action === "edit")
      return {
        panel: elements.dataResourceForm,
        revealKey: `data-resource:edit:${id}`,
        rowOwned: true,
      };
    if (action === "version")
      return {
        panel: elements.dataVersionForm,
        revealKey: `data-resource:version:${id}`,
        rowOwned: true,
      };
  }
  if (launcher.closest("#collection-adapters-body")) {
    const action = launcher.dataset.collectionAdapterAction;
    if (action === "edit")
      return {
        panel: elements.collectionAdapterForm,
        revealKey: `collection-adapter:edit:${id}`,
        rowOwned: true,
      };
    if (action === "session")
      return {
        panel: elements.collectionSessionForm,
        revealKey: `collection-session:new:${id}`,
        rowOwned: true,
      };
  }
  if (launcher.closest("#adapters-body")) {
    const action = launcher.dataset.adapterAction;
    if (action === "view" || action === "edit") {
      return {
        panel: elements.adapterEditor,
        revealKey: `adapter:${action}:${id}`,
        rowOwned: true,
      };
    }
  }
  return null;
}

function initializeDisclosureLauncher(launcher, intent) {
  if (!launcher || !intent?.panel) return;
  const modal = intent.panel.closest("dialog[data-panel-dialog]");
  if (modal) {
    launcher.classList.remove("disclosure-launcher");
    launcher.removeAttribute("aria-expanded");
    launcher.setAttribute("aria-haspopup", "dialog");
    launcher.setAttribute("aria-controls", modal.id);
    return;
  }
  launcher.classList.add("disclosure-launcher");
  launcher.setAttribute("aria-controls", intent.panel.id);
  if (activeDisclosure?.launcher !== launcher)
    launcher.setAttribute("aria-expanded", "false");
}

function synchronizeDisclosureLaunchers(root = document) {
  const launchers = [];
  if (root.matches?.("button, a")) launchers.push(root);
  launchers.push(...(root.querySelectorAll?.("button, a") || []));
  let replacement = null;
  for (const launcher of launchers) {
    const intent = disclosureIntentForLauncher(launcher);
    if (!intent?.panel) continue;
    initializeDisclosureLauncher(launcher, intent);
    if (
      activeDisclosure &&
      !activeDisclosure.launcher?.isConnected &&
      intent.panel === activeDisclosure.panel &&
      intent.revealKey === activeDisclosure.revealKey
    )
      replacement = launcher;
  }
  if (!activeDisclosure || activeDisclosure.launcher?.isConnected) return;
  if (!replacement) {
    closeActiveDisclosure({ restoreFocus: false });
    return;
  }
  const previousLauncher = activeDisclosure.launcher;
  restoreDisclosureLauncher(previousLauncher, activeDisclosure.panel);
  activeDisclosure.launcher = replacement;
  setDisclosureLauncherActive(
    replacement,
    activeDisclosure.panel,
    activeDisclosure.token,
  );
  revealedPanelLaunchers.set(activeDisclosure.panel, replacement);
  if (activeDisclosure.rowOwned)
    mountRowDisclosure(activeDisclosure.panel, replacement);
}

let disclosureSyncScheduled = false;
function scheduleDisclosureLauncherSync() {
  if (disclosureSyncScheduled) return;
  disclosureSyncScheduled = true;
  queueMicrotask(() => {
    disclosureSyncScheduled = false;
    synchronizeDisclosureLaunchers();
  });
}

function installDisclosureBehavior() {
  document.querySelectorAll("dialog[data-panel-dialog]").forEach((dialog) => {
    dialog.addEventListener("close", () => {
      if (dialog.open) return;
      const panel = dialog.querySelector(
        ":scope > form, :scope > aside, :scope > section",
      );
      if (panel === elements.runAttemptDetail)
        closeRunAttemptDisclosure({ restoreFocus: false, markDismissed: true });
      else hideRevealedPanel(panel, null, { restoreFocus: false });
    });
  });
  const closeTargets = {
    "close-experiment-detail": elements.experimentDetail,
    "close-run-detail": elements.runDetail,
    "close-evaluation-detail": elements.evaluationDetail,
    "close-collection-import": elements.collectionImportDrawer,
    "close-collection-session-detail": elements.collectionSessionDetail,
    "close-collection-session-form": elements.collectionSessionForm,
    "close-data-resource-form": elements.dataResourceForm,
    "cancel-data-resource": elements.dataResourceForm,
    "close-data-version-form": elements.dataVersionForm,
    "close-data-import-form": elements.dataImportForm,
    "close-data-derivation-form": elements.dataDerivationForm,
    "close-collection-adapter": elements.collectionAdapterForm,
    "close-adapter-editor": elements.adapterEditor,
  };
  const closeSelector = Object.keys(closeTargets)
    .map((id) => `#${id}`)
    .join(", ");
  document.addEventListener(
    "click",
    (event) => {
      const closeControl = event.target.closest?.(closeSelector);
      if (closeControl) {
        event.preventDefault();
        event.stopImmediatePropagation();
        const panel = closeTargets[closeControl.id];
        closeDisclosurePanel(panel);
        if (panel === elements.dataResourceForm) {
          resetDataResourceEditor(false);
          elements.showDataResourceForm.textContent = "New";
        }
        return;
      }
      const launcher = event.target.closest("button, a");
      const intent = disclosureIntentForLauncher(launcher);
      if (!intent?.panel) return;
      initializeDisclosureLauncher(launcher, intent);
      const result = toggleDisclosure(
        intent.panel,
        intent.revealKey,
        launcher,
        { rowOwned: intent.rowOwned === true },
      );
      if (!result.opened) {
        event.preventDefault();
        event.stopImmediatePropagation();
      }
    },
    true,
  );
  document.addEventListener("keydown", (event) => {
    if (
      event.key !== "Escape" ||
      event.defaultPrevented ||
      SkynetDialog.fullscreenOwnsEscape()
    )
      return;
    if (document.querySelector("dialog[open]")) return;
    if (activeRunAttemptDisclosure) {
      event.preventDefault();
      closeRunAttemptDisclosure();
      return;
    }
    if (!activeDisclosure) return;
    event.preventDefault();
    closeActiveDisclosure();
  });
  const disclosureObserver = new MutationObserver((records) => {
    for (const record of records) {
      if (
        record.type === "attributes" &&
        record.attributeName === "hidden" &&
        record.target === activeDisclosure?.panel &&
        // An old row's close can be observed before the new row's click handler.
        activeDisclosure.revealed &&
        record.target.hidden
      )
        closeActiveDisclosure();
      if (record.type === "childList") {
        scheduleDisclosureLauncherSync();
        if (activeRunAttemptDisclosure)
          queueMicrotask(remountActiveRunAttemptDisclosure);
      }
    }
  });
  disclosureObserver.observe(document.body, {
    subtree: true,
    childList: true,
    attributes: true,
    attributeFilter: ["hidden"],
  });
  synchronizeDisclosureLaunchers();
}

function startAdapterCreate(launcher = null) {
  adapterEditorState = {
    mode: "create",
    id: null,
    adapter: null,
    versions: [],
  };
  elements.adapterEditor.reset();
  elements.adapterEditor.querySelector(".dialog-body").hidden = false;
  elements.adapterEditor.hidden = false;
  elements.adapterEditorTitle.textContent = "New adapter";
  elements.adapterEditorSlug.value = "new-adapter";
  elements.adapterEditorName.value = "New adapter";
  elements.adapterDescription.value = "";
  elements.adapterChangeNote.value = "Initial adapter version";
  elements.adapterManifest.value = formatManifest({
    schema_version: "skynet.adapter/v1",
    slug: "new-adapter",
    display_name: "New adapter",
    description: "",
    aliases: [],
    repository_patterns: [],
    default_repository: null,
    legacy_handler: null,
    runtime: {
      allowed_backends: ["uv", "conda", "apptainer", "existing"],
      recommended_backend: null,
    },
    capabilities: {
      name: "new-adapter",
      version: 1,
      runtime_backends: ["uv", "conda", "apptainer", "existing"],
      supports_multi_gpu_single_node: true,
      supports_resume: false,
      supports_checkpoint_signal: false,
      supports_evaluation_resume: false,
      minimum_gpus: 1,
      recommended_gpus: 1,
      maximum_gpus: 1,
      evaluation_adapters: [],
    },
    train: {
      argv: [],
      static_args: [],
      parameter_flags: {},
      supported_canonical_fields: [],
      resume_argv: [],
      checkpoint_globs: [],
      environment: {},
      required_values: [],
    },
    evaluations: [],
    warnings: [],
    todos: [],
  });
  elements.adapterValidationRepository.value = "";
  elements.adapterValidationRevision.value = "";
  renderAdapterVersions([]);
  setAdapterEditorMode("create");
  revealPanel(elements.adapterEditor, {
    focusTarget: elements.adapterEditorSlug,
    launcher,
  });
}

async function openAdapter(id, editable = false, launcher = null) {
  const revealLauncher = launcher || currentRevealLauncher();
  const requestToken = disclosureToken(elements.adapterEditor, revealLauncher);
  elements.adapterEditor.hidden = false;
  elements.adapterEditorMode.textContent = editable
    ? "NEW IMMUTABLE VERSION"
    : "ADAPTER DETAIL";
  elements.adapterEditorTitle.textContent = "Loading adapter...";
  elements.adapterEditorStatus.textContent = "Loading";
  elements.adapterEditor.querySelector(".dialog-body").hidden = true;
  revealPanel(elements.adapterEditor, {
    focusTarget: elements.adapterEditorTitle,
    launcher: revealLauncher,
  });
  try {
    const payload = await api(`/api/adapters/${encodeURIComponent(id)}`);
    if (!disclosureTokenIsCurrent(elements.adapterEditor, requestToken)) return;
    const adapter = entityFrom(payload, "adapter");
    const versions = Array.isArray(adapter.versions)
      ? adapter.versions
      : listFrom(payload, ["versions"]);
    adapterEditorState = {
      mode: editable ? "edit" : "view",
      id: adapterId(adapter) || id,
      adapter,
      versions,
    };
    const manifest = adapterManifest(adapter);
    const defaults = repositoryDefaults(manifest, adapter);
    elements.adapterEditorSlug.value =
      manifest.slug ||
      adapter.seed_key ||
      adapter.slug ||
      adapterId(adapter) ||
      id;
    elements.adapterEditorName.value = adapter.name || adapter.label || id;
    elements.adapterManifest.value = formatManifest(manifest);
    elements.adapterDescription.value =
      adapter.description || manifest.description || "";
    elements.adapterChangeNote.value = "";
    elements.adapterValidationRepository.value = defaults.url;
    const jobRepoMatches =
      defaults.url && defaults.url === elements.experimentSource.value.trim();
    elements.adapterValidationRevision.value = jobRepoMatches
      ? elements.experimentRevision.value
      : "";
    elements.adapterEditorTitle.textContent =
      adapter.name || adapter.label || id;
    renderAdapterVersions(versions);
    elements.adapterEditor.querySelector(".dialog-body").hidden = false;
    setAdapterEditorMode(
      editable && !adapterArchived(adapter) && adapter.editable !== false
        ? "edit"
        : "view",
    );
    revealPanel(elements.adapterEditor, {
      focusTarget: elements.adapterEditorTitle,
      launcher: revealLauncher,
    });
  } catch (error) {
    if (!disclosureTokenIsCurrent(elements.adapterEditor, requestToken)) return;
    closeDisclosurePanel(elements.adapterEditor, revealLauncher);
    showToast(`Adapter detail failed: ${error.message}`, true);
  }
}

async function saveAdapter(event) {
  event.preventDefault();
  if (!elements.adapterEditor.reportValidity()) return;
  let manifest;
  try {
    manifest = normalizedEditorManifest();
  } catch (error) {
    showToast(error.message, true);
    elements.adapterManifest.focus();
    return;
  }
  elements.saveAdapter.disabled = true;
  try {
    let result;
    if (adapterEditorState.mode === "create") {
      result = await api("/api/adapters", {
        method: "POST",
        body: JSON.stringify({
          name: elements.adapterEditorName.value.trim(),
          manifest,
          description: elements.adapterDescription.value.trim(),
          repository_url:
            elements.adapterValidationRepository.value.trim() || null,
          change_note: elements.adapterChangeNote.value.trim() || null,
        }),
      });
    } else {
      result = await api(
        `/api/adapters/${encodeURIComponent(adapterEditorState.id)}`,
        {
          method: "PUT",
          body: JSON.stringify({
            manifest,
            name: elements.adapterEditorName.value.trim(),
            description: elements.adapterDescription.value.trim(),
            repository_url:
              elements.adapterValidationRepository.value.trim() || null,
            expected_latest_version: Number(
              adapterVersion(adapterEditorState.adapter),
            ),
            change_note: elements.adapterChangeNote.value.trim() || null,
          }),
        },
      );
    }
    const saved = entityFrom(result, "adapter");
    const id =
      adapterId(saved) ||
      adapterEditorState.id ||
      elements.adapterEditorSlug.value.trim();
    showToast(
      adapterEditorState.mode === "create"
        ? "Adapter created."
        : "Immutable adapter version created.",
    );
    await loadAdapters(true);
    await openAdapter(id, false);
  } catch (error) {
    showToast(`Adapter save failed: ${error.message}`, true);
  } finally {
    elements.saveAdapter.disabled = false;
  }
}

async function validateAdapterManifest() {
  let manifest;
  try {
    manifest = normalizedEditorManifest();
  } catch (error) {
    showToast(error.message, true);
    return;
  }
  const repository = elements.adapterValidationRepository.value.trim();
  const revision = elements.adapterValidationRevision.value.trim();
  if (!repository || !revision) {
    showToast("Repository URL and revision are required for validation.", true);
    return;
  }
  elements.validateAdapter.disabled = true;
  elements.adapterValidationReport.hidden = false;
  elements.adapterValidationReport.classList.remove("is-error", "is-success");
  elements.adapterValidationReport.textContent =
    "Validating manifest against repository...";
  try {
    const isSavedView =
      adapterEditorState.mode === "view" && adapterEditorState.id;
    const version = Number(adapterVersion(adapterEditorState.adapter));
    const result = await api(
      isSavedView
        ? `/api/adapters/${encodeURIComponent(adapterEditorState.id)}/validate`
        : "/api/adapters/validate",
      {
        method: "POST",
        body: JSON.stringify(
          isSavedView
            ? {
                repository_url: repository,
                source_revision: revision,
                gateway: elements.gateway.value,
                version_number: Number.isInteger(version) ? version : null,
              }
            : {
                manifest,
                repository,
                revision,
                gateway: elements.gateway.value,
              },
        ),
      },
    );
    elements.adapterValidationReport.textContent = JSON.stringify(
      result,
      null,
      2,
    );
    const failed =
      result.valid === false ||
      ["INVALID", "ERROR"].includes(
        String(result.status || "").toUpperCase(),
      ) ||
      (Array.isArray(result.errors) && result.errors.length > 0);
    elements.adapterValidationReport.classList.add(
      failed ? "is-error" : "is-success",
    );
  } catch (error) {
    elements.adapterValidationReport.textContent = `Validation failed: ${error.message}`;
    elements.adapterValidationReport.classList.add("is-error");
  } finally {
    elements.validateAdapter.disabled = false;
  }
}

async function cloneAdapter(id) {
  const source = adapterRows.find((adapter) => adapterId(adapter) === id);
  const name = await askUserDialog(
    "Name for the duplicated adapter:",
    `${source?.name || "Adapter"} copy`,
  );
  if (!name) return;
  const changeNote = await askUserDialog(
    "Initial version change note:",
    `Cloned from ${source?.name || id}`,
  );
  if (changeNote === null) return;
  try {
    const result = await api(`/api/adapters/${encodeURIComponent(id)}/clone`, {
      method: "POST",
      body: JSON.stringify({
        name: name.trim(),
        version_number: Number(adapterVersion(source)) || null,
        change_note: changeNote.trim() || null,
      }),
    });
    const cloned = entityFrom(result, "adapter");
    const clonedId = adapterId(cloned);
    if (!clonedId) throw new Error("Clone API returned no adapter ID.");
    showToast("Adapter duplicated.");
    await loadAdapters(true);
    await openAdapter(clonedId, true);
  } catch (error) {
    showToast(`Adapter duplication failed: ${error.message}`, true);
  }
}

async function setAdapterArchived(id, archived) {
  const verb = archived ? "Archive" : "Restore";
  if (
    !(await askUserDialog(
      `${verb} adapter ${id}? Existing experiment snapshots will not change.`,
    ))
  )
    return;
  try {
    await api(
      archived
        ? `/api/adapters/${encodeURIComponent(id)}`
        : `/api/adapters/${encodeURIComponent(id)}/restore`,
      {
        method: archived ? "DELETE" : "POST",
      },
    );
    showToast(`Adapter ${archived ? "archived" : "restored"}.`);
    closeDisclosurePanel(elements.adapterEditor, elements.addAdapter);
    await loadAdapters(true);
  } catch (error) {
    showToast(`${verb} failed: ${error.message}`, true);
  }
}

function evaluationSuiteId(suite) {
  return String(suite?.id || suite?.slug || suite?.name || "");
}

function selectedEvaluationSuite() {
  return (
    evaluationSuites.find(
      (suite) => evaluationSuiteId(suite) === elements.evaluationSuite.value,
    ) || null
  );
}

function evaluationSuiteScope(runId = "") {
  const normalizedRunId = String(runId || "").trim();
  return normalizedRunId ? `run:${normalizedRunId}` : "global";
}

function populateEvaluationSuites(preferredSuiteId = "", { runId = "" } = {}) {
  if (!evaluationSuites.length) {
    elements.evaluationSuite.innerHTML =
      '<option value="">No suites available</option>';
    elements.evaluationSuiteStatus.textContent = runId
      ? "No registered evaluation suites are available."
      : "Evaluation suite API is unavailable or empty.";
    elements.evaluationSuite.setCustomValidity(
      runId
        ? "No compatible evaluation suites are available for this training run."
        : "No evaluation suites are available.",
    );
    updateEvaluationEnvironmentFromSuite();
    return;
  }
  const desiredSuiteId = String(
    pendingEvaluationSuiteId ||
      preferredSuiteId ||
      elements.evaluationSuite.value ||
      "",
  );
  setSelectOptions(
    elements.evaluationSuite,
    evaluationSuites.map((suite) => ({
      ...suite,
      label: `${suite.label || suite.name}${suite.compatibility ? ` — ${suite.compatibility.label}` : ""}`,
    })),
  );
  elements.evaluationSuite.insertAdjacentHTML(
    "afterbegin",
    '<option value="">Choose a suite...</option>',
  );
  const desiredExists = evaluationSuites.some(
    (suite) => evaluationSuiteId(suite) === desiredSuiteId,
  );
  const firstCompatibleSuiteId = runId
    ? evaluationSuiteId(
        evaluationSuites.find(
          (suite) => suite.is_default && evaluationSuiteIsRunnable(suite),
        ) || {},
      )
    : "";
  const selectedSuiteId = desiredExists
    ? desiredSuiteId
    : firstCompatibleSuiteId;
  elements.evaluationSuite.value = selectedSuiteId;
  elements.evaluationSuite.setCustomValidity("");
  elements.evaluationSuite.removeAttribute("aria-invalid");
  const selectedSuite = evaluationSuites.find(
    (suite) => evaluationSuiteId(suite) === selectedSuiteId,
  );
  const selectionMessage =
    runId && selectedSuiteId && !desiredExists
      ? desiredSuiteId
        ? `Suite ${desiredSuiteId} is not compatible with this run. Selected ${selectedSuite?.label || selectedSuite?.name || selectedSuiteId}, the first compatible suite.`
        : `Selected ${selectedSuite?.label || selectedSuite?.name || selectedSuiteId}, the first compatible suite for this run.`
      : desiredSuiteId && !desiredExists
        ? `Unsupported: the previously selected suite is not compatible with this run. Choose a listed suite.`
        : `${evaluationSuites.length} registered suites available. Choose a suite to check compatibility.`;
  elements.evaluationSuiteStatus.textContent = selectionMessage;
  updateEvaluationEnvironmentFromSuite({
    preserveTasks: desiredExists && selectedSuiteId === desiredSuiteId,
  });
  if (elements.evaluationRunId.value.trim())
    scheduleEvaluationTargetValidation();
}

async function loadEvaluationSuites(
  force = false,
  explicitRunId = undefined,
  { apply = true } = {},
) {
  const runId = String(
    explicitRunId === undefined
      ? elements.evaluationRunId.value
      : explicitRunId || "",
  ).trim();
  const scope = evaluationSuiteScope(runId);
  const desiredSuiteId = String(
    pendingEvaluationSuiteId || elements.evaluationSuite.value || "",
  );
  const currentFormScope = () =>
    evaluationSuiteScope(elements.evaluationRunId.value);
  const applySuites = (suites, error = null) => {
    if (!apply) return;
    if (!runId) {
      setSelectOptions(elements.experimentEvaluationSuites, suites);
      if (Array.isArray(pendingEvaluationSuiteDefaults))
        applyEvaluationSuiteDefaults(pendingEvaluationSuiteDefaults);
    }
    if (currentFormScope() !== scope) return;
    evaluationSuites = suites;
    evaluationSuitesScope = scope;
    populateEvaluationSuites(desiredSuiteId, { runId });
    const reason = evaluationSuiteReasons.get(scope);
    if (error)
      elements.evaluationSuiteStatus.textContent = `Could not load evaluation suites: ${error.message}`;
    else if (!suites.length && reason)
      elements.evaluationSuiteStatus.textContent = reason;
  };

  if (!force && evaluationSuitesCache.has(scope)) {
    const suites = evaluationSuitesCache.get(scope);
    applySuites(suites);
    return suites;
  }

  let requestPromise = !force ? evaluationSuitesPromises.get(scope) : null;
  if (!requestPromise) {
    const generation = (evaluationSuiteRequestGenerations.get(scope) || 0) + 1;
    evaluationSuiteRequestGenerations.set(scope, generation);
    const endpoint = runId
      ? `/api/evaluation-suites?run_id=${encodeURIComponent(runId)}`
      : "/api/evaluation-suites";
    requestPromise = (async () => {
      try {
        const payload = await api(endpoint);
        const suites = listFrom(payload, ["evaluation_suites", "suites"]);
        if (evaluationSuiteRequestGenerations.get(scope) === generation) {
          evaluationSuitesCache.set(scope, suites);
          evaluationSuiteReasons.set(
            scope,
            (payload.unavailable_suites || [])
              .map((item) => item.reason)
              .filter(Boolean)
              .join(" "),
          );
        }
        return { suites, error: null, generation };
      } catch (error) {
        return { suites: [], error, generation };
      }
    })();
    evaluationSuitesPromises.set(scope, requestPromise);
    requestPromise.finally(() => {
      if (evaluationSuitesPromises.get(scope) === requestPromise)
        evaluationSuitesPromises.delete(scope);
    });
  }

  const result = await requestPromise;
  if (evaluationSuiteRequestGenerations.get(scope) === result.generation) {
    applySuites(result.suites, result.error);
  }
  if (!apply && result.error) throw result.error;
  return result.suites;
}

async function refreshAfterDeletion(kind) {
  if (["dataset", "prepared", "local-copy"].includes(kind)) {
    SkynetDialog.close(document.getElementById("prepared-dataset-dialog"));
    await Promise.all([
      loadDataRegistry(true),
      loadDataBundles(true),
      window.refreshPreparedDatasets?.(),
    ]);
    return;
  }
  if (kind === "adapter" || kind === "suite") {
    // Invalidate pending responses too, so a pre-delete response cannot revive an option.
    for (const [scope, generation] of evaluationSuiteRequestGenerations) {
      evaluationSuiteRequestGenerations.set(scope, generation + 1);
    }
    evaluationSuitesCache.clear();
    evaluationSuitesPromises.clear();
    evaluationSuiteReasons.clear();
    if (kind === "adapter") await loadAdapters(true);
    await loadEvaluationCatalog(true);
    await loadEvaluationSuites(true, "");
    if (elements.evaluationRunId.value.trim()) await loadEvaluationSuites(true);
    return;
  }
  await Promise.all([
    loadExperiments(true),
    loadRuns(true),
    loadEvaluations(true),
  ]);
}

function evaluationCatalogTasks(suite) {
  const config = evaluationSuiteConfig(suite);
  const labels = new Map(
    (suite.task_options || config.task_options || []).map((task) => [
      task.id,
      task.label,
    ]),
  );
  return evaluationSuiteTasks(suite).map((task) => ({
    ...task,
    label: labels.get(task.id) || task.label,
  }));
}

function evaluationCatalogDescription(suite) {
  return (
    evaluationSuiteConfig(suite).task_selection_reason ||
    suite.task_selection_reason ||
    suite.description ||
    ""
  );
}

function openEvaluationSuite(id, launcher) {
  const suite = evaluationCatalogRows.find(
    (suite) => String(suite.id) === String(id),
  );
  if (!suite)
    return showToast(
      "This suite is no longer available. Refresh the list.",
      true,
    );
  const panel = document.getElementById("evaluation-suite-detail");
  const tasks = evaluationCatalogTasks(suite);
  document.getElementById("evaluation-suite-detail-title").textContent =
    suite.label || suite.name;
  document.getElementById("evaluation-suite-detail-meta").innerHTML =
    keyValueHtml([
      ["Suite", suite.name],
      ["Suite version", suite.version || suite.suite_version],
      ["Environment", suite.evaluator || suite.evaluator_adapter],
      ["Updated", formatDate(suite.updated_at || suite.created_at)],
    ]);
  const description = document.getElementById("evaluation-suite-description");
  description.textContent = evaluationCatalogDescription(suite);
  description.hidden = !description.textContent;
  document.getElementById("evaluation-suite-tasks").innerHTML = tasks.length
    ? tasks
        .map(
          (task) =>
            `<tr><td class="wrap-cell">${escapeHtml(task.label)}</td><td class="wrap-cell"><code>${escapeHtml(task.id)}</code></td></tr>`,
        )
        .join("")
    : emptyRow(
        2,
        evaluationSuiteConfig(suite).task_source === "training_dataset"
          ? "Tasks are resolved from the selected training dataset."
          : "No tasks registered.",
      );
  revealPanel(panel, {
    launcher,
    focusTarget: document.getElementById("evaluation-suite-detail-title"),
  });
}

function renderEvaluationCatalog() {
  const query = document
    .querySelector("#evaluation-suite-search")
    .value.trim()
    .toLowerCase();
  const rows = evaluationCatalogRows.filter((suite) =>
    [
      suite.label,
      suite.name,
      suite.version || suite.suite_version,
      suite.evaluator || suite.evaluator_adapter,
      evaluationCatalogDescription(suite),
      ...evaluationCatalogTasks(suite).flatMap((task) => [task.id, task.label]),
    ]
      .join(" ")
      .toLowerCase()
      .includes(query),
  );
  document.querySelector("#evaluation-suite-count").textContent = query
    ? `${rows.length} of ${evaluationCatalogRows.length} suites`
    : `${rows.length} suites`;
  document.querySelector("#evaluation-suites-body").innerHTML =
    rows
      .map((suite) => {
        const tasks = evaluationCatalogTasks(suite);
        const taskCount = tasks.length
          ? `${tasks.length} task${tasks.length === 1 ? "" : "s"}`
          : evaluationSuiteConfig(suite).task_source === "training_dataset"
            ? "From training data"
            : "0 tasks";
        return `<tr>
      <td><strong>${escapeHtml(suite.label || suite.name)}</strong><span class="secondary">${escapeHtml(suite.name)}</span></td>
      <td>${escapeHtml(suite.version || suite.suite_version)}</td>
      <td>${escapeHtml(suite.evaluator || suite.evaluator_adapter)}</td>
      <td>${taskCount}</td>
      <td>${escapeHtml(formatDate(suite.updated_at || suite.created_at))}</td>
      <td class="row-actions"><button type="button" data-suite-view="${escapeHtml(suite.id)}" aria-haspopup="dialog" aria-controls="evaluation-suite-detail-dialog">View</button>${suite.can_delete ? `<button type="button" data-delete-kind="suite" data-delete-id="${escapeHtml(suite.id)}">Delete</button>` : ""}</td>
    </tr>`;
      })
      .join("") ||
    emptyRow(
      6,
      query
        ? "No suites match the filter."
        : "No evaluation suites registered.",
    );
}

async function loadEvaluationCatalog(force = false) {
  const request = ++evaluationCatalogRequest;
  const errorBox = document.querySelector("#evaluation-catalog-error");
  clearNotice(errorBox);
  elements.refreshEvaluations.disabled = true;
  try {
    // The registry has its own state: browsing it must not replace the run's compatible suites.
    const suites = await loadEvaluationSuites(force, "", { apply: false });
    if (request !== evaluationCatalogRequest) return;
    evaluationCatalogRows = suites;
    renderEvaluationCatalog();
  } catch (error) {
    if (request !== evaluationCatalogRequest) return;
    showNotice(
      errorBox,
      `Evaluation suites could not be loaded: ${error.message}`,
    );
    document.querySelector("#evaluation-suite-count").textContent =
      "Unavailable";
    document.querySelector("#evaluation-suites-body").innerHTML = emptyRow(
      6,
      "Evaluation suites could not be loaded.",
    );
  } finally {
    if (request === evaluationCatalogRequest)
      elements.refreshEvaluations.disabled = false;
  }
}

function resetSourceSelect(select, message) {
  select.innerHTML = `<option value="">${escapeHtml(message)}</option>`;
  select.value = "";
  select.disabled = true;
}

function setSourceRefStatus(message, state = "") {
  elements.sourceRefStatus.textContent = message;
  elements.sourceRefStatus.className = `source-ref-status${state ? ` is-${state}` : ""}`;
}

function setRuntimeStatus(message, state = "") {
  elements.runtimeInspectionStatus.textContent = message;
  elements.runtimeInspectionStatus.className = `runtime-inspection-status${state ? ` is-${state}` : ""}`;
}

function resetRuntimeInspection(
  message = "Select a commit to inspect its runtime.",
  state = "empty",
) {
  ++runtimeInspectionRequest;
  runtimeInspectionKey = "";
  runtimeCandidates = [];
  runtimeProfiles = [];
  clearRepositoryInputOptions({ render: true });
  elements.experimentRuntime.innerHTML = `<option value="">${escapeHtml(message)}</option>`;
  elements.experimentRuntime.value = "";
  elements.experimentRuntime.disabled = true;
  elements.experimentRuntimeProfileSelect.innerHTML =
    '<option value="">Choose a runtime backend first</option>';
  elements.experimentRuntimeProfileSelect.value = "";
  elements.experimentRuntimeProfileSelect.disabled = true;
  elements.experimentRuntimeProfile.value = "";
  elements.experimentRuntimeProfile.disabled = false;
  elements.experimentRuntimeProfile.required = false;
  elements.runtimeProfileHelp.textContent =
    "A detected runtime fills this from repository evidence. Manual choices may require an explicit value.";
  elements.runtimeEvidence.hidden = true;
  elements.runtimeEvidence.innerHTML = "";
  setRuntimeStatus(message, state);
  updateExperimentSubmitState();
}

function normalizeRuntimeProfiles(payload) {
  const rows = Array.isArray(payload?.runtime_profiles)
    ? payload.runtime_profiles
    : [];
  return rows
    .map((row) => ({
      id: String(row.id || "").trim(),
      label: String(row.label || row.id || "Unnamed runtime profile"),
      description: String(row.description || ""),
      backend: String(row.backend || "")
        .trim()
        .toLowerCase(),
      environmentPath: row.environment_path || null,
      containerImage: row.container_image || null,
      lockFile: row.lock_file || null,
      resolvedLocation:
        row.resolved_location ||
        row.environment_path ||
        row.container_image ||
        row.lock_file ||
        null,
      versions: row.versions || {},
      status: row.runtime_verified ? "runtime_verified" : "configured",
      runtimeVerified: Boolean(row.runtime_verified),
      verification: row.verification || {},
    }))
    .filter((row) => row.id && row.backend);
}

function runtimeEvidenceItems(candidate) {
  const evidence = Array.isArray(candidate.evidence)
    ? candidate.evidence
    : candidate.evidence
      ? [candidate.evidence]
      : [];
  return evidence.map((item) =>
    typeof item === "string" ? item : compactJson(item, 180),
  );
}

function normalizeRuntimeCandidates(payload) {
  const rawRows = listFrom(payload, [
    "runtime_candidates",
    "candidates",
    "runtimes",
  ]);
  const merged = new Map();
  rawRows.forEach((raw, index) => {
    const row = typeof raw === "string" ? { type: raw } : raw || {};
    const value = String(
      row.type || row.runtime || row.id || row.name || "",
    ).trim();
    if (!value) return;
    const status = String(row.status || "").toLowerCase();
    const runnable =
      row.runnable !== false &&
      !["blocked", "invalid", "unavailable", "unsupported"].includes(status);
    const confidence = String(
      row.confidence || row.strength || (row.strong ? "strong" : "unknown"),
    ).toLowerCase();
    const candidate = {
      value,
      label: String(row.label || row.display_name || row.name || value),
      runnable,
      confidence,
      score: Number(row.score ?? row.confidence_score),
      evidence: runtimeEvidenceItems(row),
      reason: row.reason || row.message || "",
      profile:
        row.profile ||
        row.configuration?.profile ||
        row.configuration?.environment_path ||
        row.configuration?.container_image ||
        row.configuration?.lock_file ||
        null,
      manual: false,
      requiresProfile: false,
      index,
    };
    if (!merged.has(value)) merged.set(value, candidate);
    else {
      const current = merged.get(value);
      current.runnable ||= candidate.runnable;
      current.evidence.push(...candidate.evidence);
      if (!current.profile) current.profile = candidate.profile;
      if (candidate.confidence === "strong" || candidate.confidence === "high")
        current.confidence = candidate.confidence;
    }
  });
  return [...merged.values()];
}

function adapterAllowedRuntimeBackends(adapter = selectedAdapter()) {
  const manifest = adapterManifest(adapter);
  const sources = [
    manifest.runtime?.allowed_backends,
    manifest.capabilities?.runtime_backends,
    adapter?.runtime,
    adapter?.runtimes,
    adapter?.capabilities?.runtime_backends,
  ];
  const raw = sources.find(
    (value) =>
      Array.isArray(value) || value instanceof Set || typeof value === "string",
  );
  const values =
    raw instanceof Set
      ? [...raw]
      : Array.isArray(raw)
        ? raw
        : typeof raw === "string"
          ? raw.split(",")
          : [];
  return [
    ...new Set(
      values
        .map((value) => String(value).trim().toLowerCase())
        .map((value) => (value === "container" ? "apptainer" : value))
        .filter(Boolean),
    ),
  ];
}

function runtimeDisplayName(backend) {
  return (
    {
      uv: "uv",
      conda: "Conda",
      apptainer: "Apptainer",
      existing: "Existing environment",
    }[backend] || backend
  );
}

function manualRuntimeCandidate(backend, rejectedCandidate = null) {
  const evidence = [
    "Allowed by the selected adapter manifest; not automatically detected",
  ];
  if (rejectedCandidate?.evidence?.length) {
    evidence.push(
      `Automatic evidence was not runnable: ${rejectedCandidate.evidence.join(" / ")}`,
    );
  }
  return {
    value: backend,
    label: `${runtimeDisplayName(backend)} (manual)`,
    runnable: true,
    confidence: "manual selection",
    score: Number.NaN,
    evidence,
    reason: "",
    profile: null,
    manual: true,
    requiresProfile: backend !== "existing",
  };
}

function prepareRuntimeChoices(candidates) {
  const allowed = new Set(adapterAllowedRuntimeBackends());
  candidates.forEach((candidate) => {
    if (allowed.size && !allowed.has(candidate.value)) {
      candidate.runnable = false;
      candidate.reason = `The selected adapter does not allow ${candidate.value}.`;
    }
  });
  runtimeProfiles.forEach((profile) => {
    if (allowed.size && !allowed.has(profile.backend)) return;
    const index = candidates.findIndex(
      (candidate) => candidate.value === profile.backend,
    );
    if (index < 0) candidates.push(manualRuntimeCandidate(profile.backend));
    else if (!candidates[index].runnable) {
      candidates.splice(
        index,
        1,
        manualRuntimeCandidate(profile.backend, candidates[index]),
      );
    }
  });
  const strong = candidates.filter(
    (candidate) => candidate.runnable && isStrongRuntimeCandidate(candidate),
  );
  if (strong.length) return candidates;

  allowed.forEach((backend) => {
    const index = candidates.findIndex(
      (candidate) => candidate.value === backend,
    );
    if (index < 0) candidates.push(manualRuntimeCandidate(backend));
    else if (!candidates[index].runnable)
      candidates.splice(
        index,
        1,
        manualRuntimeCandidate(backend, candidates[index]),
      );
  });
  return candidates;
}

function selectedRuntimeProfile() {
  const id = elements.experimentRuntimeProfileSelect.value;
  return runtimeProfiles.find((profile) => profile.id === id) || null;
}

function populateRuntimeProfileSelect(backend) {
  const matching = runtimeProfiles.filter(
    (profile) => profile.backend === backend,
  );
  elements.experimentRuntimeProfileSelect.innerHTML = [
    '<option value="">Custom or repository-detected configuration</option>',
    ...matching.map(
      (profile) =>
        `<option value="${escapeHtml(profile.id)}">${escapeHtml(profile.label)} / ${escapeHtml(profile.status.replaceAll("_", " "))}</option>`,
    ),
  ].join("");
  elements.experimentRuntimeProfileSelect.value = "";
  elements.experimentRuntimeProfileSelect.disabled = !matching.length;
}

function isStrongRuntimeCandidate(candidate) {
  return (
    ["strong", "high", "exact", "locked"].includes(candidate.confidence) ||
    (Number.isFinite(candidate.score) && candidate.score >= 0.8)
  );
}

function renderRuntimeEvidence() {
  if (!runtimeCandidates.length) {
    elements.runtimeEvidence.hidden = true;
    elements.runtimeEvidence.innerHTML = "";
    return;
  }
  elements.runtimeEvidence.hidden = false;
  elements.runtimeEvidence.innerHTML = runtimeCandidates
    .map((candidate) => {
      const evidence = candidate.evidence.length
        ? candidate.evidence.join(" / ")
        : candidate.reason || "No evidence detail returned";
      const selected = candidate.value === elements.experimentRuntime.value;
      return `
      <div class="runtime-candidate ${selected ? "is-selected" : ""} ${candidate.runnable ? "" : "is-blocked"}">
        <strong>${escapeHtml(candidate.label)}</strong>
        <span>${
          candidate.manual
            ? `manual / ${candidate.requiresProfile ? "configuration required" : "explicit selection"}`
            : `${escapeHtml(candidate.confidence)} / detected ${candidate.runnable ? "runnable" : "blocked"}`
        }</span>
        <p>${escapeHtml(evidence)}</p>
      </div>`;
    })
    .join("");
}

function applyRuntimeSelection(explicit = true) {
  const candidate = runtimeCandidates.find(
    (row) => row.value === elements.experimentRuntime.value,
  );
  if (!candidate) return;
  populateRuntimeProfileSelect(candidate.value);
  applyRuntimeProfileSelection(explicit);
}

function applyRuntimeProfileSelection(explicit = true) {
  const candidate = runtimeCandidates.find(
    (row) => row.value === elements.experimentRuntime.value,
  );
  if (!candidate) {
    updateExperimentSubmitState();
    return;
  }
  const namedProfile = selectedRuntimeProfile();
  if (namedProfile) {
    elements.experimentRuntimeProfile.required = false;
    elements.experimentRuntimeProfile.disabled = true;
    elements.experimentRuntimeProfile.value =
      namedProfile.resolvedLocation || "Resolved by operator profile";
    const versions = Object.entries(namedProfile.versions)
      .map(([name, version]) => `${name} ${version}`)
      .join(", ");
    elements.runtimeProfileHelp.textContent = `${namedProfile.status.replaceAll("_", " ")}: ${namedProfile.id}${versions ? ` / ${versions}` : ""}. The complete profile is snapshotted at submission.`;
  } else if (candidate.manual) {
    elements.experimentRuntimeProfile.disabled = false;
    elements.experimentRuntimeProfile.required = Boolean(
      candidate.requiresProfile,
    );
    elements.experimentRuntimeProfile.value = "";
    if (candidate.value === "conda") {
      elements.experimentRuntimeProfile.placeholder =
        "/path/to/conda/environment";
      elements.runtimeProfileHelp.textContent =
        "Required: provide an existing Conda environment path/profile.";
    } else if (candidate.value === "apptainer") {
      elements.experimentRuntimeProfile.placeholder =
        "/path/to/image.sif or immutable image reference";
      elements.runtimeProfileHelp.textContent =
        "Required: provide an Apptainer image path or immutable reference.";
    } else if (candidate.value === "existing") {
      elements.experimentRuntimeProfile.placeholder =
        "Optional environment detail";
      elements.runtimeProfileHelp.textContent =
        "No path is required. This explicit choice uses the compute node's existing environment.";
    } else {
      elements.experimentRuntimeProfile.placeholder =
        "Path to the pinned runtime input";
      elements.runtimeProfileHelp.textContent =
        "Required: provide the pinned runtime input for this manual choice.";
    }
  } else {
    elements.experimentRuntimeProfile.disabled = false;
    elements.experimentRuntimeProfile.required = false;
    elements.experimentRuntimeProfile.placeholder =
      "Resolved from repository evidence";
    elements.runtimeProfileHelp.textContent =
      "Resolved from dependency files at the selected exact commit.";
    if (candidate.profile)
      elements.experimentRuntimeProfile.value = candidate.profile;
  }
  renderRuntimeEvidence();
  setRuntimeStatus(
    namedProfile
      ? `Selected named runtime profile: ${namedProfile.label} (${namedProfile.status.replaceAll("_", " ")}).`
      : candidate.manual
        ? `Selected manual runtime: ${candidate.label}. ${candidate.requiresProfile ? "Complete its required configuration." : "No profile path is required."}`
        : `${explicit ? "Selected detected runtime" : "Auto-selected single strong candidate"}: ${candidate.label}.`,
    explicit ? "selected" : "success",
  );
  updateExperimentSubmitState();
}

function adapterRepositoryInspectionFingerprint(adapter) {
  const choiceSources = declaredAdapterInputFields(adapter)
    .filter((field) => field.choice_source)
    .map((field) => ({ path: field.path, choice_source: field.choice_source }));
  return adapterManifestFingerprint({
    choice_sources: choiceSources,
    allowed_runtime_backends: adapterAllowedRuntimeBackends(adapter).sort(),
  });
}

function repositoryInspectionKey(
  repository,
  revision,
  projectSubdirectory,
  adapter,
) {
  return [
    repository,
    revision.toLowerCase(),
    projectSubdirectory,
    adapterRepositoryInspectionFingerprint(adapter),
  ].join("\n");
}

async function inspectRepositoryRuntime(forceRefresh = false) {
  const repository = elements.experimentSource.value.trim();
  const revision = elements.experimentRevision.value.trim();
  if (!repository || !revision) {
    resetRuntimeInspection();
    return;
  }
  const request = ++runtimeInspectionRequest;
  const projectSubdirectory = elements.experimentWorkdir.value.trim() || ".";
  const adapter = selectedAdapter();
  const adapterScope = adapterDeclaredScope(adapter);
  const key = repositoryInspectionKey(
    repository,
    revision,
    projectSubdirectory,
    adapter,
  );
  if (
    !forceRefresh &&
    runtimeInspectionKey === key &&
    repositoryInputOptionsState?.scopeKey ===
      adapterInputOptionsScopeKey(
        adapter,
        repository,
        revision,
        projectSubdirectory,
      )
  ) {
    if (repositoryInputOptionsState && adapterScope) {
      repositoryInputOptionsState = {
        ...repositoryInputOptionsState,
        requestedRevision: revision,
        adapterStableId: adapterScope.stableId,
        adapterVersionId: adapterScope.versionId,
        manifestHash: adapterScope.manifestHash,
        scopeKey: adapterInputOptionsScopeKey(
          adapter,
          repository,
          repositoryInputOptionsState.resolvedCommit,
          projectSubdirectory,
        ),
      };
    }
    renderAdapterDeclaredFields(adapter);
    updateExperimentSubmitState();
    return;
  }
  runtimeInspectionKey = "";
  runtimeCandidates = [];
  clearRepositoryInputOptions({ render: true });
  elements.experimentRuntime.innerHTML = `<option value="">${forceRefresh ? "Refreshing repository inspection..." : "Restoring saved runtime..."}</option>`;
  elements.experimentRuntime.disabled = true;
  elements.runtimeEvidence.hidden = true;
  setRuntimeStatus(
    forceRefresh
      ? `Refreshing repository inspection for ${revision.slice(0, 12)}...`
      : `Restoring database-cached inspection for ${revision.slice(0, 12)}...`,
    "loading",
  );
  try {
    const query = new URLSearchParams({
      repo_url: repository,
      revision,
      project_subdirectory: projectSubdirectory,
      gateway: elements.gateway.value,
    });
    if (adapterScope?.stableId) query.set("adapter_id", adapterScope.stableId);
    if (adapterScope?.versionId && adapterScope.versionId !== "unversioned") {
      query.set("adapter_version_id", adapterScope.versionId);
    }
    const { payload, fromCache } = await sourceMetadataRequest(
      `/api/source/inspect?${query.toString()}`,
      { forceRefresh },
    );
    if (request !== runtimeInspectionRequest) return;
    const currentAdapterScope = adapterDeclaredScope(selectedAdapter());
    if (currentAdapterScope?.key !== adapterScope?.key) return;
    const resolvedRevision = firstValue(
      payload.commit,
      payload.resolved_commit,
      payload.resolved_revision,
    );
    if (!resolvedRevision) {
      throw new Error(
        "Repository inspection did not return a resolved commit.",
      );
    }
    const resolvedCommit = String(resolvedRevision).trim().toLowerCase();
    repositoryInputOptionsState = {
      repository,
      requestedRevision: revision,
      resolvedCommit,
      projectSubdirectory,
      adapterStableId: adapterScope?.stableId || "",
      adapterVersionId: adapterScope?.versionId || "",
      manifestHash: adapterScope?.manifestHash || "",
      scopeKey: adapterInputOptionsScopeKey(
        adapter,
        repository,
        resolvedCommit,
        projectSubdirectory,
      ),
      options: normalizeRepositoryInputOptions(payload),
    };
    renderAdapterDeclaredFields(selectedAdapter());
    runtimeProfiles = normalizeRuntimeProfiles(payload);
    runtimeCandidates = prepareRuntimeChoices(
      normalizeRuntimeCandidates(payload),
    );
    const runnable = runtimeCandidates.filter(
      (candidate) => candidate.runnable,
    );
    elements.experimentRuntime.innerHTML = [
      '<option value="">Choose a runtime...</option>',
      ...runtimeCandidates.map(
        (candidate) =>
          `<option value="${escapeHtml(candidate.value)}" ${candidate.runnable ? "" : "disabled"}>${escapeHtml(candidate.label)}${candidate.manual ? "" : ` / detected ${escapeHtml(candidate.confidence)}`}${candidate.runnable ? "" : " / unavailable"}</option>`,
      ),
    ].join("");
    elements.experimentRuntime.disabled = !runnable.length;
    runtimeInspectionKey = key;
    const strong = runnable.filter(isStrongRuntimeCandidate);
    if (strong.length === 1) {
      elements.experimentRuntime.value = strong[0].value;
      applyRuntimeSelection(false);
    } else if (!runnable.length) {
      setRuntimeStatus(
        "No runnable runtime candidate was returned. Add detection rules to the adapter or repository manifest.",
        "error",
      );
    } else {
      if (!strong.length) {
        const manualNames = runnable
          .filter((candidate) => candidate.manual)
          .map((candidate) => candidate.label)
          .join(", ");
        setRuntimeStatus(
          `No reproducible automatic runtime candidate was found. Choose an explicit manual option allowed by the adapter${manualNames ? `: ${manualNames}` : ""}.`,
          "warning",
        );
      } else {
        setRuntimeStatus(
          `${strong.length} strong detected candidates found. Choose one explicitly; there is no silent fallback.`,
          "warning",
        );
      }
      renderRuntimeEvidence();
    }
    if (fromCache) {
      elements.runtimeInspectionStatus.textContent +=
        " Database inspection reused; Refresh refs to query the repository again.";
    }
  } catch (error) {
    if (request !== runtimeInspectionRequest) return;
    resetRuntimeInspection(
      `Runtime inspection failed: ${error.message}`,
      "error",
    );
  }
}

function updateSourceCommitMeta() {
  const sha = elements.experimentRevision.value;
  const commit = sourceCommits.find((row) => row.sha === sha);
  if (!commit) {
    elements.sourceCommitMeta.textContent = "No commit selected.";
    elements.sourceCommitMeta.removeAttribute("title");
    return;
  }
  const details = [
    commit.sha,
    commit.author,
    commit.timestamp ? formatDate(commit.timestamp) : null,
  ].filter(Boolean);
  elements.sourceCommitMeta.textContent = details.join(" / ");
  if (commit.subject) elements.sourceCommitMeta.title = commit.subject;
  else elements.sourceCommitMeta.removeAttribute("title");
}

async function loadSourceCommits(
  preferredSha = "",
  owningBranchRequest = sourceBranchRequest,
  forceRefresh = false,
) {
  const repository = elements.experimentSource.value.trim();
  const branch = elements.experimentBranch.value;
  const cacheKey = `${repository}\n${branch}`;
  if (!forceRefresh && sourceCommitsKey === cacheKey && sourceCommits.length) {
    const selectedSha = preferredSha || elements.experimentRevision.value;
    if (
      !selectedSha ||
      !sourceCommits.some((commit) => commit.sha === selectedSha)
    ) {
      elements.experimentRevision.value = "";
      updateSourceCommitMeta();
      resetRuntimeInspection(
        "Choose an exact commit; no commit was selected automatically.",
        "empty",
      );
      setSourceRefStatus(
        `${sourceCommits.length} commits are cached for ${branch}. Choose an exact commit.`,
        "warning",
      );
      return;
    }
    elements.experimentRevision.value = selectedSha;
    updateSourceCommitMeta();
    saveSourceSelection(repository, branch, selectedSha);
    await inspectRepositoryRuntime();
    setSourceRefStatus(
      `${sourceBranches.length} branch${sourceBranches.length === 1 ? "" : "es"} / ${sourceCommits.length} recent commits restored from cache.`,
    );
    return;
  }
  const request = ++sourceCommitRequest;
  sourceCommits = [];
  resetSourceSelect(
    elements.experimentRevision,
    branch
      ? forceRefresh
        ? "Refreshing commits..."
        : "Restoring saved commits..."
      : "Choose a branch first",
  );
  updateSourceCommitMeta();
  if (!repository || !branch) return;

  setSourceRefStatus(
    forceRefresh
      ? `Refreshing commits from ${branch}...`
      : `Restoring cached commits for ${branch}...`,
    "loading",
  );
  try {
    const path = `/api/source/commits?repo_url=${encodeURIComponent(repository)}&branch=${encodeURIComponent(branch)}&limit=50`;
    const { payload, fromCache } = await sourceMetadataRequest(path, {
      forceRefresh,
    });
    if (
      request !== sourceCommitRequest ||
      owningBranchRequest !== sourceBranchRequest
    )
      return;
    if (!Array.isArray(payload.commits)) {
      throw new Error("Commit response did not contain a commits array.");
    }
    const commits = payload.commits;
    sourceCommits = commits
      .map((commit) => ({
        sha: String(commit.sha || "").trim(),
        short_sha: String(commit.short_sha || commit.sha || "").trim(),
        subject: String(commit.subject || "").trim(),
        author: String(commit.author || "").trim(),
        timestamp: commit.timestamp || null,
      }))
      .filter((commit) => commit.sha);
    sourceCommitsKey = cacheKey;

    if (!sourceCommits.length) {
      resetSourceSelect(elements.experimentRevision, "No commits found");
      setSourceRefStatus(`No commits were returned for ${branch}.`, "empty");
      return;
    }

    elements.experimentRevision.innerHTML = [
      '<option value="">Choose a commit...</option>',
      ...sourceCommits.map((commit) => {
        const summary = commit.subject ? ` / ${commit.subject}` : "";
        return `<option value="${escapeHtml(commit.sha)}">${escapeHtml(commit.short_sha || commit.sha.slice(0, 8))}${escapeHtml(summary)}</option>`;
      }),
    ].join("");
    elements.experimentRevision.disabled = false;
    if (preferredSha) {
      if (!sourceCommits.some((commit) => commit.sha === preferredSha)) {
        elements.experimentRevision.value = "";
        updateSourceCommitMeta();
        resetRuntimeInspection(
          `Requested commit ${preferredSha.slice(0, 12)} is unavailable on ${branch}. Choose an available commit.`,
          "error",
        );
        setSourceRefStatus(
          `Requested commit ${preferredSha} is unavailable on ${branch}. No replacement was selected.`,
          "error",
        );
        return;
      }
      elements.experimentRevision.value = preferredSha;
    } else {
      elements.experimentRevision.value = "";
      updateSourceCommitMeta();
      resetRuntimeInspection(
        "Choose an exact commit; no commit was selected automatically.",
        "empty",
      );
      setSourceRefStatus(
        `${sourceCommits.length} commits loaded from ${branch}. Choose an exact commit.`,
        "warning",
      );
      return;
    }
    updateSourceCommitMeta();
    saveSourceSelection(repository, branch, preferredSha);
    await inspectRepositoryRuntime(forceRefresh);
    setSourceRefStatus(
      `${sourceBranches.length} branch${sourceBranches.length === 1 ? "" : "es"} / ${sourceCommits.length} recent commits loaded from ${branch}.${fromCache ? " Database commit metadata reused; Refresh refs to query again." : ""}`,
    );
  } catch (error) {
    if (
      request !== sourceCommitRequest ||
      owningBranchRequest !== sourceBranchRequest
    )
      return;
    resetSourceSelect(elements.experimentRevision, "Commits unavailable");
    updateSourceCommitMeta();
    setSourceRefStatus(
      `Could not load commits for ${branch}: ${error.message}`,
      "error",
    );
  }
}

async function loadSourceBranches(
  preserveSelection = false,
  forceRefresh = false,
) {
  window.clearTimeout(sourceUrlTimer);
  const repository = elements.experimentSource.value.trim();
  if (
    !forceRefresh &&
    sourceBranchesKey === repository &&
    sourceBranches.length
  ) {
    const savedSelection = savedSourceSelection(repository);
    const selectedBranch =
      elements.experimentBranch.value || savedSelection.branch;
    if (
      !selectedBranch ||
      !sourceBranches.some((branch) => branch.name === selectedBranch)
    ) {
      elements.experimentBranch.value = "";
      setSourceRefStatus(
        "Cached branches are available. Choose a branch explicitly.",
        "warning",
      );
      return;
    }
    elements.experimentBranch.value = selectedBranch;
    const branchTip =
      sourceBranches.find((branch) => branch.name === selectedBranch)?.sha ||
      "";
    const selectedCommit =
      elements.experimentRevision.value ||
      savedSelection.commits?.[selectedBranch] ||
      branchTip;
    await loadSourceCommits(selectedCommit, sourceBranchRequest);
    return;
  }
  const previousBranch = preserveSelection
    ? elements.experimentBranch.value
    : "";
  const previousSha = preserveSelection
    ? elements.experimentRevision.value
    : "";
  const request = ++sourceBranchRequest;
  ++sourceCommitRequest;
  sourceBranches = [];
  sourceCommits = [];
  resetRuntimeInspection(
    repository ? "Choose a branch and commit." : "Enter a repository URL.",
  );
  resetSourceSelect(
    elements.experimentBranch,
    repository
      ? forceRefresh
        ? "Refreshing branches..."
        : "Restoring saved branches..."
      : "Enter a repository URL",
  );
  resetSourceSelect(elements.experimentRevision, "Choose a branch first");
  updateSourceCommitMeta();

  if (!repository) {
    setSourceRefStatus(
      "Enter a repository URL to load branches and commits.",
      "empty",
    );
    return;
  }

  elements.sourceRefRefresh.disabled = true;
  setSourceRefStatus(
    forceRefresh
      ? "Refreshing repository metadata through the selected SSH gateway..."
      : "Restoring repository metadata from the database cache...",
    "loading",
  );
  try {
    const path = `/api/source/branches?repo_url=${encodeURIComponent(repository)}`;
    const { payload } = await sourceMetadataRequest(path, { forceRefresh });
    if (request !== sourceBranchRequest) return;
    if (!Array.isArray(payload.branches)) {
      throw new Error("Branch response did not contain a branches array.");
    }
    const branches = payload.branches;
    const uniqueBranches = new Map();
    branches.forEach((branch) => {
      const name = String(branch.name || "").trim();
      if (name && !uniqueBranches.has(name)) {
        uniqueBranches.set(name, {
          name,
          sha: String(branch.sha || branch.commit || "").trim(),
        });
      }
    });
    sourceBranches = [...uniqueBranches.values()];
    sourceBranchesKey = repository;

    if (!sourceBranches.length) {
      resetSourceSelect(elements.experimentBranch, "No branches found");
      setSourceRefStatus(
        "No branches were returned for this repository.",
        "empty",
      );
      return;
    }

    elements.experimentBranch.innerHTML = [
      '<option value="">Choose a branch...</option>',
      ...sourceBranches.map((branch) => {
        const tip = branch.sha ? ` @ ${branch.sha.slice(0, 8)}` : "";
        return `<option value="${escapeHtml(branch.name)}">${escapeHtml(branch.name + tip)}</option>`;
      }),
    ].join("");
    elements.experimentBranch.disabled = false;
    const defaultBranch = String(payload.default_branch || "").trim();
    rememberSourceSelection(
      repository,
      payload.selection || { branch: "", commits: {} },
    );
    const savedSelection = savedSourceSelection(repository);
    const selectedBranch =
      previousBranch || savedSelection.branch || defaultBranch;
    if (!selectedBranch) {
      elements.experimentBranch.value = "";
      setSourceRefStatus(
        "Branches loaded, but the repository declared no default branch. Choose one explicitly.",
        "warning",
      );
      return;
    }
    if (!sourceBranches.some((branch) => branch.name === selectedBranch)) {
      elements.experimentBranch.value = "";
      setSourceRefStatus(
        `Requested branch "${selectedBranch}" is unavailable. No replacement was selected.`,
        "error",
      );
      return;
    }
    elements.experimentBranch.value = selectedBranch;
    const branchTip =
      sourceBranches.find((branch) => branch.name === selectedBranch)?.sha ||
      "";
    const selectedCommit =
      previousSha || savedSelection.commits?.[selectedBranch] || branchTip;
    saveSourceSelection(repository, selectedBranch, selectedCommit);
    await loadSourceCommits(selectedCommit, request, forceRefresh);
  } catch (error) {
    if (request !== sourceBranchRequest) return;
    resetSourceSelect(elements.experimentBranch, "Branches unavailable");
    resetSourceSelect(elements.experimentRevision, "Choose a branch first");
    updateSourceCommitMeta();
    setSourceRefStatus(
      `Could not load repository branches: ${error.message}`,
      "error",
    );
  } finally {
    if (request === sourceBranchRequest)
      elements.sourceRefRefresh.disabled = false;
  }
}

function scheduleSourceBranchLoad() {
  window.clearTimeout(sourceUrlTimer);
  ++sourceBranchRequest;
  ++sourceCommitRequest;
  sourceBranches = [];
  sourceCommits = [];
  sourceBranchesKey = "";
  sourceCommitsKey = "";
  resetRuntimeInspection("Repository changed; choose an exact commit.");
  resetSourceSelect(elements.experimentBranch, "Repository changed");
  resetSourceSelect(elements.experimentRevision, "Choose a branch first");
  updateSourceCommitMeta();
  if (!elements.experimentSource.value.trim()) {
    setSourceRefStatus(
      "Enter a repository URL to load branches and commits.",
      "empty",
    );
    return;
  }
  setSourceRefStatus(
    "Repository changed. Database metadata will be reused when the value is committed; Refresh refs forces a repository query.",
    "warning",
  );
}

function experimentPayload() {
  const declaredOverrides = collectAdapterDeclaredOverrides();
  const overrides = [
    ...declaredOverrides.declaredLines,
    ...declaredOverrides.advancedLines,
  ];
  const adapter = selectedAdapter();
  const runtimeCandidate = runtimeCandidates.find(
    (candidate) => candidate.value === elements.experimentRuntime.value,
  );
  const namedRuntimeProfile = selectedRuntimeProfile();
  const runtimeProfile = elements.experimentRuntimeProfile.value.trim();
  const runtime = {
    type: runtimeCandidate?.value || "",
    profile: namedRuntimeProfile?.id || runtimeProfile || null,
  };
  if (namedRuntimeProfile) runtime.profile_id = namedRuntimeProfile.id;
  if (!namedRuntimeProfile && runtimeProfile && runtime.type === "uv")
    runtime.lock_file = runtimeProfile;
  if (
    !namedRuntimeProfile &&
    runtimeProfile &&
    ["conda", "existing"].includes(runtime.type)
  )
    runtime.environment_path = runtimeProfile;
  if (!namedRuntimeProfile && runtimeProfile && runtime.type === "apptainer")
    runtime.container_image = runtimeProfile;
  const experimentName = elements.experimentName.value.trim();
  const wandbConnection = trackingConnections.get("wandb") || {};
  const mlflowConnection = trackingConnections.get("mlflow") || {};
  const preservedTracking = loadedExperimentCanonicalContext?.tracking || {};
  const preservedProviders = Array.isArray(preservedTracking.providers)
    ? preservedTracking.providers
    : [];
  const preservedWandb =
    preservedProviders.find(
      (provider) =>
        String(provider?.provider || provider?.type || "").toLowerCase() ===
        "wandb",
    ) || {};
  const preservedMlflow =
    preservedProviders.find(
      (provider) =>
        String(provider?.provider || provider?.type || "").toLowerCase() ===
        "mlflow",
    ) || {};
  const trackingProviders = [];
  if (elements.wandbEnabled.checked) {
    trackingProviders.push({
      provider: "wandb",
      enabled: true,
      entity: preservedWandb.entity || wandbConnection.entity || null,
      project: elements.wandbProject.value.trim() || experimentName,
      run_name_template: elements.wandbRunName.value.trim() || null,
    });
  }
  if (elements.mlflowEnabled.checked) {
    trackingProviders.push({
      provider: "mlflow",
      enabled: true,
      tracking_uri:
        elements.mlflowUri.value.trim() ||
        preservedMlflow.tracking_uri ||
        mlflowConnection.tracking_uri ||
        mlflowConnection.base_url ||
        null,
      experiment: elements.mlflowExperiment.value.trim() || experimentName,
      run_name_template: elements.mlflowRunName.value.trim() || null,
    });
  }
  const trainingNumber = (element) =>
    element.disabled ? null : numberOrNull(element);
  const hyperparameters = Object.fromEntries(
    Object.entries({
      learning_rate: trainingNumber(elements.hpLearningRate),
      batch_size: trainingNumber(elements.hpBatchSize),
      batch_semantics: elements.hpBatchSemantics.value,
      gradient_accumulation: trainingNumber(elements.hpGradAcc),
      num_workers: trainingNumber(elements.hpNumWorkers),
      precision: enforceSupportedPrecision(),
      max_steps: trainingNumber(elements.hpMaxSteps),
    }).filter(
      ([, value]) =>
        value !== null &&
        value !== undefined &&
        value !== "" &&
        value !== "adapter-default",
    ),
  );
  const selectedEvaluationSuiteIds = elements.evaluationEnabled.checked
    ? selectedValues(elements.experimentEvaluationSuites)
    : [];
  const loadedEvaluation = loadedExperimentCanonicalContext?.evaluation;
  const preserveLoadedEvaluation = Boolean(
    elements.evaluationEnabled.checked &&
      Array.isArray(loadedEvaluation?.specs) &&
      JSON.stringify([...selectedEvaluationSuiteIds].sort()) ===
        JSON.stringify([...loadedEvaluation.suiteIds].sort()),
  );
  const preservedCheckpoint =
    loadedExperimentCanonicalContext?.checkpoint || {};
  const preservedResources = loadedExperimentCanonicalContext?.resources || {};
  return {
    schema_version: "1",
    name: elements.experimentName.value.trim(),
    adapter: adapterId(adapter),
    source: {
      repository: elements.experimentSource.value.trim(),
      revision: elements.experimentRevision.value.trim(),
      project_subdirectory: elements.experimentWorkdir.value.trim() || ".",
      adapter_version: Number(adapterVersion(adapter)),
    },
    ...(selectedExperimentDataBundle()?.selections
      ? { data_selections: selectedExperimentDataBundle().selections }
      : {}),
    runtime,
    hyperparameters,
    native_overrides: overrides,
    resources: {
      gateway: elements.gateway.value,
      account: preservedResources.account || "rl2-lab",
      ...(preservedResources.partition
        ? { partition: preservedResources.partition }
        : {}),
      queue_policy: elements.resourcePolicy.value,
      node_mode: "auto",
      node: null,
      gpu_mode: elements.gpuMode.value,
      gpus_per_node:
        elements.gpuMode.value === "manual"
          ? numberOrNull(elements.experimentGpuCount)
          : null,
      gpu_type: elements.experimentGpuType.value,
      nodes: numberOrNull(elements.resourceNodes),
      cpus_per_task: numberOrNull(elements.resourceCpus),
      memory_gb: numberOrNull(elements.resourceMemory),
      time_limit: elements.resourceTime.value.trim(),
    },
    checkpoint: {
      mode: elements.checkpointMode.value,
      path:
        elements.checkpointMode.value === "none"
          ? null
          : elements.checkpointPath.value.trim(),
    },
    checkpoint_save_steps: elements.checkpointSaveSteps.disabled
      ? null
      : numberOrNull(elements.checkpointSaveSteps),
    checkpoint_warning_seconds: numberOrNull(
      document.querySelector("#checkpoint-warning-seconds"),
    ),
    checkpoint_keep_last: preservedCheckpoint.keep_last ?? null,
    checkpoint_final_selector: preservedCheckpoint.final_selector ?? null,
    remove_training_state_after_success:
      preservedCheckpoint.remove_training_state_after_success ?? null,
    max_attempts: numberOrNull(elements.checkpointMaxAttempts),
    auto_resume: elements.checkpointAutoResume.checked,
    sweep: {
      definition: elements.sweepDefinition.value.trim() || null,
    },
    tracking: {
      providers: trackingProviders,
      native_tracking: preservedTracking.native_tracking ?? "preserve",
      tags: preservedTracking.tags ?? {},
      offline_spool: preservedTracking.offline_spool ?? true,
      provider:
        trackingProviders.length === 1 ? trackingProviders[0].provider : null,
      enabled: trackingProviders.length > 0,
      uri: elements.mlflowEnabled.checked
        ? mlflowConnection.tracking_uri || mlflowConnection.base_url || null
        : null,
      experiment: elements.mlflowEnabled.checked
        ? elements.mlflowExperiment.value.trim() || experimentName
        : null,
      run_name_template:
        trackingProviders.length === 1
          ? trackingProviders[0].run_name_template
          : null,
    },
    evaluation: {
      enabled: elements.evaluationEnabled.checked,
      suite_ids: preserveLoadedEvaluation ? [] : selectedEvaluationSuiteIds,
      ...(preserveLoadedEvaluation
        ? { canonical_specs: loadedEvaluation.specs }
        : {}),
    },
  };
}

function validateExperiment({ notify = true, batchValidation = null } = {}) {
  const reject = (message) => {
    if (notify) showToast(message, true);
    return false;
  };
  let sweepError = "";
  const rawSweep = elements.sweepDefinition.value.trim();
  if (rawSweep) {
    try {
      const sweep = JSON.parse(rawSweep);
      if (!sweep || Array.isArray(sweep) || typeof sweep !== "object")
        sweepError = "Sweep must be a JSON object.";
    } catch {
      sweepError = 'Sweep must be valid JSON, for example {"seed":[1,2,3]}.';
    }
  }
  elements.sweepDefinition.setCustomValidity(sweepError);
  if (sweepError) return reject(sweepError);
  elements.hpLearningRate.setCustomValidity(
    !elements.hpLearningRate.disabled &&
      elements.hpLearningRate.value !== "" &&
      Number(elements.hpLearningRate.value) <= 0
      ? "Learning rate must be greater than zero."
      : "",
  );
  const batchState = batchValidation || updateBatchCompatibility();
  if (!batchState.valid) return reject(batchState.errors[0]);
  const declared = collectAdapterDeclaredOverrides();
  if (declared.errors.length) return reject(declared.errors[0]);
  if (!elements.experimentForm.checkValidity()) {
    const invalid = [...elements.experimentForm.elements].find(
      (control) => control.willValidate && !control.validity.valid,
    );
    const label =
      invalid?.labels?.[0]?.textContent?.trim() ||
      invalid?.name ||
      invalid?.id ||
      "Experiment field";
    return reject(
      `${label}: ${invalid?.validationMessage || "Enter a valid value."}`,
    );
  }
  const selectedRevision = elements.experimentRevision.value.trim();
  if (
    !selectedRevision ||
    !sourceCommits.some((commit) => commit.sha === selectedRevision)
  ) {
    return reject("Select a commit loaded from the repository.");
  }
  const inspectionKey = repositoryInspectionKey(
    elements.experimentSource.value.trim(),
    selectedRevision,
    elements.experimentWorkdir.value.trim() || ".",
    selectedAdapter(),
  );
  const runtime = runtimeCandidates.find(
    (candidate) => candidate.value === elements.experimentRuntime.value,
  );
  if (runtimeInspectionKey !== inspectionKey || !runtime || !runtime.runnable) {
    return reject(
      "Inspect the selected commit and explicitly choose a runnable runtime.",
    );
  }
  if (
    !selectedRuntimeProfile() &&
    runtime.manual &&
    runtime.requiresProfile &&
    !elements.experimentRuntimeProfile.value.trim()
  ) {
    const requirement =
      runtime.value === "conda"
        ? "a Conda environment path/profile"
        : runtime.value === "apptainer"
          ? "an Apptainer image path/reference"
          : "a pinned runtime input";
    return reject(
      `The manual ${runtimeDisplayName(runtime.value)} runtime requires ${requirement}.`,
    );
  }
  if (
    elements.checkpointMode.value !== "none" &&
    !elements.checkpointPath.value.trim()
  ) {
    return reject("A checkpoint path or run ID is required.");
  }
  for (const provider of selectedTrackingProviders()) {
    const connection = trackingConnections.get(provider);
    if (!connection || connection.connected !== true) {
      return reject(
        ["error", "unknown", "unavailable"].includes(connection?.status)
          ? `${trackingProviderLabel(provider)} connection status is unavailable. Refresh Settings to verify it.`
          : `${trackingProviderLabel(provider)} is selected but not connected. Connect and test it in Settings.`,
      );
    }
    if (provider === "wandb" && !connection.entity) {
      return reject(
        "The W&B connection did not return a verified entity. Reconnect it in Settings.",
      );
    }
  }
  if (
    elements.evaluationEnabled.checked &&
    !selectedValues(elements.experimentEvaluationSuites).length
  ) {
    return reject("Select at least one evaluation suite.");
  }
  return true;
}

let loadedExperimentOrigin = null;
let experimentPreviewSignature = null;
let experimentPreviewScripts = [];

function invalidateExperimentPreview() {
  experimentPreviewSignature = null;
  experimentPreviewScripts = [];
  beginLogViewUpdate(elements.experimentScriptPreview);
  clearExperimentPreviewAfterLoad();
}

function experimentPreviewIsCurrent() {
  return (
    experimentPreviewSignature !== null &&
    experimentPreviewSignature === JSON.stringify(experimentPayload())
  );
}

function renderSelectedExperimentScript() {
  const selected = document.querySelector("#experiment-preview-variant");
  const full = document.querySelector("#experiment-preview-full").checked;
  const script =
    experimentPreviewScripts[Number(selected.value)] ||
    "# No runnable variant was generated.";
  const displayed = full
    ? script
    : script
        .split("\n")
        .map((line) =>
          line.length > 1200
            ? `# Embedded payload omitted (${line.length.toLocaleString()} characters). Download the exact script or enable full script.`
            : line,
        )
        .join("\n");
  updateLogView(elements.experimentScriptPreview, displayed);
}

function updateExperimentSubmitState() {
  const presetMode = elements.experimentPresetDialog.open;
  const batchValidation = updateBatchCompatibility();
  const ready =
    !experimentBusy &&
    validateExperiment({ notify: false, batchValidation }) &&
    experimentPreviewIsCurrent();
  const existingExperiment = matchingExperimentForPayload(experimentPayload());
  const latestRevision = Number(existingExperiment?.latest_revision_number);
  const nextRevision = Number.isFinite(latestRevision)
    ? latestRevision + 1
    : null;
  if (existingExperiment) {
    const revisionLabel =
      nextRevision === null ? "a new revision" : `revision ${nextRevision}`;
    elements.saveExperimentButton.textContent = "Create draft revision";
    elements.submitExperimentButton.textContent = "Create revision and submit";
    const origin =
      loadedExperimentOrigin?.id ===
      String(existingExperiment.id || existingExperiment.experiment_id)
        ? `Loaded revision ${loadedExperimentOrigin.revision}. `
        : "";
    elements.experimentRevisionIntent.textContent = `${origin}This project/name already exists. The action will create ${revisionLabel}; submitted revisions remain locked.`;
  } else {
    elements.saveExperimentButton.textContent = "Create draft";
    elements.submitExperimentButton.textContent = "Create and submit";
    elements.experimentRevisionIntent.textContent =
      "Submitted specifications are immutable. Changes create a new experiment revision.";
  }
  if (presetMode)
    elements.saveExperimentButton.textContent = experimentBusy
      ? "Creating…"
      : "Create preset";
  elements.saveExperimentButton.type = presetMode ? "submit" : "button";
  elements.saveExperimentButton.classList.toggle("button-accent", presetMode);
  elements.saveExperimentButton.classList.toggle("button-quiet", !presetMode);
  elements.submitExperimentButton.hidden = presetMode;
  elements.experimentPreviewButton.hidden = presetMode;
  elements.experimentPreviewButton.disabled =
    experimentBusy || !batchValidation.valid;
  elements.saveExperimentButton.disabled =
    experimentBusy || !batchValidation.valid;
  elements.submitExperimentButton.disabled = presetMode || !ready;
  elements.newExperimentPreset.disabled = experimentBusy;
  const batchBlockedTitle = batchValidation.errors[0] || "";
  elements.experimentPreviewButton.title = batchBlockedTitle;
  elements.saveExperimentButton.title = batchBlockedTitle;
  elements.submitExperimentButton.title = ready
    ? existingExperiment
      ? "Create a new immutable experiment revision and submit its Training Runs to Slurm"
      : "Create this experiment and submit its Training Runs to Slurm"
    : "Complete the required fields and preview the current configuration before submitting";
}

function setExperimentBusy(busy) {
  experimentBusy = busy;
  elements.experimentPresetDialog.dataset.blockClose = String(busy);
  elements.experimentPresetDialog.querySelector(
    "[data-dialog-close]",
  ).disabled = busy;
  updateExperimentSubmitState();
}

// The preset dialog and Submit share one editor, including its runtime/data
// validation and unsaved inputs. Move the live nodes; never clone form controls.
let presetEditorHomes = [];
function openExperimentPreset() {
  if (experimentBusy || elements.experimentPresetDialog.open) return;
  const destination = document.getElementById("experiment-preset-editor");
  presetEditorHomes = [
    elements.experimentsError,
    document.getElementById("experiment-composer"),
  ].map((node) => {
    const home = document.createComment("experiment editor home");
    node.before(home);
    destination.append(node);
    return { node, home };
  });
  elements.experimentName.value = "";
  invalidateExperimentPreview();
  clearNotice(elements.experimentsError);
  SkynetDialog.open(elements.experimentPresetDialog, {
    launcher: elements.newExperimentPreset,
  });
  updateExperimentSubmitState();
  renderTrackingNamePreview();
  elements.experimentName.focus({ preventScroll: true });
}

function restoreExperimentEditor() {
  if (elements.experimentPresetDialog.open) return;
  for (const { node, home } of presetEditorHomes) home.replaceWith(node);
  presetEditorHomes = [];
  updateExperimentSubmitState();
}

async function previewExperiment() {
  const scope = "experiment-preview";
  invalidateExperimentPreview();
  if (!validateExperiment()) {
    updateExperimentSubmitState();
    return;
  }
  const signature = JSON.stringify(experimentPayload());
  setExperimentBusy(true);
  try {
    const result = await api("/api/experiments/preview", {
      method: "POST",
      body: JSON.stringify(experimentPayload()),
    });
    if (signature !== JSON.stringify(experimentPayload())) return;
    clearNotificationScope(scope);
    const responseScripts = Array.isArray(result.scripts)
      ? result.scripts.filter((script) => typeof script === "string")
      : [];
    const runnableScripts = responseScripts.filter(
      experimentPreviewScriptIsRunnable,
    );
    const variantCount = result.variant_count ?? responseScripts.length;
    const warnings = Array.isArray(result.warnings) ? result.warnings : [];
    const blockers = Array.isArray(result.blockers) ? result.blockers : [];
    const shapeErrors = experimentPreviewShapeErrors(result);
    experimentPreviewScripts = runnableScripts;
    const variantSelect = document.querySelector("#experiment-preview-variant");
    document.querySelector("#experiment-preview-variant-field").hidden =
      experimentPreviewScripts.length < 2;
    variantSelect.innerHTML = runnableScripts
      .map(
        (_, index) => `<option value="${index}">Variant ${index + 1}</option>`,
      )
      .join("");
    variantSelect.disabled = !runnableScripts.length;
    renderSelectedExperimentScript();
    experimentPreviewSignature =
      !blockers.length && !shapeErrors.length && runnableScripts.length > 0
        ? signature
        : null;
    const warningList = document.querySelector("#experiment-preview-warnings");
    warningList.replaceChildren();
    warnings.forEach((warning) => {
      const item = document.createElement("li");
      item.textContent =
        typeof warning === "string" ? warning : JSON.stringify(warning);
      warningList.append(item);
    });
    warningList.hidden = !warnings.length;
    elements.experimentPreviewMeta.textContent = `${variantCount} variant${variantCount === 1 ? "" : "s"} / ${runnableScripts.length} runnable / ${blockers.length} blocked${warnings.length ? ` / ${warnings.length} warning(s)` : ""}`;
    elements.experimentPreviewBlockerList.replaceChildren();
    blockers.forEach((blocker) => {
      const item = document.createElement("li");
      const reasons = Array.isArray(blocker?.reasons)
        ? blocker.reasons.join("; ")
        : "No reason returned.";
      item.textContent = `${blocker?.variant || "Variant"}: ${reasons}`;
      elements.experimentPreviewBlockerList.append(item);
    });
    shapeErrors.forEach((message) => {
      const item = document.createElement("li");
      item.textContent = `Malformed preview response: ${message}.`;
      elements.experimentPreviewBlockerList.append(item);
    });
    elements.experimentPreviewBlockers.hidden =
      blockers.length === 0 && shapeErrors.length === 0;
    elements.experimentPreviewEmpty.hidden = true;
    elements.experimentPreviewPanel.hidden = false;
  } catch (error) {
    if (signature === JSON.stringify(experimentPayload()))
      showToast(`Preview failed: ${error.message}`, true, { scope });
  } finally {
    setExperimentBusy(false);
  }
}

function submissionFeedback(payload) {
  const records = Array.isArray(payload?.submitted)
    ? payload.submitted
    : Array.isArray(payload?.experiment?.runs)
      ? payload.experiment.runs
      : Array.isArray(payload?.runs)
        ? payload.runs
        : payload?.run
          ? [payload.run]
          : payload?.run_id
            ? [payload]
            : [];
  const failed = records.filter((row) =>
    ["FAILED", "BLOCKED", "SUBMISSION_FAILED"].includes(
      normalizedRunState(row.status),
    ),
  );
  const pending = records.some(
    (row) =>
      ["SUBMITTING", "PENDING"].includes(normalizedRunState(row.status)) &&
      !row.slurm_job_id &&
      !row.latest_attempt?.slurm_job_id,
  );
  if (failed.length)
    return {
      error: true,
      message: failed
        .map(
          (row) =>
            `${row.run_id || row.id || "Run"}: ${row.error || row.latest_attempt?.slurm_reason || row.blockers?.join("; ") || row.status}`,
        )
        .join("\n"),
    };
  return {
    error: false,
    message:
      pending || !records.length
        ? "Submission is still being confirmed. Monitor the attempt before retrying."
        : "Submission processed. Check the Training Run state and Slurm job ID below.",
  };
}

function showSubmissionFeedback(payload) {
  const feedback = submissionFeedback(payload);
  if (feedback.error)
    showNotice(elements.runsError, `Submission failed: ${feedback.message}`);
  else showToast(feedback.message);
}

function createdExperimentId(payload) {
  const experiment = entityFrom(payload, "experiment");
  return (
    experiment.id ||
    experiment.experiment_id ||
    payload.id ||
    payload.experiment_id
  );
}

function submittedTrainingRunRecords(...payloads) {
  const records = [];
  const append = (candidate) => {
    if (typeof candidate === "string" || typeof candidate === "number") {
      records.push({ id: String(candidate) });
      return;
    }
    if (!candidate || typeof candidate !== "object" || Array.isArray(candidate))
      return;
    if (candidate.id || candidate.run_id) records.push(candidate);
  };
  payloads.filter(Boolean).forEach((payload) => {
    [
      payload.run,
      payload.submission?.run,
      payload.experiment?.run,
      payload.revision?.run,
    ].forEach(append);
    [
      payload.runs,
      payload.run_ids,
      payload.submission?.runs,
      payload.experiment?.runs,
      payload.revision?.runs,
    ]
      .filter(Array.isArray)
      .forEach((items) => items.forEach(append));
    const variants =
      [
        payload.variants,
        payload.experiment?.variants,
        payload.revision?.variants,
      ].find(Array.isArray) || [];
    variants.forEach((variant) => {
      append(variant?.run);
      if (Array.isArray(variant?.runs)) variant.runs.forEach(append);
      if (variant?.run_id)
        append({
          id: variant.run_id,
          experiment_revision_number: variant.experiment_revision_number,
        });
    });
  });
  const unique = new Map();
  records.forEach((record) => {
    const id = String(record.id || record.run_id || "").trim();
    if (!id) return;
    const existing = unique.get(id);
    unique.set(
      id,
      existing ? { ...existing, ...record, id } : { ...record, id },
    );
  });
  return [...unique.values()];
}

function explicitlySubmittedTrainingRunRecords(...payloads) {
  const exactPayloads = [];
  payloads.filter(Boolean).forEach((payload) => {
    exactPayloads.push({
      run: payload.run,
      runs:
        payload.runs ||
        (Array.isArray(payload.submitted) ? payload.submitted : undefined),
      run_ids: [
        ...(Array.isArray(payload.run_ids) ? payload.run_ids : []),
        ...(Array.isArray(payload.submission?.run_ids)
          ? payload.submission.run_ids
          : []),
      ],
      submission: payload.submission
        ? {
            run: payload.submission.run,
            runs: payload.submission.runs,
            run_ids: payload.submission.run_ids,
          }
        : null,
    });
    if (payload.run_id) exactPayloads.push({ run: { id: payload.run_id } });
    if (payload.primary_run_id)
      exactPayloads.push({
        run: { id: payload.primary_run_id, primary: true },
      });
    if (payload.submission?.run_id)
      exactPayloads.push({ run: { id: payload.submission.run_id } });
    if (payload.submission?.primary_run_id) {
      exactPayloads.push({
        run: { id: payload.submission.primary_run_id, primary: true },
      });
    }
  });
  return submittedTrainingRunRecords(...exactPayloads);
}

function resetTrainingRunListFilters() {
  elements.runSearch.value = "";
  elements.runStatusFilter.value = "all";
}

async function openSubmittedTrainingRun({
  experimentId,
  revisionNumber = null,
  responses = [],
}) {
  let records = explicitlySubmittedTrainingRunRecords(...responses);
  let resolution = "submit response";
  const normalizedRevision =
    revisionNumber == null || revisionNumber === ""
      ? NaN
      : Number(revisionNumber);
  if (!records.length && experimentId) {
    if (!Number.isFinite(normalizedRevision)) {
      resetTrainingRunListFilters();
      await activateTab("runs");
      await loadRuns(true);
      showNotice(
        elements.runsError,
        "Submission returned, but the submit response returned no Training Run ID and no exact experiment revision was available for resolution.",
      );
      return;
    }
    try {
      const detail = await api(
        `/api/experiments/${encodeURIComponent(experimentId)}`,
      );
      records = submittedTrainingRunRecords(detail).filter(
        (record) =>
          Number(
            record.experiment_revision_number ?? record.revision_number,
          ) === normalizedRevision &&
          String(
            record.experiment_id || record.experiment?.id || experimentId,
          ) === String(experimentId),
      );
      resolution = `experiment ${experimentId} revision ${normalizedRevision}`;
    } catch (error) {
      resetTrainingRunListFilters();
      await activateTab("runs");
      showNotice(
        elements.runsError,
        `Submission returned, but its new Training Run could not be resolved: ${error.message}`,
      );
      return;
    }
  }
  resetTrainingRunListFilters();
  await activateTab("runs");
  await loadRuns(true);
  if (!records.length) {
    showNotice(
      elements.runsError,
      `Submission returned, but ${resolution} did not identify a Training Run.`,
    );
    return;
  }
  const primaryRecords = records.filter(
    (record) => record.primary === true || record.is_primary === true,
  );
  const selectedRecords =
    primaryRecords.length === 1 ? primaryRecords : records;
  if (selectedRecords.length !== 1) {
    showNotice(
      elements.runsError,
      `Training submission created ${records.length} Training Runs, but ${resolution} did not identify one primary run. Choose the intended run explicitly.`,
    );
    return;
  }
  const runId = String(selectedRecords[0].id || selectedRecords[0].run_id);
  const created = runRows.find((run) => String(run.id || run.run_id) === runId);
  if (created) {
    resetTrainingRunListFilters();
    renderRuns();
  }
  const launcher = [
    ...elements.runsBody.querySelectorAll("[data-run-action='view']"),
  ].find((candidate) => String(candidate.dataset.id) === runId);
  if (!created || !launcher) {
    showNotice(
      elements.runsError,
      `Training Run ${runId} was created but is not present in the Training Run list response.`,
    );
    return;
  }
  clearNotice(elements.runsError);
  await viewRun(runId, launcher);
}

function normalizeExperimentIdentityPart(value, fallback = "") {
  return String(value ?? fallback)
    .trim()
    .toLowerCase();
}

function normalizeExperimentRepository(value) {
  return normalizeExperimentIdentityPart(value)
    .replace(/\/+$/, "")
    .replace(/\.git$/, "");
}

function experimentProjectIdentity(experiment) {
  const source = experiment?.source || {};
  const repository =
    source.repository ||
    experiment?.repository ||
    experiment?.project ||
    experiment?.project_name ||
    "";
  const projectSubdirectory =
    source.project_subdirectory ||
    experiment?.project_subdirectory ||
    experiment?.workdir ||
    ".";
  return `${normalizeExperimentRepository(repository)}::${normalizeExperimentIdentityPart(projectSubdirectory, ".")}`;
}

function matchingExperimentForPayload(payload) {
  const name = normalizeExperimentIdentityPart(payload?.name);
  const project = experimentProjectIdentity(payload);
  if (!name || project.startsWith("::")) return null;
  const matches = experimentRows.filter(
    (experiment) =>
      normalizeExperimentIdentityPart(experiment.name) === name &&
      experimentProjectIdentity(experiment) === project,
  );
  return (
    matches.sort(
      (left, right) =>
        Number(right.latest_revision_number || 0) -
        Number(left.latest_revision_number || 0),
    )[0] || null
  );
}

function explicitBoolean(value) {
  if (typeof value === "boolean") return value;
  const normalized = normalizeExperimentIdentityPart(value);
  if (["true", "1", "yes"].includes(normalized)) return true;
  if (["false", "0", "no"].includes(normalized)) return false;
  return null;
}

function normalizedLifecycleState(value) {
  return normalizeExperimentIdentityPart(value).replace(/[_-]+/g, " ");
}

function experimentLifecycle(experiment) {
  const lifecycleStates = [
    experiment?.lifecycle,
    experiment?.submission_lifecycle,
    experiment?.submission_state,
    experiment?.status,
    experiment?.latest_revision?.lifecycle,
  ].map(normalizedLifecycleState);
  const lockDeclarations = [
    experiment?.locked,
    experiment?.is_locked,
    experiment?.latest_revision?.locked,
  ]
    .map(explicitBoolean)
    .filter((value) => value !== null);
  const submittedDeclaration = explicitBoolean(experiment?.submitted);
  const explicitlyUnsubmitted =
    lockDeclarations.includes(false) ||
    submittedDeclaration === false ||
    lifecycleStates.some((state) =>
      ["draft", "unsubmitted", "not submitted"].includes(state),
    );
  const explicitlyLocked =
    lockDeclarations.includes(true) ||
    submittedDeclaration === true ||
    lifecycleStates.some((state) =>
      ["submitted", "locked", "submitted locked"].includes(state),
    );
  const submittedAt = firstValue(
    experiment?.submitted_at,
    experiment?.latest_revision_submitted_at,
    experiment?.latest_revision?.submitted_at,
  );
  const locked =
    !explicitlyUnsubmitted && (explicitlyLocked || Boolean(submittedAt));
  return {
    locked,
    label: locked ? "Submitted / locked" : "Draft",
  };
}

function jobStatusLabel(job) {
  return job?.display_status || job?.status || job?.state || "unknown";
}

function normalizedRunState(value) {
  return String(value || "")
    .trim()
    .toUpperCase()
    .replace(/[\s-]+/g, "_");
}

function runAttemptCount(run) {
  const declared = run?.attempt_count ?? run?.attempts_count;
  if (declared !== null && declared !== undefined && declared !== "") {
    const count = Number(declared);
    if (Number.isInteger(count) && count >= 0) return count;
  }
  if (Array.isArray(run?.attempts)) return runAttemptRecords(run).length;
  return null;
}

function runAttemptRecords(run) {
  const attempts = Array.isArray(run?.attempts)
    ? run.attempts.filter(
        (attempt) =>
          attempt && typeof attempt === "object" && !Array.isArray(attempt),
      )
    : [];
  const latest = run?.latest_attempt;
  if (
    latest &&
    typeof latest === "object" &&
    !Array.isArray(latest) &&
    !attempts.some(
      (attempt) =>
        attempt === latest ||
        (attempt.id && latest.id && attempt.id === latest.id),
    )
  )
    attempts.push(latest);
  const stages = new Map(
    (run?.stages || []).map((stage) => [stage.id, stage.stage_type]),
  );
  return attempts.filter((attempt) => {
    const stageType = normalizedRunState(
      attempt.stage_type || stages.get(attempt.stage_id),
    );
    return stageType !== "EVALUATE" && !attempt.evaluation_id;
  });
}

function attemptHasSlurmSubmission(attempt) {
  if (!attempt || typeof attempt !== "object") return false;
  if (
    attempt.slurm_job_id ||
    attempt.slurm_array_job_id ||
    attempt.job_id ||
    attempt.submitted_at
  )
    return true;
  return [
    "SUBMITTED",
    "PENDING",
    "RUNNING",
    "REQUEUED",
    "COMPLETED",
    "SUCCEEDED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "PREEMPTED",
  ].includes(normalizedRunState(attempt.status || attempt.state));
}

function runHasExplicitPreflightFailure(run, attempts) {
  if (
    ["PREFLIGHT_FAILED", "SUBMISSION_FAILED"].includes(
      normalizedRunState(run?.status || run?.state),
    )
  ) {
    return true;
  }
  if (
    explicitBoolean(run?.preflight_failed) === true ||
    explicitBoolean(run?.submission_failed) === true
  ) {
    return true;
  }
  const declaredFailure = firstValue(
    run?.preflight_error,
    run?.preflight_failure,
    run?.submission_error,
    run?.submission_failure,
    run?.submit_error,
  );
  if (
    declaredFailure !== null &&
    declaredFailure !== undefined &&
    declaredFailure !== ""
  )
    return true;
  return attempts.some(
    (attempt) =>
      ["PREFLIGHT_FAILED", "SUBMISSION_FAILED"].includes(
        normalizedRunState(attempt.status || attempt.state),
      ) &&
      !attempt.slurm_job_id &&
      !attempt.slurm_array_job_id &&
      !attempt.job_id,
  );
}

function variantRunDisplayState(run) {
  const rawState = jobStatusLabel(run);
  const state = normalizedRunState(rawState);
  const attempts = runAttemptRecords(run);
  const attemptCount = runAttemptCount(run);
  const explicitlyZeroAttempts = attemptCount === 0;
  const hasAttemptEvidence =
    !explicitlyZeroAttempts &&
    ((attemptCount !== null && attemptCount > 0) ||
      attempts.length > 0 ||
      Boolean(run?.slurm_job_id || run?.slurm_array_job_id || run?.job_id));
  const hasSubmittedAttempt =
    !explicitlyZeroAttempts &&
    (attempts.some(attemptHasSlurmSubmission) ||
      Boolean(
        run?.slurm_job_id ||
          run?.slurm_array_job_id ||
          run?.job_id ||
          run?.submitted_at,
      ) ||
      (attemptCount !== null &&
        attemptCount > 0 &&
        ["PENDING", "SUBMITTED"].includes(state)));
  const preflightFailed = runHasExplicitPreflightFailure(run, attempts);
  if (preflightFailed && !hasSubmittedAttempt) return "PREFLIGHT FAILED";
  if (!hasAttemptEvidence || (state === "PENDING" && !hasSubmittedAttempt)) {
    return preflightFailed ? "PREFLIGHT FAILED" : "NOT SUBMITTED";
  }
  return rawState;
}

async function createExperiment(shouldSubmit) {
  if (experimentBusy) return;
  const presetMode = elements.experimentPresetDialog.open;
  if (presetMode) shouldSubmit = false;
  if (!validateExperiment()) return;
  if (shouldSubmit && !experimentPreviewIsCurrent()) {
    showToast("Preview the current configuration before submitting.", true);
    return;
  }
  const payload = experimentPayload();
  const existingExperiment = matchingExperimentForPayload(payload);
  if (presetMode && existingExperiment) {
    showToast(
      "A preset with this name already exists for this project. Choose a different name.",
      true,
    );
    elements.experimentName.focus();
    return;
  }
  const existingId =
    existingExperiment?.id || existingExperiment?.experiment_id;
  const latestRevision = Number(existingExperiment?.latest_revision_number);
  const nextRevision = Number.isFinite(latestRevision)
    ? latestRevision + 1
    : null;
  const revisionLabel =
    nextRevision === null ? "a new revision" : `revision ${nextRevision}`;
  if (shouldSubmit) {
    const confirmation = existingId
      ? `Create and submit ${revisionLabel} of experiment "${payload.name}"? The submitted revision will be locked.`
      : `Create and submit experiment "${payload.name}"? The submitted specification will be locked.`;
    if (!(await askUserDialog(confirmation))) return;
  }
  setExperimentBusy(true);
  let saved = false;
  try {
    const result = existingId
      ? await api(
          `/api/experiments/${encodeURIComponent(existingId)}/revisions`,
          {
            method: "POST",
            body: JSON.stringify({
              spec: payload,
              submit: shouldSubmit,
              gateway: elements.gateway.value,
            }),
          },
        )
      : await api("/api/experiments", {
          method: "POST",
          body: JSON.stringify(payload),
        });
    const id = createdExperimentId(result);
    let submissionResult = shouldSubmit && existingId ? result : null;
    let createdRevisionNumber = null;
    if (existingId) {
      const returnedRevision =
        result.revision_number ??
        result.latest_revision_number ??
        result.revision?.revision_number ??
        nextRevision;
      createdRevisionNumber = returnedRevision;
      const createdRevisionLabel = returnedRevision
        ? `revision ${returnedRevision}`
        : "a new revision";
      if (!shouldSubmit)
        showToast(
          `Experiment ${existingId} ${createdRevisionLabel} created as a draft.`,
        );
    } else if (shouldSubmit) {
      if (!id)
        throw new Error(
          "Experiment was created but the API returned no experiment ID.",
        );
      const submitted = await api(
        `/api/experiments/${encodeURIComponent(id)}/submit`,
        {
          method: "POST",
          body: JSON.stringify({ gateway: elements.gateway.value }),
        },
      );
      submissionResult = submitted;
      createdRevisionNumber =
        submitted.revision_number ??
        submitted.experiment?.latest_revision?.revision_number;
      // Report transport failures after opening the resulting run, without claiming success.
    } else {
      showToast(
        presetMode
          ? `Preset "${payload.name}" created.`
          : `Experiment ${id || payload.name} saved as a draft.`,
      );
    }
    saved = true;
    invalidateExperimentPreview();
    if (presetMode) {
      const search = document.getElementById("experiment-search");
      search.value = "";
      delete search.dataset.datasetId;
      delete search.dataset.presetIds;
    }
    await loadExperiments(true);
    if (shouldSubmit) {
      await openSubmittedTrainingRun({
        experimentId: existingId || id,
        revisionNumber: createdRevisionNumber,
        responses: [submissionResult, result],
      });
      showSubmissionFeedback(submissionResult);
    }
  } catch (error) {
    showToast(
      `${shouldSubmit ? "Submission" : "Save"} failed: ${error.message}`,
      true,
    );
  } finally {
    setExperimentBusy(false);
    if (presetMode && saved)
      SkynetDialog.close(elements.experimentPresetDialog);
  }
}

async function submitDraftExperiment(id, launcher = null) {
  const experiment = experimentRows.find(
    (candidate) =>
      String(candidate.id || candidate.experiment_id) === String(id),
  );
  if (!experiment || experimentLifecycle(experiment).locked) return;
  const revisionNumber = experiment.latest_revision_number;
  const revisionLabel = revisionNumber ? ` revision ${revisionNumber}` : "";
  if (
    !(await askUserDialog(
      `Submit experiment "${experiment.name || id}"${revisionLabel}? The submitted specification will be locked.`,
    ))
  )
    return;
  if (launcher) launcher.disabled = true;
  try {
    const submitted = await api(
      `/api/experiments/${encodeURIComponent(id)}/submit`,
      {
        method: "POST",
        body: JSON.stringify({ gateway: elements.gateway.value }),
      },
    );

    await loadExperiments(true);
    await openSubmittedTrainingRun({
      experimentId: id,
      revisionNumber,
      responses: [submitted],
    });
    showSubmissionFeedback(submitted);
  } catch (error) {
    showToast(`Submission failed: ${error.message}`, true);
  } finally {
    if (launcher?.isConnected) launcher.disabled = false;
  }
}

function loadedCanonicalPath(root, path) {
  let value = root;
  for (const segment of String(path || "")
    .split(".")
    .filter(Boolean)) {
    if (
      !value ||
      typeof value !== "object" ||
      !Object.prototype.hasOwnProperty.call(value, segment)
    ) {
      return { present: false, value: undefined };
    }
    value = value[segment];
  }
  return { present: true, value };
}

function loadedCanonicalObject(value, label) {
  if (typeof value === "string") {
    try {
      value = JSON.parse(value);
    } catch (error) {
      throw new Error(`${label} is not valid JSON: ${error.message}`);
    }
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} is missing or is not an object.`);
  }
  return value;
}

function setLoadedControlValue(control, value) {
  control.value = value === null || value === undefined ? "" : String(value);
}

function loadedCanonicalPathIsExplicit(spec, path) {
  const explicitParameters = Array.isArray(spec?.intent?.explicit_parameters)
    ? spec.intent.explicit_parameters.map(String)
    : [];
  return explicitParameters.some(
    (requestedPath) =>
      requestedPath === path ||
      requestedPath.startsWith(`${path}.`) ||
      path.startsWith(`${requestedPath}.`),
  );
}

function setLoadedSelectValue(control, value, label, aliases = {}) {
  if (value === null || value === undefined || value === "") {
    control.value = "";
    return;
  }
  const raw = String(value);
  const desired = aliases[raw] || raw;
  const option = [...control.options].find(
    (candidate) => candidate.value === desired,
  );
  if (!option || option.disabled) {
    throw new Error(
      `${label} value “${raw}” is not available in the current form.`,
    );
  }
  control.value = desired;
}

function requestedExperimentRevision(experiment, requestedNumber) {
  const revisions = Array.isArray(experiment.revisions)
    ? experiment.revisions
    : [];
  const wanted =
    requestedNumber === null ||
    requestedNumber === undefined ||
    requestedNumber === ""
      ? null
      : Number(requestedNumber);
  let revision = null;
  if (wanted !== null) {
    revision =
      revisions.find(
        (candidate) =>
          Number(candidate.revision_number ?? candidate.number) === wanted,
      ) || null;
    if (
      !revision &&
      Number(
        experiment.latest_revision?.revision_number ??
          experiment.latest_revision_number,
      ) === wanted
    ) {
      revision = experiment.latest_revision;
    }
    if (!revision)
      throw new Error(
        `Experiment revision ${wanted} was not returned by the API.`,
      );
  } else {
    revision = experiment.latest_revision || revisions[0] || experiment;
  }
  const rawSpec =
    revision?.requested_spec_json ??
    revision?.requested_spec ??
    experiment.requested_spec_json;
  return {
    revision,
    spec: loadedCanonicalObject(
      rawSpec,
      "The experiment requested specification",
    ),
    revisionNumber:
      revision?.revision_number ?? experiment.latest_revision_number ?? wanted,
  };
}

function pinnedAdapterFromExperiment(source) {
  let manifest = source.adapter_manifest;
  if (typeof manifest === "string") {
    try {
      manifest = JSON.parse(manifest);
    } catch (error) {
      throw new Error(
        `The pinned adapter manifest is invalid JSON: ${error.message}`,
      );
    }
  }
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    throw new Error(
      "This experiment revision has no pinned adapter manifest; loading it would silently use a different adapter version.",
    );
  }
  const liveAdapter = adapterRows.find(
    (adapter) =>
      adapterId(adapter) === String(source.adapter_id || "") ||
      String(adapter?.slug || adapter?.seed_key || "") ===
        String(source.adapter || ""),
  );
  const stableId = String(
    source.adapter_id ||
      adapterId(liveAdapter) ||
      source.adapter ||
      manifest.slug ||
      "",
  ).trim();
  const versionNumber = source.adapter_version;
  const versionId = String(source.adapter_version_id || "").trim();
  if (
    !stableId ||
    versionNumber === null ||
    versionNumber === undefined ||
    versionNumber === "" ||
    !versionId
  ) {
    throw new Error(
      "The experiment revision does not identify an exact adapter ID and version.",
    );
  }
  return {
    id: stableId,
    slug: source.adapter || manifest.slug || liveAdapter?.slug || stableId,
    name:
      manifest.display_name || liveAdapter?.name || source.adapter || stableId,
    label:
      manifest.display_name || liveAdapter?.label || source.adapter || stableId,
    status: "active",
    manifest_valid: true,
    _pinnedExperiment: true,
    selected_version: {
      id: versionId,
      version_number: versionNumber,
      manifest,
      manifest_sha256: source.adapter_manifest_sha256 || "",
    },
  };
}

function installPinnedSourceRevision(source, origin = "experiment") {
  const repository = String(source.repository || "").trim();
  const revision = String(source.revision || "").trim();
  if (!repository || !revision)
    throw new Error(
      "The experiment does not contain an exact repository and commit.",
    );
  ++sourceBranchRequest;
  ++sourceCommitRequest;
  sourceBranchesKey = "";
  sourceCommitsKey = "";
  sourceBranches = [{ name: "__pinned_experiment_commit__", sha: revision }];
  sourceCommits = [
    {
      sha: revision,
      short_sha: revision.slice(0, 8),
      subject: `Pinned by ${origin}`,
      author: "",
      timestamp: null,
    },
  ];
  elements.experimentSource.value = repository;
  elements.experimentWorkdir.value = source.project_subdirectory || ".";
  elements.experimentBranch.innerHTML =
    '<option value="__pinned_experiment_commit__">Pinned commit (branch not recorded)</option>';
  elements.experimentBranch.value = "__pinned_experiment_commit__";
  elements.experimentBranch.disabled = false;
  elements.experimentBranch.dataset.pinnedCommit = "true";
  elements.experimentRevision.innerHTML = `<option value="${escapeHtml(revision)}">${escapeHtml(revision.slice(0, 8))} / pinned ${escapeHtml(origin)} commit</option>`;
  elements.experimentRevision.value = revision;
  elements.experimentRevision.disabled = false;
  updateSourceCommitMeta();
  setSourceRefStatus(
    `Exact commit restored from the ${origin}. Use Refresh refs to choose another repository revision.`,
  );
}

function loadedSweepDefinition(sweep) {
  if (!sweep) return "";
  const strategy = String(sweep.strategy || "grid");
  if (strategy !== "grid")
    throw new Error(
      `Sweep strategy “${strategy}” cannot be represented by this configuration form.`,
    );
  if (Array.isArray(sweep.variants) && sweep.variants.length) {
    throw new Error(
      "An explicit sweep variant list cannot be represented by this configuration form.",
    );
  }
  if (
    sweep.max_parallel !== null &&
    sweep.max_parallel !== undefined &&
    Number(sweep.max_parallel) !== 2
  ) {
    throw new Error(
      `Sweep max_parallel=${sweep.max_parallel} cannot be represented by this configuration form.`,
    );
  }
  if (
    sweep.confirmation_threshold !== null &&
    sweep.confirmation_threshold !== undefined &&
    Number(sweep.confirmation_threshold) !== 20
  )
    throw new Error(
      `Sweep confirmation_threshold=${sweep.confirmation_threshold} cannot be represented by this configuration form.`,
    );
  const axes =
    sweep.axes && typeof sweep.axes === "object" && !Array.isArray(sweep.axes)
      ? sweep.axes
      : {};
  Object.entries(axes).forEach(([name, values]) => {
    if (!Array.isArray(values))
      throw new Error(`Sweep axis “${name}” is not an array.`);
  });
  const definition = { ...axes };
  const seeds = Array.isArray(sweep.seeds) ? sweep.seeds : [];
  if (seeds.length && !(seeds.length === 1 && Number(seeds[0]) === 42))
    definition.seeds = seeds;
  return Object.keys(definition).length
    ? JSON.stringify(definition, null, 2)
    : "";
}

function loadedNativeFieldValue(nativeSpec, path) {
  if (path === "native.argv")
    return { present: Array.isArray(nativeSpec.argv), value: nativeSpec.argv };
  if (path === "native.resume_argv")
    return {
      present: Array.isArray(nativeSpec.resume_argv),
      value: nativeSpec.resume_argv,
    };
  if (path.startsWith("native.config.")) {
    return loadedCanonicalPath(
      nativeSpec.config || {},
      path.slice("native.config.".length),
    );
  }
  if (path.startsWith("native.overrides.")) {
    const suffix = path.slice("native.overrides.".length);
    const overrides = nativeSpec.overrides || {};
    if (Object.prototype.hasOwnProperty.call(overrides, suffix))
      return { present: true, value: overrides[suffix] };
    return loadedCanonicalPath(overrides, suffix);
  }
  return { present: false, value: undefined };
}

function flattenLoadedNativeConfig(prefix, value, output) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const entries = Object.entries(value);
    if (entries.length) {
      entries.forEach(([key, nested]) =>
        flattenLoadedNativeConfig(`${prefix}.${key}`, nested, output),
      );
      return;
    }
  }
  output.push([prefix, value]);
}

function hydrateLoadedNativeSpec(nativeSpec = {}) {
  const fields = adapterInputFields();
  const scope = adapterDeclaredScopeValues(renderedAdapterDeclaredScope, true);
  const claimedKeys = new Set([
    "config.initial_checkpoint",
    "config.initial_checkpoint_mode",
  ]);
  for (const field of fields) {
    const descriptor = adapterOverrideDescriptor(field.path);
    if (descriptor.error) throw new Error(descriptor.error);
    claimedKeys.add(descriptor.key);
    const loaded = loadedNativeFieldValue(nativeSpec, field.path);
    if (!loaded.present) continue;
    if (field.sensitive)
      throw new Error(
        `${field.label} is sensitive and cannot be restored into the browser.`,
      );
    const control = document.getElementById(adapterFieldControlId(field.path));
    if (!control)
      throw new Error(
        `${field.label} could not be rendered for the pinned adapter.`,
      );
    setAdapterFieldControlValue(control, field, loaded.value);
    if (
      control.tagName === "SELECT" &&
      control.value === "" &&
      loaded.value !== ""
    ) {
      throw new Error(
        `${field.label} value ${JSON.stringify(loaded.value)} is unavailable at the pinned commit.`,
      );
    }
    scope?.set(field.path, {
      kind: field.kind,
      raw: adapterFieldRawValue(control),
      touched: true,
    });
  }

  const advanced = [];
  const flattenedConfig = [];
  flattenLoadedNativeConfig("config", nativeSpec.config || {}, flattenedConfig);
  const collidesWithClaimed = (key) =>
    [...claimedKeys].some(
      (claimed) =>
        claimed === key ||
        claimed.startsWith(`${key}.`) ||
        key.startsWith(`${claimed}.`),
    );
  flattenedConfig.forEach(([key, value]) => {
    if (!collidesWithClaimed(key))
      advanced.push(`${key}=${JSON.stringify(value)}`);
  });
  Object.entries(nativeSpec.overrides || {}).forEach(([key, value]) => {
    if (!collidesWithClaimed(key))
      advanced.push(`${key}=${JSON.stringify(value)}`);
  });
  if (Array.isArray(nativeSpec.argv) && !claimedKeys.has("argv"))
    advanced.push(`argv=${JSON.stringify(nativeSpec.argv)}`);
  if (Array.isArray(nativeSpec.resume_argv) && !claimedKeys.has("resume_argv"))
    advanced.push(`resume_argv=${JSON.stringify(nativeSpec.resume_argv)}`);
  elements.nativeOverrides.value = advanced.join("\n");
  const validation = validateAdapterDeclaredFields({
    focus: false,
    notify: false,
  });
  if (validation === false)
    throw new Error(
      "The pinned adapter-native values are not valid in the current form.",
    );
  renderCommonHyperparameterDefaults();
}

function loadedEvaluationSuiteIds(specs) {
  if (!specs.length) return [];
  const optionIds = new Set(
    [...elements.experimentEvaluationSuites.options].map(
      (option) => option.value,
    ),
  );
  return specs.map((evaluation) => {
    const requestedNames = [
      evaluation.suite,
      evaluation.suite_id,
      evaluation.name,
    ]
      .flatMap((value) =>
        value && typeof value === "object"
          ? [value.id, value.slug, value.name]
          : [value],
      )
      .filter(Boolean)
      .map(String);
    const requestedVersion = firstValue(
      evaluation.suite_version,
      evaluation.version,
    );
    const matches = evaluationSuites.filter((suite) => {
      const names = [suite.id, suite.slug, suite.name, suite.label, suite.suite]
        .filter(Boolean)
        .map(String);
      if (!requestedNames.some((name) => names.includes(name))) return false;
      if (
        requestedVersion === null ||
        requestedVersion === undefined ||
        requestedVersion === ""
      )
        return true;
      const version = firstValue(suite.suite_version, suite.version);
      return (
        version !== null &&
        version !== undefined &&
        String(version) === String(requestedVersion)
      );
    });
    if (matches.length !== 1) {
      const label = requestedNames[0] || "unnamed suite";
      throw new Error(
        matches.length
          ? `Evaluation suite “${label}” is ambiguous in the current registry.`
          : `Evaluation suite “${label}” at version “${requestedVersion || "unspecified"}” is not registered.`,
      );
    }
    const id = evaluationSuiteId(matches[0]);
    if (!optionIds.has(id))
      throw new Error(
        `Evaluation suite “${id}” is not available in the configuration selector.`,
      );
    return id;
  });
}

function clearExperimentPreviewAfterLoad() {
  elements.experimentPreviewPanel.hidden = true;
  elements.experimentPreviewEmpty.hidden = false;
  elements.experimentPreviewMeta.textContent = "";
  elements.experimentPreviewBlockers.hidden = true;
  elements.experimentPreviewBlockerList.replaceChildren();
  elements.experimentScriptPreview.textContent = "";
}

async function hydrateExperimentConfiguration(spec, request) {
  const source = loadedCanonicalObject(spec.source, "Experiment source");
  const runtime = loadedCanonicalObject(spec.runtime, "Experiment runtime");
  const train = loadedCanonicalObject(
    spec.train,
    "Experiment training configuration",
  );
  const resources = loadedCanonicalObject(
    spec.resources,
    "Experiment resources",
  );
  const nativeSpec =
    spec.native && typeof spec.native === "object" ? spec.native : {};
  const checkpoint =
    train.checkpoint && typeof train.checkpoint === "object"
      ? train.checkpoint
      : {};
  const tracking =
    spec.tracking && typeof spec.tracking === "object" ? spec.tracking : {};
  const evaluations = Array.isArray(spec.evaluation) ? spec.evaluation : [];

  loadedExperimentCanonicalContext = null;
  loadedExperimentAdapterSnapshot = pinnedAdapterFromExperiment(source);
  const pinnedAdapterOptionId = adapterOptionId(
    loadedExperimentAdapterSnapshot,
  );
  populateExperimentAdapters(pinnedAdapterOptionId);
  elements.experimentAdapter.value = pinnedAdapterOptionId;
  const adapterScope = adapterDeclaredScope(loadedExperimentAdapterSnapshot);
  if (adapterScope) adapterDeclaredValueState.delete(adapterScope.key);
  applySelectedAdapter({ loadSource: false });

  setLoadedControlValue(elements.experimentName, spec.name);
  installPinnedSourceRevision(source);

  populateExperimentDataBundles();
  const pinnedInputs = spec.data?.bundle?.assignments || [];
  experimentInputSlots().forEach((slot, index) => {
    const pinned = pinnedInputs.find(
      (item) =>
        item.role === slot.role && Number(item.position || 0) === slot.position,
    );
    const control = index
      ? document.getElementById("experiment-data-input-" + index)
      : elements.experimentDataBundle;
    const match =
      pinned &&
      trainingDatasetRows.find((dataset) => {
        const candidate = dataset.assignments[0];
        return (
          candidate?.version?.manifest_sha256 ===
            pinned.version?.manifest_sha256 &&
          candidate.resource?.name === pinned.resource?.name &&
          candidate.config?.location?.path === pinned.config?.location?.path
        );
      });
    if (pinned && !match)
      throw new Error(
        "Saved " +
          slot.role.replaceAll("_", " ") +
          " files are no longer available at their registered location. Choose replacement data explicitly.",
      );
    control.value = match?.id || "";
    control.setCustomValidity("");
    control.removeAttribute("aria-invalid");
  });
  populateExperimentDataBundles();
  renderAdapterDeclaredFields();

  const batch =
    train.batch && typeof train.batch === "object" ? train.batch : {};
  setLoadedControlValue(elements.hpLearningRate, train.learning_rate);
  setLoadedControlValue(elements.hpBatchSize, batch.value);
  setLoadedSelectValue(
    elements.hpBatchSemantics,
    batch.declared_semantics,
    "Batch semantics",
  );
  setLoadedControlValue(elements.hpGradAcc, batch.gradient_accumulation_steps);
  setLoadedControlValue(elements.hpNumWorkers, train.num_workers_per_rank);
  setLoadedControlValue(elements.hpMaxSteps, train.max_steps);
  const precision =
    train.precision === null || train.precision === undefined
      ? "adapter-default"
      : String(train.precision).toLowerCase();
  setLoadedSelectValue(elements.hpPrecision, precision, "Precision", {
    bfloat16: "bf16",
    float32: "fp32",
    float16: "fp16",
  });

  if (!restoreSshGatewayPreference()) {
    setLoadedSelectValue(elements.gateway, resources.gateway, "SSH gateway");
  }
  setLoadedSelectValue(
    elements.resourcePolicy,
    resources.queue_policy,
    "Queue policy",
  );
  const gpu =
    resources.gpu && typeof resources.gpu === "object" ? resources.gpu : {};
  const gpuMode = gpu.mode === "explicit" ? "manual" : gpu.mode || "auto";
  setLoadedSelectValue(elements.gpuMode, gpuMode, "GPU allocation");
  setLoadedControlValue(elements.experimentGpuCount, gpu.count);
  const gpuType = firstValue(gpu.gpu_type, gpu.type, gpu.profile);
  setLoadedSelectValue(
    elements.experimentGpuType,
    gpuType === "any" ? "auto" : gpuType,
    "GPU type",
  );
  setLoadedControlValue(elements.resourceNodes, resources.nodes);
  setLoadedControlValue(elements.resourceCpus, resources.cpus_per_task);
  setLoadedControlValue(elements.resourceMemory, resources.memory_gb);
  setLoadedControlValue(elements.resourceTime, resources.time_limit);

  setLoadedControlValue(
    elements.checkpointSaveSteps,
    loadedCanonicalPathIsExplicit(spec, "train.checkpoint.save_every_steps")
      ? checkpoint.save_every_steps
      : null,
  );
  setLoadedControlValue(
    document.querySelector("#checkpoint-warning-seconds"),
    checkpoint.save_before_timeout_seconds ?? 300,
  );
  setLoadedControlValue(
    elements.checkpointMaxAttempts,
    checkpoint.max_attempts,
  );
  elements.checkpointAutoResume.checked = Boolean(checkpoint.auto_resume);
  const initialCheckpoint = nativeSpec.config?.initial_checkpoint;
  const initialMode =
    nativeSpec.config?.initial_checkpoint_mode ||
    (initialCheckpoint ? "weights" : "none");
  setLoadedSelectValue(
    elements.checkpointMode,
    initialMode,
    "Initial checkpoint",
    { fresh: "none" },
  );
  setLoadedControlValue(elements.checkpointPath, initialCheckpoint);

  const providers = Array.isArray(tracking.providers)
    ? tracking.providers
    : tracking.provider
      ? [{ ...tracking, provider: tracking.provider }]
      : [];
  const enabledProviders = providers.filter(
    (provider) => provider?.enabled !== false,
  );
  const unsupportedProviders = enabledProviders.filter(
    (provider) =>
      !["wandb", "mlflow"].includes(
        String(provider?.provider || provider?.type || "").toLowerCase(),
      ),
  );
  if (unsupportedProviders.length) {
    throw new Error(
      `Tracking provider “${unsupportedProviders[0].provider || unsupportedProviders[0].type}” cannot be represented by this form.`,
    );
  }
  const wandbProviders = enabledProviders.filter(
    (provider) =>
      String(provider?.provider || provider?.type || "").toLowerCase() ===
      "wandb",
  );
  const mlflowProviders = enabledProviders.filter(
    (provider) =>
      String(provider?.provider || provider?.type || "").toLowerCase() ===
      "mlflow",
  );
  if (wandbProviders.length > 1 || mlflowProviders.length > 1)
    throw new Error(
      "Duplicate tracking providers cannot be represented by this form.",
    );
  const wandb = wandbProviders[0] || {};
  const mlflow = mlflowProviders[0] || {};
  elements.wandbEnabled.checked = Boolean(wandbProviders.length);
  setLoadedControlValue(elements.wandbProject, wandb.project);
  setLoadedControlValue(elements.wandbRunName, wandb.run_name_template);
  elements.mlflowEnabled.checked = Boolean(mlflowProviders.length);
  setLoadedControlValue(elements.mlflowUri, mlflow.tracking_uri);
  setLoadedControlValue(elements.mlflowExperiment, mlflow.experiment);
  setLoadedControlValue(elements.mlflowRunName, mlflow.run_name_template);

  setLoadedControlValue(
    elements.sweepDefinition,
    loadedSweepDefinition(spec.sweep),
  );
  const suiteIds = loadedEvaluationSuiteIds(evaluations);
  elements.evaluationEnabled.checked = suiteIds.length > 0;
  [...elements.experimentEvaluationSuites.options].forEach((option) => {
    option.selected = suiteIds.includes(option.value);
  });

  await inspectRepositoryRuntime(false);
  if (request !== experimentConfigurationLoadRequest) return false;
  const backend = String(runtime.backend || runtime.type || "")
    .toLowerCase()
    .replace("container", "apptainer");
  const runtimeCandidate = runtimeCandidates.find(
    (candidate) => candidate.value === backend && candidate.runnable,
  );
  if (!runtimeCandidate)
    throw new Error(
      `Pinned runtime backend “${backend || "missing"}” is not runnable for this adapter and commit.`,
    );
  elements.experimentRuntime.value = backend;
  applyRuntimeSelection(true);
  const profileId = firstValue(
    runtime.profile_id,
    runtime.profile_snapshot?.id,
  );
  if (profileId) {
    const profile = runtimeProfiles.find(
      (candidate) =>
        candidate.id === String(profileId) && candidate.backend === backend,
    );
    if (!profile)
      throw new Error(
        `Pinned runtime profile “${profileId}” is not available from the cached repository inspection.`,
      );
    elements.experimentRuntimeProfileSelect.value = profile.id;
    applyRuntimeProfileSelection(true);
  } else {
    const customProfile = firstValue(
      runtime.lock_file,
      runtime.environment_path,
      runtime.container_image,
      runtime.profile,
    );
    if (customProfile !== null && customProfile !== undefined)
      elements.experimentRuntimeProfile.value = String(customProfile);
    if (
      runtimeCandidate.manual &&
      runtimeCandidate.requiresProfile &&
      !elements.experimentRuntimeProfile.value.trim()
    ) {
      throw new Error(
        `Pinned runtime backend “${backend}” requires a runtime profile or path.`,
      );
    }
  }

  renderAdapterDeclaredFields();
  hydrateLoadedNativeSpec(nativeSpec);
  updateTrainingPresetLabel();
  loadedExperimentCanonicalContext = {
    tracking: { ...tracking, providers },
    checkpoint: { ...checkpoint },
    resources: { ...resources },
    evaluation: { specs: evaluations, suiteIds },
  };
  updateExperimentFields();
  updateBatchCompatibility();
  renderTrackingNamePreview();
  updateExperimentSubmitState();
  invalidateExperimentPreview();
  clearNotice(elements.experimentsError);
  return true;
}

async function loadExperimentConfiguration(
  id,
  revisionNumber,
  launcher = null,
) {
  const scope = "experiment-load";
  invalidateExperimentPreview();
  const request = ++experimentConfigurationLoadRequest;
  const originalLabel = launcher?.textContent || "Load configuration";
  if (launcher) {
    launcher.disabled = true;
    launcher.textContent = "Loading...";
  }
  try {
    const payload = await api(`/api/experiments/${encodeURIComponent(id)}`);
    if (request !== experimentConfigurationLoadRequest) return;
    const experiment = entityFrom(payload, "experiment");
    const loaded = requestedExperimentRevision(experiment, revisionNumber);
    const hydrated = await hydrateExperimentConfiguration(
      { ...loaded.spec, name: loaded.spec.name || experiment.name },
      request,
    );
    if (!hydrated || request !== experimentConfigurationLoadRequest) return;
    loadedExperimentOrigin = {
      id: String(id),
      revision: loaded.revisionNumber || "latest",
    };
    updateExperimentSubmitState();
    elements.experimentName.focus({ preventScroll: true });
    elements.experimentName.scrollIntoView({
      behavior: "smooth",
      block: "center",
      inline: "nearest",
    });
    showToast(
      `Loaded ${loaded.spec.name || experiment.name || id}, revision ${loaded.revisionNumber || "latest"}.`,
      false,
      { scope },
    );
  } catch (error) {
    if (request !== experimentConfigurationLoadRequest) return;
    loadedExperimentCanonicalContext = null;
    showNotice(
      elements.experimentsError,
      `Experiment configuration could not be loaded: ${error.message}`,
      { scope },
    );
    showToast(`Experiment configuration load failed: ${error.message}`, true, {
      scope,
    });
  } finally {
    if (launcher?.isConnected) {
      launcher.disabled = false;
      launcher.textContent = originalLabel;
    }
  }
}

function renderExperiments() {
  const query = document
    .getElementById("experiment-search")
    .value.trim()
    .toLowerCase();
  const datasetId =
    document.getElementById("experiment-search").dataset.datasetId;
  const matchedPresetIds =
    document.getElementById("experiment-search").dataset.presetIds;
  const presetIds = matchedPresetIds ? JSON.parse(matchedPresetIds) : null;
  const rows = experimentRows.filter((row) =>
    datasetId
      ? presetIds
        ? presetIds.includes(row.id)
        : (row.dataset_ids || []).includes(datasetId)
      : [
          row.name,
          row.id,
          row.adapter_id,
          JSON.stringify(row.adapter),
          experimentLifecycle(row).label,
        ]
          .join(" ")
          .toLowerCase()
          .includes(query),
  );
  elements.experimentCount.textContent = `${rows.length} of ${experimentRows.length} presets`;
  if (!rows.length) {
    elements.experimentsBody.innerHTML = emptyRow(
      9,
      query
        ? "No experiment presets match the filter."
        : "No presets yet. Choose New to create one.",
    );
    return;
  }
  elements.experimentsBody.innerHTML = rows
    .map((experiment) => {
      const id = experiment.id || experiment.experiment_id;
      const source = experiment.source || {};
      const revision =
        experiment.revision ||
        source.revision ||
        experiment.git_revision ||
        "-";
      const adapter =
        experiment.adapter?.id ||
        experiment.adapter_id ||
        experiment.adapter ||
        "-";
      const variantCount =
        experiment.variant_count ?? experiment.variants?.length ?? 0;
      const runs = experiment.run_count ?? experiment.runs?.length ?? 0;
      const lifecycle = experimentLifecycle(experiment);
      const revisionNumber = experiment.latest_revision_number;
      const secondaryIdentity = [
        revisionNumber ? `Experiment revision ${revisionNumber}` : "",
      ]
        .filter(Boolean)
        .join(" / ");
      return `
        <tr>
          <td><span class="node-name">${escapeHtml(experiment.name || id)}</span><span class="secondary">${escapeHtml(secondaryIdentity)}</span></td>
          <td><button type="button" class="text-button" data-preset-datasets="${escapeHtml(id)}">${(experiment.dataset_ids || []).length} dataset${(experiment.dataset_ids || []).length === 1 ? "" : "s"}</button></td>
          <td>${escapeHtml(adapter)}<span class="secondary">${escapeHtml(shortId(revision))}</span></td>
          <td>${escapeHtml(variantCount)}</td>
          <td>${escapeHtml(runs)}</td>
          <td>${statusPill(lifecycle.label)}</td>
          <td>${trackingLinksHtml(experiment)}</td>
          <td>${escapeHtml(formatDate(experiment.updated_at || experiment.created_at))}</td>
          <td class="row-actions">
            <button type="button" data-experiment-action="load" data-id="${escapeHtml(id)}" data-revision-number="${escapeHtml(revisionNumber || "")}">Load configuration</button>
            ${
              lifecycle.locked
                ? `<button type="button" data-experiment-action="view" data-id="${escapeHtml(id)}" aria-controls="experiment-detail" aria-expanded="false">View variants</button>`
                : `<button type="button" data-experiment-action="submit" data-id="${escapeHtml(id)}">Submit</button>`
            }
            <button type="button" data-delete-kind="experiment" data-delete-id="${escapeHtml(id)}">Delete</button>
          </td>
        </tr>`;
    })
    .join("");
}

let trainingInputsPromise = null;
function loadTrainingInputs(force = false) {
  if (!trainingInputsPromise) {
    trainingInputsPromise = Promise.all([
      loadAdapters(force),
      loadDataBundles(force),
    ]).finally(() => {
      trainingInputsPromise = null;
    });
  }
  return trainingInputsPromise;
}

async function loadExperiments(force = false) {
  if (loadedTabs.has("experiments") && !force) return;
  elements.refreshExperiments.disabled = true;
  clearNotice(elements.experimentsError);
  await Promise.all([
    loadTrainingInputs(force),
    loadEvaluationSuites(force, ""),
    loadTrackingConnections(force).catch(() => undefined),
  ]);
  try {
    const payload = await api("/api/experiments");
    experimentRows = listFrom(payload, ["experiments"]);
    renderExperiments();
    updateExperimentSubmitState();
    loadedTabs.add("experiments");
  } catch (error) {
    experimentRows = [];
    elements.experimentsBody.innerHTML = emptyRow(
      8,
      "Experiment data could not be loaded.",
    );
    elements.experimentCount.textContent = "Unavailable";
    showNotice(
      elements.experimentsError,
      `Experiment API unavailable: ${error.message}`,
    );
  } finally {
    elements.refreshExperiments.disabled = false;
  }
}

function renderVariants(payload, launcher = null) {
  const experiment = entityFrom(payload, "experiment");
  const variants = listFrom(experiment, ["variants"]);
  const runs = listFrom(experiment, ["runs"]);
  const runsByVariant = new Map();
  const addRun = (variantId, run) => {
    if (!variantId || !run) return;
    const key = String(variantId);
    const grouped = runsByVariant.get(key) || [];
    const runId = run.id || run.run_id;
    if (
      !runId ||
      !grouped.some(
        (candidate) =>
          String(candidate.id || candidate.run_id) === String(runId),
      )
    ) {
      grouped.push(run);
    }
    runsByVariant.set(key, grouped);
  };
  runs.forEach((run) => {
    const variantId = run.variant_id || run.variant?.id;
    addRun(variantId, run);
  });
  elements.experimentDetailTitle.textContent =
    experiment.name || "Experiment variants";
  const lifecycle = experimentLifecycle(experiment);
  elements.experimentDetailMeta.textContent = [
    experiment.id || experiment.experiment_id || "",
    experiment.latest_revision_number
      ? `Revision ${experiment.latest_revision_number}`
      : "",
    lifecycle.label,
  ]
    .filter(Boolean)
    .join(" / ");
  renderTrackingDetail(elements.experimentDetailTracking, experiment);
  revealPanel(elements.experimentDetail, {
    focusTarget: elements.experimentDetailTitle,
    launcher,
  });
  if (!variants.length) {
    elements.variantsBody.innerHTML = emptyRow(
      5,
      "No expanded variants were returned.",
    );
    return;
  }
  elements.variantsBody.innerHTML = variants
    .map((variant, index) => {
      const variantId = variant.id || "";
      const variantRuns = [...(runsByVariant.get(String(variantId)) || [])];
      (Array.isArray(variant.runs) ? variant.runs : []).forEach((run) =>
        addRun(variantId, run),
      );
      if (variant.run) addRun(variantId, variant.run);
      if (variant.run_id)
        addRun(variantId, {
          ...variant,
          id: variant.run_id,
          status: variant.status || variant.state,
        });
      const orderedRuns = [
        ...(runsByVariant.get(String(variantId)) || variantRuns),
      ].sort(
        (left, right) =>
          (Date.parse(right.updated_at || right.created_at || 0) || 0) -
          (Date.parse(left.updated_at || left.created_at || 0) || 0),
      );
      const latestRun = orderedRuns[0] || null;
      const latestState = latestRun ? variantRunDisplayState(latestRun) : null;
      const result =
        variant.result || variant.primary_result || latestRun?.result;
      const runList = orderedRuns.length
        ? orderedRuns
            .map((run) => {
              const runId = run.id || run.run_id || "-";
              const runNumber = run.run_number;
              const label = runNumber
                ? `Training Run ${runNumber}`
                : shortId(runId, 18);
              const secondary = runNumber ? shortId(runId, 18) : "";
              return `<div><span class="node-name">${escapeHtml(label)}</span>${secondary ? `<span class="secondary">${escapeHtml(secondary)}</span>` : ""}${trackingLinksHtml(run, { compact: true })}</div>`;
            })
            .join("")
        : '<span class="secondary">No runs</span>';
      return `
      <tr>
        <td><span class="node-name">${escapeHtml(variant.name || `variant-${index + 1}`)}</span><span class="secondary">${escapeHtml(variant.id || "")}</span></td>
        <td class="wrap-cell"><code>${escapeHtml(compactJson(variant.parameters || variant.overrides || {}, 180))}</code></td>
        <td class="wrap-cell">${runList}</td>
        <td>${latestRun ? statusPill(latestState) : '<span class="secondary">No runs</span>'}</td>
        <td>${escapeHtml(compactJson(result, 120))}</td>
      </tr>`;
    })
    .join("");
}

async function viewExperiment(id, launcher = null) {
  const revealLauncher = launcher || currentRevealLauncher();
  const requestToken = disclosureToken(
    elements.experimentDetail,
    revealLauncher,
  );
  try {
    const payload = await api(`/api/experiments/${encodeURIComponent(id)}`);
    if (!disclosureTokenIsCurrent(elements.experimentDetail, requestToken))
      return;
    renderVariants(payload, revealLauncher);
  } catch (error) {
    if (!disclosureTokenIsCurrent(elements.experimentDetail, requestToken))
      return;
    closeDisclosurePanel(elements.experimentDetail, revealLauncher);
    showToast(`Experiment detail failed: ${error.message}`, true);
  }
}

function progressPercent(value) {
  const raw =
    typeof value === "object" && value !== null
      ? (value.fraction ?? value.percent ?? value.completed_ratio)
      : value;
  const number = Number(raw);
  if (!Number.isFinite(number)) return null;
  return Math.max(0, Math.min(100, number <= 1 ? number * 100 : number));
}

function progressCell(progress, label) {
  const percent = progressPercent(progress);
  if (percent === null)
    return `<span class="secondary">${escapeHtml(label || "Unknown")}</span>`;
  return `
    <div class="progress-cell">
      <div class="progress-track"><i style="width:${percent}%"></i></div>
      <span>${escapeHtml(label || `${Math.round(percent)}%`)}</span>
    </div>`;
}

function progressSummaryLabel(summary) {
  if (!summary || typeof summary !== "object") return null;
  const completed =
    summary.completed === null || summary.completed === undefined
      ? Number.NaN
      : Number(summary.completed);
  const total =
    summary.total === null || summary.total === undefined
      ? Number.NaN
      : Number(summary.total);
  if (Number.isFinite(completed) && Number.isFinite(total)) {
    return `${completed}/${total} ${summary.unit || "items"}`;
  }
  if (["complete", "not_applicable"].includes(summary.eta_state))
    return "Recorded count unavailable";
  if (Number.isFinite(total))
    return `Waiting for ${summary.unit || "progress"} data`;
  return "Progress unknown";
}

function compactEtaDuration(value) {
  const seconds = Math.max(0, Math.ceil(Number(value)));
  if (!Number.isFinite(seconds)) return null;
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  if (seconds < 86400) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h${minutes ? ` ${minutes}m` : ""}`;
  }
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  return `${days}d${hours ? ` ${hours}h` : ""}`;
}

function etaReasonLabel(reason) {
  return (
    {
      terminal_success: "Finished",
      terminal_without_completion: "Job ended before completion",
      work_complete: "All recorded work is complete",
      resolved_max_steps_unavailable: "Resolved max_steps is unavailable",
      total_work_unavailable: "Total work is unavailable",
      attempt_not_started: "No attempt has started",
      attempt_not_running: "Waiting for the current attempt",
      waiting_for_retry: "Waiting for another attempt",
      waiting_for_progress: "Waiting for recorded progress",
      insufficient_progress_samples: "More current-attempt samples are needed",
      current_attempt_timing_unavailable:
        "Current-attempt timing is unavailable",
      current_attempt_progress_rate: "Estimated from current-attempt progress",
    }[reason] || String(reason || "ETA unavailable").replaceAll("_", " ")
  );
}

function etaPresentation(summary) {
  const state = String(summary?.eta_state || "unknown");
  const duration = compactEtaDuration(summary?.eta_seconds);
  if (state === "estimating" && duration !== null) {
    return {
      state,
      label: `~${duration}`,
      detail: `~${duration} remaining`,
      reason: etaReasonLabel(summary.eta_reason),
    };
  }
  if (state === "complete") {
    return {
      state,
      label: "Complete",
      detail: "Complete",
      reason: etaReasonLabel(summary?.eta_reason),
    };
  }
  if (state === "waiting") {
    return {
      state,
      label: "Waiting",
      detail: `Waiting - ${etaReasonLabel(summary?.eta_reason)}`,
      reason: etaReasonLabel(summary?.eta_reason),
    };
  }
  if (state === "not_applicable") {
    return {
      state,
      label: "Not applicable",
      detail: `Not applicable - ${etaReasonLabel(summary?.eta_reason)}`,
      reason: etaReasonLabel(summary?.eta_reason),
    };
  }
  return {
    state: "unknown",
    label: "Unknown",
    detail: `Unknown - ${etaReasonLabel(summary?.eta_reason)}`,
    reason: etaReasonLabel(summary?.eta_reason),
  };
}

function etaCell(summary) {
  const eta = etaPresentation(summary);
  return `<span class="eta-value" data-eta-state="${escapeHtml(eta.state)}" title="${escapeHtml(eta.reason)}">${escapeHtml(eta.label)}</span>`;
}

function runResourceLabel(run) {
  const resources = run.resources || {};
  const latest = run.latest_attempt || {};
  const gpus =
    latest.gpu_count ??
    resources.gpu?.count ??
    resources.gpus_per_node ??
    resources.gpu_count ??
    run.gpu_count ??
    run.gpus;
  const nodes = resources.nodes ?? run.nodes;
  const gpuType =
    latest.gpu_type ||
    resources.gpu?.type ||
    resources.gpu_type ||
    run.gpu_type;
  const parts = [];
  if (gpus) parts.push(`${gpus}x ${gpuType || "GPU"}`);
  if (nodes) parts.push(`${nodes} node${Number(nodes) === 1 ? "" : "s"}`);
  return parts.join(" / ") || "-";
}

const modelIORequests = new WeakMap();

function renderModelIO(section, summary, message = "") {
  if (!section) return;
  const pending = modelIORequests.get(section);
  if (pending) clearTimeout(pending.timer);
  modelIORequests.delete(section);
  section.innerHTML = `<h3>Inputs and outputs</h3><div class="key-value-grid">${summary ? keyValueHtml(summary.entries || []) : ""}</div><p class="secondary">${escapeHtml(message || summary?.note || "Select an adapter.")}</p>`;
}

function requestModelIO(section, manifest, values = {}, bundleId = null) {
  if (!section) return;
  const previous = modelIORequests.get(section);
  if (previous) clearTimeout(previous.timer);
  const request = {};
  renderModelIO(section, null, "Reading input and output sizes…");
  modelIORequests.set(section, request);
  request.timer = setTimeout(async () => {
    try {
      // Send only declarations; embedded runtime scripts are not needed here.
      const train = manifest?.train || {};
      const summary = await api("/api/model-io/preview", {
        method: "POST",
        body: JSON.stringify({
          manifest: {
            slug: manifest?.slug,
            train: {
              model_io: train.model_io,
              data_requirements: train.data_requirements,
              input_fields: (train.input_fields || [])
                .filter((field) => !field.sensitive)
                .map(({ path, default: value }) => ({ path, default: value })),
            },
          },
          values,
          ...(bundleId
            ? {
                data_selections:
                  selectedExperimentDataBundle()?.selections || [],
              }
            : {}),
        }),
      });
      if (modelIORequests.get(section) === request)
        renderModelIO(section, summary);
    } catch (error) {
      if (modelIORequests.get(section) === request)
        renderModelIO(section, null, `Sizes unavailable: ${error.message}`);
    }
  }, 180);
}

function refreshAdapterModelIO() {
  const section = document.getElementById("adapter-model-io");
  try {
    requestModelIO(section, JSON.parse(elements.adapterManifest.value));
  } catch {
    renderModelIO(section, null, "Enter a valid manifest to see sizes.");
  }
}

function refreshExperimentModelIO() {
  const adapter = selectedAdapter();
  const section = document.getElementById("experiment-model-io");
  if (!adapter) {
    renderModelIO(section, null);
    return;
  }
  const values = {};
  for (const field of declaredAdapterInputFields(adapter)) {
    if (field.sensitive) continue;
    const control = document.getElementById(adapterFieldControlId(field.path));
    if (!control) continue;
    const parsed = parseAdapterDeclaredValue(control, field);
    if (parsed.error) {
      renderModelIO(
        section,
        null,
        "Complete the model settings to resolve sizes.",
      );
      return;
    }
    if (parsed.present) values[field.path] = parsed.value;
  }
  if (elements.nativeOverrides.value.trim()) {
    renderModelIO(
      section,
      null,
      "Advanced native values are set. Exact sizes depend on those overrides.",
    );
    return;
  }
  const bundle = selectedExperimentDataBundle();
  if (bundle && !experimentBundleCompatibility(bundle, adapter).compatible) {
    renderModelIO(
      section,
      null,
      "Choose a compatible dataset to resolve sizes.",
    );
    return;
  }
  requestModelIO(section, adapterManifest(adapter), values, bundle?.id || null);
}

function trainingAdapterLabel(run = {}, attempt = null) {
  const snapshot = attempt?.execution_snapshot_json;
  const source =
    snapshot?.resolved_spec?.source || run.resolved_spec_json?.source || {};
  const name = snapshot?.adapter?.slug || source.adapter || run.adapter_name;
  const version =
    snapshot?.adapter?.version ?? source.adapter_version ?? run.adapter_version;
  return name
    ? `${name}${version != null ? ` · v${version}` : ""}`
    : "Not recorded";
}

function filteredRuns() {
  const query = elements.runSearch.value.trim().toLowerCase();
  const stateFilter = elements.runStatusFilter.value;
  return runRows.filter((run) => {
    const state = jobStatusLabel(run).toLowerCase();
    const haystack = [
      run.id,
      run.run_id,
      run.experiment_name,
      run.experiment_id,
      run.variant_name,
      run.variant_id,
      trainingAdapterLabel(run),
      run.latest_attempt?.slurm_job_id,
      run.slurm_job_id,
    ]
      .join(" ")
      .toLowerCase();
    const matchesState =
      stateFilter === "all" ||
      (stateFilter === "completed"
        ? ["succeeded", "completed"].includes(state)
        : state === stateFilter);
    return (!query || haystack.includes(query)) && matchesState;
  });
}

function runRowDescriptor(run) {
  const id = String(run.id || run.run_id || "");
  const state = jobStatusLabel(run);
  const attempt =
    run.attempt ??
    run.attempt_number ??
    run.latest_attempt?.attempt_number ??
    run.latest_attempt?.number ??
    "-";
  const attemptCount = runAttemptCount(run);
  const attemptLabel =
    attempt === "-"
      ? "-"
      : `#${attempt}${attemptCount && Number(attemptCount) !== Number(attempt) ? ` of ${attemptCount}` : ""}`;
  const progressSummary = run.progress_summary;
  const progress = progressSummary ?? run.progress ?? run.metrics?.progress;
  const progressLabel =
    progressSummaryLabel(progressSummary) ||
    run.progress_label ||
    run.current_step;
  const eta =
    String(state).toUpperCase() === "RUNNING"
      ? `<div class="progress-eta">ETA ${etaCell(progressSummary)}</div>`
      : "";
  const runLabel = trainingRunName(run);
  const runSecondary = [run.kind || run.stage || "train"]
    .filter(Boolean)
    .join(" / ");
  const experimentRevision = run.experiment_revision_number
    ? `Revision ${run.experiment_revision_number}`
    : "";
  const variantLabel = run.variant_name || run.variant_id || "-";
  const active =
    activeDisclosure?.panel === elements.runDetail && activeRunDetailId === id;
  return {
    id,
    cells: [
      {
        html: `<span class="node-name">${escapeHtml(runLabel)}</span><span class="secondary">${escapeHtml(runSecondary)}</span>`,
      },
      {
        html: `${escapeHtml(run.experiment_name || run.experiment_id || "-")}<span class="secondary">${escapeHtml(trainingAdapterLabel(run))}</span><span class="secondary">${escapeHtml([experimentRevision, variantLabel].filter(Boolean).join(" / "))}</span>`,
      },
      {
        html:
          trainingDataValue(run).map(valueHtml).join("<br>") ||
          escapeHtml(savedTrainingDataLabel(run)),
      },
      { html: statusPill(state) },
      {
        html: executionHtml(run),
      },
      { html: escapeHtml(runResourceLabel(run)) },
      { html: escapeHtml(attemptLabel) },
      {
        html: run.status_detail
          ? `<span class="secondary" title="${escapeHtml(run.status_detail)}">Cluster acceptance unconfirmed</span>`
          : `${eta}${progressCell(progress, progressLabel)}`,
      },
      { html: escapeHtml(formatDate(run.latest_attempt?.started_at)) },
      {
        html: escapeHtml(
          progressSummary?.elapsed_seconds == null
            ? "-"
            : (compactEtaDuration(progressSummary.elapsed_seconds) ?? "-"),
        ),
      },
      { html: trackingLinksHtml(run) },
      {
        className: "row-actions",
        preserve: active,
        html: `<button type="button" data-run-action="view" data-id="${escapeHtml(id)}" aria-controls="run-detail" aria-expanded="false">View</button>${["CREATED", "SUBMITTING", "SUBMITTED", "PENDING", "PENDING_SLURM", "QUEUED", "CONFIGURING", "RUNNING", "CANCELLING", "RESUMING", "RETRYING", "RETRY_PENDING", "REQUEUED"].includes(state) ? cancellationActionButton("run", id, run.manual_actions || { cancel: { enabled: state !== "CANCELLING" } }, { label: "Cancel", className: "" }) : `<button type="button" data-delete-kind="run" data-delete-id="${escapeHtml(id)}">Delete</button>`}`,
      },
    ],
  };
}

function renderRuns({ background = false } = {}) {
  const knownStates = new Set(
    [...elements.runStatusFilter.options].map((option) => option.value),
  );
  for (const run of runRows) {
    const state = jobStatusLabel(run).toLowerCase();
    if (!state || state === "succeeded" || knownStates.has(state)) continue;
    elements.runStatusFilter.add(
      new Option(
        state.replaceAll("_", " ").replace(/^./, (char) => char.toUpperCase()),
        state,
      ),
    );
    knownStates.add(state);
  }
  const rows = filteredRuns();
  const descriptors = rows.map(runRowDescriptor);
  const countLabel =
    rows.length === runRows.length
      ? `${runRows.length} Training Run${runRows.length === 1 ? "" : "s"}`
      : `${rows.length} of ${runRows.length} Training Runs`;
  const panel = elements.runsBody.closest(".panel") || elements.runsBody;
  commitPanelRefresh(
    panel,
    "runs-list",
    {
      countLabel,
      rows: descriptors.map(({ id, cells }) => ({
        id,
        cells: cells.map(({ html, className }) => ({ html, className })),
      })),
    },
    () => {
      setTextIfChanged(elements.runCount, countLabel);
      const existingRows = new Map(
        [
          ...elements.runsBody.querySelectorAll(
            ":scope > tr[data-run-row][data-run-id]",
          ),
        ].map((row) => [String(row.dataset.runId), row]),
      );
      const visibleIds = new Set(descriptors.map(({ id }) => id));
      if (
        activeDisclosure?.panel === elements.runDetail &&
        !visibleIds.has(String(activeRunDetailId))
      ) {
        closeActiveDisclosure({ restoreFocus: false });
      }
      const orderedRows = descriptors.map((descriptor) => {
        const row =
          existingRows.get(descriptor.id) || document.createElement("tr");
        row.dataset.runRow = "";
        row.dataset.runId = descriptor.id;
        patchTableRow(row, descriptor.cells);
        existingRows.delete(descriptor.id);
        return row;
      });
      existingRows.forEach((row) => row.remove());
      elements.runsBody
        .querySelectorAll(
          ":scope > tr:not([data-run-row]):not(.row-disclosure-companion)",
        )
        .forEach((row) => row.remove());
      if (!orderedRows.length) {
        const row = document.createElement("tr");
        row.className = "empty-row";
        patchTableRow(row, [
          {
            colSpan: 12,
            html: escapeHtml(
              runRows.length
                ? "No Training Runs match the filters."
                : "No Training Runs have been created.",
            ),
          },
        ]);
        elements.runsBody.append(row);
        return;
      }
      reconcileTableSequence(elements.runsBody, orderedRows);
      synchronizeDisclosureLaunchers(elements.runsBody);
    },
    { background },
  );
}

let runProgressRefreshPending = false;
let runProgressRefreshTimer = null;
function scheduleRunProgressRefresh() {
  window.clearTimeout(runProgressRefreshTimer);
  runProgressRefreshTimer = null;
  const hasActiveRun = runRows.some((run) =>
    RUN_DETAIL_ACTIVE_STATES.has(normalizedRunState(run.status || run.state)),
  );
  if (
    (!runProgressRefreshPending && !hasActiveRun) ||
    activeTab !== "runs" ||
    document.visibilityState !== "visible"
  )
    return;
  runProgressRefreshTimer = window.setTimeout(
    () => {
      runProgressRefreshTimer = null;
      if (activeTab === "runs" && document.visibilityState === "visible")
        loadRuns(true, {
          background: true,
          progressOnly: runProgressRefreshPending,
        }).catch(() => {});
    },
    runProgressRefreshPending ? 1500 : 5000,
  );
}

async function loadRuns(
  force = false,
  { background = false, progressOnly = false } = {},
) {
  const scope = "runs-list";
  if (loadedTabs.has("runs") && !force) {
    scheduleRunProgressRefresh();
    return;
  }
  if (!background) {
    elements.refreshRuns.disabled = true;
  }
  try {
    const payload = await api(
      progressOnly ? "/api/runs?refresh_progress=false" : "/api/runs",
    );
    runProgressRefreshPending = payload.progress_refresh_pending === true;
    runRows = listFrom(payload, ["runs"]);
    renderRuns({ background });
    loadedTabs.add("runs");
    clearNotificationScope(scope);
  } catch (error) {
    if (background) {
      showNotice(
        elements.runsError,
        `Training Run refresh failed: ${error.message}`,
        { scope },
      );
      throw error;
    }
    runRows = [];
    elements.runsBody.innerHTML = emptyRow(
      12,
      "Training Run data could not be loaded.",
    );
    elements.runCount.textContent = "Unavailable";
    showNotice(
      elements.runsError,
      `Training Run API unavailable: ${error.message}`,
      { scope },
    );
  } finally {
    if (!background) elements.refreshRuns.disabled = false;
    scheduleRunProgressRefresh();
  }
}

function linkedValue(kind, id, text, extra = {}) {
  return { kind, id, text: text || "Not recorded", ...extra };
}
function copyValue(text) {
  return { text: text || "Not recorded", copy: Boolean(text) };
}
function valueHtml(value) {
  if (!value || typeof value !== "object") return escapeHtml(value ?? "—");
  const text = escapeHtml(value.text ?? "—");
  if (value.kind && value.id)
    return `<button type="button" class="entity-link" data-entity-kind="${escapeHtml(value.kind)}" data-entity-id="${escapeHtml(value.id)}"${value.checkpoint ? ` data-checkpoint-id="${escapeHtml(value.checkpoint)}"` : ""}>${text}</button>`;
  if (value.copy)
    return `<span class="copy-value">${text}<button type="button" class="entity-link" data-copy-value="${escapeHtml(value.text)}" aria-label="Copy ${text}">Copy</button></span>`;
  return text;
}
function keyValueHtml(entries) {
  return entries
    .map(
      ([key, value]) =>
        `<div class="key-value"><span>${escapeHtml(key)}</span><strong>${valueHtml(value)}</strong></div>`,
    )
    .join("");
}
function executionContext(
  entity,
  attempt = entity.latest_attempt || latestEvaluationAttempt(entity) || {},
) {
  const resources = entity.resources || {};
  return {
    queue:
      attempt.partition_name ||
      resources.partition ||
      entity.partition ||
      "Not assigned",
    account: attempt.account || resources.account || "Not recorded",
    node: attempt.node_list || attempt.node || "Not allocated",
    gateway: attempt.gateway || "Not recorded",
    reason: queueReasonLabel(attempt),
    job:
      attempt.slurm_job_id || attempt.slurm_array_job_id || "Not accepted yet",
  };
}
function executionHtml(entity, attempt) {
  const context = executionContext(entity, attempt);
  return `<span class="job-id">${escapeHtml(context.job)}</span><span class="secondary">Queue: ${escapeHtml(context.queue)}</span><span class="secondary">Node: ${escapeHtml(context.node)}</span>`;
}
function executionEntries(entity, attempt) {
  const context = executionContext(entity, attempt);
  return [
    ["Slurm job", copyValue(context.job)],
    ["Queue / partition", context.queue],
    ["Slurm account", context.account],
    ["Compute node", context.node],
    ["SSH gateway", context.gateway],
    ...(context.reason !== "-" ? [["Scheduler reason", context.reason]] : []),
  ];
}
function trainingRunName(run) {
  return (
    run.name ||
    [
      run.experiment_name,
      `Training run ${run.run_number || run.training_run_number || ""}`.trim(),
    ]
      .filter(Boolean)
      .join(" · ")
  );
}
function evaluationName(evaluation) {
  return (
    evaluation.name ||
    `${evaluation.suite_label || evaluation.suite_name || evaluation.suite || "Evaluation"} · ${shortId(evaluation.id || evaluation.evaluation_id, 8)}`
  );
}
function trainingDataValue(run) {
  return (run.training_data || []).map((item) =>
    linkedValue(
      "dataset",
      item.resource_id,
      [
        item.name,
        item.format,
        item.episodes == null
          ? null
          : `${item.episodes} episode${item.episodes === 1 ? "" : "s"}`,
      ]
        .filter(Boolean)
        .join(" · "),
    ),
  );
}
function renderRunCheckpoints(run, selectedId) {
  const section = document.getElementById("run-checkpoints");
  const list = document.getElementById("run-checkpoint-list");
  const checkpoints = run.checkpoints || [];
  section.hidden = false;
  list.innerHTML = checkpoints.length
    ? `<table><thead><tr><th>Checkpoint</th><th>Step</th><th>State</th><th>Created</th><th>Storage path</th></tr></thead><tbody>${checkpoints.map((item) => `<tr data-checkpoint-row="${escapeHtml(item.id)}" tabindex="-1"><td>${escapeHtml(item.path?.split("/").pop() || item.checkpoint_type)}<span class="secondary">${valueHtml(copyValue(item.id))}</span></td><td>${escapeHtml(item.training_step ?? "—")}</td><td>${statusPill(item.status || "AVAILABLE")}</td><td>${escapeHtml(formatDate(item.created_at))}</td><td>${valueHtml(copyValue(item.path))}</td></tr>`).join("")}</tbody></table>`
    : '<p class="secondary">No checkpoint has been recorded.</p>';
  if (selectedId)
    list
      .querySelector(`[data-checkpoint-row="${CSS.escape(selectedId)}"]`)
      ?.focus();
}

document
  .getElementById("adapter-search")
  .addEventListener("input", renderAdapters);
document
  .getElementById("adapter-show-archived")
  .addEventListener("change", renderAdapters);
document
  .getElementById("experiment-search")
  .addEventListener("input", renderExperiments);
document.addEventListener("click", async (event) => {
  const copy = event.target.closest("[data-copy-value]");
  if (copy) {
    try {
      await navigator.clipboard.writeText(copy.dataset.copyValue);
      showToast("Copied.");
    } catch {
      showToast("Copy unavailable. Select and copy the displayed value.", true);
    }
    return;
  }
  const link = event.target.closest("[data-entity-kind]");
  if (!link || link.disabled) return;
  const { entityKind: kind, entityId: id, checkpointId } = link.dataset;
  link.disabled = true;
  try {
    closeActiveDisclosure({ restoreFocus: false });
    for (const dialog of document.querySelectorAll("dialog[open]"))
      SkynetDialog.close(dialog);
    if (kind === "run") {
      await activateTab("experiments", true, "runs");
      const launcher =
        elements.runsBody.querySelector(
          `[data-run-action="view"][data-id="${CSS.escape(id)}"]`,
        ) || document.getElementById("experiment-tab-runs");
      toggleDisclosure(elements.runDetail, `run:${id}`, launcher, {
        rowOwned: true,
      });
      await viewRun(id, launcher);
      if (checkpointId) {
        await new Promise(requestAnimationFrame);
        if (activeRunDetailId === id)
          document
            .querySelector(
              `[data-checkpoint-row="${CSS.escape(checkpointId)}"]`,
            )
            ?.focus();
      }
    } else if (kind === "experiment") {
      await activateTab("experiments", true, "submit");
      const launcher =
        elements.experimentsBody.querySelector(
          `[data-experiment-action="view"][data-id="${CSS.escape(id)}"]`,
        ) || document.getElementById("experiment-tab-submit");
      toggleDisclosure(
        elements.experimentDetail,
        `experiment:${id}`,
        launcher,
        { rowOwned: true },
      );
      await viewExperiment(id, launcher);
    } else if (kind === "adapter") {
      await activateTab("experiments", true, "adapters");
      await loadAdapters();
      const adapter = adapterRows.find(
        (item) => adapterId(item) === id || adapterManifest(item).slug === id,
      );
      if (!adapter)
        throw new Error(
          "This adapter is no longer registered. The training run retains its original configuration.",
        );
      await openAdapter(adapterId(adapter), false);
    } else if (kind === "suite") {
      await activateTab("evaluations", true, "suites");
      await loadEvaluationCatalog(true);
      openEvaluationSuite(id, link);
    } else if (kind === "dataset") {
      await activateTab("data", true, "registry");
      await window.openPreparedDataset(id);
    } else if (kind === "evaluation") {
      await activateTab("evaluations", true, "runs");
      await loadEvaluations(true);
      const launcher =
        elements.evaluationsBody.querySelector(
          `[data-evaluation-action="view"][data-id="${CSS.escape(id)}"]`,
        ) || document.getElementById("evaluation-tab-runs");
      toggleDisclosure(
        elements.evaluationDetail,
        `evaluation:${id}`,
        launcher,
        { rowOwned: true },
      );
      await viewEvaluation(id, launcher);
    } else if (kind === "prepared") {
      const job = await api(`/api/data/exports/${encodeURIComponent(id)}`);
      await activateTab("data", true, "registry");
      await window.openPreparedDataset((job.export || job).resource_id);
    } else if (kind === "recording") {
      await activateTab("data", true, "recording");
      const search = document.getElementById("simulation-recordings-search");
      search.value = id;
      search.dispatchEvent(new Event("input", { bubbles: true }));
    }
  } catch (error) {
    showToast(error.message, true);
  } finally {
    if (link.isConnected) link.disabled = false;
  }
});

function queueReasonLabel(attempt) {
  const reason = String(attempt?.slurm_reason || "").trim();
  const descriptions = {
    QOSGrpGRES: "Waiting for the group's GPU quota",
    Resources: "Waiting for matching resources to become available",
    Priority: "Waiting behind higher-priority jobs",
    Dependency: "Waiting for a prerequisite job",
  };
  if (descriptions[reason]) return `${descriptions[reason]} (${reason})`;
  if (reason) return reason;
  return String(attempt?.status || attempt?.state).toUpperCase() === "PENDING"
    ? "The scheduler has not reported a reason yet."
    : "-";
}

function attemptExitCodeLabel(attempt) {
  const status = String(attempt?.status || attempt?.state || "").toUpperCase();
  const terminal = [
    "SUCCEEDED",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMED_OUT",
    "PREEMPTED",
  ];
  return terminal.includes(status)
    ? runAttemptValue(attempt, "exit_code")
    : "Available when the attempt ends";
}

function runAttemptValue(attempt, ...keys) {
  for (const key of keys) {
    const value = attempt?.[key];
    if (value !== undefined && value !== null && value !== "") return value;
  }
  return null;
}

function runAttemptNestedValue(attempt, group, ...keys) {
  const nested = attempt?.[group];
  if (!nested || typeof nested !== "object" || Array.isArray(nested))
    return null;
  return runAttemptValue(nested, ...keys);
}

function runAttemptDurationLabel(attempt) {
  const persisted = runAttemptValue(
    attempt,
    "duration_seconds",
    "elapsed_seconds",
    "duration",
    "elapsed",
  );
  if (persisted !== null) {
    const seconds = Number(persisted);
    if (!Number.isFinite(seconds)) return String(persisted);
    const whole = Math.max(0, Math.round(seconds));
    const hours = Math.floor(whole / 3600);
    const minutes = Math.floor((whole % 3600) / 60);
    const remainder = whole % 60;
    return [
      hours ? `${hours}h` : "",
      minutes || hours ? `${minutes}m` : "",
      `${remainder}s`,
    ]
      .filter(Boolean)
      .join(" ");
  }
  const started = new Date(
    runAttemptValue(attempt, "started_at", "created_at") || "",
  ).getTime();
  const ended = new Date(
    runAttemptValue(attempt, "ended_at", "finished_at") || "",
  ).getTime();
  if (!Number.isFinite(started) || !Number.isFinite(ended) || ended < started)
    return "-";
  return runAttemptDurationLabel({
    duration_seconds: (ended - started) / 1000,
  });
}

function runAttemptResourceLabel(attempt) {
  const resources = attempt?.resources;
  if (typeof resources === "string" && resources.trim()) return resources;
  const gpuCount =
    runAttemptNestedValue(
      attempt,
      "resources",
      "gpus_per_node",
      "gpu_count",
      "gpus",
    ) ?? runAttemptValue(attempt, "gpus_per_node", "gpu_count", "gpus");
  const gpuType =
    runAttemptNestedValue(attempt, "resources", "gpu_type", "gres") ??
    runAttemptValue(attempt, "gpu_type", "gres");
  const cpuCount =
    runAttemptNestedValue(
      attempt,
      "resources",
      "cpus_per_task",
      "cpu_count",
      "cpus",
    ) ?? runAttemptValue(attempt, "cpus_per_task", "cpu_count", "cpus");
  const memoryGb =
    runAttemptNestedValue(attempt, "resources", "memory_gb", "mem_gb") ??
    runAttemptValue(attempt, "memory_gb", "mem_gb");
  const nodeCount =
    runAttemptNestedValue(attempt, "resources", "nodes", "node_count") ??
    runAttemptValue(attempt, "nodes", "node_count");
  const parts = [];
  if (gpuCount !== null) parts.push(`${gpuCount}x ${gpuType || "GPU"}`);
  else if (gpuType) parts.push(String(gpuType));
  if (cpuCount !== null) parts.push(`${cpuCount} CPU`);
  if (memoryGb !== null) parts.push(`${memoryGb} GB`);
  if (nodeCount !== null)
    parts.push(`${nodeCount} node${Number(nodeCount) === 1 ? "" : "s"}`);
  return parts.join(" / ") || "-";
}

function runAttemptPath(attempt, ...keys) {
  const direct = runAttemptValue(attempt, ...keys);
  if (direct !== null) return direct;
  const logs = attempt?.logs;
  return logs && typeof logs === "object" && !Array.isArray(logs)
    ? runAttemptValue(logs, ...keys)
    : null;
}

const commonHyperparameterSourceLabels = {
  user_override: "user override",
  adapter_default: "adapter default",
  repository_config: "repository config",
  legacy_derived: "legacy-derived",
  unresolved: "unresolved",
  not_applicable: "not applicable",
};

function commonHyperparameterEntries(payload) {
  if (!payload || typeof payload !== "object") return {};
  const entries = payload.entries;
  if (Array.isArray(entries)) {
    return Object.fromEntries(
      entries.flatMap((entry) => {
        if (!entry || typeof entry !== "object") return [];
        const key = entry.key ?? entry.name ?? entry.field ?? entry.path;
        return key ? [[String(key), entry]] : [];
      }),
    );
  }
  if (entries && typeof entries === "object") return entries;
  return payload;
}

function commonHyperparameterSourceLabel(source, legacyValue) {
  if (source === undefined || source === null || source === "") {
    return legacyValue ? "legacy-derived" : "unresolved";
  }
  const sourceValue =
    source && typeof source === "object"
      ? (source.type ?? source.kind ?? source.name ?? JSON.stringify(source))
      : source;
  const normalized = String(sourceValue)
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, "_");
  const aliases = {
    explicit: "user_override",
    explicit_spec: "user_override",
    user: "user_override",
    requested: "user_override",
    manifest_default: "adapter_default",
    adapter_manifest_default: "adapter_default",
    adapter: "adapter_default",
    repository: "repository_config",
    repository_inspection: "repository_config",
    repo_config: "repository_config",
    legacy: "legacy_derived",
  };
  const canonical = aliases[normalized] || normalized;
  return (
    commonHyperparameterSourceLabels[canonical] ||
    `unknown source: ${String(sourceValue)}`
  );
}

function commonHyperparameter(payload, provenancePayload, ...keys) {
  const entries = commonHyperparameterEntries(payload);
  const provenanceEntries = commonHyperparameterEntries(provenancePayload);
  for (const key of keys) {
    if (!Object.prototype.hasOwnProperty.call(entries, key)) continue;
    const raw = entries[key];
    const provenance = provenanceEntries[key];
    const provenanceRecord =
      provenance && typeof provenance === "object" && !Array.isArray(provenance)
        ? provenance
        : {};
    const provenanceSource =
      provenanceRecord.status === "not_applicable"
        ? "not_applicable"
        : provenanceRecord.status === "unresolved"
          ? "unresolved"
          : provenanceRecord.source;
    const structured =
      raw &&
      typeof raw === "object" &&
      !Array.isArray(raw) &&
      [
        "value",
        "effective_value",
        "resolved_value",
        "source",
        "provenance",
        "origin",
      ].some((field) => Object.prototype.hasOwnProperty.call(raw, field));
    if (!structured) {
      return {
        value: raw,
        source: commonHyperparameterSourceLabel(
          provenanceSource,
          raw !== undefined && raw !== null && raw !== "",
        ),
      };
    }
    const value = Object.prototype.hasOwnProperty.call(raw, "value")
      ? raw.value
      : Object.prototype.hasOwnProperty.call(raw, "effective_value")
        ? raw.effective_value
        : raw.resolved_value;
    return {
      value,
      source: commonHyperparameterSourceLabel(
        raw.source ?? raw.provenance ?? raw.origin ?? provenanceSource,
        false,
      ),
    };
  }
  return { value: null, source: "unresolved" };
}

function commonHyperparameterLabel(entry) {
  let value = entry.value;
  if (value === undefined || value === null || value === "") value = "-";
  else if (typeof value === "object") value = JSON.stringify(value);
  return `${String(value)} (${entry.source})`;
}

const pendingCommonHyperparameterResolutions = new Set();

function commonHyperparameterResolutionKey(runId, attemptId) {
  return `${runId}\u001f${attemptId}`;
}

function commonHyperparameterResolutionLabel(resolution) {
  const status = String(
    typeof resolution === "string" ? resolution : resolution?.status || "",
  )
    .trim()
    .toLowerCase();
  if (status === "snapshotted") return "";
  const labels = {
    enriched_receipt:
      "Actual values were recovered from stored repository evidence.",
    available_from_cache:
      "Actual values are available from the repository-inspection cache.",
    requires_explicit_fetch:
      "Actual values require network inspection of the pinned repository.",
    unavailable: "Actual values cannot be resolved from the stored evidence.",
  };
  return (
    labels[status] ||
    `Actual-value resolution status is unknown: ${status || "missing"}.`
  );
}

function activeRunAttemptMatches(runId, attemptId) {
  return (
    String(activeRunDetailId || "") === runId &&
    String(activeRunAttemptDisclosure?.runId || "") === runId &&
    String(activeRunAttemptDisclosure?.attemptId || "") === attemptId
  );
}

function setCommonHyperparameterResolutionError(key, message) {
  const controls = elements.runAttemptHyperparameters.querySelector(
    "[data-common-hyperparameter-resolution]",
  );
  if (!controls || controls.dataset.resolutionKey !== key) return;
  const status = controls.querySelector(
    "[data-common-hyperparameter-resolution-status]",
  );
  const button = controls.querySelector(
    "[data-resolve-common-hyperparameters]",
  );
  if (status) {
    status.textContent = `Resolution failed: ${message}`;
    status.setAttribute("role", "alert");
  }
  if (button) {
    button.disabled = false;
    button.removeAttribute("aria-busy");
    button.textContent = "Resolve actual values";
  }
}

async function resolveRunAttemptCommonHyperparameters(record, action) {
  const disclosure = activeRunAttemptDisclosure;
  const runId = String(disclosure?.runId || activeRunDetailId || "");
  const attemptId = String(
    disclosure?.attemptId ||
      record.attemptId ||
      runAttemptValue(record.attempt, "id", "attempt_id") ||
      "",
  );
  const scope = `run-values:${runId}:${attemptId}`;
  if (!runId || !attemptId || !activeRunAttemptMatches(runId, attemptId)) {
    showNotice(
      elements.runsError,
      "Cannot resolve values because the selected run attempt is no longer open.",
      { scope },
    );
    return;
  }
  if (
    !action ||
    typeof action !== "object" ||
    String(action.method).toUpperCase() !== "POST" ||
    !action.path
  ) {
    const message =
      "The server returned an invalid common-hyperparameter resolution action.";
    showNotice(elements.runsError, message, { scope });
    showToast(message, true, { scope });
    return;
  }
  const body =
    action.body && typeof action.body === "object" ? action.body : {};
  if (body.allow_network === true) {
    const gateway = String(body.gateway || "").trim();
    const confirmed = await askUserDialog(
      `Resolving actual values requires network inspection of the pinned repository${gateway ? ` through ${gateway}` : ""}. Continue?`,
    );
    if (!confirmed) return;
  }

  const key = commonHyperparameterResolutionKey(runId, attemptId);
  if (pendingCommonHyperparameterResolutions.has(key)) return;
  pendingCommonHyperparameterResolutions.add(key);
  const scrollPosition = { left: window.scrollX, top: window.scrollY };
  const controls = elements.runAttemptHyperparameters.querySelector(
    "[data-common-hyperparameter-resolution]",
  );
  const status = controls?.querySelector(
    "[data-common-hyperparameter-resolution-status]",
  );
  const button = controls?.querySelector(
    "[data-resolve-common-hyperparameters]",
  );
  if (status) {
    status.textContent =
      body.allow_network === true
        ? "Inspecting the pinned repository over the network..."
        : "Resolving actual values from cached evidence...";
  }
  if (button) {
    button.disabled = true;
    button.setAttribute("aria-busy", "true");
    button.textContent = "Resolving...";
  }

  try {
    await api(String(action.path), {
      method: String(action.method),
      body: JSON.stringify(body),
    });
    if (!activeRunAttemptMatches(runId, attemptId)) return;
    const detailPayload = await api(`/api/runs/${encodeURIComponent(runId)}`);
    if (!activeRunAttemptMatches(runId, attemptId)) return;
    pendingCommonHyperparameterResolutions.delete(key);
    const detail = renderRunDetail(detailPayload, runId, {
      preserveAttempt: true,
    });
    if (!activeRunAttemptMatches(runId, attemptId)) {
      throw new Error(
        "The resolved attempt could not be restored after refresh.",
      );
    }
    startRunDetailPolling(runId, detail.state);
    window.scrollTo(scrollPosition.left, scrollPosition.top);
    showToast("Actual common hyperparameter values resolved.", false, {
      scope,
    });
  } catch (error) {
    pendingCommonHyperparameterResolutions.delete(key);
    const message = error instanceof Error ? error.message : String(error);
    setCommonHyperparameterResolutionError(key, message);
    showNotice(
      elements.runsError,
      `Common hyperparameter resolution failed: ${message}`,
      { scope },
    );
    showToast(`Common hyperparameter resolution failed: ${message}`, true, {
      scope,
    });
  } finally {
    pendingCommonHyperparameterResolutions.delete(key);
  }
}

function renderCommonHyperparameterResolution(record) {
  const resolution = record.attempt?.common_hyperparameters_resolution;
  const runId = String(
    activeRunAttemptDisclosure?.runId || activeRunDetailId || "",
  );
  const attemptId = String(
    activeRunAttemptDisclosure?.attemptId ||
      record.attemptId ||
      runAttemptValue(record.attempt, "id", "attempt_id") ||
      "",
  );
  const key = commonHyperparameterResolutionKey(runId, attemptId);
  const controls = document.createElement("div");
  controls.className = "section-tools";
  controls.style.gridColumn = "1 / -1";
  controls.style.justifyContent = "space-between";
  controls.dataset.commonHyperparameterResolution = "true";
  controls.dataset.resolutionKey = key;

  const status = document.createElement("p");
  status.className = "quiet-label";
  status.dataset.commonHyperparameterResolutionStatus = "true";
  status.setAttribute("aria-live", "polite");
  status.textContent = commonHyperparameterResolutionLabel(resolution);
  controls.append(status);

  const action = resolution?.action;
  if (action && typeof action === "object") {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "button button-outline";
    button.dataset.resolveCommonHyperparameters = "true";
    const pending = pendingCommonHyperparameterResolutions.has(key);
    button.disabled = pending;
    button.textContent = pending ? "Resolving..." : "Resolve actual values";
    if (pending) button.setAttribute("aria-busy", "true");
    button.addEventListener("click", () =>
      resolveRunAttemptCommonHyperparameters(record, action),
    );
    controls.append(button);
  }
  elements.runAttemptHyperparameters.append(controls);
}

function renderRunAttemptMetadata(record) {
  const attempt = record.attempt;
  renderModelIO(
    document.getElementById("run-attempt-model-io"),
    attempt.model_io,
    attempt.model_io ? "" : "Sizes were not recorded.",
  );
  const error = attemptFailureDetail(attempt);
  const hyperparameters =
    attempt.common_hyperparameters &&
    typeof attempt.common_hyperparameters === "object"
      ? attempt.common_hyperparameters
      : {};
  const hyperparameterProvenance =
    attempt.common_hyperparameter_provenance &&
    typeof attempt.common_hyperparameter_provenance === "object"
      ? attempt.common_hyperparameter_provenance
      : {};
  elements.runAttemptDetailTitle.textContent = `Attempt ${record.attemptNumber}`;
  elements.runAttemptDetailMeta.innerHTML = keyValueHtml([
    ["Attempt", record.attemptNumber],
    ["Adapter", record.adapterLabel || trainingAdapterLabel({}, attempt)],
    ["Attempt ID", runAttemptValue(attempt, "id", "attempt_id")],
    ["State", jobStatusLabel(attempt)],
    ["Scheduler reason", queueReasonLabel(attempt)],
    ["Slurm job", runAttemptValue(attempt, "slurm_job_id", "job_id")],
    ["SSH gateway", runAttemptValue(attempt, "gateway")],
    ["Compute node", runAttemptValue(attempt, "node", "node_list", "nodelist")],
    [
      "Account",
      runAttemptValue(attempt, "account") ??
        runAttemptNestedValue(attempt, "resources", "account"),
    ],
    [
      "Partition",
      runAttemptValue(attempt, "partition") ??
        runAttemptNestedValue(attempt, "resources", "partition"),
    ],
    [
      "QoS",
      runAttemptValue(attempt, "qos") ??
        runAttemptNestedValue(attempt, "resources", "qos"),
    ],
    ["Resources", runAttemptResourceLabel(attempt)],
    ["Created", formatDate(runAttemptValue(attempt, "created_at"))],
    ["Submitted", formatDate(runAttemptValue(attempt, "submitted_at"))],
    ["Started", formatDate(runAttemptValue(attempt, "started_at"))],
    ["Ended", formatDate(runAttemptValue(attempt, "ended_at", "finished_at"))],
    ["Duration", runAttemptDurationLabel(attempt)],
    ["Exit code", attemptExitCodeLabel(attempt)],
    ["Signal", runAttemptValue(attempt, "signal", "term_signal")],
    [
      "Checkpoint",
      runAttemptPath(attempt, "checkpoint_path", "resume_checkpoint_path"),
    ],
    [
      "Working directory",
      runAttemptPath(attempt, "working_directory", "workdir", "chdir"),
    ],
    ["Sbatch file", runAttemptPath(attempt, "sbatch_path", "script_path")],
    ["Stdout file", runAttemptPath(attempt, "stdout_path", "output_path")],
    ["Stderr file", runAttemptPath(attempt, "stderr_path", "error_path")],
    ["Failure detail", error],
  ]);
  renderTrackingDetail(elements.runAttemptDetailTracking, {
    tracking_links: record.trackingLinks || [],
  });
  const commonHyperparameters = [
    [
      "Learning rate",
      commonHyperparameter(
        hyperparameters,
        hyperparameterProvenance,
        "learning_rate",
        "train.learning_rate",
      ),
    ],
    [
      "Batch size",
      commonHyperparameter(
        hyperparameters,
        hyperparameterProvenance,
        "batch_size",
        "batch_value",
        "train.batch.value",
      ),
    ],
    [
      "Batch semantics",
      commonHyperparameter(
        hyperparameters,
        hyperparameterProvenance,
        "batch_semantics",
        "declared_semantics",
        "train.batch.declared_semantics",
      ),
    ],
    [
      "Gradient accumulation",
      commonHyperparameter(
        hyperparameters,
        hyperparameterProvenance,
        "gradient_accumulation",
        "gradient_accumulation_steps",
        "train.batch.gradient_accumulation_steps",
      ),
    ],
    [
      "Data workers",
      commonHyperparameter(
        hyperparameters,
        hyperparameterProvenance,
        "num_workers",
        "num_workers_per_rank",
        "train.num_workers_per_rank",
      ),
    ],
    [
      "Precision",
      commonHyperparameter(
        hyperparameters,
        hyperparameterProvenance,
        "precision",
        "train.precision",
      ),
    ],
    [
      "Max steps / iterations",
      commonHyperparameter(
        hyperparameters,
        hyperparameterProvenance,
        "max_steps",
        "max_iterations",
        "train.max_steps",
      ),
    ],
    ...(["max_epochs", "epochs", "train.max_epochs"].some(
      (key) =>
        Object.prototype.hasOwnProperty.call(
          commonHyperparameterEntries(hyperparameters),
          key,
        ) ||
        Object.prototype.hasOwnProperty.call(
          commonHyperparameterEntries(hyperparameterProvenance),
          key,
        ),
    )
      ? [
          [
            "Max epochs",
            commonHyperparameter(
              hyperparameters,
              hyperparameterProvenance,
              "max_epochs",
              "epochs",
              "train.max_epochs",
            ),
          ],
        ]
      : []),
  ].filter(
    ([, entry]) =>
      String(entry.source || "")
        .trim()
        .toLowerCase()
        .replace(/[\s-]+/g, "_") !== "not_applicable",
  );
  const commonHyperparameterSection =
    elements.runAttemptHyperparameters.closest(".attempt-hyperparameters");
  if (commonHyperparameterSection)
    commonHyperparameterSection.hidden = commonHyperparameters.length === 0;
  elements.runAttemptHyperparameters.innerHTML = commonHyperparameters.length
    ? keyValueHtml(
        commonHyperparameters.map(([label, entry]) => [
          label,
          commonHyperparameterLabel(entry),
        ]),
      )
    : "";
  const adapterSettings = Object.entries(attempt.adapter_settings || {});
  const fieldLabels = new Map(
    (record.adapterFields || []).map((field) => [field.path, field.label]),
  );
  elements.runAttemptAdapterSection.hidden = adapterSettings.length === 0;
  elements.runAttemptAdapterSettings.innerHTML = keyValueHtml(
    adapterSettings.map(([path, value]) => [
      fieldLabels.get(path) ||
        path.replace(/^native\.(config|overrides)\./, "").replaceAll("_", " "),
      displayCommonHyperparameterDefault(value),
    ]),
  );
  if (commonHyperparameters.length)
    renderCommonHyperparameterResolution(record);
}

function setRunAttemptLauncherActive(launcher, active) {
  if (!launcher) return;
  launcher.classList.add("disclosure-launcher");
  launcher.classList.toggle("is-active", active);
  launcher.setAttribute("aria-controls", "run-attempt-detail-dialog");
  launcher.setAttribute("aria-haspopup", "dialog");
  launcher.setAttribute("aria-expanded", active ? "true" : "false");
  launcher.textContent = "Detail";
  const row = launcher.closest("[data-attempt-row]");
  row?.classList.toggle("is-active", active);
}

const RUN_DETAIL_ACTIVE_STATES = new Set([
  "SUBMITTING",
  "SUBMITTED",
  "PENDING",
  "RUNNING",
  "REQUEUED",
  "RETRY_PENDING",
  "CANCELLING",
]);
const RUN_DETAIL_POLL_INTERVAL_MS = 5000;
const RUN_DETAIL_FINAL_POLL_DELAY_MS = 2000;
let runDetailLatestAttemptId = null;
let runDetailFollowLatest = true;
let runDetailAttemptDismissed = false;
let runDetailPollGeneration = 0;
let runDetailPollTimer = null;
let runDetailPollInFlight = null;
let runDetailPollRunId = null;

function stopRunDetailPolling({ clearStatus = true } = {}) {
  runDetailPollGeneration += 1;
  window.clearTimeout(runDetailPollTimer);
  runDetailPollTimer = null;
  runDetailPollInFlight = null;
  runDetailPollRunId = null;
}

function closeRunAttemptDisclosure({
  restoreFocus = true,
  markDismissed = false,
} = {}) {
  const closing = activeRunAttemptDisclosure;
  activeRunAttemptDisclosure = null;
  runAttemptDisclosureSequence += 1;
  if (markDismissed && closing) runDetailAttemptDismissed = true;
  if (closing) setRunAttemptLauncherActive(closing.launcher, false);
  if (elements.runAttemptDetail && !elements.runAttemptDetail.hidden) {
    hideRevealedPanel(elements.runAttemptDetail, closing?.launcher || null, {
      restoreFocus,
    });
  } else if (restoreFocus && closing?.launcher?.isConnected) {
    closing.launcher.focus({ preventScroll: true });
  }
  elements.runAttemptDetailTitle.textContent = "Attempt";
  elements.runAttemptDetailMeta.innerHTML = "";
  renderTrackingDetail(elements.runAttemptDetailTracking, {});
  elements.runAttemptHyperparameters.innerHTML = "";
  elements.runAttemptStdoutStatus.textContent = "";
  elements.runAttemptStderrStatus.textContent = "";
  updateLogView(
    elements.runAttemptStdoutLog,
    "Select an attempt to load stdout.",
  );
  updateLogView(
    elements.runAttemptStderrLog,
    "Select an attempt to load stderr.",
  );
  return Boolean(closing);
}

function resetRunAttemptContext(runId = null) {
  stopRunDetailPolling();
  closeRunAttemptDisclosure({ restoreFocus: false });
  runDetailAttemptRecords.clear();
  activeRunDetailId =
    runId === null || runId === undefined ? null : String(runId);
  runDetailLatestAttemptId = null;
  runDetailFollowLatest = true;
  runDetailAttemptDismissed = false;
}

function runAttemptDisclosureIsCurrent(token, runId, attemptKey) {
  return Boolean(
    activeRunAttemptDisclosure &&
      activeRunAttemptDisclosure.token === token &&
      activeRunAttemptDisclosure.runId === String(runId) &&
      activeRunAttemptDisclosure.attemptKey === attemptKey,
  );
}

function runAttemptRecordById(attemptId) {
  return (
    [...runDetailAttemptRecords.values()].find(
      (record) => String(record.attemptId) === String(attemptId),
    ) || null
  );
}

const logViewUpdateGenerations = new WeakMap();

function beginLogViewUpdate(element) {
  const generation = (logViewUpdateGenerations.get(element) || 0) + 1;
  logViewUpdateGenerations.set(element, generation);
  return generation;
}

function logViewUpdateIsCurrent(element, generation) {
  return logViewUpdateGenerations.get(element) === generation;
}

function logStreamStatus(content, loadedLabel = "loaded") {
  if (!content) return "empty";
  if (content.startsWith("Waiting for Slurm to create ")) return "waiting";
  if (/^No (stdout|stderr) log file was produced/.test(content))
    return "not produced";
  return loadedLabel;
}

function logViewViewport(element) {
  return (
    element.closest(
      ".log-view, .log-output, .preview-code, .code-block, [data-log-viewport]",
    ) || element
  );
}

function updateLogView(element, content, { generation = null } = {}) {
  const writeGeneration = generation ?? beginLogViewUpdate(element);
  if (!logViewUpdateIsCurrent(element, writeGeneration)) return false;

  const next = String(content);
  if (element.textContent === next) return true;

  const viewport = logViewViewport(element);
  const top = viewport.scrollTop;
  const left = viewport.scrollLeft;
  const bottomDistance = Math.max(
    0,
    viewport.scrollHeight - viewport.clientHeight - top,
  );
  const followBottom = bottomDistance <= 24;

  element.textContent = next;
  if (!logViewUpdateIsCurrent(element, writeGeneration) || !element.isConnected)
    return false;
  const maxTop = Math.max(0, viewport.scrollHeight - viewport.clientHeight);
  const maxLeft = Math.max(0, viewport.scrollWidth - viewport.clientWidth);
  viewport.scrollTop = followBottom
    ? maxTop
    : Math.min(
        maxTop,
        top <= maxTop ? top : Math.max(0, maxTop - bottomDistance),
      );
  viewport.scrollLeft = Math.min(maxLeft, left);
  return true;
}

async function loadRunAttemptLogs(
  record,
  token,
  { showLoading = true, retainOnError = false, pollGeneration = null } = {},
) {
  const runId = activeRunDetailId;
  const attemptKey = record.attemptKey;
  const stdoutGeneration = beginLogViewUpdate(elements.runAttemptStdoutLog);
  const stderrGeneration = beginLogViewUpdate(elements.runAttemptStderrLog);
  if (showLoading) {
    setTextIfChanged(elements.runAttemptStdoutStatus, "loading");
    setTextIfChanged(elements.runAttemptStderrStatus, "loading");
    updateLogView(elements.runAttemptStdoutLog, "Loading stdout...", {
      generation: stdoutGeneration,
    });
    updateLogView(elements.runAttemptStderrLog, "Loading stderr...", {
      generation: stderrGeneration,
    });
  }
  const base = `/api/runs/${encodeURIComponent(runId)}/attempts/${encodeURIComponent(record.attemptId)}/logs`;
  const [stdoutResult, stderrResult] = await Promise.allSettled([
    api(`${base}?stream=stdout`),
    api(`${base}?stream=stderr`),
  ]);
  if (!runAttemptDisclosureIsCurrent(token, runId, attemptKey)) return false;
  if (pollGeneration !== null && pollGeneration !== runDetailPollGeneration)
    return false;
  if (stdoutResult.status === "fulfilled") {
    const content = logContent(stdoutResult.value);
    if (
      updateLogView(
        elements.runAttemptStdoutLog,
        content || "No stdout output.",
        { generation: stdoutGeneration },
      )
    ) {
      setTextIfChanged(
        elements.runAttemptStdoutStatus,
        logStreamStatus(content, "stdout loaded"),
      );
    }
  } else {
    if (!retainOnError) {
      if (
        updateLogView(
          elements.runAttemptStdoutLog,
          `Unable to load stdout: ${stdoutResult.reason.message}`,
          { generation: stdoutGeneration },
        )
      ) {
        setTextIfChanged(
          elements.runAttemptStdoutStatus,
          "refresh failed; retrying",
        );
      }
    } else if (
      logViewUpdateIsCurrent(elements.runAttemptStdoutLog, stdoutGeneration)
    ) {
      setTextIfChanged(
        elements.runAttemptStdoutStatus,
        "refresh failed; retrying",
      );
    }
  }
  if (stderrResult.status === "fulfilled") {
    const content = logContent(stderrResult.value);
    if (
      updateLogView(
        elements.runAttemptStderrLog,
        content || "No stderr output.",
        { generation: stderrGeneration },
      )
    ) {
      setTextIfChanged(
        elements.runAttemptStderrStatus,
        logStreamStatus(content, "stderr loaded"),
      );
    }
  } else {
    if (!retainOnError) {
      if (
        updateLogView(
          elements.runAttemptStderrLog,
          `Unable to load stderr: ${stderrResult.reason.message}`,
          { generation: stderrGeneration },
        )
      ) {
        setTextIfChanged(
          elements.runAttemptStderrStatus,
          "refresh failed; retrying",
        );
      }
    } else if (
      logViewUpdateIsCurrent(elements.runAttemptStderrLog, stderrGeneration)
    ) {
      setTextIfChanged(
        elements.runAttemptStderrStatus,
        "refresh failed; retrying",
      );
    }
  }
  return (
    stdoutResult.status === "fulfilled" && stderrResult.status === "fulfilled"
  );
}

async function openRunAttemptDetail(
  attemptKey,
  launcher,
  {
    focus = true,
    loadLogs = true,
    showLoading = true,
    followLatest = null,
  } = {},
) {
  const record = runDetailAttemptRecords.get(attemptKey);
  if (!record || !activeRunDetailId || !launcher) return;
  closeRunAttemptDisclosure({ restoreFocus: false });
  const token = ++runAttemptDisclosureSequence;
  const runId = activeRunDetailId;
  activeRunAttemptDisclosure = {
    token,
    runId,
    attemptKey,
    attemptId: String(record.attemptId),
    launcher,
  };
  runDetailAttemptDismissed = false;
  runDetailFollowLatest =
    followLatest === null
      ? String(record.attemptId) === String(runDetailLatestAttemptId)
      : Boolean(followLatest);
  setRunAttemptLauncherActive(launcher, true);
  renderRunAttemptMetadata(record);
  if (focus) {
    revealPanel(elements.runAttemptDetail, {
      focusTarget: elements.runAttemptDetailTitle,
      launcher,
    });
  } else {
    elements.runAttemptDetail.hidden = false;
    SkynetDialog.open(elements.runAttemptDetail.closest("dialog"), {
      launcher,
    });
  }
  if (loadLogs) await loadRunAttemptLogs(record, token, { showLoading });
}

async function toggleRunAttemptDetail(attemptKey, launcher) {
  if (
    activeRunAttemptDisclosure &&
    activeRunAttemptDisclosure.runId === activeRunDetailId &&
    activeRunAttemptDisclosure.attemptKey === attemptKey
  ) {
    closeRunAttemptDisclosure({ markDismissed: true });
    return;
  }
  await openRunAttemptDetail(attemptKey, launcher);
}

function savedTrainingDataLabel(run) {
  if (run.training_data?.length)
    return run.training_data
      .map((item) =>
        [
          item.name,
          item.format,
          item.episodes != null
            ? `${item.episodes} episode${item.episodes === 1 ? "" : "s"}`
            : null,
        ]
          .filter(Boolean)
          .join(" · "),
      )
      .join("; ");
  const spec =
    run.resolved_spec_json || run.spec_snapshot_json || run.spec_json || {};
  const inputs = spec.data?.bundle?.assignments || [];
  return (
    inputs
      .filter((a) => a.role === "training_data")
      .map((a) => {
        const version = a.version || {},
          metadata = version.metadata || {};
        const name =
          metadata.display_name || spec.data.bundle.name || a.resource?.name;
        const episodes = Array.isArray(metadata.episodes)
          ? metadata.episodes.length
          : (metadata.num_episodes ?? metadata.episodes);
        return [
          name,
          version.format,
          episodes != null
            ? `${episodes} episode${Number(episodes) === 1 ? "" : "s"}`
            : null,
        ]
          .filter(Boolean)
          .join(" · ");
      })
      .join("; ") || "Not recorded"
  );
}

function renderRunDetailContent(payload, id, { preserveAttempt = false } = {}) {
  const run = entityFrom(payload, "run");
  const attempts = runAttemptRecords({
    ...run,
    attempts: Array.isArray(run.attempts)
      ? run.attempts
      : listFrom(payload, ["attempts"]),
  });
  const selectedAttemptId = preserveAttempt
    ? activeRunAttemptDisclosure?.attemptId
    : null;
  const state = run.status || run.state || "unknown";
  elements.runDetail.dataset.runState = String(state).toUpperCase();
  setTextIfChanged(
    elements.runDetailTitle,
    run.name ||
      (run.run_number ? `Training Run ${run.run_number}` : run.id || id),
  );
  setHtmlIfChanged(
    elements.runDetailMeta,
    keyValueHtml([
      ["Training run ID", copyValue(run.id)],
      [
        "Experiment",
        linkedValue("experiment", run.experiment_id, run.experiment_name),
      ],
      ["Experiment revision", run.experiment_revision_number],
      [
        "Adapter",
        linkedValue(
          "adapter",
          run.adapter_id ||
            run.resolved_spec_json?.adapter?.id ||
            run.adapter_name,
          trainingAdapterLabel(run),
        ),
      ],
      ...(trainingDataValue(run).length
        ? trainingDataValue(run).map((value) => ["Training data", value])
        : [["Training data", savedTrainingDataLabel(run)]]),
      ["Runtime", copyValue(run.runtime_profile)],
      ["Code commit", copyValue(run.source_commit)],
      ...executionEntries(run),
      ["Output storage path", copyValue(run.run_directory)],
      ["Variant", run.variant_name || run.variant_id],
      ["Training Run number", run.run_number],
      [
        "Restarted from run",
        run.restarted_from_run_id
          ? linkedValue(
              "run",
              run.restarted_from_run_id,
              "View source training run",
            )
          : "—",
      ],
      ["Training Run state", jobStatusLabel(run)],
      [
        "Progress",
        run.status_detail || progressSummaryLabel(run.progress_summary),
      ],
      ...(String(state).toUpperCase() === "RUNNING"
        ? [
            [
              "ETA",
              run.status_detail
                ? "—"
                : etaPresentation(run.progress_summary).detail,
            ],
          ]
        : []),
      ["Checkpoint", run.checkpoint_path || run.latest_checkpoint],
      ["Started", formatDate(run.latest_attempt?.started_at)],
      [
        "Elapsed",
        run.progress_summary?.elapsed_seconds == null
          ? "—"
          : compactEtaDuration(run.progress_summary.elapsed_seconds),
      ],
      ["Created", formatDate(run.created_at)],
      ["Updated", formatDate(run.updated_at)],
    ]),
  );
  renderRunCheckpoints(run);
  renderTrackingDetail(elements.runDetailTracking, run);
  const manualActions =
    run.manual_actions && typeof run.manual_actions === "object"
      ? run.manual_actions
      : {};
  const trackingActions =
    run.tracking_actions && typeof run.tracking_actions === "object"
      ? Object.values(run.tracking_actions)
      : [];
  const runActionButton = (action, label, className) => {
    const metadata = manualActions[action];
    const recovering =
      action === "recover_submission" && submissionRecoveryRequests.has(id);
    if (recovering) label = "Recovering submission…";
    const resuming = action === "resume" && runResumeRequests.has(id);
    if (resuming) label = "Submitting new attempt…";
    const enabled = metadata?.enabled === true && !recovering && !resuming;
    const reason =
      typeof metadata?.reason === "string" && metadata.reason.trim()
        ? metadata.reason.trim()
        : enabled
          ? ""
          : "This action is not available for this run.";
    const accessibleLabel = reason ? `${label}: ${reason}` : label;
    return `<button class="${className}" type="button" data-run-action="${action}" data-run-action-enabled="${enabled}" data-run-action-reason="${escapeHtml(reason)}" data-id="${escapeHtml(id)}" title="${escapeHtml(reason)}" aria-label="${escapeHtml(accessibleLabel)}"${enabled ? "" : " disabled"}>${label}</button>`;
  };
  const trackingActionButtons = trackingActions
    .filter((action) => action?.enabled === true)
    .map((action) => {
      const provider = String(action.provider || "").toLowerCase();
      const label =
        action.label || `Connect ${trackingProviderLabel(provider)}`;
      return `<button class="button button-outline" type="button" data-run-action="attach-tracking" data-run-action-enabled="true" data-tracking-provider="${escapeHtml(provider)}" data-tracking-action-path="${escapeHtml(action.path || "")}" data-id="${escapeHtml(id)}">${escapeHtml(label)}</button>`;
    })
    .join("");
  setHtmlIfChanged(
    elements.runDetailActions,
    `
      ${manualActions.recover_submission?.enabled ? runActionButton("recover_submission", "Recover unconfirmed submission", "button button-accent") : ""}
      ${runEvaluationActionButton(run, id)}
      ${trackingActionButtons}
      ${runActionButton("resume", /no usable|from scratch|from the beginning|initial pinned/i.test(manualActions.resume?.reason || "") ? "Restart training from beginning / new attempt" : "Resume Training Run / new attempt", "button button-outline")}
      ${runActionButton("rerun", "Start new Training Run from pinned variant", "button button-outline")}
      ${cancellationActionButton("run", id, manualActions)}`,
  );
  const existingAttemptRows = new Map(
    [
      ...elements.attemptsBody.querySelectorAll(
        ":scope > tr[data-attempt-row][data-attempt-key]",
      ),
    ].map((row) => [String(row.dataset.attemptKey), row]),
  );
  const retainedAttemptKeys = new Set();
  const orderedAttemptRows = [];
  runDetailAttemptRecords.clear();
  attempts.forEach((attempt, index) => {
    const attemptNumber =
      attempt.attempt_number ?? attempt.number ?? attempt.attempt ?? index + 1;
    const attemptId = attempt.id ?? attempt.attempt_id ?? attemptNumber;
    const attemptKey = String(attemptId);
    runDetailAttemptRecords.set(attemptKey, {
      attempt,
      attemptId,
      attemptNumber,
      attemptKey,
      adapterLabel: trainingAdapterLabel(run, attempt),
      adapterFields:
        (
          attempt.execution_snapshot_json?.adapter?.manifest ||
          run.resolved_spec_json?.source?.adapter_manifest
        )?.train?.input_fields || [],
      trackingLinks: Array.isArray(run.tracking_links)
        ? run.tracking_links
        : [],
    });
    retainedAttemptKeys.add(attemptKey);
    const row =
      existingAttemptRows.get(attemptKey) || document.createElement("tr");
    row.dataset.attemptRow = "";
    row.dataset.attemptKey = attemptKey;
    const attemptActive =
      activeRunAttemptDisclosure?.attemptId === String(attemptId);
    patchTableRow(row, [
      { html: escapeHtml(attemptNumber) },
      { html: statusPill(jobStatusLabel(attempt)) },
      { html: escapeHtml(attempt.slurm_job_id || attempt.job_id || "-") },
      {
        html: escapeHtml(
          [
            `Queue: ${attempt.partition_name || "Not recorded"}`,
            `Node: ${attempt.node || attempt.node_list || "Not allocated"}`,
            `SSH: ${attempt.gateway || "Not recorded"}`,
          ].join(" · ") || "-",
        ),
      },
      { html: escapeHtml(formatDate(attempt.started_at)) },
      { html: escapeHtml(formatDate(attempt.ended_at || attempt.finished_at)) },
      {
        className: "wrap-cell",
        html: escapeHtml(
          attempt.error ||
            attempt.failure_reason ||
            attempt.slurm_reason ||
            "-",
        ),
      },
      {
        className: "row-actions",
        preserve: attemptActive,
        html: `<button type="button" data-attempt-action="view" data-attempt-key="${escapeHtml(attemptKey)}" aria-controls="run-attempt-detail-dialog" aria-haspopup="dialog" aria-expanded="false">Detail</button>`,
      },
    ]);
    orderedAttemptRows.push(row);
  });
  existingAttemptRows.forEach((row, attemptKey) => {
    if (!retainedAttemptKeys.has(attemptKey)) row.remove();
  });
  elements.attemptsBody
    .querySelectorAll(":scope > tr.empty-row")
    .forEach((row) => row.remove());
  if (orderedAttemptRows.length) {
    const companion = elements.runAttemptDetail.closest(
      "tr.row-disclosure-companion",
    );
    const activeRow = activeRunAttemptDisclosure
      ? orderedAttemptRows.find(
          (row) =>
            row.dataset.attemptKey === activeRunAttemptDisclosure.attemptKey,
        )
      : null;
    reconcileTableSequence(
      elements.attemptsBody,
      orderedAttemptRows.flatMap((row) =>
        row === activeRow && companion ? [row, companion] : [row],
      ),
    );
  } else {
    const template = document.createElement("template");
    template.innerHTML = emptyRow(8, "No attempt records were returned.");
    elements.attemptsBody.append(template.content);
  }
  const latestRecord = attempts.length
    ? [...runDetailAttemptRecords.values()][runDetailAttemptRecords.size - 1]
    : null;
  runDetailLatestAttemptId = latestRecord
    ? String(latestRecord.attemptId)
    : null;
  if (preserveAttempt && activeRunAttemptDisclosure) {
    const selectedRecord = runAttemptRecordById(selectedAttemptId);
    if (selectedRecord) {
      const launcher = elements.attemptsBody.querySelector(
        `[data-attempt-key="${CSS.escape(selectedRecord.attemptKey)}"] [data-attempt-action="view"]`,
      );
      if (launcher) {
        activeRunAttemptDisclosure.attemptKey = selectedRecord.attemptKey;
        activeRunAttemptDisclosure.attemptId = String(selectedRecord.attemptId);
        activeRunAttemptDisclosure.launcher = launcher;
        setRunAttemptLauncherActive(launcher, true);
        renderRunAttemptMetadata(selectedRecord);
      }
    } else {
      closeRunAttemptDisclosure({ restoreFocus: false });
    }
  }
  if (preserveAttempt && activeRunAttemptDisclosure)
    remountActiveRunAttemptDisclosure();
  elements.runDetail.hidden = false;
  return { run, state: String(state).toUpperCase(), latestRecord };
}

function updateRunSummary(run, id) {
  const index = runRows.findIndex(
    (item) => String(item.id || item.run_id) === String(id),
  );
  if (index < 0) return;
  runRows[index] = { ...runRows[index], ...run };
  const summaryRow = elements.runsBody.querySelector(
    `:scope > tr[data-run-id="${CSS.escape(String(id))}"]`,
  );
  if (summaryRow)
    patchTableRow(summaryRow, runRowDescriptor(runRows[index]).cells);
}

function renderRunDetail(
  payload,
  id,
  { preserveAttempt = false, background = false } = {},
) {
  const run = entityFrom(payload, "run");
  const attempts = runAttemptRecords({
    ...run,
    attempts: Array.isArray(run.attempts)
      ? run.attempts
      : listFrom(payload, ["attempts"]),
  });
  const panel = elements.runDetail.closest("dialog") || elements.runDetail;
  let rendered = null;
  commitPanelRefresh(
    panel,
    `run-detail:${id}`,
    { run, attempts },
    () => {
      updateRunSummary(run, id);
      rendered = renderRunDetailContent(payload, id, { preserveAttempt });
    },
    { background },
  );
  if (rendered) return rendered;
  return {
    run,
    state: String(run.status || run.state || "unknown").toUpperCase(),
    latestRecord: runDetailLatestAttemptId
      ? runAttemptRecordById(runDetailLatestAttemptId)
      : null,
  };
}

function logContent(payload) {
  if (typeof payload === "string") return payload;
  if (!payload || typeof payload !== "object") return "";
  return (
    payload.stderr ||
    payload.log ||
    payload.content ||
    payload.text ||
    JSON.stringify(payload, null, 2)
  );
}

async function viewRun(id, launcher = null) {
  const scope = `run-detail:${id}`;
  const revealLauncher = launcher || currentRevealLauncher();
  resetRunAttemptContext(id);
  const requestToken = disclosureToken(elements.runDetail, revealLauncher);
  elements.runDetailTitle.textContent = trainingRunName(
    runRows.find((run) => run.id === id) || {},
  );
  elements.runDetailActions.innerHTML = "";
  elements.runDetailTracking.replaceChildren();
  elements.runDetailMeta.innerHTML = keyValueHtml([
    ["Status", "Loading run..."],
  ]);
  elements.attemptsBody.innerHTML = emptyRow(8, "Loading attempts...");
  revealPanel(elements.runDetail, {
    focusTarget: elements.runDetailTitle,
    launcher: revealLauncher,
    scroll: false,
  });
  const detailResult = await Promise.resolve(
    api(`/api/runs/${encodeURIComponent(id)}`),
  )
    .then((value) => ({ status: "fulfilled", value }))
    .catch((reason) => ({ status: "rejected", reason }));
  if (!disclosureTokenIsCurrent(elements.runDetail, requestToken)) return;
  if (detailResult.status === "fulfilled") {
    try {
      const detail = renderRunDetail(detailResult.value, id);
      if (!disclosureTokenIsCurrent(elements.runDetail, requestToken)) return;
      clearNotificationScope(scope);
      startRunDetailPolling(id, detail.state);
    } catch (error) {
      elements.runDetailMeta.innerHTML = keyValueHtml([
        ["Detail error", error.message],
      ]);
      elements.attemptsBody.innerHTML = emptyRow(
        8,
        "Training Run detail could not be rendered.",
      );
      showToast(`Training Run detail failed: ${error.message}`, true, {
        scope,
      });
    }
  } else {
    elements.runDetailMeta.innerHTML = keyValueHtml([
      ["Detail error", detailResult.reason.message],
    ]);
    elements.attemptsBody.innerHTML = emptyRow(
      8,
      "Training Run detail unavailable.",
    );
    startRunDetailPolling(id, null, {
      initialDelay: RUN_DETAIL_POLL_INTERVAL_MS,
    });
  }
}

async function attachRunTracking(id, provider, actionPath, button) {
  const scope = `run-tracking:${id}:${provider}`;
  if (!id || !provider || !actionPath) {
    showToast(
      "Tracking attachment action is incomplete; refresh the Training Run detail.",
      true,
      { scope },
    );
    return;
  }
  const expectedPrefix = `/api/runs/${encodeURIComponent(id)}/tracking/`;
  if (
    !actionPath.startsWith(expectedPrefix) ||
    !actionPath.endsWith("/attach")
  ) {
    showToast(
      "Tracking attachment action was rejected because its server path is invalid.",
      true,
      { scope },
    );
    return;
  }
  const originalLabel = button.textContent;
  const requestToken = disclosureToken(elements.runDetail);
  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  button.textContent = `Connecting ${trackingProviderLabel(provider)}...`;
  try {
    const payload = await api(actionPath, { method: "POST" });
    if (
      activeRunDetailId === String(id) &&
      !elements.runDetail.hidden &&
      disclosureTokenIsCurrent(elements.runDetail, requestToken)
    ) {
      const detail = renderRunDetail(payload, id, { preserveAttempt: true });
      startRunDetailPolling(id, detail.state);
    } else {
      updateRunSummary(entityFrom(payload, "run"), id);
    }
    showToast(
      `${trackingProviderLabel(provider)} connected to Training Run ${shortId(id)}.`,
      false,
      { scope },
    );
  } catch (error) {
    const message = `Tracking attachment failed for Training Run ${shortId(id)}: ${error.message}`;
    showNotice(elements.runsError, message, { scope });
    showToast(message, true, { scope });
    if (button.isConnected) {
      button.disabled = false;
      button.removeAttribute("aria-busy");
      button.textContent = originalLabel;
    }
  }
}

function runDetailPollEligible(runId) {
  return Boolean(
    runId &&
      activeTab === "runs" &&
      document.visibilityState === "visible" &&
      !elements.runDetail.hidden &&
      activeRunDetailId === String(runId),
  );
}

function scheduleRunDetailPoll(
  delay,
  { finalCycle = false, includeList = true } = {},
) {
  window.clearTimeout(runDetailPollTimer);
  const generation = runDetailPollGeneration;
  const runId = runDetailPollRunId;
  if (!runDetailPollEligible(runId)) return;
  runDetailPollTimer = window.setTimeout(
    () => pollRunDetail(generation, runId, { finalCycle, includeList }),
    delay,
  );
}

function startRunDetailPolling(
  runId,
  state = null,
  { initialDelay = null, includeList = true } = {},
) {
  runDetailPollGeneration += 1;
  window.clearTimeout(runDetailPollTimer);
  runDetailPollTimer = null;
  runDetailPollInFlight = null;
  runDetailPollRunId = String(runId);
  const active =
    state === null || RUN_DETAIL_ACTIVE_STATES.has(String(state).toUpperCase());
  const delay =
    initialDelay ??
    (active ? RUN_DETAIL_POLL_INTERVAL_MS : RUN_DETAIL_FINAL_POLL_DELAY_MS);
  scheduleRunDetailPoll(delay, { finalCycle: !active, includeList });
}

async function pollRunDetail(
  generation,
  runId,
  { finalCycle = false, includeList = true } = {},
) {
  const scope = `run-detail:${runId}`;
  if (generation !== runDetailPollGeneration || !runDetailPollEligible(runId))
    return;
  if (
    runDetailPollInFlight &&
    runDetailPollInFlight.generation === generation &&
    runDetailPollInFlight.runId === String(runId)
  ) {
    scheduleRunDetailPoll(250, { finalCycle, includeList });
    return;
  }
  const pollOwner = { generation, runId: String(runId) };
  runDetailPollInFlight = pollOwner;
  try {
    const detailResult = await Promise.resolve(
      api(`/api/runs/${encodeURIComponent(runId)}`),
    )
      .then((value) => ({ status: "fulfilled", value }))
      .catch((reason) => ({ status: "rejected", reason }));
    if (generation !== runDetailPollGeneration || !runDetailPollEligible(runId))
      return;
    if (detailResult.status === "rejected") {
      showNotice(
        elements.runsError,
        `Live run update failed; displayed data may be stale: ${detailResult.reason.message}. Retrying automatically.`,
        { scope },
      );
      scheduleRunDetailPoll(RUN_DETAIL_POLL_INTERVAL_MS, { includeList: true });
      return;
    }
    clearNotificationScope(scope);
    const detail = renderRunDetail(detailResult.value, runId, {
      preserveAttempt: true,
      background: true,
    });
    if (!detail.latestRecord) {
      closeRunAttemptDisclosure({ restoreFocus: false });
    } else if (activeRunAttemptDisclosure && !runDetailAttemptDismissed) {
      let selectedRecord = activeRunAttemptDisclosure
        ? runAttemptRecordById(activeRunAttemptDisclosure.attemptId)
        : null;
      if (
        !selectedRecord ||
        (runDetailFollowLatest &&
          String(selectedRecord.attemptId) !==
            String(detail.latestRecord.attemptId))
      ) {
        const launcher = elements.attemptsBody.querySelector(
          `[data-attempt-key="${CSS.escape(detail.latestRecord.attemptKey)}"] [data-attempt-action="view"]`,
        );
        if (launcher) {
          await openRunAttemptDetail(detail.latestRecord.attemptKey, launcher, {
            focus: false,
            loadLogs: false,
            showLoading: true,
            followLatest: true,
          });
          selectedRecord = detail.latestRecord;
        }
      }
      if (selectedRecord && activeRunAttemptDisclosure) {
        await loadRunAttemptLogs(
          selectedRecord,
          activeRunAttemptDisclosure.token,
          {
            showLoading: false,
            retainOnError: true,
            pollGeneration: generation,
          },
        );
      }
    }
    if (generation !== runDetailPollGeneration || !runDetailPollEligible(runId))
      return;
    const active = RUN_DETAIL_ACTIVE_STATES.has(detail.state);
    if (active) {
      scheduleRunDetailPoll(RUN_DETAIL_POLL_INTERVAL_MS, { includeList: true });
    } else if (!finalCycle) {
      scheduleRunDetailPoll(RUN_DETAIL_FINAL_POLL_DELAY_MS, {
        finalCycle: true,
        includeList: true,
      });
    } else {
      runDetailPollRunId = null;
      runDetailPollTimer = null;
    }
  } finally {
    if (runDetailPollInFlight === pollOwner) runDetailPollInFlight = null;
  }
}

async function refreshRunsPage() {
  await loadRuns(true);
  if (activeRunDetailId && !elements.runDetail.hidden && activeTab === "runs") {
    startRunDetailPolling(activeRunDetailId, null, {
      initialDelay: 0,
      includeList: false,
    });
  }
}

function evaluationScalarValues(value, output = []) {
  if (typeof value === "string" || typeof value === "number") {
    const text = String(value).trim();
    if (text) output.push(text);
  } else if (Array.isArray(value)) {
    value.forEach((item) => evaluationScalarValues(item, output));
  } else if (value && typeof value === "object") {
    Object.values(value).forEach((item) =>
      evaluationScalarValues(item, output),
    );
  }
  return output;
}

function evaluationValuesForKeys(
  value,
  keyPattern,
  output = [],
  seen = new Set(),
) {
  if (!value || typeof value !== "object" || seen.has(value)) return output;
  seen.add(value);
  for (const [key, nested] of Object.entries(value)) {
    const normalizedKey = key.replace(/([a-z])([A-Z])/g, "$1_$2").toLowerCase();
    if (keyPattern.test(normalizedKey)) evaluationScalarValues(nested, output);
    evaluationValuesForKeys(nested, keyPattern, output, seen);
  }
  return [...new Set(output)];
}

function normalizedEvaluationValue(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

function evaluationMetadataIncludes(metadata, value) {
  const needle = normalizedEvaluationValue(value);
  return (
    needle.length >= 3 && normalizedEvaluationValue(metadata).includes(needle)
  );
}

function evaluationSuiteConfig(suite) {
  return suite?.config_json &&
    typeof suite.config_json === "object" &&
    !Array.isArray(suite.config_json)
    ? suite.config_json
    : {};
}

function evaluationSuiteIsRunnable(suite) {
  const config = evaluationSuiteConfig(suite);
  const unavailable = (value) =>
    value === false ||
    value === 0 ||
    String(value || "").toLowerCase() === "false";
  return (
    (!suite?.compatibility || suite.compatibility.ready === true) &&
    !unavailable(suite?.enabled) &&
    !unavailable(suite?.runnable) &&
    !unavailable(config.enabled) &&
    !unavailable(config.runnable)
  );
}

function evaluationSuitePreference(run, suite) {
  if (suite?.is_default === true) return 4;
  const config = evaluationSuiteConfig(suite);
  const runMetadata = JSON.stringify(run || {});
  const suiteIdentity = [
    suite?.id,
    suite?.slug,
    suite?.name,
    suite?.suite,
    config.suite,
    config.name,
  ];
  const compatibility = [
    suite?.environment,
    suite?.default_environment,
    suite?.evaluator_adapter,
    suite?.evaluator,
    suite?.adapter,
    suite?.dataset,
    suite?.data_bundle,
    config.evaluator_adapter,
    config.evaluator,
    config.adapter,
    config.environment,
    config.default_environment,
    config.simulator,
    config.benchmark,
    config.data,
    config.dataset,
    config.data_bundle,
  ];
  const canonicalDefault =
    suite?.is_default === true ||
    suite?.default === true ||
    suite?.canonical === true ||
    config.is_default === true ||
    config.default === true ||
    config.canonical === true;
  if (
    evaluationScalarValues(suiteIdentity).some((value) =>
      evaluationMetadataIncludes(runMetadata, value),
    )
  )
    return 3;
  if (
    evaluationScalarValues(compatibility).some((value) =>
      evaluationMetadataIncludes(runMetadata, value),
    )
  )
    return 2;
  return canonicalDefault ? 1 : 0;
}

function preferredEvaluationSuite(run) {
  return (
    evaluationSuites
      .map((suite, index) => ({ suite, index }))
      .filter(({ suite }) => evaluationSuiteIsRunnable(suite))
      .sort(
        (left, right) =>
          evaluationSuitePreference(run, right.suite) -
            evaluationSuitePreference(run, left.suite) ||
          left.index - right.index,
      )[0]?.suite || null
  );
}

function evaluationCheckpointForRun(run) {
  const checkpoints = Array.isArray(run?.checkpoints) ? run.checkpoints : [];
  const available = [...checkpoints].reverse().filter((checkpoint) => {
    if (!checkpoint || typeof checkpoint !== "object") return false;
    const path = String(checkpoint.path || "").trim();
    const status = String(checkpoint.status || "")
      .trim()
      .toUpperCase();
    const registeredAvailable =
      status === "AVAILABLE" || (!status && checkpoint.available === true);
    return (
      Boolean(path) &&
      registeredAvailable &&
      checkpoint.available !== false &&
      !checkpoint.pruned_at
    );
  });
  const selectedInference =
    available.find((checkpoint) => {
      const kind = String(
        checkpoint.checkpoint_type || checkpoint.type || "",
      ).toUpperCase();
      return (
        checkpoint.is_selected_for_inference === true &&
        (!kind || kind === "INFERENCE")
      );
    }) ||
    available.find(
      (checkpoint) => checkpoint.is_selected_for_inference === true,
    ) ||
    available.find(
      (checkpoint) =>
        String(
          checkpoint.checkpoint_type || checkpoint.type || "",
        ).toUpperCase() === "INFERENCE",
    );
  return String(selectedInference?.path || "").trim();
}

const RUN_EVALUATION_CHECKPOINT_REQUIRED =
  "A registered checkpoint is required";

function runEvaluationAction(run) {
  if (run?.evaluation_blocker)
    return {
      enabled: false,
      reason: run.evaluation_blocker,
      checkpointPath: "",
    };
  const checkpointPath = evaluationCheckpointForRun(run);
  return checkpointPath
    ? { enabled: true, reason: "", checkpointPath }
    : {
        enabled: false,
        reason: RUN_EVALUATION_CHECKPOINT_REQUIRED,
        checkpointPath: "",
      };
}

function runEvaluationActionButton(run, id) {
  const action = runEvaluationAction(run);
  const accessibleLabel = action.enabled
    ? "Start evaluation"
    : `Start evaluation: ${action.reason}`;
  return `<button type="button" class="button" data-run-action="evaluate" data-run-action-enabled="${action.enabled}" data-run-action-reason="${escapeHtml(action.reason)}" data-id="${escapeHtml(id)}" title="${escapeHtml(action.reason)}" aria-label="${escapeHtml(accessibleLabel)}"${action.enabled ? "" : " disabled"}>Start evaluation</button>`;
}

function evaluationSuiteTasks(suite) {
  const config = evaluationSuiteConfig(suite);
  const declared =
    Array.isArray(suite?.tasks) && suite.tasks.length
      ? suite.tasks
      : Array.isArray(config.tasks)
        ? config.tasks
        : [];
  const seen = new Set();
  return declared.flatMap((task) => {
    const id =
      typeof task === "string"
        ? task.trim()
        : String(
            firstValue(task?.id, task?.task_id, task?.slug, task?.name, ""),
          ).trim();
    if (!id || seen.has(id)) return [];
    seen.add(id);
    const label =
      typeof task === "string"
        ? id
        : String(firstValue(task?.label, task?.title, task?.name, id));
    return [{ id, label }];
  });
}

function checkedEvaluationTaskIds() {
  return [
    ...elements.evaluationTasksOptions.querySelectorAll("input:checked"),
  ].map((input) => input.value);
}

function updateEvaluationTaskLabel() {
  const total =
    elements.evaluationTasksOptions.querySelectorAll("input").length;
  const selected = checkedEvaluationTaskIds().length;
  elements.evaluationTasksFilterLabel.textContent = selected
    ? `${selected} of ${total} task${total === 1 ? "" : "s"}`
    : renderedEvaluationTaskPolicy.mode === "single" && total > 1
      ? "Choose one task"
      : "Suite default";
}

function populateEvaluationTasks(suite, { preserve = false } = {}) {
  const suiteId = evaluationSuiteId(suite);
  const selected =
    preserve && suiteId === renderedEvaluationTaskSuiteId
      ? new Set(checkedEvaluationTaskIds())
      : new Set();
  const tasks = evaluationSuiteTasks(suite);
  renderedEvaluationTaskSuiteId = suiteId;
  elements.evaluationTasksOptions.innerHTML = tasks
    .map(
      (task) => `
    <label class="filter-option">
      <input type="checkbox" value="${escapeHtml(task.id)}"${selected.has(task.id) ? " checked" : ""}>
      <span>${escapeHtml(task.label)}</span>
      ${task.label === task.id ? "" : `<code>${escapeHtml(task.id)}</code>`}
    </label>`,
    )
    .join("");
  const disabled = !tasks.length;
  elements.evaluationTasksFilter.classList.toggle("is-disabled", disabled);
  elements.evaluationTasksFilter.setAttribute(
    "aria-disabled",
    String(disabled),
  );
  elements.evaluationTasksAll.disabled = disabled;
  elements.evaluationTasksClear.disabled = disabled;
  if (disabled) elements.evaluationTasksFilter.open = false;
  elements.evaluationTasksStatus.textContent = suite
    ? disabled
      ? "No task catalog is declared; the suite or evaluator adapter default will be used."
      : `${tasks.length} task${tasks.length === 1 ? "" : "s"} declared. Leave all unchecked to use the suite default.`
    : "Choose a suite to load its task catalog.";
  updateEvaluationTaskLabel();
}

function updateEvaluationEnvironmentFromSuite({ preserveTasks = false } = {}) {
  const suite = selectedEvaluationSuite();
  const config = evaluationSuiteConfig(suite);
  if (suite?.compatibility) {
    const report = suite.compatibility;
    elements.evaluationSuiteStatus.textContent = `${report.label}: ${
      report.messages?.length
        ? report.messages.join(" ")
        : report.executor === "recorded_simulator"
          ? "Metadata matches. Model loading and one simulator step will be checked when the job starts."
          : "Native integration available. Runtime checks are performed by its evaluator."
    }`;
  }
  const environment = firstValue(
    suite?.environment,
    suite?.default_environment,
    config.environment,
    config.default_environment,
    config.simulator,
    suite?.evaluator_adapter,
    suite?.evaluator,
    config.evaluator,
    "",
  );
  elements.evaluationEnvironment.value = suite ? String(environment || "") : "";
  updateEvaluationGpuOptions(suite);
  const episodeLimit = Number(config.maximum_episodes_per_task || 10000);
  elements.evaluationEpisodes.max = String(episodeLimit);
  if (Number(elements.evaluationEpisodes.value) > episodeLimit)
    elements.evaluationEpisodes.value = String(episodeLimit);
  populateEvaluationTasks(suite, { preserve: preserveTasks });
  const limit = Number(suite?.maximum_parallelism || 1);
  elements.evaluationParallelism.max = String(limit);
  elements.evaluationParallelism.readOnly = limit === 1;
  if (Number(elements.evaluationParallelism.value) > limit)
    elements.evaluationParallelism.value = String(limit);
  document.querySelector("#evaluation-parallelism-help").textContent =
    limit > 1
      ? "One GPU per worker, in one Slurm allocation."
      : "This evaluator runs one worker.";
}

function updateEvaluationGpuOptions(suite) {
  const select = document.getElementById("evaluation-resource-gpu");
  const previous = select.value;
  const allowed = Array.isArray(suite?.allowed_gpu_types)
    ? new Set(suite.allowed_gpu_types)
    : null;
  const options = Array.from(elements.experimentGpuType.options).filter(
    (option) =>
      option.value !== "auto" && (!allowed || allowed.has(option.value)),
  );
  select.replaceChildren(...options.map((option) => option.cloneNode(true)));
  if (options.some((option) => option.value === previous))
    select.value = previous;
  else select.value = options[0]?.value || "";
  select.disabled = !options.length;
  if (!options.length)
    select.add(new Option("No compatible GPU configured", ""));
}

function initializeEvaluationResources() {
  const container = document.querySelector("#evaluation-resources");
  // Reuse the training form's resource controls, validation and options.
  for (const [source, name, label] of [
    ["resource-policy", "queue", "Queue policy"],
    ["experiment-gpu-type", "gpu", "GPU type"],
    ["resource-cpus", "cpus", "CPUs / worker"],
    ["resource-memory", "memory", "Memory / worker (GB)"],
    ["resource-time", "time", "Wall time"],
  ]) {
    const field = document
      .getElementById(source)
      .closest(".field")
      .cloneNode(true);
    const input = field.querySelector("input, select");
    input.id = `evaluation-resource-${name}`;
    input.required = true;
    const title = field.querySelector("label");
    title.htmlFor = input.id;
    title.textContent = label;
    if (name === "gpu") {
      input.querySelector('option[value="auto"]')?.remove();
      input.value = "a40";
    }
    input.addEventListener("input", () => scheduleEvaluationTargetValidation());
    input.addEventListener("change", () =>
      scheduleEvaluationTargetValidation(),
    );
    container.append(field);
  }
}
initializeEvaluationResources();

function validateEvaluationResources() {
  if (!document.getElementById("evaluation-resource-gpu").value) {
    document.querySelector("#evaluation-resource-summary").textContent =
      "No compatible GPU is configured for this suite.";
    return false;
  }
  const time = document.querySelector("#evaluation-resource-time");
  const raw = time.value.trim();
  const match = /^(?:(\d+)-)?(\d{1,2}):(\d{2}):(\d{2})$/.exec(raw);
  const seconds = match
    ? Number(match[1] || 0) * 86400 +
      Number(match[2]) * 3600 +
      Number(match[3]) * 60 +
      Number(match[4])
    : 0;
  const valid = Boolean(
    match && Number(match[3]) < 60 && Number(match[4]) < 60 && seconds > 0,
  );
  time.setCustomValidity(
    valid ? "" : "Enter a positive wall time as HH:MM:SS or D-HH:MM:SS.",
  );
  time.setAttribute("aria-invalid", String(!valid));
  const budget = document.querySelector("#evaluation-resource-summary");
  if (!valid) budget.textContent = time.validationMessage;
  return valid;
}

function evaluationResources() {
  const value = (name) =>
    document.getElementById(`evaluation-resource-${name}`).value;
  return {
    gateway: elements.gateway.value,
    queue_policy: value("queue"),
    nodes: 1,
    gpu: { mode: "explicit", count: 1, type: value("gpu") },
    cpus_per_task: Number(value("cpus")),
    memory_gb: Number(value("memory")),
    time_limit: value("time").trim(),
  };
}

function evaluationTargetSignature() {
  return JSON.stringify([
    elements.evaluationRunId.value.trim(),
    elements.evaluationCheckpoint.value.trim(),
    elements.evaluationSuite.value,
    elements.evaluationEnvironment.value,
    checkedEvaluationTaskIds(),
    elements.evaluationArgv.value.trim(),
    elements.evaluationResumeArgv.value.trim(),
    elements.gateway.value,
    elements.evaluationEpisodes.value,
    elements.evaluationSeeds.value,
    elements.evaluationParallelism.value,
    elements.evaluationHeadless.checked,
    evaluationResources(),
  ]);
}

function setEvaluationFieldValidation(element, status, state, message) {
  status.className = `field-validation${state ? ` is-${state}` : ""}`;
  status.textContent = message;
  if (state === "invalid") element.setAttribute("aria-invalid", "true");
  else element.removeAttribute("aria-invalid");
}

function evaluationSeedsValid() {
  const raw = elements.evaluationSeeds.value.trim();
  const valid =
    raw !== "" &&
    raw
      .split(",")
      .every(
        (seed) =>
          /^\d+$/.test(seed.trim()) && Number.isSafeInteger(Number(seed)),
      );
  const message = valid
    ? ""
    : "Enter comma-separated nonnegative integer seeds, for example 0,1,2.";
  elements.evaluationSeeds.setCustomValidity(message);
  const hint = document.querySelector("#evaluation-seeds-error");
  if (hint) {
    hint.textContent = message;
    hint.hidden = valid;
  }
  return valid;
}

function updateEvaluationSubmitState() {
  const currentIsValid =
    evaluationTargetValidationState.valid &&
    evaluationTargetValidationState.planValid &&
    evaluationTargetValidationState.signature === evaluationTargetSignature();
  elements.submitEvaluation.disabled =
    evaluationSubmitting ||
    evaluationTargetValidationState.pending ||
    !currentIsValid ||
    !evaluationSeedsValid() ||
    !elements.evaluationForm.checkValidity();
}

async function validateEvaluationTarget(request, signature) {
  const runId = elements.evaluationRunId.value.trim();
  const checkpointPath = elements.evaluationCheckpoint.value.trim();
  const suiteId = elements.evaluationSuite.value;
  if (!validateEvaluationResources()) {
    evaluationTargetValidationState = {
      pending: false,
      valid: false,
      planValid: false,
      signature,
    };
    updateEvaluationSubmitState();
    return false;
  }
  let commandOverrides;
  try {
    commandOverrides = evaluationCommandOverrides();
  } catch (error) {
    if (
      request !== evaluationTargetValidationRequest ||
      signature !== evaluationTargetSignature()
    )
      return false;
    setEvaluationFieldValidation(
      elements.evaluationPlanStatus,
      elements.evaluationPlanStatus,
      "invalid",
      error.message,
    );
    evaluationTargetValidationState = {
      pending: false,
      valid: false,
      planValid: false,
      signature,
    };
    updateEvaluationSubmitState();
    return false;
  }
  try {
    const result = await api("/api/evaluations/validate-target", {
      method: "POST",
      timeoutMs: 30000,
      body: JSON.stringify({
        run_id: runId,
        checkpoint_path: checkpointPath || null,
        suite_id: suiteId || null,
        environment: elements.evaluationEnvironment.value || null,
        tasks: checkedEvaluationTaskIds(),
        gateway: elements.gateway.value,
        episodes_per_task: numberOrNull(elements.evaluationEpisodes),
        seeds: elements.evaluationSeeds.value
          .split(",")
          .map((seed) => Number(seed.trim())),
        parallelism: numberOrNull(elements.evaluationParallelism),
        headless: elements.evaluationHeadless.checked,
        resources: evaluationResources(),
        argv: commandOverrides.argv,
        resume_argv: commandOverrides.resumeArgv,
      }),
    });
    if (
      request !== evaluationTargetValidationRequest ||
      signature !== evaluationTargetSignature()
    )
      return false;
    const errors =
      result.errors && typeof result.errors === "object" ? result.errors : {};
    if (!checkpointPath && result.resolved_checkpoint_path) {
      elements.evaluationCheckpoint.value = String(
        result.resolved_checkpoint_path,
      );
    }
    const finalSignature = evaluationTargetSignature();
    const runValid = result.run_valid === true;
    const checkpointValid = result.checkpoint_valid === true;
    const suiteValid = Boolean(suiteId) && result.suite_valid === true;
    const planValid = Boolean(suiteId) && result.plan_valid === true;
    const valid =
      result.valid === true &&
      runValid &&
      checkpointValid &&
      suiteValid &&
      planValid;
    setEvaluationFieldValidation(
      elements.evaluationRunId,
      elements.evaluationRunValidation,
      runValid ? "valid" : "invalid",
      runValid
        ? `Valid run${result.run_state ? ` (${result.run_state})` : ""}.`
        : String(errors.run_id || "Run is not valid for evaluation."),
    );
    setEvaluationFieldValidation(
      elements.evaluationCheckpoint,
      elements.evaluationCheckpointValidation,
      checkpointValid ? "valid" : "invalid",
      checkpointValid
        ? checkpointPath
          ? "Registered checkpoint is available for this run."
          : "Resolved an available registered checkpoint from this run."
        : String(
            errors.checkpoint_path ||
              "Checkpoint is not available for this run.",
          ),
    );
    const resources = result.resolved_resources;
    const budget = document.querySelector("#evaluation-resource-summary");
    budget.textContent = resources
      ? `Total: ${resources.gpu?.count || 1} ${String(resources.gpu?.type || "any").toUpperCase()} GPU${resources.gpu?.count === 1 ? "" : "s"} · ${resources.cpus_per_task} CPUs · ${resources.memory_gb} GB RAM · ${resources.time_limit}${resources.node?.name ? ` · Node: ${resources.node.name}` : ""}`
      : "Allocation has not been resolved. Validate a compatible checkpoint and evaluation suite.";
    const evaluator =
      result.evaluator && typeof result.evaluator === "object"
        ? result.evaluator
        : null;
    const evaluatorLabel = evaluator
      ? `${evaluator.slug || evaluator.id || "evaluator"} / v${evaluator.version ?? evaluator.version_id ?? "?"}`
      : null;
    const blockers = Array.isArray(result.plan_blockers)
      ? result.plan_blockers.filter(Boolean)
      : [];
    const planMessage = !suiteId
      ? "Choose a suite to resolve its evaluator implementation."
      : planValid
        ? result.plan_source === "manual_override"
          ? "Ready: manual evaluator command override."
          : result.compatibility?.executor === "recorded_simulator"
            ? "Ready to submit. Model loading, camera/joint inputs and one simulator step will be verified before scoring."
            : String(
                result.plan_message ||
                  `Ready: ${evaluatorLabel || "registered evaluator adapter"}.`,
              )
        : String(
            errors.suite_id ||
              errors.environment ||
              result.plan_message ||
              blockers.join(" ") ||
              "No runnable evaluator implementation resolved for this suite.",
          );
    setEvaluationFieldValidation(
      elements.evaluationPlanStatus,
      elements.evaluationPlanStatus,
      planValid ? "valid" : suiteId ? "invalid" : "",
      planMessage,
    );
    evaluationLastValidatedAt = Date.now();
    evaluationTargetValidationState = {
      pending: false,
      valid,
      planValid,
      signature: finalSignature,
    };
    updateEvaluationSubmitState();
    return valid;
  } catch (error) {
    if (
      request !== evaluationTargetValidationRequest ||
      signature !== evaluationTargetSignature()
    )
      return false;
    setEvaluationFieldValidation(
      elements.evaluationRunId,
      elements.evaluationRunValidation,
      "",
      "Run validation could not be completed.",
    );
    setEvaluationFieldValidation(
      elements.evaluationCheckpoint,
      elements.evaluationCheckpointValidation,
      "",
      "Checkpoint validation could not be completed.",
    );
    setEvaluationFieldValidation(
      elements.evaluationPlanStatus,
      elements.evaluationPlanStatus,
      "invalid",
      `Validation failed: ${error.message}`,
    );
    evaluationTargetValidationState = {
      pending: false,
      valid: false,
      planValid: false,
      signature,
    };
    updateEvaluationSubmitState();
    return false;
  }
}

let evaluationLastValidatedAt = 0;
function scheduleEvaluationTargetValidation({ immediate = false } = {}) {
  window.clearTimeout(evaluationTargetValidationTimer);
  if (!evaluationSeedsValid()) {
    updateEvaluationSubmitState();
    return;
  }
  const signature = evaluationTargetSignature();
  if (
    !immediate &&
    evaluationTargetValidationState.valid &&
    evaluationTargetValidationState.signature === signature &&
    Date.now() - evaluationLastValidatedAt < 10000
  )
    return;
  const request = ++evaluationTargetValidationRequest;
  const runId = elements.evaluationRunId.value.trim();
  if (!runId) {
    setEvaluationFieldValidation(
      elements.evaluationRunId,
      elements.evaluationRunValidation,
      "invalid",
      "Enter a training run ID.",
    );
    setEvaluationFieldValidation(
      elements.evaluationCheckpoint,
      elements.evaluationCheckpointValidation,
      "",
      "A registered checkpoint will be resolved from the run.",
    );
    setEvaluationFieldValidation(
      elements.evaluationPlanStatus,
      elements.evaluationPlanStatus,
      "",
      elements.evaluationSuite.value
        ? "Enter a valid run to resolve the evaluator plan."
        : "Choose a suite to resolve its evaluator implementation.",
    );
    evaluationTargetValidationState = {
      pending: false,
      valid: false,
      planValid: false,
      signature,
    };
    updateEvaluationSubmitState();
    return;
  }
  setEvaluationFieldValidation(
    elements.evaluationRunId,
    elements.evaluationRunValidation,
    "pending",
    "Checking run...",
  );
  setEvaluationFieldValidation(
    elements.evaluationCheckpoint,
    elements.evaluationCheckpointValidation,
    "pending",
    "Checking checkpoint...",
  );
  setEvaluationFieldValidation(
    elements.evaluationPlanStatus,
    elements.evaluationPlanStatus,
    "pending",
    elements.evaluationSuite.value
      ? "Resolving evaluator implementation..."
      : "Choose a suite to resolve its evaluator implementation.",
  );
  evaluationTargetValidationState = {
    pending: true,
    valid: false,
    planValid: false,
    signature,
  };
  updateEvaluationSubmitState();
  evaluationTargetValidationTimer = window.setTimeout(
    () => validateEvaluationTarget(request, signature),
    immediate ? 0 : 350,
  );
}

async function startEvaluationForRun(id) {
  const requestedRunId = String(id || "").trim();
  if (!requestedRunId) return;
  const request = ++evaluationPrefillRequest;

  let run = null;
  let loadError = null;
  try {
    const payload = await api(
      `/api/runs/${encodeURIComponent(requestedRunId)}`,
    );
    run = entityFrom(payload, "run");
    const returnedRunId = String(run?.id || run?.run_id || "").trim();
    if (returnedRunId !== requestedRunId)
      throw new Error("Run detail did not match the selected run.");
  } catch (error) {
    loadError = error;
  }
  if (request !== evaluationPrefillRequest) return;

  if (loadError) {
    showToast(`Start evaluation unavailable: ${loadError.message}`, true);
    return;
  }
  const evaluationAction = runEvaluationAction(run);
  if (!evaluationAction.enabled) {
    showToast(evaluationAction.reason, true);
    return;
  }

  clearNotice(elements.evaluationsError);
  elements.evaluationRunId.value = "";
  elements.evaluationCheckpoint.value = "";
  elements.evaluationSuite.value = "";
  pendingEvaluationSuiteId = "";
  updateEvaluationEnvironmentFromSuite();
  scheduleEvaluationTargetValidation();

  elements.evaluationRunId.value = requestedRunId;
  elements.evaluationCheckpoint.value = evaluationAction.checkpointPath;
  scheduleEvaluationTargetValidation({ immediate: true });
  evaluationSuitesCache.delete(evaluationSuiteScope(requestedRunId));
  activateTab("evaluations", true, "submit");
  await loadEvaluationSuites(false, requestedRunId);
  if (request !== evaluationPrefillRequest) return;

  const suite = preferredEvaluationSuite(run || {}) || null;
  pendingEvaluationSuiteId = suite ? evaluationSuiteId(suite) : "";
  elements.evaluationSuite.value = pendingEvaluationSuiteId;
  updateEvaluationEnvironmentFromSuite();
  pendingEvaluationSuiteId = "";
  scheduleEvaluationTargetValidation({ immediate: true });

  const warnings = [];
  if (!suite)
    warnings.push(
      elements.evaluationSuiteStatus.textContent ||
        "No compatible evaluation suite is available for this run.",
    );
  if (warnings.length)
    showNotice(elements.evaluationsError, warnings.join(" "));
  else clearNotice(elements.evaluationsError);
  window.requestAnimationFrame(() => {
    elements.evaluationForm.scrollIntoView({
      block: "nearest",
      inline: "nearest",
    });
    elements.evaluationRunId.focus({ preventScroll: true });
  });
}

async function recoverRunSubmission(id, button) {
  const scope = `run-recovery:${id}`;
  if (submissionRecoveryRequests.has(id)) return;
  if (
    !(await askUserDialog(
      `Recover this unconfirmed submission through ${elements.gateway.value}? The exact saved script and original submission identity will be reused.`,
    ))
  )
    return;
  submissionRecoveryRequests.add(id);
  button.disabled = true;
  button.textContent = "Recovering submission…";
  try {
    const result = await api(
      `/api/runs/${encodeURIComponent(id)}/recover-submission`,
      {
        method: "POST",
        body: JSON.stringify({ gateway: elements.gateway.value }),
      },
    );
    await loadRuns(true);
    if (activeRunDetailId === id)
      startRunDetailPolling(id, null, { initialDelay: 0, includeList: false });
    showToast(
      `Submission recovered: Slurm job ${result.slurm_job_id}.`,
      false,
      { scope },
    );
  } catch (error) {
    showNotice(
      elements.runsError,
      `Submission recovery failed: ${error.message}`,
      { scope },
    );
  } finally {
    submissionRecoveryRequests.delete(id);
    if (activeRunDetailId === id)
      startRunDetailPolling(id, null, { initialDelay: 0, includeList: false });
  }
}

async function resumeRun(id, reason = "") {
  if (runResumeRequests.has(id)) return;
  let submitted = false;
  const scope = `run-resume:${id}`;
  const actionLabel = /no usable|from scratch|initial pinned/i.test(reason)
    ? "Restart training from beginning / new attempt"
    : "Resume Training Run / new attempt";
  const explanation = reason.trim() ? `\n\n${reason.trim()}` : "";
  runResumeRequests.add(id);
  try {
    if (!(await askUserDialog(`${actionLabel} for run ${id}?${explanation}`)))
      return;
    const button = elements.runDetailActions.querySelector(
      '[data-run-action="resume"]',
    );
    if (button?.dataset.id === id) {
      button.disabled = true;
      button.textContent = "Submitting new attempt…";
    }
    const result = await api(`/api/runs/${encodeURIComponent(id)}/resume`, {
      method: "POST",
      body: JSON.stringify({ mode: "resume", gateway: elements.gateway.value }),
    });
    submitted = true;
    clearNotificationScope(scope);
    resetTrainingRunListFilters();
    await loadRuns(true);
    if (activeRunDetailId === id && !elements.runDetail.hidden) {
      stopRunDetailPolling({ clearStatus: false });
      const payload = await api(`/api/runs/${encodeURIComponent(id)}`);
      if (activeRunDetailId === id && !elements.runDetail.hidden) {
        runDetailFollowLatest = true;
        const detail = renderRunDetail(payload, id, { preserveAttempt: true });
        if (activeRunAttemptDisclosure && detail.latestRecord) {
          const latest = detail.latestRecord;
          const launcher = elements.attemptsBody.querySelector(
            `[data-attempt-key="${CSS.escape(latest.attemptKey)}"] [data-attempt-action="view"]`,
          );
          await openRunAttemptDetail(latest.attemptKey, launcher, {
            focus: false,
            followLatest: true,
          });
        }
        startRunDetailPolling(id, detail.state);
      }
    } else {
      const launcher = elements.runsBody.querySelector(
        `[data-run-action="view"][data-id="${CSS.escape(id)}"]`,
      );
      await viewRun(id, launcher);
    }
    const feedback = submissionFeedback(result);
    if (feedback.error)
      showNotice(elements.runsError, `Attempt failed: ${feedback.message}`, {
        scope,
      });
    else showToast(feedback.message, false, { scope });
  } catch (error) {
    showToast(
      submitted
        ? `Attempt submitted, but its display could not refresh: ${error.message}. Use Refresh to check its status.`
        : `Resume failed: ${error.message}`,
      true,
      { scope },
    );
  } finally {
    runResumeRequests.delete(id);
    if (activeRunDetailId === id)
      startRunDetailPolling(id, null, { initialDelay: 0 });
  }
}

function cancellationRequestKey(kind, id) {
  return `${kind}:${id}`;
}

function cancellationActionButton(kind, id, manualActions, options = {}) {
  const metadata = manualActions?.cancel;
  if (!metadata || typeof metadata !== "object") return "";
  const actionEnabled = metadata.enabled === true;
  const reason =
    typeof metadata.reason === "string" && metadata.reason.trim()
      ? metadata.reason.trim()
      : actionEnabled
        ? ""
        : "Cancellation is not available for this job.";
  const pending = cancellationRequests.has(cancellationRequestKey(kind, id));
  const label =
    options.label ||
    (kind === "run" ? "Cancel Training Run" : "Cancel evaluation");
  const visibleLabel = pending ? "Cancellation requested..." : label;
  const accessibleLabel = pending
    ? `${label}: cancellation request in progress`
    : reason
      ? `${label}: ${reason}`
      : label;
  return `<button class="${escapeHtml(options.className ?? "button button-danger")}" type="button" data-cancel-action data-cancel-kind="${escapeHtml(kind)}" data-cancel-action-enabled="${actionEnabled}" data-cancel-label="${escapeHtml(label)}" data-cancel-reason="${escapeHtml(reason)}" data-id="${escapeHtml(id)}" title="${escapeHtml(reason)}" aria-label="${escapeHtml(accessibleLabel)}"${pending ? ' aria-busy="true"' : ""}${actionEnabled && !pending ? "" : " disabled"}>${escapeHtml(visibleLabel)}</button>`;
}

function restoreCancellationButton(kind, id) {
  document
    .querySelectorAll(
      `[data-cancel-kind="${CSS.escape(String(kind))}"][data-id="${CSS.escape(String(id))}"]`,
    )
    .forEach((button) => {
      button.disabled = button.dataset.cancelActionEnabled !== "true";
      button.textContent = button.dataset.cancelLabel || "Cancel";
      button.removeAttribute("aria-busy");
    });
}

async function requestCancellation(kind, id, button, reason = "") {
  const key = cancellationRequestKey(kind, id);
  const notificationScope = `cancel:${key}`;
  if (button.disabled || cancellationRequests.has(key)) return;
  const subject = kind === "run" ? "Training Run" : "evaluation";
  const explanation = reason.trim() ? `\n\n${reason.trim()}` : "";
  const confirmed = await askUserDialog(
    `Request cancellation for ${subject} ${id}?${explanation}\n\nRecorded progress, results, and logs will be retained.`,
  );
  if (!confirmed) return;

  cancellationRequests.add(key);
  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  button.textContent = "Cancellation requested...";
  try {
    const payload = await api(
      kind === "run"
        ? `/api/runs/${encodeURIComponent(id)}/cancel`
        : `/api/evaluations/${encodeURIComponent(id)}/cancel`,
      { method: "POST" },
    );
    cancellationRequests.delete(key);
    showToast(`Cancellation requested for ${subject} ${id}.`, false, {
      scope: notificationScope,
    });
    if (kind === "run") {
      const returnedRun = payload?.run;
      if (returnedRun && typeof returnedRun === "object") {
        const index = runRows.findIndex(
          (run) => String(run.id || run.run_id) === String(id),
        );
        if (index >= 0) {
          runRows[index] = { ...runRows[index], ...returnedRun };
          renderRuns({ background: true });
        }
      }
      if (activeRunDetailId === String(id) && !elements.runDetail.hidden) {
        startRunDetailPolling(
          id,
          returnedRun?.status || returnedRun?.state || "CANCELLING",
          { initialDelay: 0 },
        );
      }
      return;
    }

    const returnedEvaluation = payload?.evaluation;
    if (returnedEvaluation && typeof returnedEvaluation === "object") {
      updateEvaluationRow(returnedEvaluation, { background: true });
    }
    scheduleEvaluationListPolling(0);
    if (
      activeEvaluationDetailId === String(id) &&
      !elements.evaluationDetail.hidden
    ) {
      const launcher =
        activeDisclosure?.panel === elements.evaluationDetail
          ? activeDisclosure.launcher
          : null;
      await viewEvaluation(id, launcher, { polling: true });
    }
  } catch (error) {
    cancellationRequests.delete(key);
    restoreCancellationButton(kind, id);
    showToast(`Cancellation request failed: ${error.message}`, true, {
      scope: notificationScope,
    });
  }
}

function handleCancellationAction(event) {
  const button = event.target.closest("[data-cancel-action]");
  if (!button) return;
  requestCancellation(
    button.dataset.cancelKind,
    button.dataset.id,
    button,
    button.dataset.cancelReason || "",
  );
}

async function rerunRun(id, reason = "") {
  const scope = `run-rerun:${id}`;
  const actionLabel = "Start new Training Run from pinned variant";
  const explanation = reason.trim() ? `\n\n${reason.trim()}` : "";
  if (!(await askUserDialog(`${actionLabel} for run ${id}?${explanation}`)))
    return;
  try {
    const result = await api(`/api/runs/${encodeURIComponent(id)}/rerun`, {
      method: "POST",
      body: JSON.stringify({ gateway: elements.gateway.value }),
    });
    const run = entityFrom(result, "run");
    const newRunId = run.id || run.run_id || result.id || result.run_id;
    showToast(
      `New Training Run${newRunId ? ` ${newRunId}` : ""} created from the pinned variant of ${id}.`,
      false,
      { scope },
    );
    await loadRuns(true);
    if (newRunId) {
      const launcher = [
        ...elements.runsBody.querySelectorAll("[data-run-action='view']"),
      ].find((candidate) => String(candidate.dataset.id) === String(newRunId));
      if (launcher) await viewRun(newRunId, launcher);
    }
  } catch (error) {
    showToast(`New Training Run failed: ${error.message}`, true, { scope });
  }
}

function evaluationPrimaryResult(evaluation) {
  const aggregate = evaluation.aggregate || [];
  const successRate = aggregate.find(
    (metric) => metric.metric === "success_rate" && !metric.task,
  );
  if (successRate)
    return `Success rate: ${(Number(successRate.mean) * 100).toFixed(1)}%`;
  const primaryMetric = aggregate.find((metric) => metric.mean !== undefined);
  if (primaryMetric)
    return `${primaryMetric.metric}: ${Number(primaryMetric.mean).toPrecision(4)}`;
  const result =
    evaluation.result || evaluation.results || evaluation.metrics || {};
  if (evaluation.primary_result !== undefined) return evaluation.primary_result;
  for (const key of ["success_rate", "score", "mean_reward", "accuracy"]) {
    if (key === "success_rate" && result[key] !== undefined)
      return `Success rate: ${(Number(result[key]) * 100).toFixed(1)}%`;
    if (result[key] !== undefined) return `${key}: ${result[key]}`;
  }
  return Object.keys(result).length
    ? compactJson(result, 80)
    : "No result recorded";
}

function evaluationRowCells(evaluation) {
  const id = evaluation.id || evaluation.evaluation_id;
  const state = jobStatusLabel(evaluation);
  const suite =
    evaluation.suite_label ||
    evaluation.suite_name ||
    evaluation.suite_id ||
    evaluation.suite ||
    "-";
  const environment =
    evaluation.environment || evaluation.evaluator_adapter || "-";
  const progressSummary = evaluation.progress_summary;
  const completed = Number(
    progressSummary?.completed ??
      evaluation.progress_completed ??
      evaluation.completed_episodes,
  );
  const total = Number(
    progressSummary?.total ??
      evaluation.progress_total ??
      evaluation.total_episodes,
  );
  const progress =
    progressSummary ||
    (Number.isFinite(completed) && Number.isFinite(total) && total > 0
      ? completed / total
      : (evaluation.progress ?? evaluation.completed_episodes_ratio));
  const progressLabel =
    progressSummaryLabel(progressSummary) ||
    evaluation.progress_label ||
    (Number.isFinite(completed) && Number.isFinite(total)
      ? `${completed}/${total}`
      : null);
  return [
    `<span class="node-name">${escapeHtml(evaluationName(evaluation))}</span>`,
    `${valueHtml(linkedValue("run", evaluation.run_id, trainingRunName(evaluation)))}<span class="secondary">${valueHtml(linkedValue("run", evaluation.run_id, evaluation.checkpoint_path?.split("/").pop() || "Checkpoint details", { checkpoint: evaluation.checkpoint_id }))}</span>`,
    `${valueHtml(linkedValue("suite", evaluation.evaluation_suite_id || evaluation.suite_id, suite))}<span class="secondary">${escapeHtml(environment)}</span>`,
    statusPill(state),
    executionHtml(evaluation),
    evaluation.status_detail
      ? escapeHtml(evaluation.status_detail)
      : `${state.toUpperCase() === "RUNNING" ? `<div class="progress-eta">ETA ${etaCell(progressSummary)}</div>` : ""}${progressCell(progress, progressLabel)}`,
    escapeHtml(evaluationPrimaryResult(evaluation)),
    escapeHtml(
      formatDate(
        evaluation.latest_attempt?.started_at ||
          latestEvaluationAttempt(evaluation)?.started_at,
      ),
    ),
    escapeHtml(
      progressSummary?.elapsed_seconds == null
        ? "—"
        : compactEtaDuration(progressSummary.elapsed_seconds),
    ),
  ];
}

function evaluationLifecycleAction(evaluation, id) {
  const state = String(
    evaluation.status || evaluation.state || "",
  ).toUpperCase();
  if (!EVALUATION_ACTIVE_STATES.has(state)) {
    return `<button type="button" data-delete-kind="evaluation" data-delete-id="${escapeHtml(id)}">Delete</button>`;
  }
  const cancelling = state === "CANCELLING";
  const actions = cancelling
    ? {
        cancel: {
          enabled: false,
          reason: "Cancellation has already been requested.",
        },
      }
    : evaluation.manual_actions || { cancel: { enabled: true } };
  return cancellationActionButton("evaluation", id, actions, {
    label: cancelling ? "Cancelling…" : "Cancel",
    className: "",
  });
}

function patchEvaluationRow(row, descriptor) {
  patchTableRow(row, descriptor.cells);
  // Keep the open Results launcher while updating the adjacent lifecycle action.
  setHtmlIfChanged(
    row.querySelector("[data-evaluation-lifecycle]"),
    descriptor.lifecycleAction,
  );
}

function evaluationRowDescriptor(evaluation) {
  const id = String(evaluation.id || evaluation.evaluation_id || "");
  const active =
    activeDisclosure?.panel === elements.evaluationDetail &&
    activeEvaluationDetailId === id;
  const lifecycleAction = evaluationLifecycleAction(evaluation, id);
  return {
    id,
    lifecycleAction,
    cells: [
      ...evaluationRowCells(evaluation).map((html) => ({ html })),
      {
        className: "row-actions",
        preserve: active,
        html: `<button type="button" data-evaluation-action="view" data-id="${escapeHtml(id)}">Results</button><span data-evaluation-lifecycle>${lifecycleAction}</span>`,
      },
    ],
  };
}

function patchEvaluationSummaryRow(evaluation) {
  const descriptor = evaluationRowDescriptor(evaluation);
  const row = elements.evaluationsBody.querySelector(
    `:scope > tr[data-evaluation-id="${CSS.escape(descriptor.id)}"]`,
  );
  if (!row) return false;
  patchEvaluationRow(row, descriptor);
  return true;
}

function renderEvaluations({ background = false } = {}) {
  const stateFilter = document.querySelector("#evaluation-state-filter");
  const knownStates = new Set(
    [...stateFilter.options].map((option) => option.value),
  );
  for (const row of evaluationRows) {
    const status = jobStatusLabel(row).toUpperCase();
    if (!status || knownStates.has(status)) continue;
    const option = document.createElement("option");
    option.value = status;
    option.textContent = status;
    stateFilter.append(option);
    knownStates.add(status);
  }
  const query = document
    .querySelector("#evaluation-search")
    .value.trim()
    .toLowerCase();
  const filter = document.querySelector("#evaluation-state-filter").value;
  const filtered = evaluationRows.filter(
    (row) =>
      (!query ||
        [
          row.id,
          row.run_id,
          row.experiment_name,
          row.suite_name,
          row.suite_id,
          row.latest_attempt?.slurm_job_id,
        ]
          .join(" ")
          .toLowerCase()
          .includes(query)) &&
      (filter === "all" || jobStatusLabel(row).toUpperCase() === filter),
  );
  const descriptors = filtered.map(evaluationRowDescriptor);
  const countLabel = `${filtered.length} of ${evaluationRows.length} evaluations`;
  const panel =
    elements.evaluationsBody.closest(".panel") || elements.evaluationsBody;
  commitPanelRefresh(
    panel,
    "evaluations-list",
    {
      countLabel,
      rows: descriptors.map(({ id, cells }) => ({
        id,
        cells: cells.map(({ html, className }) => ({ html, className })),
      })),
    },
    () => {
      setTextIfChanged(elements.evaluationCount, countLabel);
      const existingRows = new Map(
        [
          ...elements.evaluationsBody.querySelectorAll(
            ":scope > tr[data-evaluation-id]",
          ),
        ].map((row) => [String(row.dataset.evaluationId), row]),
      );
      const visibleIds = new Set(descriptors.map(({ id }) => id));
      if (
        activeDisclosure?.panel === elements.evaluationDetail &&
        !visibleIds.has(String(activeEvaluationDetailId))
      ) {
        closeActiveDisclosure({ restoreFocus: false });
      }
      const orderedRows = descriptors.map((descriptor) => {
        const row =
          existingRows.get(descriptor.id) || document.createElement("tr");
        row.dataset.evaluationId = descriptor.id;
        patchEvaluationRow(row, descriptor);
        existingRows.delete(descriptor.id);
        return row;
      });
      existingRows.forEach((row) => row.remove());
      elements.evaluationsBody
        .querySelectorAll(
          ":scope > tr:not([data-evaluation-id]):not(.row-disclosure-companion)",
        )
        .forEach((row) => row.remove());
      if (!orderedRows.length) {
        const row = document.createElement("tr");
        row.className = "empty-row";
        patchTableRow(row, [
          { colSpan: 10, html: "No evaluations match the filters." },
        ]);
        elements.evaluationsBody.append(row);
        return;
      }
      const companion = elements.evaluationDetail.closest(
        "tr.row-disclosure-companion",
      );
      const activeRow = activeEvaluationDetailId
        ? orderedRows.find(
            (row) =>
              row.dataset.evaluationId === String(activeEvaluationDetailId),
          )
        : null;
      reconcileTableSequence(
        elements.evaluationsBody,
        orderedRows.flatMap((row) =>
          row === activeRow && companion ? [row, companion] : [row],
        ),
      );
      synchronizeDisclosureLaunchers(elements.evaluationsBody);
    },
    { background },
  );
}

function updateEvaluationRow(evaluation, { background = false } = {}) {
  const id = String(evaluation.id || evaluation.evaluation_id || "");
  const index = evaluationRows.findIndex(
    (row) => String(row.id || row.evaluation_id) === id,
  );
  if (index < 0) return false;
  evaluationRows[index] = { ...evaluationRows[index], ...evaluation };
  const panel =
    elements.evaluationsBody.closest(".panel") || elements.evaluationsBody;
  commitPanelRefresh(
    panel,
    `evaluation-summary:${id}`,
    evaluationRowCells(evaluationRows[index]),
    () => {
      patchEvaluationSummaryRow(evaluationRows[index]);
    },
    { background },
  );
  return true;
}

const EVALUATION_ACTIVE_STATES = new Set([
  "CREATED",
  "SUBMITTING",
  "PENDING",
  "PENDING_SLURM",
  "SUBMITTED",
  "QUEUED",
  "CONFIGURING",
  "RUNNING",
  "REQUEUED",
  "RETRY_PENDING",
  "CANCELLING",
]);
const EVALUATION_LIST_POLL_INTERVAL_MS = 5000;

function revalidateFinishedEvaluation(previous, next) {
  const runId = elements.evaluationRunId.value.trim();
  if (
    previous.some(
      (row) =>
        String(row.run_id) === runId &&
        EVALUATION_ACTIVE_STATES.has(
          String(row.status || row.state).toUpperCase(),
        ),
    ) &&
    !next.some(
      (row) =>
        String(row.run_id) === runId &&
        EVALUATION_ACTIVE_STATES.has(
          String(row.status || row.state).toUpperCase(),
        ),
    )
  ) {
    scheduleEvaluationTargetValidation({ immediate: true });
  }
}

function stopEvaluationListPolling() {
  window.clearTimeout(evaluationListPollTimer);
  evaluationListPollTimer = null;
}

function evaluationListPollEligible() {
  return Boolean(
    document.visibilityState === "visible" &&
      activeTab === "evaluation-runs" &&
      (evaluationProgressRefreshPending ||
        evaluationRows.some((evaluation) =>
          EVALUATION_ACTIVE_STATES.has(
            String(evaluation.status || evaluation.state || "").toUpperCase(),
          ),
        )),
  );
}

function scheduleEvaluationListPolling(
  delay = EVALUATION_LIST_POLL_INTERVAL_MS,
) {
  stopEvaluationListPolling();
  if (!evaluationListPollEligible()) return;
  evaluationListPollTimer = window.setTimeout(async () => {
    evaluationListPollTimer = null;
    if (!evaluationListPollEligible()) return;
    if (!evaluationListPollInFlight) {
      evaluationListPollInFlight = api(
        evaluationProgressRefreshPending
          ? "/api/evaluations?refresh_progress=false"
          : "/api/evaluations",
      )
        .then((payload) => {
          evaluationProgressRefreshPending =
            payload.progress_refresh_pending === true;
          if (
            elements.evaluationsError.textContent.startsWith(
              "Live evaluation update failed",
            )
          )
            clearNotice(elements.evaluationsError);
          const next = listFrom(payload, ["evaluations"]);
          revalidateFinishedEvaluation(evaluationRows, next);
          evaluationRows = next;
          renderEvaluations({ background: true });
        })
        .catch((error) =>
          showNotice(
            elements.evaluationsError,
            `Live evaluation update failed; displayed data may be stale: ${error.message}. Retrying automatically.`,
          ),
        )
        .finally(() => {
          evaluationListPollInFlight = null;
        });
    }
    await evaluationListPollInFlight;
    scheduleEvaluationListPolling();
  }, delay);
}

async function loadEvaluations(force = false) {
  if (loadedTabs.has("evaluations") && !force) {
    scheduleEvaluationListPolling();
    return;
  }
  elements.refreshEvaluations.disabled = true;
  clearNotice(elements.evaluationsError);
  await loadEvaluationSuites(force);
  try {
    const runPayload = await api("/api/runs");
    const choices = listFrom(runPayload, ["runs"]);
    document.querySelector("#evaluation-run-options").innerHTML = choices
      .map(
        (run) =>
          `<option value="${escapeHtml(run.id)}">${escapeHtml(run.experiment_name || run.id)} / ${escapeHtml(run.status)}</option>`,
      )
      .join("");
  } catch (error) {
    showNotice(
      elements.evaluationsError,
      `Training run suggestions could not be loaded: ${error.message}. You can enter a known run ID manually.`,
    );
  }
  try {
    const payload = await api("/api/evaluations");
    evaluationProgressRefreshPending =
      payload.progress_refresh_pending === true;
    const next = listFrom(payload, ["evaluations"]);
    revalidateFinishedEvaluation(evaluationRows, next);
    evaluationRows = next;
    renderEvaluations();
    loadedTabs.add("evaluations");
    scheduleEvaluationListPolling();
  } catch (error) {
    evaluationRows = [];
    elements.evaluationsBody.innerHTML = emptyRow(
      9,
      "Evaluation data could not be loaded.",
    );
    elements.evaluationCount.textContent = "Unavailable";
    showNotice(
      elements.evaluationsError,
      `Evaluation API unavailable: ${error.message}`,
    );
  } finally {
    elements.refreshEvaluations.disabled = false;
  }
}

function evaluationPayload() {
  const seeds = elements.evaluationSeeds.value
    .split(",")
    .map((seed) => seed.trim())
    .filter(Boolean)
    .map(Number);
  const tasks = checkedEvaluationTaskIds();
  const { argv, resumeArgv } = evaluationCommandOverrides();
  return {
    run_id: elements.evaluationRunId.value.trim() || null,
    checkpoint_path: elements.evaluationCheckpoint.value.trim() || null,
    suite_id: elements.evaluationSuite.value,
    environment: elements.evaluationEnvironment.value,
    tasks,
    episodes_per_task: numberOrNull(elements.evaluationEpisodes),
    seeds,
    parallelism: numberOrNull(elements.evaluationParallelism),
    headless: elements.evaluationHeadless.checked,
    resources: evaluationResources(),
    auto_resume: elements.evaluationAutoResume.checked,
    max_attempts: numberOrNull(elements.evaluationMaxAttempts),
    gateway: elements.gateway.value,
    argv,
    resume_argv: resumeArgv,
  };
}

function evaluationCommandOverrides() {
  let argv = [];
  let resumeArgv = [];
  if (elements.evaluationArgv.value.trim()) {
    argv = JSON.parse(elements.evaluationArgv.value);
    if (!Array.isArray(argv) || argv.some((item) => typeof item !== "string")) {
      throw new Error("Evaluator argv must be a JSON list of strings.");
    }
  }
  if (elements.evaluationResumeArgv.value.trim()) {
    resumeArgv = JSON.parse(elements.evaluationResumeArgv.value);
    if (
      !Array.isArray(resumeArgv) ||
      resumeArgv.some((item) => typeof item !== "string")
    ) {
      throw new Error("Resume argv must be a JSON list of strings.");
    }
  }
  return { argv, resumeArgv };
}

async function createEvaluation(event) {
  event.preventDefault();
  if (!elements.evaluationForm.reportValidity()) return;
  if (
    evaluationTargetValidationState.pending ||
    !evaluationTargetValidationState.valid ||
    evaluationTargetValidationState.signature !== evaluationTargetSignature()
  ) {
    scheduleEvaluationTargetValidation({ immediate: true });
    showToast(
      "Wait for a valid training run and checkpoint before creating the evaluation.",
      true,
    );
    return;
  }
  let payload;
  try {
    payload = evaluationPayload();
  } catch (error) {
    showToast(error.message, true);
    return;
  }
  if (!payload.run_id) {
    showToast(
      "Provide a training run ID. External checkpoints must first be registered as a run.",
      true,
    );
    elements.evaluationRunId.focus();
    return;
  }
  if (payload.seeds.some((seed) => !Number.isInteger(seed))) {
    showToast("Seeds must be comma-separated integers.", true);
    elements.evaluationSeeds.focus();
    return;
  }
  evaluationSubmitting = true;
  updateEvaluationSubmitState();
  try {
    const result = await api("/api/evaluations", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    const evaluation = entityFrom(result, "evaluation");
    const evaluationId = evaluation.id || evaluation.evaluation_id;
    if (!evaluationId)
      throw new Error(
        "Evaluation was created but the API returned no evaluation ID.",
      );
    // Submission creates an active stage, so the previous readiness result is stale.
    evaluationLastValidatedAt = 0;
    scheduleEvaluationTargetValidation({ immediate: true });
    showToast(`Evaluation ${evaluationId} created.`);
    document.querySelector("#evaluation-search").value = "";
    document.querySelector("#evaluation-state-filter").value = "all";
    await activateTab("evaluations", true, "runs");
    await loadEvaluations(true);
    const launcher = [
      ...elements.evaluationsBody.querySelectorAll(
        "[data-evaluation-action='view']",
      ),
    ].find(
      (candidate) => String(candidate.dataset.id) === String(evaluationId),
    );
    if (!launcher) {
      showNotice(
        elements.evaluationsError,
        `Evaluation ${evaluationId} was created but is not present in the evaluation list response.`,
      );
    } else {
      clearNotice(elements.evaluationsError);
      launcher.click();
    }
  } catch (error) {
    showToast(`Evaluation creation failed: ${error.message}`, true);
  } finally {
    evaluationSubmitting = false;
    updateEvaluationSubmitState();
  }
}

function stopEvaluationDetailPolling() {
  window.clearTimeout(evaluationDetailPollTimer);
  evaluationDetailPollTimer = null;
  activeEvaluationDetailId = null;
  closeEvaluationRollout();
  const video = document.querySelector("#evaluation-rollout-video");
  if (video && !video.paused) video.pause();
}

function latestEvaluationAttempt(evaluation) {
  const attempts = Array.isArray(evaluation.attempts)
    ? evaluation.attempts
    : [];
  return (
    [...attempts].sort(
      (left, right) =>
        Number(right.attempt_number || 0) - Number(left.attempt_number || 0),
    )[0] || null
  );
}

function attemptFailureDetail(attempt) {
  const status = String(attempt?.status || attempt?.state || "").toUpperCase();
  const failed = ["FAILED", "TIMED_OUT", "PREEMPTED", "CANCELLED"].includes(
    status,
  );
  const value =
    runAttemptValue(attempt, "error", "failure_reason", "error_json") ??
    (failed ? runAttemptValue(attempt, "slurm_reason") : null);
  if (value && typeof value === "object") return compactJson(value, 240);
  return value;
}

let activeEvaluationRollout = null;
let evaluationRolloutGeneration = 0;

function evaluationEpisodeRate(episode) {
  return episode.success === true
    ? "100%"
    : episode.success === false
      ? "0%"
      : "—";
}

function evaluationEpisodeResult(episode) {
  if (typeof episode.success === "boolean")
    return evaluationEpisodeRate(episode);
  const metric = Object.entries(episode.metrics_json || {}).find(
    ([, value]) => typeof value === "number" && Number.isFinite(value),
  );
  return metric ? `${metric[0]}: ${Number(metric[1]).toPrecision(4)}` : "—";
}

function closeEvaluationRollout() {
  window.SkynetEpisodeViewer?.close("rollout-episode-viewer");
  const dialog = document.querySelector("#evaluation-rollout-dialog");
  activeEvaluationRollout = null;
  evaluationRolloutGeneration += 1;
  const video = document.querySelector("#evaluation-rollout-video");
  if (video?.dataset.videoKey) {
    video.pause();
    video.removeAttribute("src");
    delete video.dataset.videoKey;
    video.load();
  }
  if (dialog?.open) SkynetDialog.close(dialog);
}

document
  .querySelector("#evaluation-rollout-dialog")
  .addEventListener("close", closeEvaluationRollout);

function evaluationEpisodeAttempt(evaluation, episode) {
  const job = episode.metrics_json?.slurm_job_id;
  return (
    evaluation.attempts?.find(
      (attempt) => job && String(attempt.slurm_job_id) === String(job),
    ) || latestEvaluationAttempt(evaluation)
  );
}

function renderEvaluationAttempt(attempt) {
  const headings = [
    "Attempt",
    "State",
    "Slurm job",
    "Execution",
    "SSH gateway",
    "Started",
    "Ended",
    "Error",
  ];
  const values = attempt
    ? [
        escapeHtml(attempt.attempt_number ?? "—"),
        statusPill(jobStatusLabel(attempt)),
        escapeHtml(attempt.slurm_job_id || "—"),
        `Queue: ${escapeHtml(attempt.partition_name || "Not assigned")}<span class="secondary">Node: ${escapeHtml(attempt.node_list || attempt.node || "Not allocated")}</span>`,
        escapeHtml(attempt.gateway || "Not recorded"),
        escapeHtml(formatDate(attempt.started_at)),
        escapeHtml(formatDate(attempt.finished_at || attempt.ended_at)),
        escapeHtml(attemptFailureDetail(attempt) || "—"),
      ]
    : [];
  setHtmlIfChanged(
    elements.evaluationAttemptMeta,
    `<table><thead><tr>${headings.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>` +
      (values.length
        ? `<tr>${values.map((v) => `<td>${v}</td>`).join("")}</tr>`
        : '<tr><td colspan="8">No Slurm attempt recorded.</td></tr>') +
      "</tbody></table>",
  );
}

function renderEvaluationRolloutModal(evaluation, episode) {
  const video = document.querySelector("#evaluation-rollout-video");
  const empty = document.querySelector("#evaluation-rollout-empty");
  setTextIfChanged(
    document.querySelector("#evaluation-rollout-title"),
    `Episode ${Number(episode.episode_index) + 1} · ${episode.task}`,
  );
  const key = `${evaluation.id}:${episode.id}:${episode.video_path || ""}`;
  video.hidden = !episode.video_path;
  empty.hidden = Boolean(episode.video_path);
  empty.textContent = !EVALUATION_ACTIVE_STATES.has(
    String(evaluation.status || evaluation.state).toUpperCase(),
  )
    ? "No video was recorded for this episode."
    : "The video will appear when this episode finishes.";
  if (video.dataset.videoKey !== key) {
    video.pause();
    video.dataset.videoKey = key;
    if (episode.video_path)
      video.src = `/api/evaluations/${encodeURIComponent(evaluation.id)}/episodes/${encodeURIComponent(episode.id)}/video`;
    else video.removeAttribute("src");
    video.load();
    window.SkynetEpisodeViewer?.open(
      "rollout-episode-viewer",
      "evaluation-rollout-video",
      `/api/evaluations/${encodeURIComponent(evaluation.id)}/episodes/${encodeURIComponent(episode.id)}`,
    );
  }
  video.onerror = () => {
    empty.hidden = false;
    empty.textContent =
      "Unable to load the video. Close and reopen Detail to retry.";
  };
  const values = [
    ["Task", episode.task],
    ["Episode", Number(episode.episode_index) + 1],
    ["State", episode.status],
    ["Seed", episode.seed],
    ["Success rate", evaluationEpisodeRate(episode)],
    ["Reward", episode.reward ?? "—"],
    ["Steps", episode.episode_length ?? "—"],
    ["Failure", episode.failure_reason || "—"],
    ...Object.entries(episode.metrics_json || {}).filter(
      ([, value]) => typeof value === "number" && Number.isFinite(value),
    ),
  ];
  setHtmlIfChanged(
    document.querySelector("#evaluation-result-summary"),
    "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>" +
      values
        .map(
          ([key, value]) =>
            `<tr><td>${escapeHtml(key)}</td><td>${escapeHtml(value)}</td></tr>`,
        )
        .join("") +
      "</tbody></table>",
  );
  updateLogView(
    elements.evaluationResultJson,
    JSON.stringify(episode, null, 2),
  );
  renderEvaluationAttempt(evaluationEpisodeAttempt(evaluation, episode));
}

async function loadEvaluationRolloutLogs(
  evaluation,
  episode,
  generation,
  { polling = false } = {},
) {
  const base = `/api/evaluations/${encodeURIComponent(evaluation.id)}/episodes/${encodeURIComponent(episode.id)}/logs`;
  const targets = [
    ["stdout", elements.evaluationStdoutLog, elements.evaluationStdoutStatus],
    ["stderr", elements.evaluationStderrLog, elements.evaluationStderrStatus],
  ];
  await Promise.all(
    targets.map(async ([stream, log, status]) => {
      const update = beginLogViewUpdate(log);
      if (!polling) {
        status.textContent = "loading";
        updateLogView(log, `Loading ${stream}…`, { generation: update });
      }
      try {
        const result = await api(`${base}?stream=${stream}`);
        if (
          generation !== evaluationRolloutGeneration ||
          !activeEvaluationRollout
        )
          return;
        const content = logContent(result);
        updateLogView(log, content || `No ${stream} output.`, {
          generation: update,
        });
        status.textContent = logStreamStatus(content, "loaded");
      } catch (error) {
        if (
          generation !== evaluationRolloutGeneration ||
          !activeEvaluationRollout
        )
          return;
        status.textContent = "unavailable";
        updateLogView(log, `Unable to load ${stream}: ${error.message}`, {
          generation: update,
        });
      }
    }),
  );
}

function openEvaluationRollout(evaluation, episode, launcher) {
  closeEvaluationRollout();
  activeEvaluationRollout = {
    evaluationId: String(evaluation.id),
    episodeId: String(episode.id),
  };
  const generation = ++evaluationRolloutGeneration;
  document.querySelector("#evaluation-rollout-detail").hidden = false;
  renderEvaluationRolloutModal(evaluation, episode);
  SkynetDialog.open(document.querySelector("#evaluation-rollout-dialog"), {
    launcher,
  });
  void loadEvaluationRolloutLogs(evaluation, episode, generation);
}

function renderEvaluationRollouts(evaluation) {
  const episodes = evaluation.episodes || [];
  const body = document.querySelector("#evaluation-rollouts-body");
  const loaded = Array.isArray(evaluation.episodes);
  const signature = refreshContentSignature({ loaded, episodes });
  const section = document.querySelector("#evaluation-rollouts");
  setTextIfChanged(
    document.querySelector("#evaluation-rollout-count"),
    loaded
      ? `${episodes.length} episode${episodes.length === 1 ? "" : "s"}`
      : "Loading episodes…",
  );
  if (section.dataset.rolloutSignature !== signature) {
    section.dataset.rolloutSignature = signature;
    const rows = new Map(
      [...body.querySelectorAll("tr[data-episode-id]")].map((row) => [
        row.dataset.episodeId,
        row,
      ]),
    );
    const ordered = episodes.map((episode) => {
      const id = String(episode.id);
      const row = rows.get(id) || document.createElement("tr");
      row.dataset.episodeId = id;
      patchTableRow(row, [
        {
          html: `<strong>Episode ${Number(episode.episode_index) + 1}</strong><span class="secondary">${escapeHtml(episode.task)}</span>`,
        },
        { html: statusPill(episode.status) },
        { html: escapeHtml(episode.seed) },
        { html: escapeHtml(evaluationEpisodeResult(episode)) },
        {
          className: "row-actions",
          preserve: true,
          html: `<button type="button" data-rollout-detail="${escapeHtml(id)}" aria-haspopup="dialog" aria-controls="evaluation-rollout-dialog">Detail</button>`,
        },
      ]);
      return row;
    });
    body.replaceChildren(...ordered);
    if (!episodes.length)
      body.innerHTML = `<tr><td colspan="5">${Array.isArray(evaluation.episodes) ? "No rollouts recorded." : "Loading rollouts…"}</td></tr>`;
  }
  body.onclick = (event) => {
    const button = event.target.closest("[data-rollout-detail]");
    const episode = episodes.find(
      (ep) => String(ep.id) === button?.dataset.rolloutDetail,
    );
    if (episode) openEvaluationRollout(evaluation, episode, button);
  };
  if (activeEvaluationRollout?.evaluationId === String(evaluation.id)) {
    const selected = episodes.find(
      (ep) => String(ep.id) === activeEvaluationRollout.episodeId,
    );
    if (selected) {
      renderEvaluationRolloutModal(evaluation, selected);
      void loadEvaluationRolloutLogs(
        evaluation,
        selected,
        evaluationRolloutGeneration,
        { polling: true },
      );
    } else closeEvaluationRollout();
  }
}

function evaluationDetailPollEligible(id) {
  return Boolean(
    id &&
      document.visibilityState === "visible" &&
      activeTab === "evaluation-runs" &&
      activeEvaluationDetailId === String(id) &&
      activeDisclosure?.panel === elements.evaluationDetail &&
      !elements.evaluationDetail.hidden,
  );
}

function scheduleEvaluationDetailPolling(id) {
  window.clearTimeout(evaluationDetailPollTimer);
  const expectedId = String(id);
  activeEvaluationDetailId = expectedId;
  if (!evaluationDetailPollEligible(expectedId)) return;
  evaluationDetailPollTimer = window.setTimeout(async () => {
    evaluationDetailPollTimer = null;
    if (!evaluationDetailPollEligible(expectedId)) return;
    const launcher =
      activeDisclosure?.panel === elements.evaluationDetail
        ? activeDisclosure.launcher
        : null;
    await viewEvaluation(expectedId, launcher, { polling: true });
  }, 5000);
}

function renderEvaluationDetail(evaluation) {
  const id = evaluation.id || evaluation.evaluation_id;
  patchEvaluationSummaryRow(evaluation);
  setTextIfChanged(elements.evaluationDetailTitle, evaluationName(evaluation));
  setHtmlIfChanged(
    elements.evaluationDetailActions,
    cancellationActionButton("evaluation", id, evaluation.manual_actions || {}),
  );
  setHtmlIfChanged(
    elements.evaluationDetailMeta,
    keyValueHtml([
      ["Evaluation ID", copyValue(id)],
      [
        "Training run",
        linkedValue("run", evaluation.run_id, trainingRunName(evaluation)),
      ],
      [
        "Checkpoint",
        linkedValue(
          "run",
          evaluation.run_id,
          evaluation.checkpoint_path?.split("/").pop() || "View checkpoint",
          { checkpoint: evaluation.checkpoint_id },
        ),
      ],
      ["Checkpoint storage path", copyValue(evaluation.checkpoint_path)],
      [
        "Suite",
        linkedValue(
          "suite",
          evaluation.evaluation_suite_id || evaluation.suite_id,
          evaluation.suite_label || evaluation.suite_name || evaluation.suite,
        ),
      ],
      ["Suite version", evaluation.suite_version],
      ["Environment", evaluation.evaluator_adapter || evaluation.environment],
      ["Tasks", (evaluation.task_selection_json || []).join(", ")],
      ["Seeds", (evaluation.seeds_json || []).join(", ")],
      ["Episodes per task", evaluation.episodes_per_task],
      ...executionEntries(evaluation),
      ["Result storage path", copyValue(evaluation.result_path)],
      ["State", jobStatusLabel(evaluation)],
      [
        "Progress",
        evaluation.status_detail ||
          progressSummaryLabel(evaluation.progress_summary) ||
          "Waiting",
      ],
      ...(evaluation.state === "RUNNING" || evaluation.status === "RUNNING"
        ? [["ETA", etaPresentation(evaluation.progress_summary).detail]]
        : []),
      ["Task result", evaluationPrimaryResult(evaluation)],
      [
        "Started",
        formatDate(
          evaluation.latest_attempt?.started_at ||
            latestEvaluationAttempt(evaluation)?.started_at,
        ),
      ],
      [
        "Elapsed",
        evaluation.progress_summary?.elapsed_seconds == null
          ? "—"
          : compactEtaDuration(evaluation.progress_summary.elapsed_seconds),
      ],
      ["Updated", formatDate(evaluation.updated_at || evaluation.created_at)],
    ]),
  );
  renderEvaluationRollouts(evaluation);
}

async function viewEvaluation(id, launcher = null, { polling = false } = {}) {
  if (polling && !evaluationDetailPollEligible(id)) return;
  const revealLauncher = launcher || currentRevealLauncher();
  const requestToken = disclosureToken(
    elements.evaluationDetail,
    revealLauncher,
  );
  let evaluation = evaluationRows.find(
    (row) => String(row.id || row.evaluation_id) === String(id),
  );
  if (!evaluation) return;
  if (!polling) {
    if (activeEvaluationDetailId !== String(id)) closeEvaluationRollout();
    activeEvaluationDetailId = String(id);
    renderEvaluationDetail(evaluation);
    revealPanel(elements.evaluationDetail, {
      focusTarget: elements.evaluationDetailTitle,
      launcher: revealLauncher,
    });
  }
  const errorBox = document.querySelector("#evaluation-detail-error");
  const current = () =>
    disclosureTokenIsCurrent(elements.evaluationDetail, requestToken) &&
    activeEvaluationDetailId === String(id) &&
    !elements.evaluationDetail.hidden;
  try {
    const payload = await api(`/api/evaluations/${encodeURIComponent(id)}`);
    if (!current()) return;
    evaluation = entityFrom(payload, "evaluation");
    const index = evaluationRows.findIndex(
      (row) => String(row.id || row.evaluation_id) === String(id),
    );
    if (index >= 0) {
      revalidateFinishedEvaluation([evaluationRows[index]], [evaluation]);
      evaluationRows[index] = { ...evaluationRows[index], ...evaluation };
    }
    errorBox.hidden = true;
    renderEvaluationDetail(evaluation);
  } catch (error) {
    if (!current()) return;
    errorBox.textContent = `Unable to refresh details: ${error.message}. Retrying…`;
    errorBox.hidden = false;
  }
  if (current()) scheduleEvaluationDetailPolling(id);
}

function dataResourceIdentity(resource) {
  if (!resource) return "unknown resource";
  return (
    [
      resource.provider,
      [resource.namespace, resource.name].filter(Boolean).join("/"),
    ]
      .filter(Boolean)
      .join(":") ||
    resource.id ||
    resource.resource_id ||
    "unknown resource"
  );
}

function dataVersionResource(version) {
  if (version?.resource) return version.resource;
  return (
    version?._resource ||
    dataResourceRows.find(
      (resource) =>
        String(resource.id || resource.resource_id) ===
        String(version?.resource_id),
    )
  );
}

function dataVersionLabel(version) {
  if (!version) return "unknown version";
  const identity = dataResourceIdentity(dataVersionResource(version));
  return `${identity}@${shortId(version.revision || version.id || version.version_id, 14)}`;
}

function formatDataBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes < 0) return "-";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let amount = bytes;
  let unit = -1;
  do {
    amount /= 1024;
    unit += 1;
  } while (amount >= 1024 && unit < units.length - 1);
  return `${amount.toFixed(amount >= 10 ? 1 : 2)} ${units[unit]}`;
}

function trainingDatasetLabel(dataset) {
  const metadata = dataset.assignments?.[0]?.version?.metadata || {};
  const episodes = Array.isArray(metadata.episodes)
    ? metadata.episodes.length
    : metadata.num_episodes || metadata.episodes;
  return [
    dataset.name,
    dataset.format,
    episodes != null
      ? `${episodes} episode${Number(episodes) === 1 ? "" : "s"}`
      : null,
    formatDate(dataset.created_at),
  ]
    .filter(Boolean)
    .join(" · ");
}

function datasetAlgorithmCompatibility(dataset, adapter) {
  const slot = experimentInputSlots(adapter).find(
    (item) => item.role === "training_data",
  );
  if (!slot)
    return {
      compatible: false,
      message: "This algorithm does not accept a registered training dataset.",
    };
  return experimentBundleCompatibility(
    {
      ...dataset,
      assignments: dataset.assignments.map((item) => ({
        ...item,
        role: slot.role,
        position: slot.position,
      })),
    },
    adapter,
    slot.bindings,
  );
}
function updateAlgorithmCompatibility() {
  const dataset = trainingDatasetRows.find(
    (row) => row.id === elements.experimentDataBundle.value,
  );
  for (const option of elements.experimentAdapter.options) {
    const adapter = [...adapterRows, loadedExperimentAdapterSnapshot]
      .filter(Boolean)
      .find((item) => adapterOptionId(item) === option.value);
    if (!adapter) continue;
    const result = dataset
      ? datasetAlgorithmCompatibility(dataset, adapter)
      : { compatible: true };
    option.disabled = !result.compatible;
    option.dataset.label ||= option.textContent;
    option.textContent =
      option.dataset.label + (result.compatible ? "" : " — " + result.message);
  }
  const chosen = elements.experimentAdapter.selectedOptions[0];
  const invalid = Boolean(dataset && chosen?.disabled);
  elements.experimentAdapter.setCustomValidity(
    invalid ? "Choose an algorithm compatible with the selected dataset" : "",
  );
  elements.experimentAdapter.setAttribute("aria-invalid", String(invalid));
}

function populateExperimentDataBundles() {
  const slots = experimentInputSlots(),
    adapter = selectedAdapter();
  const container = document.getElementById("experiment-extra-data-inputs");
  const previous = new Map(
    [...container.querySelectorAll("select")].map((e) => [
      e.dataset.inputSlot,
      e.value,
    ]),
  );
  previous.set(
    elements.experimentDataBundle.dataset.inputSlot || slots[0]?.key,
    elements.experimentDataBundle.value,
  );
  container.replaceChildren();
  if (!slots.length)
    slots.push({
      key: JSON.stringify(["training_data", 0]),
      role: "training_data",
      position: 0,
      bindings: [],
    });
  slots.forEach((slot, index) => {
    const label =
      slot.role === "training_data"
        ? "Training dataset"
        : slot.role.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());
    let control = elements.experimentDataBundle;
    if (index) {
      const field = document.createElement("div");
      field.className = "field span-4";
      const text = document.createElement("label");
      text.htmlFor = "experiment-data-input-" + index;
      text.textContent = label + " " + (slot.position + 1);
      control = document.createElement("select");
      control.id = text.htmlFor;
      field.append(text, control);
      container.append(field);
      control.onchange = () => {
        populateExperimentDataBundles();
        invalidateExperimentPreview();
        refreshExperimentModelIO();
      };
    } else
      document.querySelector('[for="experiment-data-bundle"]').textContent =
        label;
    control.dataset.inputSlot = slot.key;
    control.disabled = false;
    const value = previous.get(slot.key) || "";
    const options = trainingDatasetRows.map((dataset) => {
      const input = {
        ...dataset,
        assignments: dataset.assignments.map((a) => ({
          ...a,
          role: slot.role,
          position: slot.position,
        })),
      };
      return {
        dataset,
        ...experimentBundleCompatibility(input, adapter, slot.bindings),
      };
    });
    const missing = value && !options.some((o) => o.dataset.id === value);
    control.innerHTML =
      '<option value="">Choose ' +
      escapeHtml(label.toLowerCase()) +
      "…</option>" +
      (missing
        ? `<option value="${escapeHtml(value)}" disabled>Selected files are no longer registered</option>`
        : "") +
      options
        .map(
          ({ dataset, compatible, message }) =>
            `<option value="${escapeHtml(dataset.id)}" ${!compatible && index > 0 ? "disabled" : ""}>${escapeHtml(trainingDatasetLabel(dataset))}${compatible ? "" : " — " + escapeHtml(message)}</option>`,
        )
        .join("");
    control.value = value;
    const invalid =
      missing ||
      options.find((o) => o.dataset.id === value)?.compatible === false;
    control.setCustomValidity(
      invalid ? "Choose compatible registered files" : "",
    );
    control.setAttribute("aria-invalid", String(Boolean(invalid)));
  });
  updateAlgorithmCompatibility();
  const chosen = trainingDatasetRows.find(
    (row) => row.id === elements.experimentDataBundle.value,
  );
  const compatibility = chosen
    ? datasetAlgorithmCompatibility(chosen, adapter)
    : null;
  elements.experimentDataBundleStatus.textContent =
    compatibility && !compatibility.compatible
      ? "Choose a compatible algorithm below. " + compatibility.message
      : "The selected files and their source information are saved with the experiment.";
  if (renderedAdapterDeclaredScope) renderAdapterDeclaredFields();
}

function experimentBundleCompatibility(
  bundle,
  adapter = selectedAdapter(),
  selectedBindings = null,
) {
  const bindings =
    selectedBindings ||
    declaredAdapterInputFields(adapter)
      .map((field) => field.data_binding)
      .filter(Boolean);
  if (!bindings.length)
    return {
      compatible: false,
      message: "Unsupported: this adapter has no training dataset binding.",
    };
  const assignments = Array.isArray(bundle?.assignments)
    ? bundle.assignments
    : [];
  const positions = new Set();
  for (const item of assignments) {
    const role = String(item?.role || "");
    const position = Number(item?.position || 0);
    const identity = JSON.stringify([role, position]);
    if (positions.has(identity))
      return {
        compatible: false,
        message: `duplicate ${role} position ${position}`,
      };
    positions.add(identity);
    if (
      bindings.some((binding) => binding.role === role) &&
      !bindings.some(
        (binding) =>
          binding.role === role && Number(binding.position || 0) === position,
      )
    ) {
      return {
        compatible: false,
        message: `adapter cannot consume ${role} position ${position}`,
      };
    }
  }
  for (const binding of bindings) {
    const assignment = assignments.find(
      (item) =>
        String(item?.role || "") === String(binding.role || "") &&
        Number(item?.position || 0) === Number(binding.position || 0),
    );
    if (!assignment)
      return { compatible: false, message: `missing ${binding.role} role` };
    const location = assignment.config?.location;
    const locationReady =
      location?.kind === "cluster" &&
      location?.status === "AVAILABLE" &&
      location?.manifest_sha256 === assignment.version?.manifest_sha256;
    const contracts = adapterDataContracts(binding, adapter);
    const metadata = assignment.version?.metadata || {};
    if (
      contracts.length &&
      (!contracts.includes(metadata.contract) ||
        metadata.validation?.status !== "PASSED")
    )
      return {
        compatible: false,
        message: "Dataset has not passed this policy's data requirements.",
      };
    if (binding.value_path === "location.path" && !locationReady)
      return {
        compatible: false,
        message: "No verified copy on the training cluster.",
      };
    if (
      String(assignment.version?.status || "").toUpperCase() !== "READY" &&
      !locationReady
    ) {
      return {
        compatible: false,
        message: `${binding.role} is ${assignment.version?.status || "unverified"}; complete import or transfer to the cluster`,
      };
    }
    if (assignment.version?.metadata?.storage_location === "workstation") {
      return {
        compatible: false,
        message:
          "Stored on the collection workstation; transfer to the training cluster first.",
      };
    }
    const formats = Array.isArray(binding.formats)
      ? binding.formats.map((item) => String(item).toLowerCase())
      : [];
    const format = String(assignment.version?.format || "");
    if (formats.length && !formats.includes(format.toLowerCase())) {
      return { compatible: false, message: `format ${format || "undeclared"}` };
    }
    const value = datasetBindingValue(assignment, binding);
    if (!value)
      return {
        compatible: false,
        message: `missing ${binding.value_path || "version.path"}`,
      };
  }
  return { compatible: true, message: "" };
}

async function loadDataBundles(force = false) {
  if (trainingDatasetRows.length && !force) {
    populateExperimentDataBundles();
    return;
  }
  elements.experimentDataBundle.disabled = true;
  elements.experimentDataBundleStatus.textContent =
    "Loading prepared datasets…";
  try {
    const payload = await api("/api/data/selections");
    trainingDatasetRows = listFrom(payload, ["datasets"]);
    populateExperimentDataBundles();
  } catch (error) {
    elements.experimentDataBundleStatus.textContent =
      "Could not load datasets: " + error.message;
    elements.experimentDataBundle.setCustomValidity(
      "Dataset selection could not be verified",
    );
  }
}

function dataVersionStatus(version) {
  return version?.status || "READY";
}

function dataResourceTypeLabel(resource) {
  return (
    dataResourceTypes[resource.category]?.[resource.kind] ||
    resource.kind ||
    "—"
  );
}

function dataFormatLabel(format) {
  const value = String(format || "");
  if (/zarr/i.test(value)) return "Zarr";
  if (/hdf5|h5/i.test(value)) return "HDF5";
  if (/lerobot/i.test(value)) return "LeRobot";
  if (/parquet/i.test(value)) return "Parquet";
  return value || "Unknown format";
}

function dataFormatsLabel(resource) {
  const counts = new Map();
  for (const version of resource.versions || []) {
    if (version.format === "skynet.episodes/v1") continue;
    const label = dataFormatLabel(version.format);
    counts.set(label, (counts.get(label) || 0) + 1);
  }
  return [...counts]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([label, count]) => `${label} × ${count}`)
    .join(" · ");
}

function dataResourceActivity(resourceId) {
  const finished = new Set([
    "READY",
    "SUCCEEDED",
    "COMPLETED",
    "CANCELLED",
    "DELETED",
  ]);
  const imports = dataImportRows.map((job) => ({
    ...job,
    activityKind: "Import",
  }));
  const conversions = dataPreparationRows.map((job) => ({
    ...job,
    activityKind: "Conversion",
  }));
  return [...imports, ...conversions]
    .filter((job) => job.resource_id === resourceId && !finished.has(job.state))
    .sort((a, b) =>
      String(b.created_at || "").localeCompare(String(a.created_at || "")),
    )
    .map((job) => {
      const failed = ["FAILED", "DELETE_FAILED"].includes(job.state);
      const label = failed
        ? `${job.activityKind} failed`
        : `${job.activityKind}: ${String(job.state || "pending")
            .toLowerCase()
            .replaceAll("_", " ")}`;
      const action =
        job.activityKind === "Import" ? "import-detail" : "dataset";
      const id = job.activityKind === "Import" ? job.id : resourceId;
      return `<span class="secondary"><button type="button" class="text-button${failed ? " danger-text" : ""}" data-resource-action="${action}" data-id="${escapeHtml(id)}"${job.error ? ` title="${escapeHtml(job.error)}"` : ""}>${escapeHtml(label)}</button></span>`;
    })
    .join("");
}

document.addEventListener("dataset-preparation-changed", (event) => {
  dataPreparationRows = Array.isArray(event.detail) ? event.detail : [];
  renderDataResources();
});

function dataVersionPath(version) {
  const locations = version.locations || [];
  const available = locations.filter(
    (location) => location.status === "AVAILABLE" && location.path,
  );
  const location =
    available.find((location) => location.kind === "cluster") || available[0];
  return (
    location?.path ||
    (locations.length ? "No available copy" : version.path || "-")
  );
}

// Explicit recording ownership takes precedence over the original source session
// for resources made from a separate single-episode recording entry.
function resourceRecordingId(resource) {
  if (resource.category !== "dataset") return "";
  const metadata = resource.metadata || {};
  return String(
    metadata.recording_session_id ||
      (resource.provider === "collection" || metadata.managed_dataset
        ? metadata.session_id
        : "") ||
      "",
  );
}

function resourceRecordingIds(resource) {
  if (resource.category !== "dataset") return [];
  if (Array.isArray(resource.recording_ids))
    return [...new Set(resource.recording_ids)];
  const id = resourceRecordingId(resource);
  return id ? [id] : [];
}

async function showDatasetRecordings(resourceId) {
  const resource = dataResourceRows.find(
    (r) => String(r.id || r.resource_id) === String(resourceId),
  );
  if (!resource) return;
  await activateTab("data", true, "recording");
  window.filterSimulationRecordings(
    resourceRecordingIds(resource),
    resource.metadata?.display_name || resource.name,
  );
  document
    .getElementById("simulation-recordings-search")
    .focus({ preventScroll: true });
}

function recordingRegistrationSummary(recordingId) {
  const resources = dataResourceRows.filter((resource) =>
    resourceRecordingIds(resource).includes(String(recordingId)),
  );
  return {
    state: dataResourceCatalogState,
    count: new Set(
      resources.map((resource) => resource.id || resource.resource_id),
    ).size,
  };
}

function notifyRecordingRegistryChanged() {
  document.dispatchEvent(new CustomEvent("recording-registry-changed"));
}

async function showRecordingResources(recordingId) {
  await activateTab("data", true, "registry");
  const search = document.getElementById("data-resource-search");
  search.value = recordingId;
  search.dataset.recordingId = recordingId;
  delete search.dataset.resourceIds;
  // Include archived entries when needed so the list matches the linked count.
  document.getElementById("data-show-archived").checked = dataResourceRows.some(
    (resource) =>
      resourceRecordingIds(resource).includes(recordingId) &&
      resource.archived_at,
  );
  refreshDataResourceTables();
  document
    .getElementById("data-resource-create-details")
    .scrollIntoView({ block: "start" });
  search.focus({ preventScroll: true });
}

let dataCatalogView = null;
const dataCatalogFilters = new Map();
function selectDataCatalog(view) {
  if (!["registry", "files"].includes(view) || view === dataCatalogView) return;
  const search = document.getElementById("data-resource-search"),
    archived = document.getElementById("data-show-archived");
  if (dataCatalogView)
    dataCatalogFilters.set(dataCatalogView, {
      value: search.value,
      recordingId: search.dataset.recordingId,
      resourceIds: search.dataset.resourceIds,
      archived: archived.checked,
    });
  const saved = dataCatalogFilters.get(view) || { value: "", archived: false };
  if (dataCatalogView) {
    search.value = saved.value;
    archived.checked = saved.archived;
    delete search.dataset.recordingId;
    delete search.dataset.resourceIds;
    if (saved.resourceIds) search.dataset.resourceIds = saved.resourceIds;
    if (saved.recordingId) search.dataset.recordingId = saved.recordingId;
  }
  dataCatalogView = view;
  document
    .getElementById("datasets")
    .setAttribute(
      "aria-labelledby",
      view === "files" ? "data-tab-files" : "data-tab-registry",
    );
}
function isFileResource(resource) {
  return resource.category === "file";
}

function visibleDataResources() {
  const files =
    document
      .querySelector('[data-data-tab="files"]')
      ?.getAttribute("aria-selected") === "true";
  return dataResourceRows.filter(
    (resource) =>
      (document.getElementById("data-show-archived").checked ||
        !resource.archived_at) &&
      resource.category === (files ? "file" : "dataset"),
  );
}

function refreshDataResourceTables() {
  dataVersionRows = visibleDataResources().flatMap((resource) =>
    (resource.versions || []).map((version) => ({
      ...version,
      _resource: resource,
    })),
  );
  renderDataResources();
  renderDataVersions();
}

function renderDataResources() {
  const showRecording = dataCatalogView !== "files";
  document.getElementById("data-resource-recording-column").hidden =
    !showRecording;
  document.getElementById("data-resource-type-column").hidden = showRecording;
  const search = document.getElementById("data-resource-search");
  search.placeholder = showRecording
    ? "Filter name, recording or source"
    : "Filter name or source";
  const query = search.value.trim().toLowerCase();
  const recordingId = showRecording ? search.dataset.recordingId : null;
  const presetIds =
    showRecording && search.dataset.resourceIds
      ? JSON.parse(search.dataset.resourceIds)
      : null;
  const available = visibleDataResources();
  const rows = available.filter((resource) => {
    if (presetIds) return presetIds.includes(resource.id);
    if (recordingId)
      return resourceRecordingIds(resource).includes(recordingId);
    return (
      !query ||
      [
        resource.id,
        resource.resource_id,
        resource.metadata?.display_name,
        resource.namespace,
        resource.name,
        resource.kind,
        resource.provider,
        ...resourceRecordingIds(resource),
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase()
        .includes(query)
    );
  });
  const unit = dataCatalogView === "files" ? "file set" : "dataset";
  elements.dataResourceCount.hidden = showRecording;
  elements.dataResourceCount.textContent = `${query || recordingId ? rows.length + " of " : ""}${available.length} ${unit}${available.length === 1 ? "" : "s"}`;
  elements.dataResourcesBody.innerHTML = rows.length
    ? rows
        .map((resource) => {
          const id = resource.id || resource.resource_id;
          const recordingCount = resourceRecordingIds(resource).length;
          const formats = dataFormatsLabel(resource);
          return `<tr data-resource-id="${escapeHtml(id)}">
        <td><span class="node-name">${escapeHtml(resource.metadata?.display_name || [resource.namespace, resource.name].filter(Boolean).join("/"))}</span>${resource.archived_at ? `<span class="secondary">Archived</span>` : ""}${dataResourceActivity(id)}</td>
        ${showRecording ? `<td><button type="button" class="text-button" data-resource-action="recordings" data-id="${escapeHtml(id)}">${recordingCount} recording${recordingCount === 1 ? "" : "s"}</button></td>` : ""}
        ${showRecording ? "" : `<td>${escapeHtml(dataResourceTypeLabel(resource))}</td>`}
        <td>${formats ? `<button type="button" class="text-button" data-resource-action="dataset" data-id="${escapeHtml(id)}">${escapeHtml(formats)}</button>` : "—"}</td>
        <td>${escapeHtml(resource.provider || "—")}</td>
        <td>${escapeHtml(formatDate(resource.updated_at || resource.created_at))}</td>
        <td class="row-actions data-resource-row-actions"><button type="button" data-resource-action="dataset" data-id="${escapeHtml(id)}">View</button>${!resource.archived_at && resource.provider === "huggingface" ? `<button type="button" data-resource-action="import" data-id="${escapeHtml(id)}">Import</button>` : ""}${resource.archived_at ? "" : `<button type="button" data-resource-action="version" data-id="${escapeHtml(id)}">Add files</button>`}<button type="button" data-resource-action="edit" data-id="${escapeHtml(id)}">Edit</button><button type="button" data-resource-action="${resource.archived_at ? "restore" : "archive"}" data-id="${escapeHtml(id)}">${resource.archived_at ? "Restore" : "Archive"}</button>${showRecording ? `<button type="button" data-delete-kind="dataset" data-delete-id="${escapeHtml(id)}">Delete</button>` : ""}</td>
      </tr>`;
        })
        .join("")
    : emptyRow(
        6,
        query
          ? "No datasets or file sets match your filter."
          : "No datasets or files have been registered.",
      );
}

function renderDataImports() {
  const imports = inspectedDataResourceId
    ? dataImportRows.filter(
        (row) => String(row.resource_id) === inspectedDataResourceId,
      )
    : dataImportRows;
  elements.dataImportCount.textContent = `${imports.length} import job${imports.length === 1 ? "" : "s"}`;
  if (!imports.length) {
    elements.dataImportsBody.innerHTML = emptyRow(
      7,
      "No dataset import jobs have been submitted.",
    );
    return;
  }
  elements.dataImportsBody.innerHTML = imports
    .map((item) => {
      const request = item.request || {};
      const row = `<tr>
      <td><span class="node-name">${escapeHtml(request.subset || "-")}</span><span class="secondary">${escapeHtml(item.resource_id)}</span></td>
      <td><code>${escapeHtml(shortId(request.revision, 16))}</code><span class="secondary">${escapeHtml(request.format || "-")}</span></td>
      <td>${statusPill(item.state || "UNKNOWN")}</td>
      <td>${escapeHtml(item.slurm_job_id || "-")}<span class="secondary">${escapeHtml([item.gateway, item.node_list].filter(Boolean).join(" / ") || item.slurm_state || "-")}</span></td>
      <td>${escapeHtml(item.version_id ? "Registered" : "—")}</td>
      <td>${escapeHtml(formatDate(item.updated_at || item.created_at))}</td>
      <td class="row-actions"><button type="button" data-import-action="detail" data-id="${escapeHtml(item.id)}">Detail</button>${dataImportCancelButton(item)}</td>
    </tr>`;
      return row;
    })
    .join("");
  renderDataImportDetail();
}

function renderDataImportDetail() {
  const item = dataImportRows.find(
    (item) => String(item.id) === String(selectedDataImportId),
  );
  if (!item) return;
  const cached = dataImportLogCache.get(String(item.id)) || {};
  document.getElementById("data-import-detail-context").textContent =
    `${item.request?.subset || item.resource_id} · ${item.id}`;
  const error = item.error
    ? `<div class="notice notice-error">${escapeHtml(item.error)}</div>`
    : "";
  document.getElementById("data-import-detail-content").innerHTML = `
      ${error}
      <p>${statusPill(item.state)} ${dataImportCancelButton(item)}</p>
      <p>Allocation: ${escapeHtml(item.request?.cpus ?? 8)} CPUs · ${escapeHtml(item.request?.memory_gb ?? 32)} GB · ${escapeHtml(item.request?.time_limit || "legacy queue default")} · no GPUs</p>
      <p>Logs are read through the import job’s recorded gateway (${escapeHtml(item.gateway || "auto")}). <button type="button" class="button button-outline" data-import-action="logs" data-id="${escapeHtml(item.id)}"${cached.loading ? " disabled" : ""}>${cached.loading ? "Loading logs..." : "Refresh logs"}</button></p>
      <div class="key-value-grid"><div class="key-value"><span>Result path</span><strong>${escapeHtml(item.result_path || "-")}</strong></div><div class="key-value"><span>Published version</span><strong>${escapeHtml(item.version_id || "-")}</strong></div></div>
      <div class="attempt-log-grid"><section><div class="panel-heading"><h3>stdout</h3></div><pre class="log-view">${escapeHtml(cached.loading ? "Loading stdout..." : (cached.stdout ?? "Click Refresh logs to load stdout."))}</pre></section><section><div class="panel-heading"><h3>stderr</h3></div><pre class="log-view">${escapeHtml(cached.loading ? "Loading stderr..." : (cached.stderr ?? "Click Refresh logs to load stderr."))}</pre></section></div>
    `;
}

function renderDataImportBudget() {
  document.querySelector("#data-import-budget").textContent =
    `Allocation: 1 node, ${document.querySelector("#data-import-cpus").value} CPUs, ${document.querySelector("#data-import-memory").value} GB memory, no GPUs; time limit ${document.querySelector("#data-import-time").value}. Queue: ${elements.dataImportQueue.value}. Cancel is available while the import is active.`;
}

function dataImportCanCancel(item) {
  return (
    Boolean(item.slurm_job_id) &&
    ["SUBMITTED", "PENDING", "RUNNING"].includes(item.state)
  );
}

function dataImportCancelButton(item) {
  return dataImportCanCancel(item)
    ? `<button type="button" class="button button-outline" data-import-action="cancel" data-id="${escapeHtml(item.id)}">Cancel import</button>`
    : "";
}

async function cancelDataImport(id, button) {
  if (
    !(await askUserDialog(
      "Cancel this import job? Published versions and existing files are retained.",
    ))
  )
    return;
  button.disabled = true;
  try {
    await api(`/api/data/imports/${encodeURIComponent(id)}/cancel`, {
      method: "POST",
    });
    await loadDataRegistry(true);
    showToast(
      "Import cancellation requested. Refresh to confirm its final state.",
    );
  } catch (error) {
    showToast(`Import cancellation failed: ${error.message}`, true);
  } finally {
    if (button.isConnected) button.disabled = false;
  }
}

function selectDataResourceForImport(id, launcher = null) {
  const resource = dataResourceRows.find(
    (candidate) => String(candidate.id || candidate.resource_id) === String(id),
  );
  if (!resource || resource.provider !== "huggingface") {
    showToast("Only a registered Hugging Face resource can be imported.", true);
    return;
  }
  elements.dataImportForm.reset();
  renderDataImportBudget();
  elements.dataImportResourceId.value = id;
  elements.dataImportResourceLabel.textContent = `Import ${dataResourceIdentity(resource)}`;
  revealPanel(elements.dataImportForm, {
    focusTarget: elements.dataImportRevision,
    launcher,
  });
}

async function submitDataImport(event) {
  event.preventDefault();
  if (!elements.dataImportForm.reportValidity()) return;
  const resourceId = elements.dataImportResourceId.value;
  elements.submitDataImport.disabled = true;
  try {
    const result = await api(
      `/api/data/resources/${encodeURIComponent(resourceId)}/imports`,
      {
        method: "POST",
        body: JSON.stringify({
          revision: elements.dataImportRevision.value.trim().toLowerCase(),
          subset: elements.dataImportSubset.value.trim().replace(/\/\*\*$/, ""),
          format: elements.dataImportFormat.value.trim(),
          role: elements.dataImportRole.value.trim(),
          gateway: elements.dataImportGateway.value,
          queue: elements.dataImportQueue.value,
          cpus: Number(document.querySelector("#data-import-cpus").value),
          memory_gb: Number(
            document.querySelector("#data-import-memory").value,
          ),
          time_limit: document.querySelector("#data-import-time").value.trim(),
        }),
      },
    );
    const item = entityFrom(result, "import");
    selectedDataImportId = item.id;
    hideRevealedPanel(elements.dataImportForm);
    await loadDataRegistry(true);
    showToast(`Dataset import submitted as Slurm job ${item.slurm_job_id}.`);
  } catch (error) {
    showToast(`Dataset import failed: ${error.message}`, true);
  } finally {
    elements.submitDataImport.disabled = false;
  }
}

async function openDataImportDetail(id, launcher = null) {
  selectedDataImportId = id;
  renderDataImportDetail();
  SkynetDialog.open(document.getElementById("data-import-detail-dialog"), {
    launcher,
  });
  await loadDataImportLogs(id);
}

async function loadDataImportLogs(id) {
  if (dataImportLogCache.get(String(id))?.loading) return;
  dataImportLogCache.set(String(id), { loading: true });
  renderDataImportDetail();
  const load = async (stream) => {
    try {
      const response = await fetch(
        `/api/data/imports/${encodeURIComponent(id)}/logs?stream=${stream}&lines=500`,
      );
      if (!response.ok) {
        const text = await response.text();
        let detail;
        try {
          detail = JSON.parse(text);
        } catch {
          detail = text;
        }
        throw new Error(apiErrorMessage(detail) || `HTTP ${response.status}`);
      }
      return (await response.text()) || `No ${stream} output recorded.`;
    } catch (error) {
      return `Unable to load ${stream}: ${error.message}`;
    }
  };
  const [stdout, stderr] = await Promise.all([load("stdout"), load("stderr")]);
  dataImportLogCache.set(String(id), { stdout, stderr });
  if (String(selectedDataImportId || "") === String(id))
    renderDataImportDetail();
}

let inspectedDataResourceId = null;
function inspectedVersions() {
  return inspectedDataResourceId
    ? dataVersionRows.filter(
        (v) =>
          String(v.resource_id || v._resource?.id) === inspectedDataResourceId,
      )
    : dataVersionRows;
}
function openDataInspection(resourceId) {
  inspectedDataResourceId = String(resourceId);
  renderDataVersions();
  renderDataDerivations();
  renderDataImports();
  const resource = dataResourceRows.find(
    (r) => String(r.id) === inspectedDataResourceId,
  );
  document.getElementById("data-inspection-title").textContent =
    (resource?.metadata?.display_name || resource?.name || "Dataset") +
    " · Files and history";
  SkynetDialog.open(document.getElementById("data-inspection-dialog"));
}
document.addEventListener("click", async (event) => {
  const presets = event.target.closest("[data-dataset-presets]");
  if (presets) {
    SkynetDialog.close(document.getElementById("prepared-dataset-dialog"));
    await activateTab("experiments", true, "presets");
    const search = document.getElementById("experiment-search");
    search.value =
      presets.dataset.datasetLabel || presets.dataset.datasetPresets;
    search.dataset.datasetId = presets.dataset.datasetPresets;
    search.dataset.presetIds = presets.dataset.presetIds;
    renderExperiments();
    search.focus();
    return;
  }
  const datasets = event.target.closest("[data-preset-datasets]");
  if (datasets) {
    const preset = experimentRows.find(
      (row) => row.id === datasets.dataset.presetDatasets,
    );
    if (!preset) return;
    await activateTab("data", true, "registry");
    const search = document.getElementById("data-resource-search");
    delete search.dataset.recordingId;
    search.dataset.resourceIds = JSON.stringify(preset.dataset_ids || []);
    search.value = preset.name;
    document.getElementById("data-show-archived").checked =
      dataResourceRows.some(
        (row) => (preset.dataset_ids || []).includes(row.id) && row.archived_at,
      );
    refreshDataResourceTables();
    search.focus();
    return;
  }
  const recordings = event.target.closest("[data-dataset-recordings]");
  if (recordings) {
    SkynetDialog.close(document.getElementById("prepared-dataset-dialog"));
    await showDatasetRecordings(recordings.dataset.datasetRecordings);
    return;
  }
  const history = event.target.closest("[data-data-history]");
  if (history) openDataInspection(history.dataset.dataHistory);
  const recording = event.target.closest("[data-recording-link]");
  if (recording) {
    for (const id of ["prepared-dataset-dialog", "data-inspection-dialog"])
      SkynetDialog.close(document.getElementById(id));
    const input = document.getElementById("simulation-recordings-search");
    input.value = recording.dataset.recordingLink;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    activateTab("data", true, "recording");
  }
});
document
  .getElementById("data-inspection-dialog")
  ?.addEventListener("close", () => {
    inspectedDataResourceId = null;
  });

function renderDataVersions() {
  elements.dataVersionCount.textContent = `${inspectedVersions().length} registered file result${inspectedVersions().length === 1 ? "" : "s"}`;
  elements.dataVersionsBody.innerHTML = inspectedVersions().length
    ? inspectedVersions()
        .map(
          (version) => `<tr>
      <td><span class="node-name">${escapeHtml(dataVersionLabel(version))}</span><span class="secondary">${escapeHtml(version.id || version.version_id)}</span></td>
      <td>${escapeHtml(version.format || "-")}</td>
      <td>${statusPill(dataVersionStatus(version))}</td>
      <td>${escapeHtml(formatDataBytes(version.size_bytes))}</td>
      <td class="wrap-cell"><code>${escapeHtml(dataVersionPath(version))}</code></td>
      <td><code>${escapeHtml(shortId(version.manifest_sha256, 16))}</code></td>
      <td>${escapeHtml(formatDate(version.created_at))}</td>
    </tr>`,
        )
        .join("")
    : emptyRow(7, "No files have been registered.");

  const formats = new Set(
    inspectedVersions()
      .map((version) => version.format)
      .filter(Boolean),
  );
  adapterRows.forEach((adapter) =>
    declaredAdapterInputFields(adapter).forEach((field) =>
      (field.data_binding?.formats || []).forEach((format) =>
        formats.add(format),
      ),
    ),
  );
  document.querySelector("#data-format-suggestions").innerHTML = [...formats]
    .sort()
    .map((format) => `<option value="${escapeHtml(format)}"></option>`)
    .join("");
  const previousOutput = elements.dataDerivationOutput.value;
  elements.dataDerivationOutput.innerHTML = inspectedVersions().length
    ? inspectedVersions()
        .map(
          (version) =>
            `<option value="${escapeHtml(version.id || version.version_id)}">${escapeHtml(dataVersionLabel(version))}</option>`,
        )
        .join("")
    : '<option value="">No versions loaded</option>';
  elements.dataDerivationOutput.disabled = !inspectedVersions().length;
  if (
    [...elements.dataDerivationOutput.options].some(
      (option) => option.value === previousOutput,
    )
  ) {
    elements.dataDerivationOutput.value = previousOutput;
  }
}

function renderDataDerivations() {
  const versions = new Set(inspectedVersions().map((v) => v.id));
  const derivations = inspectedDataResourceId
    ? dataDerivationRows.filter((row) => versions.has(row.output_version_id))
    : dataDerivationRows;
  elements.dataDerivationCount.textContent = `${derivations.length} derivation${derivations.length === 1 ? "" : "s"}`;
  elements.dataDerivationsBody.innerHTML = derivations.length
    ? derivations
        .map((derivation) => {
          const output =
            derivation.output_version ||
            dataVersionRows.find(
              (version) =>
                String(version.id || version.version_id) ===
                String(derivation.output_version_id),
            );
          const inputs = Array.isArray(derivation.inputs)
            ? derivation.inputs
            : [];
          const inputHtml = inputs.length
            ? inputs
                .map(
                  (input) =>
                    `<div><strong>${escapeHtml(input.role || "input")}</strong>: ${escapeHtml(dataVersionLabel(input.version || dataVersionRows.find((version) => String(version.id || version.version_id) === String(input.version_id))))}</div>`,
                )
                .join("")
            : '<span class="secondary">No inputs resolved</span>';
          return `<tr>
        <td><span class="node-name">${escapeHtml(dataVersionLabel(output))}</span><span class="secondary">${escapeHtml(derivation.id || derivation.derivation_id)}</span></td>
        <td class="wrap-cell">${inputHtml}</td>
        <td class="wrap-cell">${escapeHtml(derivation.converter_repository || "-")}<span class="secondary">${escapeHtml(shortId(derivation.converter_commit, 16))}</span></td>
        <td><code>${escapeHtml(shortId(derivation.runtime_lock_sha256, 16))}</code></td>
        <td>${escapeHtml(formatDate(derivation.created_at))}</td>
      </tr>`;
        })
        .join("")
    : emptyRow(5, "No derived lineage has been recorded.");
}

let dataRegistryGeneration = 0;
window.openConvertedDataset = async (job) => {
  await activateTab("data", true, "registry");
  await loadDataRegistry(true);
  const search = document.getElementById("data-resource-search");
  delete search.dataset.recordingId;
  delete search.dataset.resourceIds;
  search.value = job.resource_id;
  refreshDataResourceTables();
  const row = elements.dataResourcesBody.querySelector("[data-resource-id]");
  row?.querySelector("button")?.focus({ preventScroll: true });
};

async function loadDataRegistry(force = false) {
  if (loadedTabs.has("datasets") && !force) return;
  const generation = ++dataRegistryGeneration;
  elements.refreshDataRegistry.disabled = true;
  clearNotice(elements.dataRegistryError);
  try {
    const [resourcePayload, importPayload, derivationPayload] =
      await Promise.all([
        api("/api/data/resources?include_archived=true"),
        api("/api/data/imports"),
        api("/api/data/derivations"),
      ]);
    if (generation !== dataRegistryGeneration) return;
    const resources = listFrom(resourcePayload, ["resources"]);
    const details = await Promise.all(
      resources.map(async (resource) => {
        if (Array.isArray(resource.versions)) return resource;
        const id = resource.id || resource.resource_id;
        try {
          return entityFrom(
            await api(`/api/data/resources/${encodeURIComponent(id)}`),
            "resource",
          );
        } catch (error) {
          throw new Error(
            `Versions for ${dataResourceIdentity(resource)} could not be loaded: ${error.message}`,
          );
        }
      }),
    );
    if (generation !== dataRegistryGeneration) return;
    dataResourceTypes = resourcePayload.resource_types || {};
    dataResourceRows = details;
    dataResourceCatalogState = "ready";
    notifyRecordingRegistryChanged();
    dataImportRows = listFrom(importPayload, ["imports"]);
    if (
      selectedDataImportId &&
      !dataImportRows.some(
        (item) => String(item.id) === String(selectedDataImportId),
      )
    )
      selectedDataImportId = null;
    dataDerivationRows = listFrom(derivationPayload, ["derivations"]);

    refreshDataResourceTables();
    renderDataImports();
    renderDataDerivations();

    loadedTabs.add("datasets");
  } catch (error) {
    if (generation !== dataRegistryGeneration) return;
    dataResourceCatalogState = "unavailable";
    notifyRecordingRegistryChanged();
    elements.dataResourcesBody.innerHTML = emptyRow(
      6,
      "Dataset registry could not be loaded.",
    );
    elements.dataImportsBody.innerHTML = emptyRow(
      7,
      "Dataset import jobs could not be loaded.",
    );
    elements.dataVersionsBody.innerHTML = emptyRow(
      7,
      "Dataset registry could not be loaded.",
    );
    elements.dataDerivationsBody.innerHTML = emptyRow(
      5,
      "Dataset registry could not be loaded.",
    );

    showNotice(
      elements.dataRegistryError,
      `Dataset Registry API unavailable: ${error.message}`,
    );
  } finally {
    if (generation === dataRegistryGeneration)
      elements.refreshDataRegistry.disabled = false;
  }
}

function parseDataJson(element, label, fallback) {
  const raw = element.value.trim();
  if (!raw) return fallback;
  try {
    return JSON.parse(raw);
  } catch (error) {
    throw new Error(`${label} must be valid JSON: ${error.message}`);
  }
}

function selectDataResourceForVersion(id, launcher = null) {
  const resource = dataResourceRows.find(
    (candidate) => String(candidate.id || candidate.resource_id) === String(id),
  );
  if (!resource) {
    closeDisclosurePanel(elements.dataVersionForm, launcher);
    showToast(
      "The selected resource is not in the current registry snapshot.",
      true,
    );
    return;
  }
  elements.dataVersionForm.reset();
  elements.dataVersionResourceId.value = id;
  elements.dataVersionResourceLabel.textContent =
    dataResourceIdentity(resource);
  elements.dataVersionStatus.value = "READY";
  revealPanel(elements.dataVersionForm, {
    focusTarget: elements.dataVersionRevision,
    launcher,
  });
}

async function createDataResource(event) {
  event.preventDefault();
  const editingId = document.querySelector("#data-resource-id")?.value.trim();
  if (editingId) {
    await updateDataResource(editingId);
    return;
  }
  if (!elements.dataResourceForm.reportValidity()) return;
  elements.createDataResource.disabled = true;
  try {
    const result = await api("/api/data/resources", {
      method: "POST",
      body: JSON.stringify({
        provider: elements.dataResourceProvider.value.trim(),
        namespace: elements.dataResourceNamespace.value.trim(),
        name: elements.dataResourceName.value.trim(),
        category: document.getElementById("data-resource-category").value,
        kind: elements.dataResourceKind.value.trim(),
        description: elements.dataResourceDescription.value.trim(),
      }),
    });
    const resource = entityFrom(result, "resource");
    const id = resource.id || resource.resource_id;
    showToast(`Resource ${dataResourceIdentity(resource)} registered.`);
    closeDisclosurePanel(
      elements.dataResourceForm,
      elements.showDataResourceForm,
    );
    elements.showDataResourceForm.textContent = "New";
    elements.dataResourceName.value = "";
    elements.dataResourceDescription.value = "";
    elements.dataResourceForm.reset();
    await loadDataRegistry(true);
    if (id) {
      if (resource.provider === "huggingface") selectDataResourceForImport(id);
      else selectDataResourceForVersion(id);
    }
  } catch (error) {
    showToast(`Resource registration failed: ${error.message}`, true);
  } finally {
    elements.createDataResource.disabled = false;
  }
}

async function createDataVersion(event) {
  event.preventDefault();
  if (!elements.dataVersionForm.reportValidity()) return;
  const resourceId = elements.dataVersionResourceId.value;
  elements.createDataVersion.disabled = true;
  try {
    const size = numberOrNull(elements.dataVersionSize);
    await api(
      `/api/data/resources/${encodeURIComponent(resourceId)}/versions`,
      {
        method: "POST",
        body: JSON.stringify({
          revision: elements.dataVersionRevision.value.trim(),
          format: elements.dataVersionFormat.value.trim(),
          path: elements.dataVersionPath.value.trim(),
          source_uri: elements.dataVersionSourceUri.value.trim() || null,
          manifest_sha256: elements.dataVersionManifest.value
            .trim()
            .toLowerCase(),
          status: elements.dataVersionStatus.value,
          size_bytes: size,
        }),
      },
    );
    showToast("Immutable resource version created.");
    hideRevealedPanel(elements.dataVersionForm);
    await loadDataRegistry(true);
  } catch (error) {
    showToast(`Version creation failed: ${error.message}`, true);
  } finally {
    elements.createDataVersion.disabled = false;
  }
}

async function createDataDerivation(event) {
  event.preventDefault();
  if (!elements.dataDerivationForm.reportValidity()) return;
  let inputs;
  let converterConfig;
  try {
    inputs = parseDataJson(
      elements.dataDerivationInputs,
      "Derivation inputs",
      [],
    );
    converterConfig = parseDataJson(
      elements.dataConverterConfig,
      "Converter config",
      {},
    );
    if (!Array.isArray(inputs) || !inputs.length)
      throw new Error("Derivation inputs must be a non-empty JSON list.");
    if (
      !converterConfig ||
      Array.isArray(converterConfig) ||
      typeof converterConfig !== "object"
    )
      throw new Error("Converter config must be a JSON object.");
  } catch (error) {
    showToast(error.message, true);
    return;
  }
  elements.createDataDerivation.disabled = true;
  try {
    await api("/api/data/derivations", {
      method: "POST",
      body: JSON.stringify({
        output_version_id: elements.dataDerivationOutput.value,
        inputs,
        converter_repository: elements.dataConverterRepository.value.trim(),
        converter_commit: elements.dataConverterCommit.value.trim(),
        converter_config: converterConfig,
        runtime_lock_sha256:
          elements.dataRuntimeLockSha.value.trim().toLowerCase() || null,
      }),
    });
    showToast("Dataset derivation lineage recorded.");
    elements.showDataDerivationForm.textContent = "Record a derivation";
    hideRevealedPanel(
      elements.dataDerivationForm,
      elements.showDataDerivationForm,
    );
    elements.dataDerivationForm.reset();
    elements.dataConverterConfig.value = "{}";
    await loadDataRegistry(true);
  } catch (error) {
    showToast(`Derivation creation failed: ${error.message}`, true);
  } finally {
    elements.createDataDerivation.disabled = false;
  }
}

const COLLECTION_ADAPTER_SCHEMA = "skynet.collection-adapter/v1";
const collectionPrepareStates = new Set([
  "DRAFT",
  "PREFLIGHTED",
  "PREFLIGHT_FAILED",
  "READY",
]);
const collectionSubmittedStates = new Set([
  "SUBMITTED",
  "PENDING",
  "RUNNING",
  "CAPTURED",
  "COMPLETED",
  "FAILED",
  "CANCELLED",
]);
const collectionTerminalStates = new Set(["COMPLETED", "FAILED", "CANCELLED"]);

function collectionPretty(value) {
  return JSON.stringify(value, null, 2);
}

function collectionAdapterManifest(adapter) {
  const raw = adapter?.manifest ?? adapter;
  if (raw === undefined || raw === null) {
    return { __parse_error: "Collection adapter manifest is missing." };
  }
  if (typeof raw === "string") {
    try {
      return JSON.parse(raw);
    } catch (error) {
      return {
        __parse_error: `Invalid collection adapter manifest JSON: ${error.message}`,
      };
    }
  }
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return {
      __parse_error: "Collection adapter manifest must be a JSON object.",
    };
  }
  return raw;
}

function collectionAdapterIdentity(adapter) {
  const manifest = collectionAdapterManifest(adapter);
  return (
    manifest.display_name ||
    adapter?.display_name ||
    manifest.key ||
    adapter?.adapter_key ||
    "Unnamed adapter"
  );
}

function collectionDefaultAdapterManifest() {
  return {
    schema_version: COLLECTION_ADAPTER_SCHEMA,
    key: "",
    display_name: "",
    version: "1",
    description: "",
    runnable: false,
    streams: [
      {
        name: "action",
        kind: "action",
        shape: null,
        dtype: "float32",
        units: null,
        frame: null,
        rate_hz: null,
        native_key: "action",
        required: true,
        metadata: {},
      },
      {
        name: "proprio",
        kind: "proprio",
        shape: null,
        dtype: "float32",
        units: null,
        frame: null,
        rate_hz: null,
        native_key: "observation.proprio",
        required: true,
        metadata: {},
      },
      {
        name: "sensor",
        kind: "sensor",
        shape: null,
        dtype: "uint8",
        units: null,
        frame: null,
        rate_hz: null,
        native_key: "observation.sensor",
        required: false,
        metadata: {},
      },
    ],
    requirements: [],
    launcher: { environment: {}, steps: [] },
    capabilities: [],
    defaults: {},
    todos: [
      "Resolve native stream shapes, semantics, and launcher before marking this adapter runnable.",
    ],
    metadata: {},
  };
}

function collectionParseArray(element, label) {
  const value = parseDataJson(element, label, []);
  if (!Array.isArray(value)) throw new Error(`${label} must be a JSON list.`);
  return value;
}

function collectionParseObject(element, label) {
  const value = parseDataJson(element, label, {});
  if (!value || Array.isArray(value) || typeof value !== "object")
    throw new Error(`${label} must be a JSON object.`);
  return value;
}

function populateCollectionAdapterSelect(selectedId = "") {
  const active = collectionAdapterRows.filter(
    (adapter) => !adapter.archived_at,
  );
  elements.collectionSessionAdapter.innerHTML = active.length
    ? `<option value="">Select a runnable adapter</option>${active
        .map((adapter) => {
          const manifest = collectionAdapterManifest(adapter);
          const id = adapter.id || adapter.adapter_id;
          const disabled = manifest.runnable ? "" : " disabled";
          const suffix = manifest.runnable ? "" : " (not runnable)";
          return `<option value="${escapeHtml(id)}"${disabled}>${escapeHtml(collectionAdapterIdentity(adapter) + suffix)}</option>`;
        })
        .join("")}`
    : '<option value="">No active adapters</option>';
  const hasRunnable = active.some(
    (adapter) => collectionAdapterManifest(adapter).runnable,
  );
  elements.collectionSessionAdapter.disabled = !hasRunnable;
  document.querySelector("#show-collection-session-form").disabled =
    !hasRunnable;
  document.querySelector("#collection-new-session-help").textContent =
    hasRunnable
      ? "Choose an adapter to fill its capture settings, then review and check setup."
      : "Live cluster collection needs a configured collection adapter. Open Recordings to review and convert saved simulation demonstrations.";
  const normalizedSelection = String(selectedId || "");
  if (
    normalizedSelection &&
    active.some(
      (adapter) =>
        String(adapter.id || adapter.adapter_id) === normalizedSelection,
    )
  ) {
    elements.collectionSessionAdapter.value = normalizedSelection;
    elements.collectionSessionAdapter.setCustomValidity("");
    elements.collectionSessionAdapter.removeAttribute("aria-invalid");
  } else if (normalizedSelection) {
    const message = `Previously selected collection adapter "${normalizedSelection}" is unavailable.`;
    elements.collectionSessionAdapter.insertAdjacentHTML(
      "afterbegin",
      `<option value="${escapeHtml(normalizedSelection)}" disabled>${escapeHtml(message)}</option>`,
    );
    elements.collectionSessionAdapter.value = normalizedSelection;
    elements.collectionSessionAdapter.setCustomValidity(message);
    elements.collectionSessionAdapter.setAttribute("aria-invalid", "true");
  }
  updateCollectionCapabilityEvidence(false);
}

function collectionSessionDefaults(manifest, gateway) {
  const defaults = JSON.parse(JSON.stringify(manifest.defaults || {}));
  const evidence = {};
  for (const item of manifest.capabilities || []) {
    const status = String(item.default_status || "unknown").toLowerCase();
    evidence[item.id] = {
      status: ["unavailable", "admin_required"].includes(status)
        ? status
        : "unknown",
      scope: item.scope || "operator",
      verified_by: null,
      checked_at: null,
      details: {},
    };
  }
  // An adapter template cannot attest to a newly created session's setup.
  return {
    ...defaults,
    config: defaults.config || {},
    capture: defaults.capture || {},
    storage: defaults.storage || {},
    resources: { ...(defaults.resources || {}), gateway },
    capabilities: evidence,
  };
}

function applyCollectionAdapterDefaults() {
  const adapter = collectionAdapterRows.find(
    (item) =>
      String(item.id || item.adapter_id) ===
      elements.collectionSessionAdapter.value,
  );
  if (!adapter) return;
  const defaults = collectionSessionDefaults(
    collectionAdapterManifest(adapter),
    elements.gateway.value,
  );
  const setValue = (name, value) => {
    elements[name].value = value ?? elements[name].defaultValue ?? "";
  };
  const fields = {
    collectionSessionTask: defaults.config.task,
    collectionSessionEmbodiment:
      defaults.config.embodiment ?? defaults.config.robot_type,
    collectionRuntimeProfile: defaults.runtime_profile,
    collectionOutputPath: defaults.storage.output_path,
    collectionNativeFormat: defaults.storage.native_format,
    collectionRegisterProvider: defaults.storage.registration?.provider,
    collectionRegisterNamespace: defaults.storage.registration?.namespace,
    collectionRegisterName: defaults.storage.registration?.name,
    collectionRegisterKind: defaults.storage.registration?.kind,
    collectionCaptureSchemaName: defaults.capture.schema_name,
    collectionCaptureSchemaVersion: defaults.capture.schema_version,
    collectionClockSource: defaults.capture.clock_source,
    collectionNominalRate: defaults.capture.nominal_rate_hz,
    collectionTimestampUnit: defaults.capture.timestamp_unit,
    collectionAlignment: defaults.capture.alignment ?? "adapter_defined",
    collectionGateway: defaults.resources.gateway,
    collectionAccount: defaults.resources.account,
    collectionPartition: defaults.resources.partition,
    collectionNode: defaults.resources.node,
    collectionGpuCount: defaults.resources.gpu_count,
    collectionGpuType: defaults.resources.gpu_type,
    collectionCpuCount: defaults.resources.cpu_count,
    collectionMemoryGb: defaults.resources.memory_gb,
    collectionTimeLimit: defaults.resources.time_limit,
  };
  Object.entries(fields).forEach(([key, value]) => setValue(key, value));
  const objects = {
    collectionConfig: defaults.config,
    collectionSoftware: defaults.software,
    collectionStorageMetadata: defaults.storage.metadata,
    collectionCaptureMetadata: defaults.capture.metadata,
    collectionCalibrationMetadata: defaults.calibration?.metadata,
    collectionCapabilities: defaults.capabilities,
  };
  Object.entries(objects).forEach(([key, value]) =>
    setValue(key, collectionPretty(value || {})),
  );
  elements.collectionCalibrationIdentity.value = "";
  elements.collectionCalibrationSha.value = "";
  elements.collectionTimestampsRecorded.checked =
    defaults.capture.timestamps_recorded ?? true;
  elements.collectionRegisterOutput.checked =
    defaults.storage.registration !== null;
  updateCollectionRegistrationFields();
  updateCollectionCapabilityEvidence(false);
}

function updateCollectionCapabilityEvidence(force = false) {
  const adapter = collectionAdapterRows.find(
    (candidate) =>
      String(candidate.id || candidate.adapter_id) ===
      String(elements.collectionSessionAdapter.value),
  );
  const declarations = Array.isArray(
    collectionAdapterManifest(adapter).capabilities,
  )
    ? collectionAdapterManifest(adapter).capabilities
    : [];
  const runnable = Boolean(
    adapter &&
      !adapter.archived_at &&
      collectionAdapterManifest(adapter).runnable,
  );
  elements.createCollectionSession.disabled =
    !runnable || elements.collectionSessionForm.dataset.submitting === "true";
  document.querySelector("#collection-session-availability").textContent =
    runnable
      ? "Create a draft first, then check setup and submit from its details."
      : "Session creation is unavailable. Select a runnable adapter; draft adapters need their launcher and stream definitions completed first.";
  if (!adapter) {
    elements.collectionCapabilitiesHelp.textContent =
      "Select an adapter to see its declared capability IDs. Use verified only with who verified it and when.";
    if (force) elements.collectionCapabilities.value = "{}";
    return;
  }
  elements.collectionCapabilitiesHelp.textContent = declarations.length
    ? `Declared by this adapter: ${declarations.map((item) => `${item.id} (${item.scope})`).join(", ")}. Unknown and admin-required evidence remains visible to preflight.`
    : "This adapter declares no capability evidence keys.";
  if (
    !force &&
    elements.collectionCapabilities.value.trim() !== "{}" &&
    elements.collectionCapabilities.value.trim()
  )
    return;
  elements.collectionCapabilities.value = collectionPretty(
    collectionSessionDefaults(
      collectionAdapterManifest(adapter),
      elements.gateway.value,
    ).capabilities,
  );
}

function renderCollectionAdapters() {
  elements.collectionAdapterCount.textContent = `${collectionAdapterRows.length} adapter${collectionAdapterRows.length === 1 ? "" : "s"}`;
  elements.collectionAdaptersBody.innerHTML = collectionAdapterRows.length
    ? collectionAdapterRows
        .map((adapter) => {
          const manifest = collectionAdapterManifest(adapter);
          const streams = Array.isArray(manifest.streams)
            ? manifest.streams
            : [];
          const streamSummary = streams.length
            ? streams
                .map(
                  (stream) =>
                    `${stream.kind || "stream"}:${stream.name || stream.native_key || "unnamed"}${Array.isArray(stream.shape) ? ` [${stream.shape.join("x")}]` : " [?]"}`,
                )
                .join(", ")
            : "No streams declared";
          const id = adapter.id || adapter.adapter_id;
          const archived = Boolean(adapter.archived_at);
          const template = collectionAdapterTemplates.find(
            (item) => item.manifest?.key === manifest.key,
          );
          const canReviewTemplate =
            !archived &&
            template &&
            template.manifest_sha256 !== adapter.manifest_sha256;
          const setup = !manifest.runnable
            ? `<details><summary>Setup required</summary><p>${escapeHtml(manifest.description || "This adapter has no verified collection launcher.")}</p><ul>${(manifest.todos || []).map((todo) => `<li>${escapeHtml(todo)}</li>`).join("")}</ul></details>`
            : "";
          return `<tr>
        <td><span class="node-name">${escapeHtml(collectionAdapterIdentity(adapter))}</span><span class="secondary">${escapeHtml(manifest.key || adapter.adapter_key || id)} @ ${escapeHtml(manifest.version || "-")}</span></td>
        <td class="wrap-cell">${escapeHtml(streamSummary)}</td>
        <td>${statusPill(manifest.__parse_error ? "INVALID" : archived ? "ARCHIVED" : manifest.runnable ? "RUNNABLE" : "DRAFT")}${setup}</td>
        <td>${escapeHtml(formatDate(adapter.updated_at || adapter.created_at))}</td>
        <td class="row-actions data-resource-row-actions">
          <button type="button" data-collection-adapter-action="edit" data-id="${escapeHtml(id)}">Edit</button>
          ${canReviewTemplate ? `<button type="button" data-collection-adapter-action="template" data-id="${escapeHtml(id)}">Review bundled setup ${escapeHtml(template.manifest.version)}</button>` : ""}
          <button type="button" data-collection-adapter-action="session" data-id="${escapeHtml(id)}"${archived || !manifest.runnable ? " disabled" : ""}>Collect</button>
          <button type="button" data-collection-adapter-action="${archived ? "restore" : "archive"}" data-id="${escapeHtml(id)}">${archived ? "Restore" : "Archive"}</button>
        </td>
      </tr>`;
        })
        .join("")
    : emptyRow(5, "No collection adapters are registered.");
  populateCollectionAdapterSelect(elements.collectionSessionAdapter.value);
}

function collectionSessionContext(session) {
  const config = session.config_snapshot || session.config || {};
  return {
    operator: config.operator || "-",
    task: config.task || "-",
    embodiment: config.embodiment || "-",
  };
}

function collectionSessionAdapterName(session) {
  const snapshot = session.adapter_snapshot || {};
  const match = collectionAdapterRows.find(
    (adapter) =>
      String(adapter.id || adapter.adapter_id) === String(session.adapter_id),
  );
  return (
    snapshot.display_name ||
    snapshot.manifest?.display_name ||
    (match ? collectionAdapterIdentity(match) : session.adapter_id || "-")
  );
}

function renderCollectionSessions() {
  elements.collectionSessionCount.textContent = `${collectionSessionRows.length} session${collectionSessionRows.length === 1 ? "" : "s"}`;
  elements.collectionSessionsBody.innerHTML = collectionSessionRows.length
    ? collectionSessionRows
        .map((session) => {
          const context = collectionSessionContext(session);
          const storage = session.storage_snapshot || {};
          const id = session.id || session.session_id;
          return `<tr>
        <td><span class="node-name">${escapeHtml(session.name || shortId(id))}</span><span class="secondary">${escapeHtml(shortId(id, 16))}</span></td>
        <td>${escapeHtml(collectionSessionAdapterName(session))}</td>
        <td><span>${escapeHtml(context.operator)}</span><span class="secondary">${escapeHtml(context.task)} / ${escapeHtml(context.embodiment)}</span></td>
        <td>${statusPill(session.status)}</td>
        <td>${escapeHtml(session.slurm_job_id || "-")}</td>
        <td class="wrap-cell"><span>${escapeHtml(storage.native_format || "-")}</span><span class="secondary">${escapeHtml(storage.output_path || "-")}</span></td>
        <td>${escapeHtml(formatDate(session.updated_at || session.created_at))}</td>
        <td class="row-actions"><button class="text-button" type="button" data-collection-session-action="view" data-id="${escapeHtml(id)}">View</button></td>
      </tr>`;
        })
        .join("")
    : emptyRow(8, "No collection sessions have been created.");
}

async function loadCollection(force = false) {
  if (loadedTabs.has("collection") && !force) return;
  elements.refreshCollection.disabled = true;
  clearNotice(elements.collectionError);
  try {
    const [adapterPayload, sessionPayload] = await Promise.all([
      api("/api/collection/adapters?include_archived=true"),
      api("/api/collection/sessions?limit=250"),
    ]);
    collectionAdapterRows = listFrom(adapterPayload, ["adapters"]);
    collectionAdapterTemplates = listFrom(adapterPayload, ["templates"]);
    collectionSessionRows = listFrom(sessionPayload, ["sessions"]);
    renderCollectionAdapters();
    renderCollectionSessions();
    loadedTabs.add("collection");
  } catch (error) {
    elements.collectionAdaptersBody.innerHTML = emptyRow(
      5,
      "Collection adapters could not be loaded.",
    );
    elements.collectionSessionsBody.innerHTML = emptyRow(
      8,
      "Collection sessions could not be loaded.",
    );
    showNotice(
      elements.collectionError,
      `Collection API unavailable: ${error.message}`,
    );
    document.querySelector("#collection-new-session-help").textContent =
      "Collection availability could not be checked. Refresh after the connection is restored.";
    document.querySelector("#show-collection-session-form").disabled = true;
  } finally {
    elements.refreshCollection.disabled = false;
  }
}

function fillCollectionAdapterForm(adapter = null, launcher = null) {
  const manifest = adapter
    ? collectionAdapterManifest(adapter)
    : collectionDefaultAdapterManifest();
  elements.collectionAdapterForm.reset();
  document.querySelector("#collection-template-review-note").hidden = true;
  elements.collectionAdapterId.value = adapter?.id || adapter?.adapter_id || "";
  elements.collectionAdapterKey.value =
    manifest.key || adapter?.adapter_key || "";
  elements.collectionAdapterKey.disabled = Boolean(adapter);
  elements.collectionAdapterName.value =
    manifest.display_name || adapter?.display_name || "";
  elements.collectionAdapterVersion.value = manifest.version || "1";
  elements.collectionAdapterRunnable.checked = Boolean(manifest.runnable);
  elements.collectionAdapterDescription.value =
    manifest.description || adapter?.description || "";
  elements.collectionAdapterStreams.value = collectionPretty(
    manifest.streams || [],
  );
  elements.collectionAdapterRequirements.value = collectionPretty(
    manifest.requirements || [],
  );
  elements.collectionAdapterLauncher.value = collectionPretty(
    manifest.launcher || { environment: {}, steps: [] },
  );
  elements.collectionAdapterCapabilities.value = collectionPretty(
    manifest.capabilities || [],
  );
  elements.collectionAdapterDefaults.value = collectionPretty(
    manifest.defaults || {},
  );
  elements.collectionAdapterTodos.value = collectionPretty(
    manifest.todos || [],
  );
  elements.collectionAdapterMetadata.value = collectionPretty(
    manifest.metadata || {},
  );
  elements.collectionAdapterEditorTitle.textContent = adapter
    ? `Edit ${collectionAdapterIdentity(adapter)}`
    : "New collection adapter";
  revealPanel(elements.collectionAdapterForm, {
    focusTarget: adapter
      ? elements.collectionAdapterName
      : elements.collectionAdapterKey,
    launcher,
  });
}

async function openCollectionAdapter(id, launcher = null) {
  const revealLauncher = launcher || currentRevealLauncher();
  const requestToken = disclosureToken(
    elements.collectionAdapterForm,
    revealLauncher,
  );
  try {
    const payload = await api(
      `/api/collection/adapters/${encodeURIComponent(id)}`,
    );
    if (!disclosureTokenIsCurrent(elements.collectionAdapterForm, requestToken))
      return;
    fillCollectionAdapterForm(entityFrom(payload, "adapter"), revealLauncher);
  } catch (error) {
    if (!disclosureTokenIsCurrent(elements.collectionAdapterForm, requestToken))
      return;
    closeDisclosurePanel(elements.collectionAdapterForm, revealLauncher);
    showToast(`Adapter could not be opened: ${error.message}`, true);
  }
}

function reviewCollectionTemplate(id, launcher) {
  const adapter = collectionAdapterRows.find(
    (item) => String(item.id || item.adapter_id) === id,
  );
  const template = collectionAdapterTemplates.find(
    (item) => item.manifest?.key === collectionAdapterManifest(adapter).key,
  );
  if (!adapter || !template) {
    showToast(
      "Bundled setup is unavailable. Refresh collection and try again.",
      true,
    );
    return;
  }
  fillCollectionAdapterForm(
    { ...adapter, manifest: template.manifest },
    launcher,
  );
  elements.collectionAdapterEditorTitle.textContent = `Review bundled ${template.manifest.display_name} setup`;
  document.querySelector("#collection-template-review-note").hidden = false;
}

async function saveCollectionAdapter(event) {
  event.preventDefault();
  if (elements.saveCollectionAdapter.disabled) return;
  const scope = "collection-adapter-save";
  if (!elements.collectionAdapterForm.reportValidity()) return;
  let manifest;
  try {
    manifest = {
      schema_version: COLLECTION_ADAPTER_SCHEMA,
      key: elements.collectionAdapterKey.value.trim(),
      display_name: elements.collectionAdapterName.value.trim(),
      version: elements.collectionAdapterVersion.value.trim(),
      description: elements.collectionAdapterDescription.value.trim(),
      runnable: elements.collectionAdapterRunnable.checked,
      streams: collectionParseArray(
        elements.collectionAdapterStreams,
        "Streams",
      ),
      requirements: collectionParseArray(
        elements.collectionAdapterRequirements,
        "Requirements",
      ),
      launcher: collectionParseObject(
        elements.collectionAdapterLauncher,
        "Launcher",
      ),
      capabilities: collectionParseArray(
        elements.collectionAdapterCapabilities,
        "Capability declarations",
      ),
      defaults: collectionParseObject(
        elements.collectionAdapterDefaults,
        "Defaults",
      ),
      todos: collectionParseArray(elements.collectionAdapterTodos, "TODOs"),
      metadata: collectionParseObject(
        elements.collectionAdapterMetadata,
        "Metadata",
      ),
    };
  } catch (error) {
    showToast(error.message, true, { scope });
    return;
  }
  const id = elements.collectionAdapterId.value;
  elements.saveCollectionAdapter.disabled = true;
  try {
    await api(
      id
        ? `/api/collection/adapters/${encodeURIComponent(id)}`
        : "/api/collection/adapters",
      {
        method: id ? "PUT" : "POST",
        body: JSON.stringify(manifest),
      },
    );
    showToast(`Collection adapter ${id ? "updated" : "created"}.`, false, {
      scope,
    });
    hideRevealedPanel(
      elements.collectionAdapterForm,
      elements.addCollectionAdapter,
    );
    loadedTabs.delete("collection");
    await loadCollection(true);
  } catch (error) {
    showToast(`Adapter save failed: ${error.message}`, true, { scope });
  } finally {
    elements.saveCollectionAdapter.disabled = false;
  }
}

async function setCollectionAdapterArchived(id, archived) {
  if (
    archived &&
    !(await askUserDialog(
      "Archive this collection adapter? Existing sessions will remain readable.",
    ))
  )
    return;
  try {
    if (archived) {
      await api(`/api/collection/adapters/${encodeURIComponent(id)}`, {
        method: "DELETE",
      });
    } else {
      await api(`/api/collection/adapters/${encodeURIComponent(id)}/restore`, {
        method: "POST",
      });
    }
    showToast(`Collection adapter ${archived ? "archived" : "restored"}.`);
    loadedTabs.delete("collection");
    await loadCollection(true);
  } catch (error) {
    showToast(
      `Adapter ${archived ? "archive" : "restore"} failed: ${error.message}`,
      true,
    );
  }
}

function updateCollectionRegistrationFields() {
  const enabled = elements.collectionRegisterOutput.checked;
  document
    .querySelectorAll(".collection-registration-field input")
    .forEach((input) => {
      input.disabled = !enabled;
      input.required = enabled;
    });
}

async function createCollectionSession(event) {
  event.preventDefault();
  if (elements.collectionSessionForm.dataset.submitting === "true") return;
  if (!elements.collectionSessionForm.reportValidity()) return;
  let config;
  let software;
  let calibrationMetadata;
  let storageMetadata;
  let captureMetadata;
  let capabilities;
  try {
    config = collectionParseObject(
      elements.collectionConfig,
      "Additional config",
    );
    software = collectionParseObject(elements.collectionSoftware, "Software");
    calibrationMetadata = collectionParseObject(
      elements.collectionCalibrationMetadata,
      "Calibration metadata",
    );
    storageMetadata = collectionParseObject(
      elements.collectionStorageMetadata,
      "Storage metadata",
    );
    captureMetadata = collectionParseObject(
      elements.collectionCaptureMetadata,
      "Capture metadata",
    );
    capabilities = collectionParseObject(
      elements.collectionCapabilities,
      "Capability evidence",
    );
  } catch (error) {
    showToast(error.message, true);
    return;
  }
  config = {
    ...config,
    operator: elements.collectionSessionOperator.value.trim(),
    task: elements.collectionSessionTask.value.trim(),
    embodiment: elements.collectionSessionEmbodiment.value.trim(),
  };
  const registration = elements.collectionRegisterOutput.checked
    ? {
        provider: elements.collectionRegisterProvider.value.trim(),
        namespace: elements.collectionRegisterNamespace.value.trim(),
        name: elements.collectionRegisterName.value.trim(),
        kind: elements.collectionRegisterKind.value.trim(),
      }
    : null;
  const payload = {
    adapter_id: elements.collectionSessionAdapter.value,
    name: elements.collectionSessionName.value.trim(),
    runtime_profile: elements.collectionRuntimeProfile.value.trim() || null,
    config,
    software,
    calibration: {
      identity: elements.collectionCalibrationIdentity.value.trim(),
      sha256: elements.collectionCalibrationSha.value.trim().toLowerCase(),
      metadata: calibrationMetadata,
    },
    storage: {
      output_path: elements.collectionOutputPath.value.trim(),
      native_format: elements.collectionNativeFormat.value.trim(),
      registration,
      metadata: storageMetadata,
    },
    resources: {
      gateway: elements.collectionGateway.value,
      account: elements.collectionAccount.value.trim(),
      partition: elements.collectionPartition.value.trim(),
      gpu_count: Number(elements.collectionGpuCount.value),
      gpu_type: elements.collectionGpuType.value.trim(),
      cpu_count: Number(elements.collectionCpuCount.value),
      memory_gb: Number(elements.collectionMemoryGb.value),
      time_limit: elements.collectionTimeLimit.value.trim(),
      node: elements.collectionNode.value.trim() || null,
    },
    capture: {
      schema_name: elements.collectionCaptureSchemaName.value.trim(),
      schema_version: elements.collectionCaptureSchemaVersion.value.trim(),
      clock_source: elements.collectionClockSource.value.trim(),
      nominal_rate_hz: Number(elements.collectionNominalRate.value),
      timestamp_unit: elements.collectionTimestampUnit.value.trim(),
      alignment: elements.collectionAlignment.value,
      timestamps_recorded: elements.collectionTimestampsRecorded.checked,
      metadata: captureMetadata,
    },
    capabilities,
  };
  elements.collectionSessionForm.dataset.submitting = "true";
  elements.createCollectionSession.disabled = true;
  try {
    const result = await api("/api/collection/sessions", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    const session = entityFrom(result, "session");
    // Return a row-mounted editor to its home before replacing the table rows.
    closeDisclosurePanel(elements.collectionSessionForm);
    elements.collectionSessionForm.reset();
    showToast("Collection draft created. Review it, then prepare and submit.");
    loadedTabs.delete("collection");
    await loadCollection(true);
    const id = session.id || session.session_id;
    const launcher = [
      ...elements.collectionSessionsBody.querySelectorAll(
        "[data-collection-session-action=view]",
      ),
    ].find((button) => button.dataset.id === id);
    if (launcher)
      toggleDisclosure(
        elements.collectionSessionDetail,
        `collection-session:${id}`,
        launcher,
        { rowOwned: true },
      );
    await viewCollectionSession(id, launcher);
  } catch (error) {
    showToast(`Session creation failed: ${error.message}`, true);
  } finally {
    delete elements.collectionSessionForm.dataset.submitting;
    updateCollectionCapabilityEvidence(false);
  }
}

function collectionActionButton(action, label, tone = "button-outline") {
  return `<button class="button ${tone}" type="button" data-collection-lifecycle-action="${action}">${label}</button>`;
}

function renderCollectionLifecycleActions(session) {
  const state = String(session.status || "DRAFT").toUpperCase();
  const buttons = [];
  if (collectionPrepareStates.has(state)) {
    buttons.push(collectionActionButton("preflight", "Preflight"));
    buttons.push(collectionActionButton("prepare", "Prepare sbatch"));
  }
  if (state === "READY")
    buttons.push(collectionActionButton("submit", "Submit", "button-accent"));
  if (collectionSubmittedStates.has(state)) {
    buttons.push(collectionActionButton("refresh", "Refresh status"));
    buttons.push(collectionActionButton("sbatch", "View sbatch"));
  }
  if (["SUBMITTED", "PENDING", "RUNNING"].includes(state))
    buttons.push(collectionActionButton("cancel", "Cancel"));
  if (!["COMPLETED", "FAILED", "CANCELLED"].includes(state))
    buttons.push(collectionActionButton("fail", "Record failure"));
  if (collectionTerminalStates.has(state))
    buttons.push(collectionActionButton("restart", "Restart as new"));
  elements.collectionSessionActions.innerHTML = buttons.join("");
}

function renderCollectionSessionDetail(session, launcher = null) {
  activeCollectionSession = session;
  const id = session.id || session.session_id;
  const state = String(session.status || "DRAFT").toUpperCase();
  const context = collectionSessionContext(session);
  const storage = session.storage_snapshot || {};
  elements.collectionSessionDetailTitle.textContent =
    session.name || `Collection ${shortId(id)}`;
  elements.collectionSessionMeta.innerHTML = [
    ["Status", session.status],
    ["Adapter", collectionSessionAdapterName(session)],
    ["Operator", context.operator],
    ["Task", context.task],
    ["Embodiment", context.embodiment],
    ["Slurm job", session.slurm_job_id || "-"],
    ["Gateway", session.gateway || session.resources_snapshot?.gateway || "-"],
    ["Native format", storage.native_format || "-"],
    ["Output path", storage.output_path || "-"],
    ["Manifest SHA-256", session.manifest_sha256 || "-"],
    ["Restart of", session.restart_of_session_id || "-"],
    ["Updated", formatDate(session.updated_at || session.created_at)],
  ]
    .map(
      ([label, value]) =>
        `<div class="key-value"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`,
    )
    .join("");
  renderCollectionLifecycleActions(session);

  const registered =
    session.registered_resource_id || session.registered_version_id;
  elements.collectionRegistryLink.hidden = !registered;
  elements.collectionRegistryLink.textContent = registered
    ? `Open registered data: ${session.registered_resource_id || "resource"} / ${session.registered_version_id || "version pending"}`
    : "Open registered data";

  elements.collectionCompleteForm.hidden = state !== "CAPTURED";
  elements.collectionCompleteForm.querySelector(
    "button[type=submit]",
  ).textContent = storage.registration
    ? "Complete and register native data"
    : "Complete without registration";
  elements.collectionCompleteManifest.value = "";
  elements.collectionCompleteRevision.value = "";
  elements.collectionCompleteSize.value = "";
  elements.collectionLoadStdout.disabled =
    !collectionSubmittedStates.has(state);
  elements.collectionLoadStderr.disabled =
    !collectionSubmittedStates.has(state);
  elements.collectionLogStatus.textContent = collectionSubmittedStates.has(
    state,
  )
    ? "Select a stream"
    : "Logs available after submission";
  updateLogView(elements.collectionSessionLog, "No log loaded.");
  updateLogView(
    elements.collectionSessionError,
    session.error ? collectionPretty(session.error) : "No error recorded.",
  );
  updateLogView(
    elements.collectionSessionResult,
    collectionPretty({
      status: session.status,
      manifest_sha256: session.manifest_sha256,
      sbatch_sha256: session.sbatch_sha256,
      canonical_manifest: session.canonical_manifest,
      storage: storage,
      registered_resource_id: session.registered_resource_id,
      registered_version_id: session.registered_version_id,
    }),
  );
  const events = Array.isArray(session.events) ? session.events : [];
  elements.collectionEventsBody.innerHTML = events.length
    ? events
        .map(
          (item) => `<tr>
    <td>${escapeHtml(item.event_type || item.type || item.action || "event")}</td>
    <td>${escapeHtml([item.old_status ?? item.from_status, item.new_status ?? item.to_status].filter(Boolean).join(" → ") || item.status || "-")}</td>
    <td class="wrap-cell">${escapeHtml(item.message || compactJson(item.details || item.payload, 240))}</td>
    <td>${escapeHtml(formatDate(item.created_at || item.timestamp))}</td>
  </tr>`,
        )
        .join("")
    : emptyRow(4, "No lifecycle events recorded.");
  revealPanel(elements.collectionSessionDetail, {
    focusTarget: elements.collectionSessionDetailTitle,
    launcher,
  });
}

async function viewCollectionSession(id, launcher = null) {
  const revealLauncher = launcher || currentRevealLauncher();
  const requestToken = disclosureToken(
    elements.collectionSessionDetail,
    revealLauncher,
  );
  try {
    const payload = await api(
      `/api/collection/sessions/${encodeURIComponent(id)}`,
    );
    if (
      !disclosureTokenIsCurrent(elements.collectionSessionDetail, requestToken)
    )
      return;
    renderCollectionSessionDetail(
      entityFrom(payload, "session"),
      revealLauncher,
    );
  } catch (error) {
    if (
      !disclosureTokenIsCurrent(elements.collectionSessionDetail, requestToken)
    )
      return;
    closeDisclosurePanel(elements.collectionSessionDetail, revealLauncher);
    showToast(`Session detail failed: ${error.message}`, true);
  }
}

function collectionSessionGateway(session) {
  return session.gateway || session.resources_snapshot?.gateway || "auto";
}

async function runCollectionLifecycleAction(action) {
  if (!activeCollectionSession) return;
  const id = activeCollectionSession.id || activeCollectionSession.session_id;
  let path = `/api/collection/sessions/${encodeURIComponent(id)}/${action}`;
  let options = { method: "POST" };
  if (["prepare", "submit"].includes(action)) {
    options.body = JSON.stringify({
      gateway: collectionSessionGateway(activeCollectionSession),
      remote_validate: true,
    });
  } else if (action === "cancel") {
    if (!(await askUserDialog("Cancel this collection job?"))) return;
    options.body = JSON.stringify({
      gateway: collectionSessionGateway(activeCollectionSession),
    });
  } else if (action === "restart") {
    options.body = JSON.stringify({ name: null });
  } else if (action === "fail") {
    const message = await askUserDialog(
      "Describe the collection failure. This is saved with the session.",
      "",
    );
    if (!message?.trim()) return;
    path = `/api/collection/sessions/${encodeURIComponent(id)}/error`;
    options.body = JSON.stringify({
      code: "OPERATOR_REPORTED",
      message: message.trim(),
      details: {},
    });
  } else if (action === "sbatch") {
    options = { method: "GET" };
  }
  try {
    const result = await api(path, options);
    if (action === "restart") {
      const restarted = entityFrom(result, "session");
      loadedTabs.delete("collection");
      await loadCollection(true);
      await viewCollectionSession(restarted.id || restarted.session_id);
    } else {
      await viewCollectionSession(id);
      updateLogView(
        elements.collectionSessionResult,
        typeof result === "string" ? result : collectionPretty(result),
      );
      loadedTabs.delete("collection");
      await loadCollection(true);
    }
    showToast(`Collection ${action} completed.`);
  } catch (error) {
    showToast(`Collection ${action} failed: ${error.message}`, true);
  }
}

async function loadCollectionLog(stream) {
  if (!activeCollectionSession) return;
  const id = activeCollectionSession.id || activeCollectionSession.session_id;
  const generation = beginLogViewUpdate(elements.collectionSessionLog);
  elements.collectionLogStatus.textContent = `Loading ${stream}...`;
  try {
    const result = await api(
      `/api/collection/sessions/${encodeURIComponent(id)}/logs?stream=${encodeURIComponent(stream)}&lines=500`,
    );
    if (
      (activeCollectionSession?.id || activeCollectionSession?.session_id) !==
      id
    )
      return;
    if (
      !updateLogView(
        elements.collectionSessionLog,
        typeof result === "string" ? result : collectionPretty(result),
        { generation },
      )
    )
      return;
    elements.collectionLogStatus.textContent = `${stream}, last 500 lines`;
  } catch (error) {
    if (
      (activeCollectionSession?.id || activeCollectionSession?.session_id) !==
      id
    )
      return;
    if (
      !updateLogView(elements.collectionSessionLog, error.message, {
        generation,
      })
    )
      return;
    elements.collectionLogStatus.textContent = `${stream} unavailable`;
  }
}

async function completeCollectionSession(event) {
  event.preventDefault();
  if (elements.collectionCompleteForm.dataset.submitting === "true") return;
  if (
    !activeCollectionSession ||
    !elements.collectionCompleteForm.reportValidity()
  )
    return;
  const id = activeCollectionSession.id || activeCollectionSession.session_id;
  const button = elements.collectionCompleteForm.querySelector(
    "button[type=submit]",
  );
  elements.collectionCompleteForm.dataset.submitting = "true";
  button.disabled = true;
  try {
    const size = numberOrNull(elements.collectionCompleteSize);
    const result = await api(
      `/api/collection/sessions/${encodeURIComponent(id)}/complete`,
      {
        method: "POST",
        body: JSON.stringify({
          manifest_sha256: elements.collectionCompleteManifest.value
            .trim()
            .toLowerCase(),
          revision: elements.collectionCompleteRevision.value.trim() || null,
          size_bytes: size,
          resource_metadata: {},
          version_metadata: {},
        }),
      },
    );
    loadedTabs.delete("collection");
    loadedTabs.delete("datasets");
    await loadCollection(true);
    await viewCollectionSession(id);
    updateLogView(elements.collectionSessionResult, collectionPretty(result));
    showToast(
      result.registered
        ? "Native capture completed and registered."
        : "Native capture completed without registration.",
    );
  } catch (error) {
    showToast(`Completion failed: ${error.message}`, true);
  } finally {
    delete elements.collectionCompleteForm.dataset.submitting;
    button.disabled = false;
  }
}

const tutorialTours = {
  experiments: {
    title: "Experiments",
    steps: [
      {
        selector: "#experiments-body",
        title: "Review existing experiments",
        instruction:
          "Start with the saved drafts and submitted experiments. Row actions appear only when records exist; this tour never opens Submit.",
      },
      {
        selector: "#experiment-name",
        title: "Name the experiment",
        instruction:
          "Use a unique, meaningful name. Naming does not save anything until you choose a save or submit action yourself.",
      },
      {
        selector: "#experiment-adapter",
        title: "Choose the code adapter",
        instruction:
          "Select only an adapter returned by the registry. It defines repository matching, runtime detection, commands, and parameter mappings.",
      },
      {
        selector: "#experiment-source",
        title: "Select the repository",
        instruction:
          "Enter the real repository URL or configured source. Do not invent a repository or place credentials in this field.",
      },
      {
        selector: "#experiment-branch",
        title: "Choose a returned branch",
        instruction:
          "Branches are loaded from the repository. Pick one of the returned values before choosing an exact commit.",
      },
      {
        selector: "#experiment-revision",
        title: "Pin the exact commit",
        instruction:
          "The revision makes the source reproducible. Use a commit returned by inspection rather than a moving branch name.",
      },
      {
        selector: "#experiment-runtime",
        title: "Confirm the detected runtime",
        instruction:
          "Choose a runnable runtime candidate based on repository evidence. If none is available, resolve the adapter or runtime profile first.",
      },
      {
        selector: "#experiment-data-bundle",
        title: "Choose training data",
        instruction:
          "Choose a prepared dataset. Its files, cameras and episode split are saved with this experiment.",
      },
      {
        selector: "#hp-learning-rate",
        title: "Set common hyperparameters",
        instruction:
          "Start with small, deliberate values. Nearby fields cover batch size, gradient accumulation, workers, precision, and maximum steps.",
      },
      {
        selector: "#resource-policy",
        title: "Choose resource policy",
        instruction:
          "Auto mode lets the service select one compatible node and GPU type. Manual choices should reflect current cluster availability.",
      },
      {
        selector: "#checkpoint-save-steps",
        title: "Define checkpoint behavior",
        instruction:
          "Set a save interval and review auto-resume and retention. A short smoke run should use a small maximum step count and time limit.",
      },
      {
        selector: "#sweep-definition",
        title: "Describe variants",
        instruction:
          "Leave this blank for one run, or use a small JSON sweep such as a seed list. Large sweeps multiply cluster work.",
      },
      {
        selector: "#mlflow-enabled",
        title: "Enable tracking only when configured",
        instruction:
          "MLflow is optional. Turn it on only when the tracking URI and experiment are known and accessible.",
      },
      {
        selector: "#evaluation-enabled",
        title: "Attach downstream evaluation",
        instruction:
          "Enable evaluation only when a compatible registered suite is available. This records intent in the experiment snapshot.",
      },
      {
        selector: "#experiment-preview-button",
        title: "Preview before any mutation",
        instruction:
          "Preview the canonical spec and sbatch manually, then inspect every path and command. The tutorial stops before Save and Submit; use those controls yourself only after review.",
      },
    ],
  },
  datasets: {
    title: "Datasets",
    steps: [
      {
        selector: "#data-resources-body",
        title: "Resources are named data items",
        instruction:
          "Browse registered datasets, simulator assets, and other sources here. Exact contents live in immutable versions.",
      },
      {
        selector: "#data-resource-provider",
        title: "Describe a missing source",
        instruction:
          "Normal imports register resources automatically. This manual form needs the real provider, owner, name, kind, and purpose.",
        reveal: ["#data-resource-form"],
      },
      {
        selector: "#data-versions-body",
        title: "Versions pin exact contents",
        instruction:
          "Each version identifies one unchanged set of files, revision, format, path, and inventory hash.",
      },
      {
        selector: "#data-version-revision",
        title: "Publish only verified files",
        instruction:
          "Use the actual upstream revision, existing canonical path, computed SHA-256, and READY state only after inventory verification. This form is normally opened from a resource row.",
        reveal: ["#data-version-form"],
      },
      {
        selector: "#data-derivations-body",
        title: "Read processing receipts",
        instruction:
          "Derivation rows show which immutable inputs and converter produced each processed output.",
      },
      {
        selector: "#data-derivation-inputs",
        title: "Pin every derivation input",
        instruction:
          "List real version IDs and their roles. Conversion workflows normally record this receipt automatically.",
        reveal: ["#data-derivation-form"],
      },
      {
        selector: "#data-converter-commit",
        title: "Pin the converter",
        instruction:
          "Record the exact converter repository and commit so the processed output can be reproduced.",
      },
    ],
  },
  collection: {
    title: "Collection",
    steps: [
      {
        selector: "#collection-adapters-body",
        title: "Start with collection adapters",
        instruction:
          "Adapters declare native action, proprioceptive, and sensor streams plus setup and launch requirements. Archived entries remain readable.",
      },
      {
        selector: "#add-collection-adapter",
        title: "Open an unsaved adapter",
        instruction:
          "This only opens a local editor. The tutorial reveals the same editor without saving, archiving, restoring, or cloning anything.",
      },
      {
        selector: "#collection-adapter-streams",
        title: "Declare native streams",
        instruction:
          "Describe each source key, kind, shape, data type, units, frame, rate, and whether it is required. Raw capture stays in this representation.",
        reveal: ["#collection-adapter-form"],
      },
      {
        selector: "#collection-adapter-capabilities",
        title: "Declare external requirements",
        instruction:
          "List host, compute, client, network, or operator capabilities. Unknown and administrator-required states should remain explicit.",
      },
      {
        selector: "#collection-sessions-body",
        title: "Review immutable sessions",
        instruction:
          "Existing sessions preserve their adapter, setup, sbatch, job identity, native output, events, and registration result.",
      },
      {
        selector: "#collection-session-adapter",
        title: "Choose a runnable adapter",
        instruction:
          "Only runnable adapters returned by the API can start a draft. An unresolved adapter remains visible in the registry but unavailable here.",
      },
      {
        selector: "#collection-session-operator",
        title: "Record collection context",
        instruction:
          "Enter the real operator. The nearby task and embodiment fields become part of the immutable config snapshot.",
      },
      {
        selector: "#collection-output-path",
        title: "Choose native staging storage",
        instruction:
          "Use a new path under the collection staging area and state the adapter-native format. Completion registers this exact output without conversion.",
      },
      {
        selector: "#collection-calibration-sha",
        title: "Pin real calibration",
        instruction:
          "Record the calibration identity and its exact 64-character SHA-256. Never fabricate this digest or paste credentials.",
      },
      {
        selector: "#collection-capture-schema-name",
        title: "Describe the capture clock and schema",
        instruction:
          "Name and version the native schema, then record its clock, nominal rate, timestamp unit, and whether timestamps are present.",
      },
      {
        selector: "#collection-alignment",
        title: "State temporal alignment",
        instruction:
          "Say whether actions precede post-state, observations precede actions, streams are synchronous, or the adapter defines alignment.",
      },
      {
        selector: "#collection-capabilities",
        title: "Record evidence honestly",
        instruction:
          "Leave evidence unknown unless it was independently checked. Verified evidence must include who checked it and when; administrator-required items stay explicit.",
      },
      {
        selector: "#collection-account",
        title: "Review Slurm allocation",
        instruction:
          "Auto gateway with rl2-lab account and partition is the normal default. Do not pin a node unless there is a specific reason.",
      },
      {
        selector: "#create-collection-session",
        title: "Stop before creating the draft",
        instruction:
          "This button writes an immutable DRAFT session. The tutorial does not press it and never preflights, prepares, submits, cancels, restarts, fails, completes, or archives anything. Exit and proceed manually.",
      },
    ],
  },
  runs: {
    title: "Training Runs",
    steps: [
      {
        selector: "#run-search",
        title: "Find a run",
        instruction:
          "Filter by experiment, variant, run ID, or Slurm job without changing any run state.",
      },
      {
        selector: "#run-status-filter",
        title: "Narrow by status",
        instruction:
          "Use the status filter to focus on failures, running jobs, or completed work. Filtering is local and read-only.",
      },
      {
        selector: "#runs-body",
        title: "Read the Training Run ledger",
        instruction:
          "Rows summarize state, Slurm identity, allocation, attempts, progress, and update time. View actions appear only when rows exist.",
      },
      {
        selector: "#run-detail-meta",
        title: "Inspect the immutable run detail",
        instruction:
          "A row's View action normally fills this panel, where Start evaluation opens Evaluations with the run ID prefilled. The tutorial reveals it without selecting a record or calling the API.",
        reveal: ["#run-detail"],
      },
      {
        selector: "#run-attempt-stderr-log",
        title: "Diagnose before retrying",
        instruction:
          "The latest attempt opens automatically. Select another attempt to inspect its stdout and stderr before acting. The tutorial never presses Resume, Retry, Cancel, or any control that can submit Slurm work.",
      },
    ],
  },
  evaluations: {
    title: "Evaluations",
    steps: [
      {
        selector: "#evaluation-run-id",
        title: "Choose an existing run",
        instruction:
          "Select the exact training run whose checkpoint should be evaluated.",
      },
      {
        selector: "#evaluation-checkpoint",
        title: "Pin the checkpoint",
        instruction:
          "Use a registered inference checkpoint or exact known path. Do not guess a checkpoint location.",
      },
      {
        selector: "#evaluation-suite",
        title: "Choose a returned suite",
        instruction:
          "Suites define evaluator behavior and environments. Select only a suite returned by the API.",
      },
      {
        selector: "#evaluation-episodes",
        title: "Start with a small evaluation",
        instruction:
          "Use one episode for a smoke check, then increase episode count only after the evaluator works.",
      },
      {
        selector: "#evaluation-seeds",
        title: "Make randomness explicit",
        instruction:
          "Enter deliberate seeds, such as 0 for a smoke check, so evaluation can be repeated.",
      },
      {
        selector: "#evaluation-parallelism",
        title: "Limit parallel work",
        instruction:
          "Start with parallelism 1 and headless mode. Increase only when the environment and cluster allocation support it.",
      },
      {
        selector: "#evaluation-auto-resume",
        title: "Plan failure recovery",
        instruction:
          "Auto-resume and maximum automatic attempts control recovery. Keep command overrides blank unless the evaluator contract requires them.",
      },
      {
        selector: "#submit-evaluation",
        title: "Stop before submission",
        instruction:
          "This button creates runnable evaluation state. The tutorial never presses it. Exit, review the complete request, and submit manually only when ready.",
      },
    ],
  },
  adapters: {
    title: "Adapters",
    steps: [
      {
        selector: "#adapters-body",
        title: "Review versioned adapters",
        instruction:
          "Adapters hold repository matching, runtime detection, commands, mappings, checkpoints, and evaluators. Row actions are record-dependent.",
      },
      {
        selector: "#add-adapter",
        title: "Open an unsaved editor",
        instruction:
          "This normally starts a local draft. The tutorial reveals the editor without creating, cloning, archiving, or restoring a record.",
      },
      {
        selector: "#adapter-editor-slug",
        title: "Set stable adapter identity",
        instruction:
          "Use a stable slug and clear name. Existing adapter keys and referenced versions remain immutable.",
        reveal: ["#adapter-editor"],
      },
      {
        selector: "#adapter-manifest",
        title: "Edit the canonical manifest",
        instruction:
          "Keep detection, runtime, launcher, and parameter declarations sourced from the real repository. Never paste secrets or invent executable commands.",
      },
      {
        selector: "#adapter-validation-repository",
        title: "Pin validation source",
        instruction:
          "Use the actual HTTPS repository and exact commit. Repository inspection is optional and should be run only with explicit consent.",
      },
      {
        selector: "#validate-adapter",
        title: "Validation is an explicit action",
        instruction:
          "Validation contacts the API and may inspect a remote repository. The tutorial does not press this button.",
      },
      {
        selector: "#save-adapter",
        title: "Stop before saving",
        instruction:
          "Saving creates a new immutable adapter version. The tutorial stops here and never saves, clones, archives, restores, or deletes anything. Exit and act manually after review.",
      },
    ],
  },
};

const interactiveTutorialTours = {
  experiments: {
    title: "Experiments",
    steps: [
      {
        id: "name",
        selector: "#experiment-name",
        gate: "field",
        title: "Name this tutorial experiment",
        instruction:
          "Enter a unique name. Use the generated value if you want a clearly marked tutorial record.",
        useValue: ({ token }) => `${token}-experiment`,
      },
      {
        id: "adapter",
        selector: "#experiment-adapter",
        gate: "field",
        title: "Choose a real adapter",
        instruction:
          "Change this to an adapter returned by the registry. The tutorial will not invent one.",
        validate: tutorialNonEmpty,
      },
      {
        id: "source",
        selector: "#experiment-source",
        gate: "field",
        title: "Use the real repository",
        instruction:
          "Enter or confirm the actual repository URL. Credentials do not belong here.",
        validate: tutorialNonEmpty,
      },
      {
        id: "branch",
        selector: "#experiment-branch",
        gate: "field",
        title: "Choose a returned branch",
        instruction: "Select a branch loaded from repository inspection.",
        validate: tutorialNonEmpty,
      },
      {
        id: "revision",
        selector: "#experiment-revision",
        gate: "field",
        title: "Pin a returned commit",
        instruction:
          "Select the exact commit. A moving branch name is not reproducible.",
        validate: tutorialNonEmpty,
      },
      {
        id: "runtime",
        selector: "#experiment-runtime",
        gate: "field",
        title: "Choose a runnable runtime",
        instruction:
          "Select a detected runnable candidate; resolve adapter evidence first if none exists.",
        validate: tutorialNonEmpty,
      },
      {
        id: "adapter-fields",
        selector: "#adapter-declared-fields",
        gate: "field",
        title: "Set adapter-declared inputs",
        instruction:
          "Review the typed values declared by this adapter. Required fields must be valid; Use tutorial values only applies examples explicitly supplied by the manifest. If none is supplied, choose the real value yourself.",
        validate: tutorialAdapterDeclaredFieldsValid,
        applyValue: tutorialApplyAdapterDeclaredValues,
      },
      {
        id: "learning-rate",
        selector: "#hp-learning-rate",
        skipIf: () => elements.hpLearningRate.disabled,
        gate: "field",
        title: "Set a smoke-test learning rate",
        instruction:
          "Enter a deliberate value. The generated value is only a starting point.",
        useValue: () => "0.0003",
        validate: tutorialPositiveNumber,
      },
      {
        id: "max-steps",
        selector: () =>
          !elements.hpMaxSteps.disabled
            ? "#hp-max-steps"
            : '#adapter-declared-fields input[data-adapter-input-path="native.config.epochs"]',
        skipIf: () =>
          elements.hpMaxSteps.disabled &&
          !document.querySelector(
            '#adapter-declared-fields input[data-adapter-input-path="native.config.epochs"]',
          ),
        gate: "field",
        title: "Keep the first run short",
        instruction:
          "Use a small step or epoch count supported by this adapter.",
        useValue: () => (elements.hpMaxSteps.disabled ? "1" : "10"),
        validate: tutorialPositiveNumber,
      },
      {
        id: "resource-policy",
        selector: "#resource-policy",
        gate: "field",
        title: "Choose resource policy",
        instruction:
          "Change or confirm automatic allocation unless you have a reason to pin hardware.",
        useValue: () => "auto",
        validate: tutorialNonEmpty,
      },
      {
        id: "preview",
        selector: "#experiment-preview-button",
        gate: "action",
        title: "Generate the real preview",
        instruction:
          "Click Preview. This performs only POST /api/experiments/preview and must return a canonical script before the tour advances.",
        request: {
          method: "POST",
          path: "/api/experiments/preview",
          success: tutorialHasExperimentPreview,
        },
      },
      {
        id: "save-boundary",
        selector: "#experiment-workspace .page-heading h1",
        title: "Saving is outside this tutorial",
        instruction:
          "The verified preview is the safe endpoint. Saving creates a permanent experiment with no cleanup API, and submitting can launch Slurm, so neither action is part of this tutorial.",
      },
    ],
  },
  datasets: {
    title: "Datasets",
    steps: [
      {
        id: "open-resource",
        selector: "#show-data-resource-form",
        gate: "action",
        title: "Open a clean resource draft",
        instruction:
          "Click until the clean resource form is open. The canonical Show/Hide control resets edit state only when opening and makes no API request.",
        localPredicate: () =>
          !document.querySelector("#data-resource-form")?.hidden,
      },
      {
        id: "provider",
        selector: "#data-resource-provider",
        gate: "field",
        title: "Choose the source provider",
        instruction:
          "This lifecycle creates a disposable named resource only. Exact file versions remain separate and immutable.",
        useValue: () => "local",
        validate: tutorialNonEmpty,
      },
      {
        id: "namespace",
        selector: "#data-resource-namespace",
        gate: "field",
        title: "Use the tutorial namespace",
        instruction:
          "Keep the disposable resource isolated from production names.",
        useValue: () => "tutorial",
        validate: tutorialNonEmpty,
      },
      {
        id: "name",
        selector: "#data-resource-name",
        gate: "field",
        title: "Give the resource a unique name",
        instruction: "The generated name carries this tutorial session token.",
        useValue: ({ token }) => `${token}-resource`,
        validate: tutorialNonEmpty,
      },
      {
        id: "kind",
        selector: "#data-resource-kind",
        gate: "field",
        title: "Declare the data kind",
        instruction:
          "Choose the role that matches the source. This example is only a demonstrations registry record, not uploaded data.",
        useValue: () => "demonstrations",
        validate: tutorialNonEmpty,
      },
      {
        id: "description",
        selector: "#data-resource-description",
        gate: "field",
        title: "Tag ownership",
        instruction:
          "The token in this description lets the tutorial verify the exact record before editing or archiving it.",
        useValue: ({ token }) => `Disposable tutorial resource ${token}`,
      },
      {
        id: "create",
        selector: "#create-data-resource",
        gate: "action",
        risk: "persistent",
        confirmVerb: "CREATE",
        title: "Create the resource",
        instruction:
          "Click Register once, type the confirmation, then click it again. Advancement requires a successful POST returning resource.id.",
        request: {
          method: "POST",
          path: "/api/data/resources",
          bind: "resourceId",
          entity: "resource",
          establishBinding: true,
          identity: tutorialResourceIdentity,
          preflight: tutorialResourceFormIdentity,
          record: { kind: "data-resource", cleanup: "archive" },
          render: ({ bindings }) =>
            tutorialBoundSelector(
              "resource-id",
              bindings.resourceId,
              '[data-resource-action="edit"]',
            ),
        },
      },
      {
        id: "read",
        selector: ({ bindings }) =>
          tutorialBoundSelector(
            "resource-id",
            bindings.resourceId,
            '[data-resource-action="edit"]',
          ),
        gate: "action",
        waitForTarget: true,
        title: "Read the exact created resource",
        instruction:
          "Click View / edit on the bound row. The exact GET response must match the bound ID and immutable tutorial name.",
        request: {
          method: "GET",
          path: ({ bindings }) =>
            `/api/data/resources/${encodeURIComponent(bindings.resourceId)}`,
          bind: "resourceId",
          entity: "resource",
          identity: tutorialResourceIdentity,
        },
      },
      {
        id: "edit-description",
        selector: "#data-resource-description",
        reveal: ["#data-resource-form"],
        gate: "field",
        title: "Edit the description",
        instruction:
          "Change the description while retaining the token so ownership remains verifiable.",
        useValue: ({ token }) =>
          `Updated disposable tutorial resource ${token}`,
        validate: tutorialContainsToken,
      },
      {
        id: "update",
        selector: "#create-data-resource",
        gate: "action",
        risk: "persistent",
        confirmVerb: "UPDATE",
        requiresOwned: "resourceId",
        title: "Save the real update",
        instruction:
          "Click Save once, type the confirmation, then click again. Advancement requires a matching PATCH and the same resource ID.",
        request: {
          method: "PATCH",
          path: ({ bindings }) =>
            `/api/data/resources/${encodeURIComponent(bindings.resourceId)}`,
          bind: "resourceId",
          entity: "resource",
          identity: tutorialResourceIdentity,
        },
      },
      {
        id: "archive",
        selector: ({ bindings }) =>
          tutorialBoundSelector(
            "resource-id",
            bindings.resourceId,
            '[data-resource-action="archive"]',
          ),
        gate: "action",
        risk: "cleanup",
        confirmVerb: "ARCHIVE",
        requiresOwned: "resourceId",
        waitForTarget: true,
        title: "Archive the tutorial resource",
        instruction:
          "Archiving is the supported cleanup; it does not delete payload bytes. Confirm and click the bound Archive control again.",
        request: {
          method: "DELETE",
          path: ({ bindings }) =>
            `/api/data/resources/${encodeURIComponent(bindings.resourceId)}`,
          bind: "resourceId",
          entity: "resource",
          identity: tutorialResourceIdentity,
          cleanupBinding: "resourceId",
        },
      },
    ],
  },
  collection: {
    title: "Collection",
    steps: [
      {
        id: "open-adapter",
        selector: "#add-collection-adapter",
        gate: "action",
        title: "Open a local adapter draft",
        instruction:
          "Click New adapter. This opens the editor without writing to the database.",
      },
      {
        id: "key",
        selector: "#collection-adapter-key",
        reveal: ["#collection-adapter-form"],
        gate: "field",
        title: "Set a unique adapter key",
        instruction: "The generated key is isolated to this tutorial session.",
        useValue: ({ token }) =>
          tutorialAdapterSlug(token, "collection-adapter"),
        validate: tutorialNonEmpty,
      },
      {
        id: "display-name",
        selector: "#collection-adapter-name",
        gate: "field",
        title: "Name the adapter",
        instruction:
          "Use a clear display name that carries the tutorial token.",
        useValue: ({ token }) => `${token} inert collection adapter`,
        validate: tutorialNonEmpty,
      },
      {
        id: "version",
        selector: "#collection-adapter-version",
        gate: "field",
        title: "Set the first version",
        instruction: "Start the disposable adapter at a deliberate version.",
        useValue: () => "0.0.1",
        validate: tutorialNonEmpty,
      },
      {
        id: "runnable",
        selector: "#collection-adapter-runnable",
        gate: "field",
        title: "Keep it non-runnable",
        instruction:
          "This tutorial adapter must remain inert and cannot launch collection work.",
        useValue: () => false,
        validate: (target) => !target.checked,
      },
      {
        id: "streams",
        selector: "#collection-adapter-streams",
        gate: "field",
        title: "Declare one typed native stream",
        instruction:
          "This inert declaration demonstrates native sensor semantics without conversion.",
        useValue: () =>
          JSON.stringify(
            [
              {
                name: "tutorial.marker",
                kind: "sensor",
                shape: [1],
                dtype: "float32",
                units: null,
                frame: null,
                rate_hz: 1,
                native_key: "tutorial.marker",
                required: false,
                metadata: {},
              },
            ],
            null,
            2,
          ),
        validate: tutorialJsonArray,
      },
      {
        id: "launcher",
        selector: "#collection-adapter-launcher",
        gate: "field",
        title: "Keep the launcher inert",
        instruction:
          "An empty step list makes the non-runnable tutorial adapter incapable of executing a process.",
        useValue: () => JSON.stringify({ environment: {}, steps: [] }, null, 2),
        validate: tutorialJsonObject,
      },
      {
        id: "description",
        selector: "#collection-adapter-description",
        gate: "field",
        title: "Tag the adapter",
        instruction:
          "Retain the token so later update and archive steps can verify ownership.",
        useValue: ({ token }) =>
          `Disposable non-runnable collection adapter ${token}`,
        validate: tutorialContainsToken,
      },
      {
        id: "metadata",
        selector: "#collection-adapter-metadata",
        gate: "field",
        title: "Record tutorial metadata",
        instruction:
          "Store only the non-secret session token under the exact tutorial_token field.",
        useValue: ({ token }) =>
          JSON.stringify({ tutorial_token: token }, null, 2),
        validate: tutorialJsonObject,
      },
      {
        id: "create",
        selector: "#save-collection-adapter",
        gate: "action",
        risk: "persistent",
        confirmVerb: "CREATE",
        title: "Create the inert adapter",
        instruction:
          "Confirm, then click Save again. A successful POST must return adapter.id.",
        request: {
          method: "POST",
          path: "/api/collection/adapters",
          bind: "collectionAdapterId",
          entity: "adapter",
          establishBinding: true,
          identity: tutorialCollectionAdapterIdentity,
          preflight: tutorialCollectionAdapterFormIdentity,
          record: { kind: "collection-adapter", cleanup: "archive" },
          render: ({ bindings }) =>
            tutorialBoundSelector(
              "collection-adapter-id",
              bindings.collectionAdapterId,
              '[data-collection-adapter-action="edit"]',
            ),
        },
      },
      {
        id: "read",
        selector: ({ bindings }) =>
          tutorialBoundSelector(
            "collection-adapter-id",
            bindings.collectionAdapterId,
            '[data-collection-adapter-action="edit"]',
          ),
        gate: "action",
        waitForTarget: true,
        title: "Read the bound adapter",
        instruction:
          "Click Edit on this exact row. The GET result must match both its bound key and structured tutorial metadata.",
        request: {
          method: "GET",
          path: ({ bindings }) =>
            `/api/collection/adapters/${encodeURIComponent(bindings.collectionAdapterId)}`,
          bind: "collectionAdapterId",
          entity: "adapter",
          identity: tutorialCollectionAdapterIdentity,
        },
      },
      {
        id: "update-version",
        selector: "#collection-adapter-version",
        reveal: ["#collection-adapter-form"],
        gate: "field",
        title: "Version the edit",
        instruction: "Change the version before saving the update.",
        useValue: () => "0.0.2",
        validate: tutorialNonEmpty,
      },
      {
        id: "update-description",
        selector: "#collection-adapter-description",
        gate: "field",
        title: "Describe the update",
        instruction:
          "Change the description while retaining the tutorial token.",
        useValue: ({ token }) =>
          `Updated disposable collection adapter ${token}`,
        validate: tutorialContainsToken,
      },
      {
        id: "update",
        selector: "#save-collection-adapter",
        gate: "action",
        risk: "persistent",
        confirmVerb: "UPDATE",
        requiresOwned: "collectionAdapterId",
        title: "Save the adapter update",
        instruction:
          "Confirm and click Save again. The matching PUT must return the same adapter ID.",
        request: {
          method: "PUT",
          path: ({ bindings }) =>
            `/api/collection/adapters/${encodeURIComponent(bindings.collectionAdapterId)}`,
          bind: "collectionAdapterId",
          entity: "adapter",
          identity: tutorialCollectionAdapterIdentity,
        },
      },
      {
        id: "archive",
        selector: ({ bindings }) =>
          tutorialBoundSelector(
            "collection-adapter-id",
            bindings.collectionAdapterId,
            '[data-collection-adapter-action="archive"]',
          ),
        gate: "action",
        risk: "cleanup",
        confirmVerb: "ARCHIVE",
        requiresOwned: "collectionAdapterId",
        waitForTarget: true,
        title: "Archive the inert adapter",
        instruction:
          "Confirm and click Archive again. This is final cleanup; the tutorial never restores it automatically.",
        request: {
          method: "DELETE",
          path: ({ bindings }) =>
            `/api/collection/adapters/${encodeURIComponent(bindings.collectionAdapterId)}`,
          bind: "collectionAdapterId",
          entity: "adapter",
          identity: tutorialCollectionAdapterIdentity,
          cleanupBinding: "collectionAdapterId",
        },
      },
      {
        id: "session-name",
        selector: "#collection-session-name",
        gate: "field",
        optional: true,
        title: "Optional immutable session draft",
        instruction:
          "You may fill a session using a real runnable adapter, operator, calibration, native format, timing, and storage path. Skip if those facts are unavailable.",
        useValue: ({ token }) => `${token}-session`,
      },
      {
        id: "session-boundary",
        selector: "#data .page-heading h1",
        title: "Session creation is outside this tutorial",
        instruction:
          "Collection sessions are immutable and have no cleanup endpoint. Prepare, submit, cancel, restart, fail, and complete can also change state or contact Slurm, so the tutorial stops before them.",
      },
    ],
  },
  runs: {
    title: "Training Runs",
    steps: [
      {
        id: "search",
        selector: "#run-search",
        gate: "field",
        title: "Search the real run ledger",
        instruction:
          "Type a real experiment, variant, run, or Slurm identifier. Filtering is local and does not alter a run.",
      },
      {
        id: "status",
        selector: "#run-status-filter",
        gate: "field",
        title: "Change the status filter",
        instruction: "Choose a status that helps locate a real record.",
      },
      {
        id: "detail",
        selector: () => '#runs-body [data-run-action="view"]',
        gate: "action",
        waitForTarget: true,
        title: "Read one real run",
        instruction:
          "Click View on a returned row. Advancement requires the matching run-detail GET and returned ID to match that selected row.",
        request: {
          method: "GET",
          path: ({ bindings }) =>
            `/api/runs/${encodeURIComponent(bindings.runId)}`,
          bind: "runId",
          bindFromTarget: true,
          entity: "run",
        },
      },
      {
        id: "limits",
        selector: "#run-detail-meta",
        reveal: ["#run-detail"],
        title: "Training Runs are operational records",
        instruction:
          "Training Runs originate from submitted experiments and cannot be deleted. Start evaluation opens Evaluations with this run selected; Resume and rerun actions can contact Slurm and are never invoked by this standard tour.",
      },
    ],
  },
  evaluations: {
    title: "Evaluations",
    steps: [
      {
        id: "run",
        selector: "#evaluation-run-id",
        gate: "field",
        title: "Choose a real run",
        instruction:
          "Enter an existing run ID. The service will reject fabricated or incompatible records.",
        validate: tutorialNonEmpty,
      },
      {
        id: "checkpoint",
        selector: "#evaluation-checkpoint",
        gate: "field",
        title: "Pin the real checkpoint",
        instruction:
          "Enter the exact registered inference checkpoint or known canonical path.",
        validate: tutorialNonEmpty,
      },
      {
        id: "suite",
        selector: "#evaluation-suite",
        gate: "field",
        title: "Choose a returned suite",
        instruction: "Change this to a suite returned by the service.",
        validate: tutorialNonEmpty,
      },
      {
        id: "episodes",
        selector: "#evaluation-episodes",
        gate: "field",
        title: "Start with one episode",
        instruction: "A one-episode smoke check limits initial work.",
        useValue: () => "1",
        validate: tutorialPositiveNumber,
      },
      {
        id: "seeds",
        selector: "#evaluation-seeds",
        gate: "field",
        title: "Make the seed explicit",
        instruction:
          "Use a deliberate seed; zero is a conventional smoke-test choice.",
        useValue: () => "0",
        validate: tutorialNonEmpty,
      },
      {
        id: "parallelism",
        selector: "#evaluation-parallelism",
        gate: "field",
        title: "Limit parallelism",
        instruction:
          "Start with one worker until the evaluator and environment are proven.",
        useValue: () => "1",
        validate: tutorialPositiveNumber,
      },
      {
        id: "submit-boundary",
        selector: "#evaluations .page-heading h1",
        title: "Submission is outside this tutorial",
        instruction:
          "Evaluation submission can create a permanent record and real Slurm work. It remains excluded until server-enforced tutorial GPU, time, and job-count caps exist.",
      },
    ],
  },
  adapters: {
    title: "Adapters",
    steps: [
      {
        id: "open",
        selector: "#add-adapter",
        gate: "action",
        title: "Open a local adapter draft",
        instruction:
          "Click New adapter. No record exists until the save step succeeds.",
      },
      {
        id: "slug",
        selector: "#adapter-editor-slug",
        reveal: ["#adapter-editor"],
        gate: "field",
        title: "Set a unique slug",
        instruction:
          "The token isolates this adapter from production definitions.",
        useValue: ({ token }) => tutorialAdapterSlug(token, "adapter"),
        validate: tutorialNonEmpty,
      },
      {
        id: "name",
        selector: "#adapter-editor-name",
        gate: "field",
        title: "Name the adapter",
        instruction: "Use a clear tutorial-only display name.",
        useValue: ({ token }) => `${token} experiment adapter`,
        validate: tutorialNonEmpty,
      },
      {
        id: "manifest",
        selector: "#adapter-manifest",
        gate: "field",
        title: "Review the generated manifest",
        instruction:
          "The normal Add flow provides the valid minimal manifest. Use tutorial value only to acknowledge that exact local text; it does not invent commands.",
        useValue: (_context, target) => target.value,
        validate: tutorialJsonObject,
      },
      {
        id: "description",
        selector: "#adapter-description",
        gate: "field",
        title: "Tag ownership",
        instruction:
          "Retain the tutorial token so read, update, and archive can stay bound to this record.",
        useValue: ({ token }) => `Disposable experiment adapter ${token}`,
        validate: tutorialContainsToken,
      },
      {
        id: "change-note",
        selector: "#adapter-change-note",
        gate: "field",
        title: "Explain the first version",
        instruction: "Record why this tutorial version exists.",
        useValue: ({ token }) => `Create disposable tutorial adapter ${token}`,
        validate: tutorialNonEmpty,
      },
      {
        id: "create",
        selector: "#save-adapter",
        gate: "action",
        risk: "persistent",
        confirmVerb: "CREATE",
        title: "Create the adapter",
        instruction:
          "Confirm and click Save again. A matching POST must return adapter.id.",
        request: {
          method: "POST",
          path: "/api/adapters",
          bind: "adapterId",
          entity: "adapter",
          establishBinding: true,
          identity: tutorialExperimentAdapterIdentity,
          preflight: tutorialExperimentAdapterFormIdentity,
          record: { kind: "experiment-adapter", cleanup: "archive" },
          render: ({ bindings }) =>
            tutorialBoundSelector(
              "adapter-id",
              bindings.adapterId,
              '[data-adapter-action="view"]',
            ),
        },
      },
      {
        id: "read",
        selector: ({ bindings }) =>
          tutorialBoundSelector(
            "adapter-id",
            bindings.adapterId,
            '[data-adapter-action="view"]',
          ),
        gate: "action",
        waitForTarget: true,
        title: "Read the bound adapter",
        instruction:
          "Click View on the exact created row. Its GET response must match the immutable tutorial slug.",
        request: {
          method: "GET",
          path: ({ bindings }) =>
            `/api/adapters/${encodeURIComponent(bindings.adapterId)}`,
          bind: "adapterId",
          entity: "adapter",
          identity: tutorialExperimentAdapterIdentity,
        },
      },
      {
        id: "edit",
        selector: "#edit-adapter",
        reveal: ["#adapter-editor"],
        gate: "action",
        title: "Enter local edit mode",
        instruction:
          "Click Edit. This changes only the local editor until Save succeeds.",
      },
      {
        id: "update-description",
        selector: "#adapter-description",
        gate: "field",
        title: "Change the description",
        instruction: "Retain the token while describing the update.",
        useValue: ({ token }) =>
          `Updated disposable experiment adapter ${token}`,
        validate: tutorialContainsToken,
      },
      {
        id: "update-note",
        selector: "#adapter-change-note",
        gate: "field",
        title: "Explain the immutable version",
        instruction:
          "Adapter updates create a new immutable version under the same record.",
        useValue: ({ token }) => `Tutorial update ${token}`,
        validate: tutorialNonEmpty,
      },
      {
        id: "update",
        selector: "#save-adapter",
        gate: "action",
        risk: "persistent",
        confirmVerb: "UPDATE",
        requiresOwned: "adapterId",
        title: "Save the new adapter version",
        instruction:
          "Confirm and click Save again. Advancement requires a matching PUT returning the same adapter ID.",
        request: {
          method: "PUT",
          path: ({ bindings }) =>
            `/api/adapters/${encodeURIComponent(bindings.adapterId)}`,
          bind: "adapterId",
          entity: "adapter",
          identity: tutorialExperimentAdapterIdentity,
        },
      },
      {
        id: "archive",
        selector: ({ bindings }) =>
          tutorialBoundSelector(
            "adapter-id",
            bindings.adapterId,
            '[data-adapter-action="archive"]',
          ),
        gate: "action",
        risk: "cleanup",
        confirmVerb: "ARCHIVE",
        requiresOwned: "adapterId",
        waitForTarget: true,
        title: "Archive the tutorial adapter",
        instruction:
          "Confirm and click the bound Archive action again. The tutorial does not restore it after cleanup.",
        request: {
          method: "DELETE",
          path: ({ bindings }) =>
            `/api/adapters/${encodeURIComponent(bindings.adapterId)}`,
          bind: "adapterId",
          entity: "adapter",
          identity: tutorialExperimentAdapterIdentity,
          cleanupBinding: "adapterId",
        },
      },
    ],
  },
};

Object.assign(tutorialTours, interactiveTutorialTours);

const tutorialState = {
  active: false,
  page: null,
  index: -1,
  target: null,
  launcher: null,
  originalFocus: null,
  originalTab: null,
  scrollX: 0,
  scrollY: 0,
  openedDetails: new Map(),
  revealed: new Map(),
  animationFrame: 0,
  settleTimer: 0,
  resizeObserver: null,
  pageObserver: null,
  renderObserver: null,
  renderTimer: 0,
  recoveryTimer: 0,
  session: null,
  sessionGeneration: null,
  gateGeneration: 0,
  pendingAttempt: null,
  inflightClaim: null,
  attemptTimer: 0,
  confirmationArmed: false,
  applyingValue: false,
  controlsBound: false,
  gateComplete: false,
  ownedVerified: new Map(),
  recovering: false,
};

const tutorialUi = {
  token: document.querySelector("#tutorial-token"),
  gateStatus: document.querySelector("#tutorial-gate-status"),
  useValue: document.querySelector("#tutorial-use-value"),
  confirmPanel: document.querySelector("#tutorial-confirm-panel"),
  confirmSummary: document.querySelector("#tutorial-confirm-summary"),
  confirmPhrase: document.querySelector("#tutorial-confirm-phrase"),
  confirmInput: document.querySelector("#tutorial-confirm-input"),
  error: document.querySelector("#tutorial-error"),
  records: document.querySelector("#tutorial-records"),
  resumePanel: document.querySelector("#tutorial-resume-panel"),
  resume: document.querySelector("#tutorial-resume"),
  restart: document.querySelector("#tutorial-restart"),
  blockers: ["top", "left", "right", "bottom"].map((side) =>
    document.querySelector(`#tutorial-blocker-${side}`),
  ),
};

function tutorialNonEmpty(target) {
  return Boolean(target.value?.trim());
}
function tutorialPositiveNumber(target) {
  return Number(target.value) > 0;
}
function tutorialContainsToken(target, context) {
  return target.value.includes(context.token);
}
function tutorialJson(target, array) {
  try {
    const value = JSON.parse(target.value);
    return array
      ? Array.isArray(value)
      : Boolean(value && typeof value === "object" && !Array.isArray(value));
  } catch {
    return false;
  }
}
function tutorialJsonArray(target) {
  return tutorialJson(target, true);
}
function tutorialJsonObject(target) {
  return tutorialJson(target, false);
}
function tutorialHasExperimentPreview(payload) {
  if (experimentPreviewShapeErrors(payload).length) return false;
  return (
    payload.blockers.length === 0 &&
    payload.scripts.length === payload.variant_count &&
    payload.scripts.every(experimentPreviewScriptIsRunnable) &&
    experimentPreviewScriptIsRunnable(payload.script)
  );
}

function tutorialResourceIdentity({ token }) {
  return [{ paths: ["name"], value: `${token}-resource` }];
}

function tutorialAdapterSlug(token, suffix) {
  // Legacy saved tutorials used uppercase timestamp characters. Keep their
  // ownership token intact while generating a valid adapter identifier.
  return `${String(token).toLowerCase()}-${suffix}`;
}

function tutorialCollectionAdapterIdentity({ token }) {
  return [
    {
      paths: ["adapter_key", "key", "manifest.key"],
      value: tutorialAdapterSlug(token, "collection-adapter"),
    },
    { paths: ["manifest.metadata.tutorial_token"], value: token },
  ];
}

function tutorialExperimentAdapterIdentity({ token }) {
  return [
    {
      paths: ["slug", "manifest.slug"],
      value: tutorialAdapterSlug(token, "adapter"),
    },
  ];
}

function tutorialResourceFormIdentity({ token }) {
  return (
    document.querySelector("#data-resource-name")?.value === `${token}-resource`
  );
}

function tutorialCollectionAdapterFormIdentity({ token }) {
  if (
    document.querySelector("#collection-adapter-key")?.value !==
    tutorialAdapterSlug(token, "collection-adapter")
  )
    return false;
  try {
    return (
      JSON.parse(
        document.querySelector("#collection-adapter-metadata")?.value || "{}",
      ).tutorial_token === token
    );
  } catch {
    return false;
  }
}

function tutorialExperimentAdapterFormIdentity({ token }) {
  return (
    document.querySelector("#adapter-editor-slug")?.value ===
    tutorialAdapterSlug(token, "adapter")
  );
}

function tutorialContext() {
  return {
    token: tutorialState.session?.token || "",
    bindings: tutorialState.session?.bindings || {},
  };
}

function tutorialBoundSelector(attribute, id, suffix = "") {
  if (!id) return "[data-tutorial-missing-record]";
  const escaped = window.CSS?.escape
    ? window.CSS.escape(String(id))
    : String(id).replace(/["\\]/g, "\\$&");
  return `[data-${attribute}="${escaped}"]${suffix ? ` ${suffix}` : ""}`;
}

function tutorialResolve(value) {
  return typeof value === "function" ? value(tutorialContext()) : value;
}

function tutorialProgressKey(page) {
  return `skynet.tutorial.progress.v2.${page}`;
}

function tutorialUniqueId(prefix) {
  if (window.crypto?.randomUUID)
    return `${prefix}-${window.crypto.randomUUID()}`;
  if (!window.crypto?.getRandomValues)
    throw new Error(
      "Secure randomness is unavailable; the interactive tutorial cannot start safely.",
    );
  const bytes = new Uint8Array(16);
  window.crypto.getRandomValues(bytes);
  return `${prefix}-${[...bytes].map((value) => value.toString(16).padStart(2, "0")).join("")}`;
}

function tutorialToken() {
  const stamp = new Date()
    .toISOString()
    .replace(/[-:]/g, "")
    .replace(/\.\d{3}Z$/, "Z");
  const suffix = tutorialUniqueId("token")
    .replace(/[^a-z0-9]/gi, "")
    .slice(-12)
    .toLowerCase();
  return `tutorial-${stamp}-${suffix}`.toLowerCase();
}

function newTutorialSession(page, retainedRecords = []) {
  const token = tutorialToken();
  const sessionGeneration = tutorialUniqueId("session");
  return {
    schema: 1,
    definitionVersion: 3,
    page,
    stepId: tutorialTours[page].steps[0]?.id || null,
    sessionId: token,
    sessionGeneration,
    token,
    completedStepIds: [],
    bindings: {},
    ownedRecords: retainedRecords,
    finished: false,
    updatedAt: new Date().toISOString(),
  };
}

function loadTutorialSession(page) {
  try {
    const saved = JSON.parse(
      window.localStorage.getItem(
        workspaceStorageKey(tutorialProgressKey(page)),
      ) || "null",
    );
    if (!saved || saved.page !== page || !saved.token) return null;
    if (saved.definitionVersion === 2) {
      saved.definitionVersion = 3;
      saved.sessionGeneration = tutorialUniqueId("migrated-session");
      saved.ownedRecords = (saved.ownedRecords || []).map((record) => ({
        ...record,
        ownerToken: record.ownerToken || record.token || saved.token,
        expectedIdentity: record.expectedIdentity || [],
        createdGeneration: record.createdGeneration || saved.sessionGeneration,
        cleanupState:
          record.cleanupState === "cleanup pending"
            ? "manual identity verification required"
            : record.cleanupState,
      }));
    }
    return saved.definitionVersion === 3 ? saved : null;
  } catch {
    return null;
  }
}

function persistTutorialSession() {
  if (!tutorialState.session) return;
  tutorialState.session.updatedAt = new Date().toISOString();
  try {
    window.localStorage.setItem(
      workspaceStorageKey(tutorialProgressKey(tutorialState.session.page)),
      JSON.stringify(tutorialState.session),
    );
  } catch {
    /* Progress persistence is optional. */
  }
}

function tutorialStep() {
  return tutorialTours[tutorialState.page]?.steps[tutorialState.index] || null;
}

function tutorialSetError(message = "") {
  tutorialUi.error.textContent = String(message).slice(0, 600);
  tutorialUi.error.hidden = !message;
}

function renderTutorialRecords() {
  if (!tutorialState.session) return;
  tutorialUi.token.textContent = tutorialState.session.token;
  const records = tutorialState.session.ownedRecords || [];
  tutorialUi.records.textContent = records.length
    ? `Records: ${records.map((record) => `${record.kind} ${record.id} owner=${record.ownerToken || "unknown"} (${record.cleanupState || (record.cleanup === "archive" ? "cleanup pending" : "retained; no cleanup API")})`).join("; ")}`
    : "";
}

function tutorialValidateField(step, target) {
  if (!target || target.disabled) return false;
  if (typeof target.checkValidity === "function" && !target.checkValidity())
    return false;
  return step.validate
    ? Boolean(step.validate(target, tutorialContext()))
    : target.type === "checkbox" || tutorialNonEmpty(target);
}

function resetTutorialAttempt(message = "") {
  if (tutorialState.attemptTimer)
    window.clearTimeout(tutorialState.attemptTimer);
  tutorialState.attemptTimer = 0;
  tutorialState.pendingAttempt = null;
  tutorialState.inflightClaim = null;
  tutorialState.confirmationArmed = false;
  tutorialUi.confirmPanel.hidden = true;
  tutorialUi.confirmInput.value = "";
  if (message && tutorialState.active)
    tutorialUi.gateStatus.textContent = message;
}

function scheduleTutorialAttemptExpiry(attempt) {
  if (tutorialState.attemptTimer)
    window.clearTimeout(tutorialState.attemptTimer);
  tutorialState.attemptTimer = window.setTimeout(
    () => {
      if (
        tutorialState.pendingAttempt !== attempt ||
        attempt.claimed ||
        !tutorialGenerationMatches(attempt)
      )
        return;
      if (Date.now() <= attempt.expiresAt) {
        scheduleTutorialAttemptExpiry(attempt);
        return;
      }
      const confirmation = attempt.confirmationDialog;
      resetTutorialAttempt(
        "Action expired; retype the confirmation and click again",
      );
      if (confirmation?.open) SkynetDialog.close(confirmation, "cancel");
    },
    Math.max(0, attempt.expiresAt - Date.now()) + 50,
  );
}

document.addEventListener("skynet:confirmation-open", (event) => {
  const attempt = tutorialState.pendingAttempt;
  const dialog = event.detail?.dialog;
  if (
    !attempt?.confirmationWindowOpen ||
    attempt.claimed ||
    !tutorialGenerationMatches(attempt) ||
    !dialog?.matches("dialog[data-app-confirmation][open]")
  )
    return;
  attempt.confirmationDialog = dialog;
  attempt.expiresAt = attempt.confirmationDeadline;
  tutorialUi.gateStatus.textContent =
    "Review the application confirmation; this action remains available for up to two minutes.";
  scheduleTutorialAttemptExpiry(attempt);
});

document.addEventListener("skynet:confirmation-close", (event) => {
  const attempt = tutorialState.pendingAttempt;
  if (
    !attempt ||
    attempt.confirmationDialog !== event.detail?.dialog ||
    !tutorialGenerationMatches(attempt)
  )
    return;
  if (!event.detail.accepted || Date.now() > attempt.confirmationDeadline) {
    if (event.detail.accepted) event.preventDefault();
    resetTutorialAttempt(
      event.detail.accepted
        ? "Confirmation expired; retype the phrase and click again"
        : "Action cancelled; retype the phrase before trying again",
    );
    return;
  }
  attempt.confirmationDialog = null;
  attempt.expiresAt = Date.now() + 8000;
  tutorialUi.gateStatus.textContent =
    "Confirmed; waiting for the exact API request";
  scheduleTutorialAttemptExpiry(attempt);
});

function invalidateTutorialGate() {
  tutorialState.gateGeneration += 1;
  resetTutorialAttempt();
  tutorialState.renderObserver?.disconnect();
  tutorialState.renderObserver = null;
  if (tutorialState.renderTimer) window.clearTimeout(tutorialState.renderTimer);
  tutorialState.renderTimer = 0;
  if (tutorialState.recoveryTimer)
    window.clearTimeout(tutorialState.recoveryTimer);
  tutorialState.recoveryTimer = 0;
  tutorialState.recovering = false;
  if (tutorialState.prefillTimer)
    window.clearTimeout(tutorialState.prefillTimer);
  tutorialState.prefillTimer = 0;
}

function tutorialGenerationMatches(snapshot) {
  return Boolean(
    tutorialState.active &&
      snapshot &&
      snapshot.sessionGeneration === tutorialState.sessionGeneration &&
      snapshot.gateGeneration === tutorialState.gateGeneration &&
      snapshot.stepId === tutorialStep()?.id,
  );
}

function tutorialClaimCanSettle(claim) {
  if (!claim || claim.settled || !tutorialGenerationMatches(claim))
    return false;
  return true;
}

function tutorialActionIsPending() {
  return Boolean(
    (tutorialState.pendingAttempt?.confirmationDialog &&
      tutorialGenerationMatches(tutorialState.pendingAttempt)) ||
      tutorialClaimCanSettle(tutorialState.inflightClaim),
  );
}

function completeTutorialGate(message = "Completed", snapshot = null) {
  if (snapshot && !tutorialGenerationMatches(snapshot)) return;
  const step = tutorialStep();
  if (!step || !tutorialState.session) return;
  if (!tutorialState.session.completedStepIds.includes(step.id))
    tutorialState.session.completedStepIds.push(step.id);
  tutorialState.gateComplete = true;
  if (!tutorialVisibleRect(tutorialState.target)) {
    tutorialState.target =
      workspaceHeading(tutorialState.page) ||
      document.querySelector(".topbar") ||
      document.body;
    SkynetDialog.placeGuide(elements.tutorialLayer, tutorialState.target);
    elements.tutorialLayer.classList.add("is-fallback");
  }
  elements.tutorialNext.disabled = false;
  elements.tutorialNext.textContent =
    tutorialState.index === tutorialTours[tutorialState.page].steps.length - 1
      ? "Finish"
      : "Next";
  tutorialUi.gateStatus.textContent = message;
  persistTutorialSession();
}

function scheduleTutorialFieldGateEvaluation(step) {
  if (step?.gate !== "field") return;
  if (tutorialState.prefillTimer)
    window.clearTimeout(tutorialState.prefillTimer);
  const snapshot = {
    sessionGeneration: tutorialState.sessionGeneration,
    gateGeneration: tutorialState.gateGeneration,
    stepId: step.id,
  };
  let attempts = 40;
  const evaluate = () => {
    tutorialState.prefillTimer = 0;
    if (!tutorialGenerationMatches(snapshot) || tutorialState.gateComplete)
      return;
    const target = tutorialInteractionTarget(step);
    if (tutorialValidateField(step, target)) {
      completeTutorialGate("Current valid value accepted", snapshot);
      return;
    }
    if (attempts-- > 0)
      tutorialState.prefillTimer = window.setTimeout(evaluate, 100);
  };
  tutorialState.prefillTimer = window.setTimeout(evaluate, 0);
}

function setupTutorialGate(step) {
  invalidateTutorialGate();
  tutorialUi.confirmPanel.hidden = true;
  tutorialUi.confirmInput.value = "";
  tutorialSetError();
  tutorialUi.resumePanel.hidden = true;
  elements.tutorialBack.hidden = false;
  elements.tutorialNext.hidden = false;
  const completed = tutorialState.session?.completedStepIds?.includes(step.id);
  tutorialState.gateComplete = Boolean(completed || !step.gate);
  elements.tutorialNext.disabled = Boolean(
    step.gate && !step.optional && !tutorialState.gateComplete,
  );
  elements.tutorialNext.textContent =
    step.optional && !completed
      ? "Skip"
      : tutorialState.index ===
          tutorialTours[tutorialState.page].steps.length - 1
        ? "Finish"
        : "Next";
  tutorialUi.gateStatus.textContent = completed
    ? "Completed earlier"
    : step.gate
      ? step.optional
        ? "Optional"
        : "Waiting for your action"
      : "Read, then continue";
  tutorialUi.useValue.hidden = !(step.useValue || step.applyValue);
  tutorialUi.useValue.textContent = step.applyValue
    ? "Use manifest tutorial values"
    : "Use tutorial value";
  renderTutorialRecords();
  tutorialState.session.stepId = step.id;
  persistTutorialSession();
  scheduleTutorialFieldGateEvaluation(step);
}

function tutorialConfirmationPhrase(step) {
  return `${step.confirmVerb || "CONFIRM"} ${tutorialState.session.token}`;
}

function showTutorialConfirmation(step) {
  const phrase = tutorialConfirmationPhrase(step);
  tutorialUi.confirmSummary.textContent =
    step.risk === "slurm"
      ? "This can create a permanent record and real cluster work. No GPU/account/time cap is available in this form, so verify the effective allocation first."
      : step.risk === "cleanup"
        ? "This archives only the exact record bound to this tutorial session."
        : "This writes a persistent database record. Review the highlighted form before proceeding.";
  tutorialUi.confirmPhrase.textContent = phrase;
  tutorialUi.confirmPanel.hidden = false;
  tutorialUi.confirmInput.value = "";
  tutorialState.confirmationArmed = false;
  tutorialUi.gateStatus.textContent = "Confirmation required";
  tutorialUi.confirmInput.focus({ preventScroll: true });
  scheduleTutorialPosition();
}

function tutorialRequestPath(request) {
  return tutorialResolve(request.path);
}
function tutorialNormalizePath(path) {
  return new URL(path, window.location.origin).pathname;
}
function tutorialPathMatches(expected, actual) {
  if (expected instanceof RegExp) return expected.test(actual);
  return tutorialNormalizePath(expected) === actual;
}

function tutorialObserveApiStart(path, options = {}) {
  const attempt = tutorialState.pendingAttempt;
  if (
    !attempt ||
    attempt.claimed ||
    !tutorialGenerationMatches(attempt) ||
    attempt.confirmationDialog?.open
  )
    return null;
  if (Date.now() > attempt.expiresAt) {
    resetTutorialAttempt(
      "Action expired; type the confirmation and click again",
    );
    return null;
  }
  const request = attempt.request;
  const method = String(options.method || "GET").toUpperCase();
  const actualPath = tutorialNormalizePath(path);
  if (
    method !== request.method ||
    !tutorialPathMatches(attempt.expectedPath, actualPath)
  )
    return null;
  const boundId = request.bind
    ? tutorialState.session.bindings[request.bind]
    : null;
  if (request.bind && !request.establishBinding) {
    if (!boundId || boundId !== attempt.bindingId) return null;
    const pathIds = actualPath
      .split("/")
      .filter(Boolean)
      .map((part) => {
        try {
          return decodeURIComponent(part);
        } catch {
          return part;
        }
      });
    if (!pathIds.includes(String(boundId))) return null;
  }
  attempt.claimed = true;
  tutorialState.pendingAttempt = null;
  if (tutorialState.attemptTimer)
    window.clearTimeout(tutorialState.attemptTimer);
  tutorialState.attemptTimer = 0;
  tutorialUi.gateStatus.textContent = `Waiting for ${method} ${actualPath}`;
  const claim = {
    ...attempt,
    method,
    path: actualPath,
    boundId,
    settled: false,
  };
  tutorialState.inflightClaim = claim;
  return claim;
}

function tutorialEntityId(payload, entity) {
  const nested =
    entity && payload && typeof payload === "object" ? payload[entity] : null;
  return (
    nested?.id ||
    nested?.[`${entity}_id`] ||
    payload?.id ||
    (entity ? payload?.[`${entity}_id`] : null) ||
    null
  );
}

function tutorialResponseEntity(payload, entity) {
  return entity && payload && typeof payload === "object"
    ? payload[entity] || payload
    : payload;
}

function tutorialValueAtPath(value, path) {
  return path
    .split(".")
    .reduce(
      (current, key) =>
        current && typeof current === "object" ? current[key] : undefined,
      value,
    );
}

function tutorialIdentityMatches(entity, rules) {
  return Boolean(
    rules?.length &&
      rules.every((rule) =>
        rule.paths.some(
          (path) => tutorialValueAtPath(entity, path) === rule.value,
        ),
      ),
  );
}

function tutorialRecordFor(binding, id) {
  return tutorialState.session.ownedRecords.find(
    (record) => record.binding === binding && String(record.id) === String(id),
  );
}

function tutorialRecoverableRecord() {
  if (!tutorialState.session) return null;
  const candidates = (tutorialState.session.ownedRecords || []).filter(
    (record) => {
      if (record.cleanup !== "archive" || record.cleanupState === "archived")
        return false;
      return tutorialTours[tutorialState.page].steps.some(
        (step) =>
          step.id === "read" &&
          step.request?.method === "GET" &&
          step.request.bind === record.binding,
      );
    },
  );
  const bound = candidates.find(
    (record) =>
      String(tutorialState.session.bindings[record.binding] || "") ===
      String(record.id),
  );
  if (bound) return bound;
  return (
    candidates.find(
      (record) => !tutorialState.session.bindings[record.binding],
    ) || null
  );
}

function tutorialRecoveryStepIndex(record, stepId) {
  if (!record) return -1;
  return tutorialTours[tutorialState.page].steps.findIndex(
    (step) => step.id === stepId && step.request?.bind === record.binding,
  );
}

function beginTutorialRecovery(record) {
  const readIndex = tutorialRecoveryStepIndex(record, "read");
  if (!record || readIndex < 0) return false;
  const existing = tutorialState.session.bindings[record.binding];
  if (existing && String(existing) !== String(record.id)) {
    tutorialSetError(
      "A newer record is already bound. Finish or restart that record before reconciling this older ledger entry.",
    );
    return false;
  }
  invalidateTutorialGate();
  tutorialState.session.bindings[record.binding] = String(record.id);
  tutorialState.session.recoveryRecordId = `${record.binding}:${record.id}`;
  tutorialState.session.stepId = "read";
  tutorialState.session.completedStepIds =
    tutorialState.session.completedStepIds.filter(
      (stepId) => stepId !== "read",
    );
  tutorialState.session.finished = false;
  tutorialState.sessionGeneration = tutorialUniqueId("recovery");
  tutorialState.session.sessionGeneration = tutorialState.sessionGeneration;
  tutorialState.ownedVerified = new Map();
  persistTutorialSession();
  showTutorialStep(readIndex, 1);
  return true;
}

function tutorialActiveRecoveryRecord() {
  const recoveryId = tutorialState.session?.recoveryRecordId;
  if (!recoveryId) return null;
  return (
    (tutorialState.session.ownedRecords || []).find(
      (record) => `${record.binding}:${record.id}` === recoveryId,
    ) || null
  );
}

function tutorialBindingIsOwned(binding) {
  const id = tutorialState.session?.bindings?.[binding];
  const record = tutorialRecordFor(binding, id);
  const verified = tutorialState.ownedVerified.get(binding);
  return Boolean(
    id &&
      record &&
      verified &&
      verified.ownerToken === record.ownerToken &&
      verified.id === String(id) &&
      JSON.stringify(verified.expectedIdentity) ===
        JSON.stringify(record.expectedIdentity),
  );
}

function tutorialRecordSuccess(
  request,
  id,
  expectedIdentity,
  ownerToken,
  entity,
) {
  if (!request.record || !id) return;
  const records = tutorialState.session.ownedRecords;
  if (
    !records.some(
      (record) =>
        record.kind === request.record.kind && String(record.id) === String(id),
    )
  ) {
    records.push({
      binding: request.bind,
      kind: request.record.kind,
      id: String(id),
      cleanup: request.record.cleanup,
      cleanupState:
        request.record.cleanup === "archive"
          ? "created_pending_verification"
          : "retained; no cleanup API",
      ownerToken,
      expectedIdentity,
      createdGeneration: tutorialState.sessionGeneration,
      capturedIdentity: expectedIdentity.map((rule) => ({
        paths: rule.paths,
        value: rule.paths
          .map((path) => tutorialValueAtPath(entity, path))
          .find((value) => value !== undefined),
      })),
    });
  }
}

function tutorialCleanupSuccess(binding, id) {
  if (!binding || !id) return;
  const record = tutorialRecordFor(binding, id);
  if (record && tutorialBindingIsOwned(binding))
    record.cleanupState = "archived";
}

function waitForTutorialRender(selector, snapshot) {
  const resolved = tutorialResolve(selector);
  const check = () => {
    if (!tutorialGenerationMatches(snapshot)) return false;
    stampTutorialRecordRows();
    if (document.querySelector(resolved)) {
      tutorialState.renderObserver?.disconnect();
      tutorialState.renderObserver = null;
      if (tutorialState.renderTimer)
        window.clearTimeout(tutorialState.renderTimer);
      tutorialState.renderTimer = 0;
      completeTutorialGate("Saved and rendered", snapshot);
      return true;
    }
    return false;
  };
  if (check()) return;
  tutorialState.renderObserver?.disconnect();
  tutorialState.renderObserver = new MutationObserver(check);
  tutorialState.renderObserver.observe(document.body, {
    childList: true,
    subtree: true,
  });
  tutorialState.renderTimer = window.setTimeout(() => {
    if (
      tutorialGenerationMatches(snapshot) &&
      tutorialState.renderObserver &&
      !check()
    )
      tutorialSetError(
        "The API succeeded, but the bound row has not appeared yet. Refresh the page and resume this tutorial.",
      );
  }, 10000);
}

function tutorialObserveApiSuccess(claim, payload) {
  if (!tutorialClaimCanSettle(claim)) return;
  claim.settled = true;
  const request = claim.request;
  if (request.success && !request.success(payload)) {
    const blockerReasons = Array.isArray(payload?.blockers)
      ? payload.blockers
          .flatMap((blocker) =>
            Array.isArray(blocker?.reasons) ? blocker.reasons : [],
          )
          .filter(Boolean)
      : [];
    tutorialSetError(
      blockerReasons.length
        ? `Preview blocked: ${blockerReasons.join("; ")}. Set the missing adapter-declared values and preview again.`
        : "The request succeeded but did not return the expected result.",
    );
    resetTutorialAttempt(
      blockerReasons.length
        ? "Preview remains blocked; correct the declared values and retry"
        : "Unexpected response; confirm and retry",
    );
    return;
  }
  const id = request.bind ? tutorialEntityId(payload, request.entity) : null;
  if (request.bind && !id) {
    tutorialSetError(
      "The request succeeded but did not return the required record ID.",
    );
    resetTutorialAttempt("Missing record ID; confirm and retry");
    return;
  }
  if (request.bind) {
    const existing = tutorialState.session.bindings[request.bind];
    if (request.establishBinding) {
      if (existing && existing !== String(id)) {
        tutorialSetError(
          "A different record is already bound to this tutorial step.",
        );
        resetTutorialAttempt("Binding mismatch; restart only after cleanup");
        return;
      }
      if (!existing) tutorialState.session.bindings[request.bind] = String(id);
    } else if (
      !existing ||
      existing !== String(id) ||
      claim.bindingId !== String(id)
    ) {
      tutorialSetError(
        "The response ID did not match the immutable bound record ID.",
      );
      resetTutorialAttempt("Bound ID mismatch; confirm and retry");
      return;
    }
  }
  const entity = tutorialResponseEntity(payload, request.entity);
  const existingRecord = request.bind
    ? tutorialRecordFor(request.bind, id)
    : null;
  const identityOwnerToken = existingRecord?.ownerToken || claim.ownerToken;
  const expectedIdentity = request.identity
    ? request.identity({
        token: identityOwnerToken,
        bindings: { ...claim.bindings },
      })
    : [];
  tutorialRecordSuccess(
    request,
    id,
    expectedIdentity,
    claim.ownerToken,
    entity,
  );
  const record = request.bind ? tutorialRecordFor(request.bind, id) : null;
  const identityToVerify = record?.expectedIdentity?.length
    ? record.expectedIdentity
    : expectedIdentity;
  renderTutorialRecords();
  persistTutorialSession();
  if (
    request.identity &&
    !request.cleanupBinding &&
    !tutorialIdentityMatches(entity, identityToVerify)
  ) {
    if (record) record.cleanupState = "manual_review";
    if (request.bind) tutorialState.ownedVerified.delete(request.bind);
    renderTutorialRecords();
    persistTutorialSession();
    resetTutorialAttempt();
    if (request.establishBinding) {
      completeTutorialGate(
        `Created ${id}; exact read verification is still required`,
        claim,
      );
      const readIndex = tutorialRecoveryStepIndex(record, "read");
      if (readIndex >= 0) tutorialState.session.stepId = "read";
      persistTutorialSession();
      tutorialSetError(
        "The create succeeded and its exact ID is retained, but its response identity did not verify. Resume cleanup and perform the bound Read step; mismatches can only be reviewed manually.",
      );
    } else {
      tutorialSetError(
        "The response did not match the exact structured tutorial identity. This record remains manual-review only and cannot be archived by the tutorial.",
      );
      tutorialUi.gateStatus.textContent = "Identity verification failed";
    }
    return;
  }
  if (
    request.bind &&
    identityToVerify.length &&
    !request.cleanupBinding &&
    !request.establishBinding
  ) {
    tutorialState.ownedVerified.set(request.bind, {
      id: String(id),
      ownerToken: record?.ownerToken || claim.ownerToken,
      expectedIdentity: identityToVerify,
    });
    if (record?.cleanup === "archive") {
      // Only this exact, identity-verified GET can reconcile missed cleanup.
      // No archive/restore request is issued as part of recovery.
      const alreadyArchived =
        request.method === "GET" && Boolean(entity?.archived_at);
      record.cleanupState = alreadyArchived ? "archived" : "cleanup pending";
      if (alreadyArchived) {
        const cleanupStep = tutorialTours[tutorialState.page].steps.find(
          (step) => step.request?.cleanupBinding === request.bind,
        );
        if (
          cleanupStep &&
          !tutorialState.session.completedStepIds.includes(cleanupStep.id)
        )
          tutorialState.session.completedStepIds.push(cleanupStep.id);
        delete tutorialState.session.recoveryRecordId;
      }
    }
  }
  tutorialCleanupSuccess(request.cleanupBinding, id);
  renderTutorialRecords();
  persistTutorialSession();
  resetTutorialAttempt();
  if (request.establishBinding) {
    completeTutorialGate(
      `Created and bound: ${id}; verify it with the exact Read step`,
      claim,
    );
    const readIndex = tutorialRecoveryStepIndex(record, "read");
    if (readIndex >= 0) tutorialState.session.stepId = "read";
    persistTutorialSession();
  } else if (request.render) waitForTutorialRender(request.render, claim);
  else
    completeTutorialGate(
      record?.cleanupState === "archived" && request.method === "GET"
        ? `Already archived: ${id}; cleanup verified by the exact Read. Continue after cleanup.`
        : id
          ? `Succeeded: ${id}`
          : "Request succeeded",
      claim,
    );
}

function tutorialObserveApiFailure(claim, error) {
  if (!tutorialClaimCanSettle(claim)) return;
  claim.settled = true;
  tutorialSetError(error?.message || "The request failed.");
  resetTutorialAttempt(
    "Request failed; correct the form, retype confirmation, and retry",
  );
}

function tutorialInteractionTarget(step) {
  const selector = tutorialResolve(step.selector);
  try {
    return document.querySelector(selector);
  } catch {
    return null;
  }
}

function tutorialEventHitsTarget(event, target) {
  return Boolean(
    target && (event.target === target || target.contains(event.target)),
  );
}

function onTutorialInteraction(event) {
  if (!tutorialState.active || tutorialState.index < 0) return;
  const step = tutorialStep();
  const target = tutorialInteractionTarget(step);
  if (
    step.gate === "field" &&
    ["input", "change"].includes(event.type) &&
    tutorialEventHitsTarget(event, target)
  ) {
    if (event.isTrusted && tutorialValidateField(step, target))
      completeTutorialGate("Input accepted");
    return;
  }
  if (
    event.type === "submit" &&
    step.gate === "action" &&
    target?.form === event.target &&
    !tutorialState.pendingAttempt
  ) {
    event.preventDefault();
    event.stopImmediatePropagation();
    tutorialSetError(
      "Use the highlighted button so the tutorial can bind this exact request.",
    );
    return;
  }
  if (
    event.type !== "click" ||
    step.gate !== "action" ||
    !tutorialEventHitsTarget(event, target) ||
    !event.isTrusted
  )
    return;
  if (step.requiresOwned && !tutorialBindingIsOwned(step.requiresOwned)) {
    event.preventDefault();
    event.stopImmediatePropagation();
    tutorialSetError(
      "Ownership must be verified by reading the exact bound record first. Use Back to repeat its Read step.",
    );
    return;
  }
  if (step.request?.preflight && !step.request.preflight(tutorialContext())) {
    event.preventDefault();
    event.stopImmediatePropagation();
    tutorialSetError(
      "The form does not contain this session's exact structured tutorial identity.",
    );
    resetTutorialAttempt("Correct the identity fields before trying again");
    return;
  }
  if (
    step.request?.establishBinding &&
    tutorialState.session.bindings[step.request.bind]
  ) {
    event.preventDefault();
    event.stopImmediatePropagation();
    tutorialSetError(
      "This create step already has a bound record. It cannot create or bind a second one.",
    );
    return;
  }
  if (step.risk && !tutorialState.confirmationArmed) {
    event.preventDefault();
    event.stopImmediatePropagation();
    showTutorialConfirmation(step);
    return;
  }
  tutorialSetError();
  if (!step.request) {
    if (step.localPredicate) {
      const snapshot = {
        sessionGeneration: tutorialState.sessionGeneration,
        gateGeneration: tutorialState.gateGeneration,
        stepId: step.id,
      };
      window.setTimeout(() => {
        if (!tutorialGenerationMatches(snapshot)) return;
        if (step.localPredicate(tutorialContext()))
          completeTutorialGate("Action completed locally", snapshot);
        else
          tutorialUi.gateStatus.textContent =
            "The requested local state is not open yet; use the highlighted control again";
      }, 0);
    } else {
      completeTutorialGate("Action completed locally");
    }
    return;
  }
  if (step.request.bindFromTarget) {
    const selectedId = target.dataset.id;
    const existing = tutorialState.session.bindings[step.request.bind];
    if (!selectedId || (existing && existing !== selectedId)) {
      event.preventDefault();
      event.stopImmediatePropagation();
      tutorialSetError("The selected row does not match the bound record.");
      return;
    }
    if (!existing)
      tutorialState.session.bindings[step.request.bind] = selectedId;
  }
  const attemptId = tutorialUniqueId("attempt");
  const expiresAt = Date.now() + 8000;
  tutorialState.pendingAttempt = {
    sessionGeneration: tutorialState.sessionGeneration,
    gateGeneration: tutorialState.gateGeneration,
    stepId: step.id,
    attemptId,
    expiresAt,
    confirmationDeadline: Date.now() + 120000,
    confirmationWindowOpen: true,
    claimed: false,
    request: step.request,
    expectedPath: tutorialRequestPath(step.request),
    bindingId: step.request.bind
      ? tutorialState.session.bindings[step.request.bind] || null
      : null,
    bindings: { ...tutorialState.session.bindings },
    ownerToken:
      tutorialRecordFor(
        step.request.bind,
        tutorialState.session.bindings[step.request.bind],
      )?.ownerToken || tutorialState.session.token,
  };
  const attempt = tutorialState.pendingAttempt;
  // A native event can run a microtask checkpoint between capture and bubble
  // listeners. Keep this window for the event task so its normal action handler
  // can open the confirmation; unrelated later tasks cannot extend it.
  window.setTimeout(() => {
    attempt.confirmationWindowOpen = false;
  }, 0);
  scheduleTutorialAttemptExpiry(attempt);
  tutorialUi.gateStatus.textContent =
    "Action accepted; waiting for its exact API request";
}

function useTutorialValue(event) {
  if (!event.isTrusted) return;
  const step = tutorialStep();
  const target = tutorialInteractionTarget(step);
  if (!step || !target || target.disabled) return;
  if (step.applyValue) {
    const focusTarget = step.applyValue(tutorialContext(), target);
    tutorialState.gateComplete = Boolean(
      tutorialState.session?.completedStepIds?.includes(step.id),
    );
    elements.tutorialNext.disabled = Boolean(
      step.gate && !step.optional && !tutorialState.gateComplete,
    );
    tutorialUi.gateStatus.textContent =
      "Manifest tutorial values inserted; validating required fields";
    scheduleTutorialFieldGateEvaluation(step);
    focusTarget?.focus({ preventScroll: true });
    return;
  }
  if (!step.useValue) return;
  const value = step.useValue(tutorialContext(), target);
  tutorialState.applyingValue = true;
  try {
    if (target.type === "checkbox") target.checked = Boolean(value);
    else target.value = String(value);
    target.dispatchEvent(new Event("input", { bubbles: true }));
    target.dispatchEvent(new Event("change", { bubbles: true }));
  } finally {
    tutorialState.applyingValue = false;
  }
  tutorialState.gateComplete = Boolean(
    tutorialState.session?.completedStepIds?.includes(step.id),
  );
  elements.tutorialNext.disabled = Boolean(
    step.gate && !step.optional && !tutorialState.gateComplete,
  );
  tutorialUi.gateStatus.textContent =
    "Value inserted; validating the highlighted field";
  scheduleTutorialFieldGateEvaluation(step);
  target.focus({ preventScroll: true });
}

function collectionTutorialResumeIndex(session, requestedIndex) {
  if (session.bindings?.collectionAdapterId || requestedIndex === 0)
    return requestedIndex;
  // Unsaved editor values are deliberately not persisted. Reopen a clean
  // draft, rather than skipping hidden fields to a Read step with no record.
  session.completedStepIds = [];
  session.stepId = "open-adapter";
  return 0;
}

function ensureInteractiveTutorialControls() {
  if (tutorialState.controlsBound) return;
  tutorialState.controlsBound = true;
  tutorialUi.useValue.addEventListener("click", useTutorialValue);
  tutorialUi.confirmInput.addEventListener("input", (event) => {
    if (!event.isTrusted || !tutorialState.active) return;
    tutorialState.confirmationArmed =
      event.target.value === tutorialConfirmationPhrase(tutorialStep());
    tutorialUi.gateStatus.textContent = tutorialState.confirmationArmed
      ? "Confirmed; click the highlighted control again"
      : "Confirmation text does not match";
  });
  tutorialUi.resume.addEventListener("click", (event) => {
    if (!event.isTrusted || !tutorialState.active || !tutorialState.session)
      return;
    const recovery = tutorialRecoverableRecord();
    if (recovery && beginTutorialRecovery(recovery)) return;
    let index = Math.max(
      0,
      tutorialTours[tutorialState.page].steps.findIndex(
        (step) => step.id === tutorialState.session.stepId,
      ),
    );
    if (tutorialState.page === "collection")
      index = collectionTutorialResumeIndex(tutorialState.session, index);
    tutorialState.session.finished = false;
    tutorialState.sessionGeneration = tutorialUniqueId("resume");
    tutorialState.session.sessionGeneration = tutorialState.sessionGeneration;
    tutorialState.ownedVerified = new Map();
    showTutorialStep(index, 1);
  });
  tutorialUi.restart.addEventListener("click", (event) => {
    if (!event.isTrusted || !tutorialState.active) return;
    invalidateTutorialGate();
    const retained = (tutorialState.session?.ownedRecords || []).map(
      (record) => ({ ...record }),
    );
    tutorialState.session = newTutorialSession(tutorialState.page, retained);
    tutorialState.sessionGeneration = tutorialState.session.sessionGeneration;
    tutorialState.ownedVerified = new Map();
    showTutorialStep(0, 1);
  });
}

function showTutorialResumeChoice() {
  invalidateTutorialGate();
  tutorialState.index = -1;
  tutorialState.target = workspaceHeading(tutorialState.page)?.querySelector(
    "h1",
  );
  tutorialState.target?.setAttribute("data-tutorial-active-target", "true");
  elements.tutorialPage.textContent = tutorialTours[tutorialState.page].title;
  elements.tutorialProgress.textContent = "Saved tutorial session";
  const recovery = tutorialRecoverableRecord();
  elements.tutorialTitle.textContent = recovery
    ? "Verify or clean up the bound record"
    : "Resume or restart";
  elements.tutorialInstruction.textContent = recovery
    ? `Resume cleanup opens the exact ${recovery.kind} ${recovery.id} Read step. You must click its real row control; only a successful exact GET with matching structured identity can unlock archive. Restart creates a new token and retains this ledger entry for later review.`
    : tutorialState.page === "collection" &&
        !tutorialState.session.bindings?.collectionAdapterId
      ? "This adapter has not been created. Resume opens a clean draft using the same tutorial token so you can re-enter the required fields. No saved record or request is replayed."
      : "Resume uses the saved step and bound record IDs but never restores field values or replays a request. Restart creates a new token and retains every prior cleanup ledger entry.";
  tutorialUi.resume.hidden = false;
  tutorialUi.resume.textContent = recovery ? "Resume cleanup" : "Resume";
  tutorialUi.useValue.hidden = true;
  tutorialUi.confirmPanel.hidden = true;
  tutorialSetError();
  tutorialUi.resumePanel.hidden = false;
  elements.tutorialBack.hidden = true;
  elements.tutorialNext.hidden = true;
  renderTutorialRecords();
  scheduleTutorialPosition();
  observeTutorialTarget();
  tutorialUi.resume.focus({ preventScroll: true });
}

function setDataResourceTypes(category, selected = "") {
  document.getElementById("data-resource-category").value = category;
  document.getElementById("data-resource-name").placeholder =
    category === "file" ? "file set name" : "dataset name";
  const select = document.getElementById("data-resource-kind");
  const types = Object.entries(dataResourceTypes[category] || {});
  select.innerHTML = types.length
    ? types
        .map(
          ([value, label]) =>
            `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`,
        )
        .join("")
    : '<option value="">Types unavailable — refresh the catalog</option>';
  if (types.some(([value]) => value === selected)) select.value = selected;
}

function resetDataResourceEditor(hide = true) {
  const form = document.querySelector("#data-resource-form");
  if (!form) return;
  form.reset();
  const files =
    document
      .querySelector('[data-data-tab="files"]')
      ?.getAttribute("aria-selected") === "true";
  form.querySelector("h2, h3").textContent = files
    ? "New files"
    : "New dataset";
  setDataResourceTypes(
    files ? "file" : "dataset",
    files ? "simulation_assets" : "demonstrations",
  );
  document.querySelector("#data-resource-id").value = "";
  for (const id of [
    "data-resource-provider",
    "data-resource-namespace",
    "data-resource-name",
    "data-resource-kind",
  ])
    document.querySelector(`#${id}`).disabled = false;
  document.querySelector("#create-data-resource").textContent = "Register";
  if (hide) closeDisclosurePanel(form, elements.showDataResourceForm);
}

async function openDataResourceEditor(id, launcher = null) {
  const revealLauncher = launcher || currentRevealLauncher();
  const requestToken = disclosureToken(
    elements.dataResourceForm,
    revealLauncher,
  );
  try {
    const payload = await api(`/api/data/resources/${encodeURIComponent(id)}`);
    if (!disclosureTokenIsCurrent(elements.dataResourceForm, requestToken))
      return;
    const resource = entityFrom(payload, "resource");
    if (String(resource.id || "") !== String(id))
      throw new Error("Resource detail ID did not match the requested record.");
    document.querySelector("#data-resource-id").value = resource.id || id;
    document.querySelector("#data-resource-provider").value =
      resource.provider || "";
    document.querySelector("#data-resource-namespace").value =
      resource.namespace || "";
    document.querySelector("#data-resource-name").value = resource.name || "";
    setDataResourceTypes(resource.category, resource.kind);
    document.querySelector("#data-resource-description").value =
      resource.description || "";
    for (const fieldId of [
      "data-resource-provider",
      "data-resource-namespace",
      "data-resource-name",
      "data-resource-kind",
    ])
      document.querySelector(`#${fieldId}`).disabled = true;
    document.querySelector("#create-data-resource").textContent = "Save";
    elements.dataResourceForm.querySelector("h2, h3").textContent =
      resource.category === "file" ? "Edit files" : "Edit dataset";
    revealPanel(document.querySelector("#data-resource-form"), {
      focusTarget: document.querySelector("#data-resource-description"),
      launcher: revealLauncher,
    });
  } catch (error) {
    if (!disclosureTokenIsCurrent(elements.dataResourceForm, requestToken))
      return;
    showToast(`Resource detail failed: ${error.message}`, true);
  }
}

async function updateDataResource(id) {
  const form = document.querySelector("#data-resource-form");
  if (!form.reportValidity()) return;
  try {
    await api(`/api/data/resources/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify({
        description: document
          .querySelector("#data-resource-description")
          .value.trim(),
      }),
    });
    closeDisclosurePanel(form, elements.showDataResourceForm);
    resetDataResourceEditor(false);
    loadedTabs.delete("datasets");
    await loadDataRegistry(true);
    showToast("Registration updated.");
  } catch (error) {
    showToast(`Resource update failed: ${error.message}`, true);
  }
}

async function restoreDataResource(id) {
  try {
    await api(`/api/data/resources/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify({ archived: false }),
    });
    await loadDataRegistry(true);
    showToast("Resource restored, including its existing versions.");
  } catch (error) {
    showToast(`Resource restore failed: ${error.message}`, true);
  }
}

async function archiveDataResource(id) {
  if (
    !(await askUserDialog(
      "Archive this resource record? Payload files are not deleted.",
    ))
  ) {
    resetTutorialAttempt(
      "Archive cancelled; type the tutorial confirmation again before retrying",
    );
    return;
  }
  try {
    await api(`/api/data/resources/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
    loadedTabs.delete("datasets");
    await loadDataRegistry(true);
    showToast("Resource archived; payload files were not deleted.");
  } catch (error) {
    showToast(`Resource archive failed: ${error.message}`, true);
  }
}

function stampTutorialRecordRows() {
  const configs = [
    ["data-resources-body", "data-resource-action", "resourceId"],
    [
      "collection-adapters-body",
      "data-collection-adapter-action",
      "collectionAdapterId",
    ],
    ["experiments-body", "data-experiment-action", "experimentId"],
    ["runs-body", "data-run-action", "runId"],
    ["evaluations-body", "data-evaluation-action", "evaluationId"],
    ["adapters-body", "data-adapter-action", "adapterId"],
  ];
  for (const [bodyId, actionAttribute, rowProperty] of configs) {
    const body = document.querySelector(`#${bodyId}`);
    if (!body) continue;
    for (const row of body.querySelectorAll("tr")) {
      // Inline activity links may identify a child job. The Actions cell
      // identifies the catalog record that owns the row.
      const action =
        row.querySelector(`.row-actions [${actionAttribute}][data-id]`) ||
        row.querySelector(`[${actionAttribute}][data-id]`);
      if (!action) continue;
      row.dataset[rowProperty] = action.dataset.id;
    }
  }
}

function initializeTutorialRecordRows() {
  const bodies = [
    "data-resources-body",
    "collection-adapters-body",
    "experiments-body",
    "runs-body",
    "evaluations-body",
    "adapters-body",
  ]
    .map((id) => document.querySelector(`#${id}`))
    .filter(Boolean);
  const observer = new MutationObserver(stampTutorialRecordRows);
  bodies.forEach((body) =>
    observer.observe(body, { childList: true, subtree: true }),
  );
  stampTutorialRecordRows();
}

function tutorialStorageKey(page) {
  return `skynet.tutorial.completed.${page}`;
}

function tutorialIsComplete(page) {
  try {
    return (
      window.localStorage.getItem(
        workspaceStorageKey(tutorialStorageKey(page)),
      ) === "1"
    );
  } catch {
    return false;
  }
}

function updateTutorialLaunchStatus(page) {
  const button = document.querySelector(`[data-tutorial-page="${page}"]`);
  if (!button) return;
  const complete = tutorialIsComplete(page);
  button.classList.toggle("is-complete", complete);
  button.title = complete
    ? `${tutorialTours[page].title} tutorial completed. Run it again.`
    : `Start the ${tutorialTours[page].title} tutorial.`;
}

function tutorialViewport() {
  const viewport = window.visualViewport;
  return {
    left: viewport?.offsetLeft || 0,
    top: viewport?.offsetTop || 0,
    width: viewport?.width || window.innerWidth,
    height: viewport?.height || window.innerHeight,
  };
}

function syncTutorialViewport() {
  const viewport = tutorialViewport();
  const style = elements.tutorialLayer.style;
  style.setProperty("--tutorial-viewport-width", `${viewport.width}px`);
  style.setProperty("--tutorial-viewport-height", `${viewport.height}px`);
  Object.assign(style, {
    left: `${viewport.left}px`,
    top: `${viewport.top}px`,
    right: "auto",
    bottom: "auto",
    width: `${viewport.width}px`,
    height: `${viewport.height}px`,
  });
  return viewport;
}

function tutorialVisibleRect(target, viewport = null) {
  if (!target?.isConnected) return null;
  const style = window.getComputedStyle(target);
  if (style.display === "none" || style.visibility === "hidden") return null;
  const rect = target.getBoundingClientRect();
  if (rect.width < 1 || rect.height < 1) return null;
  if (!viewport) return rect;
  const left = Math.max(viewport.left, rect.left);
  const top = Math.max(viewport.top, rect.top);
  const right = Math.min(viewport.left + viewport.width, rect.right);
  const bottom = Math.min(viewport.top + viewport.height, rect.bottom);
  if (right - left < 1 || bottom - top < 1) return null;
  return {
    left,
    top,
    right,
    bottom,
    width: right - left,
    height: bottom - top,
  };
}

function prepareTutorialTarget(step) {
  for (const selector of step.reveal || []) {
    const section = document.querySelector(selector);
    if (section?.hidden) {
      if (!tutorialState.revealed.has(section))
        tutorialState.revealed.set(section, true);
      revealPanel(section, { scroll: false });
    }
  }
  if (step.details) {
    const disclosure = document.querySelector(step.details);
    if (disclosure && !disclosure.open) {
      if (!tutorialState.openedDetails.has(disclosure))
        tutorialState.openedDetails.set(disclosure, false);
      disclosure.open = true;
    }
  }
  const selector = tutorialResolve(step.selector);
  let target = null;
  try {
    target = document.querySelector(selector);
  } catch {
    target = null;
  }
  if (!target) return null;
  SkynetDialog.placeGuide(elements.tutorialLayer, target);
  const collectionView = target.closest("[data-collection-view]");
  if (collectionView && typeof setCollectionView === "function")
    setCollectionView(collectionView.dataset.collectionView, {
      persist: false,
    });
  let ancestor = target.parentElement;
  while (ancestor) {
    if (ancestor.tagName === "DETAILS" && !ancestor.open) {
      if (!tutorialState.openedDetails.has(ancestor))
        tutorialState.openedDetails.set(ancestor, false);
      ancestor.open = true;
    }
    ancestor = ancestor.parentElement;
  }
  return target;
}

function scheduleTutorialPosition() {
  if (!tutorialState.active || tutorialState.animationFrame) return;
  tutorialState.animationFrame = window.requestAnimationFrame(() => {
    tutorialState.animationFrame = 0;
    positionTutorial();
  });
}

function positionTutorialBlockers(
  viewport,
  spotLeft,
  spotTop,
  spotRight,
  spotBottom,
) {
  const width = viewport.width;
  const height = viewport.height;
  const left = Math.max(0, spotLeft - viewport.left);
  const top = Math.max(0, spotTop - viewport.top);
  const right = Math.min(width, spotRight - viewport.left);
  const bottom = Math.min(height, spotBottom - viewport.top);
  const [topBlocker, leftBlocker, rightBlocker, bottomBlocker] =
    tutorialUi.blockers;
  const set = (node, x, y, w, h) =>
    Object.assign(node.style, {
      left: `${x}px`,
      top: `${y}px`,
      width: `${Math.max(0, w)}px`,
      height: `${Math.max(0, h)}px`,
    });
  if (elements.tutorialSpotlight.hidden) {
    set(topBlocker, 0, 0, width, height);
    set(leftBlocker, 0, 0, 0, 0);
    set(rightBlocker, 0, 0, 0, 0);
    set(bottomBlocker, 0, 0, 0, 0);
    return;
  }
  set(topBlocker, 0, 0, width, top);
  set(leftBlocker, 0, top, left, bottom - top);
  set(rightBlocker, right, top, width - right, bottom - top);
  set(bottomBlocker, 0, bottom, width, height - bottom);
}

function positionTutorial() {
  if (!tutorialState.active) return;
  const viewport = syncTutorialViewport();
  const rawRect = tutorialVisibleRect(tutorialState.target);
  const rect = tutorialVisibleRect(tutorialState.target, viewport);
  if (!rect) {
    if (rawRect) {
      elements.tutorialSpotlight.hidden = true;
      positionTutorialBlockers(
        viewport,
        viewport.left,
        viewport.top,
        viewport.left,
        viewport.top,
      );
      return;
    }
    if (tutorialState.gateComplete) {
      elements.tutorialSpotlight.hidden = true;
      positionTutorialBlockers(
        viewport,
        viewport.left,
        viewport.top,
        viewport.left,
        viewport.top,
      );
      return;
    }
    if (tutorialActionIsPending()) return;
    if (!tutorialState.recovering) {
      tutorialState.recovering = true;
      const snapshot = {
        sessionGeneration: tutorialState.sessionGeneration,
        gateGeneration: tutorialState.gateGeneration,
        stepId: tutorialStep()?.id,
      };
      tutorialState.recoveryTimer = window.setTimeout(() => {
        tutorialState.recoveryTimer = 0;
        tutorialState.recovering = false;
        if (tutorialGenerationMatches(snapshot))
          showTutorialUnavailableStep(tutorialStep(), tutorialState.index);
      }, 0);
    }
    return;
  }
  elements.tutorialSpotlight.hidden = false;
  const right = viewport.left + viewport.width;
  const bottom = viewport.top + viewport.height;
  const padding = 7;
  const spotLeft = Math.max(viewport.left + 2, rect.left - padding);
  const spotTop = Math.max(viewport.top + 2, rect.top - padding);
  const spotRight = Math.min(right - 2, rect.right + padding);
  const spotBottom = Math.min(bottom - 2, rect.bottom + padding);
  Object.assign(elements.tutorialSpotlight.style, {
    left: `${spotLeft - viewport.left}px`,
    top: `${spotTop - viewport.top}px`,
    width: `${Math.max(1, spotRight - spotLeft)}px`,
    height: `${Math.max(1, spotBottom - spotTop)}px`,
  });
  positionTutorialBlockers(viewport, spotLeft, spotTop, spotRight, spotBottom);

  const margin = 12;
  const gap = 12;
  const dialogRect = elements.tutorialDialog.getBoundingClientRect();
  const minLeft = viewport.left + margin;
  const maxLeft = Math.max(minLeft, right - dialogRect.width - margin);
  const left = Math.min(
    maxLeft,
    Math.max(minLeft, rect.left + rect.width / 2 - dialogRect.width / 2),
  );
  let top;
  let placement;
  if (rect.bottom + gap + dialogRect.height <= bottom - margin) {
    top = rect.bottom + gap;
    placement = "below";
  } else if (rect.top - gap - dialogRect.height >= viewport.top + margin) {
    top = rect.top - gap - dialogRect.height;
    placement = "above";
  } else {
    top = Math.max(
      viewport.top + margin,
      Math.min(
        bottom - dialogRect.height - margin,
        viewport.top + (viewport.height - dialogRect.height) / 2,
      ),
    );
    placement = "center";
  }
  elements.tutorialDialog.dataset.placement = placement;
  elements.tutorialDialog.style.left = `${left - viewport.left}px`;
  elements.tutorialDialog.style.top = `${top - viewport.top}px`;
}

function observeTutorialTarget() {
  tutorialState.resizeObserver?.disconnect();
  if (!("ResizeObserver" in window)) return;
  tutorialState.resizeObserver = new ResizeObserver(scheduleTutorialPosition);
  if (tutorialState.target)
    tutorialState.resizeObserver.observe(tutorialState.target);
  tutorialState.resizeObserver.observe(elements.tutorialDialog);
}

function focusTutorialStepTarget(target) {
  if (target.matches("input, select, textarea, button") && !target.disabled) {
    try {
      target.focus({ preventScroll: true });
    } catch {
      // Focus is advisory; the dialog remains fully operable if a browser rejects it.
    }
  }
  window.requestAnimationFrame(() => {
    if (tutorialState.active)
      elements.tutorialNext.focus({ preventScroll: true });
  });
}

function showTutorialUnavailableStep(step, index) {
  if (index === tutorialState.index && tutorialActionIsPending()) return;
  if (
    index === tutorialState.index &&
    tutorialState.gateComplete &&
    tutorialState.session?.completedStepIds.includes(step.id)
  ) {
    completeTutorialGate("Completed; continue when ready");
    return;
  }
  invalidateTutorialGate();
  tutorialState.target?.removeAttribute("data-tutorial-active-target");
  tutorialState.index = index;
  const completed = Boolean(
    tutorialState.session?.completedStepIds.includes(step.id),
  );
  tutorialState.gateComplete = completed;
  const snapshot = {
    sessionGeneration: tutorialState.sessionGeneration,
    gateGeneration: tutorialState.gateGeneration,
    stepId: step.id,
  };
  tutorialState.target =
    workspaceHeading(tutorialState.page)?.querySelector("h1") ||
    workspaceHeading(tutorialState.page) ||
    document.body;
  tutorialState.target?.setAttribute("data-tutorial-active-target", "true");
  elements.tutorialPage.textContent = tutorialTours[tutorialState.page].title;
  elements.tutorialProgress.textContent = `Step ${index + 1} of ${tutorialTours[tutorialState.page].steps.length}`;
  elements.tutorialTitle.textContent = step.title;
  elements.tutorialInstruction.textContent = completed
    ? `${step.instruction} This step is already complete; no further action is needed.`
    : `${step.instruction} The required record or control is not currently available. Refresh the page or verify the saved record to recover without replaying the action.`;
  elements.tutorialBack.hidden = false;
  elements.tutorialBack.disabled = index === 0;
  elements.tutorialNext.hidden = false;
  elements.tutorialNext.disabled = !completed;
  elements.tutorialNext.textContent =
    index === tutorialTours[tutorialState.page].steps.length - 1
      ? "Finish"
      : "Next";
  tutorialUi.useValue.hidden = true;
  tutorialUi.confirmPanel.hidden = true;
  tutorialUi.resumePanel.hidden = true;
  tutorialUi.gateStatus.textContent = completed
    ? "Completed; continue when ready"
    : "The saved control is unavailable. Verify the saved record or restart to recover.";
  tutorialUi.resumePanel.hidden = false;
  tutorialUi.resume.hidden = completed || !tutorialRecoverableRecord();
  tutorialUi.resume.textContent = "Verify saved record";
  tutorialUi.restart.hidden = false;
  SkynetDialog.placeGuide(elements.tutorialLayer, tutorialState.target);
  SkynetDialog.openGuide(elements.tutorialLayer);
  elements.tutorialSpotlight.hidden = true;
  elements.tutorialLayer.classList.add("is-fallback");
  tutorialState.session.stepId = step.id;
  persistTutorialSession();
  tutorialState.renderObserver?.disconnect();
  tutorialState.renderObserver = new MutationObserver(() => {
    if (!tutorialGenerationMatches(snapshot)) return;
    const target = tutorialInteractionTarget(step);
    if (tutorialVisibleRect(target)) {
      tutorialState.renderObserver?.disconnect();
      tutorialState.renderObserver = null;
      showTutorialStep(index, 1);
    }
  });
  tutorialState.renderObserver.observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
  });
  scheduleTutorialPosition();
  elements.tutorialBack.focus({ preventScroll: true });
}

function showTutorialStep(requestedIndex, direction = 1) {
  if (!tutorialState.active) return;
  const tour = tutorialTours[tutorialState.page];
  let index = requestedIndex;
  let target = null;
  while (index >= 0 && index < tour.steps.length) {
    const candidateStep = tour.steps[index];
    if (candidateStep.skipIf?.()) {
      index += direction;
      continue;
    }
    target = prepareTutorialTarget(candidateStep);
    if (tutorialVisibleRect(target)) break;
    if (candidateStep.waitForTarget) {
      showTutorialUnavailableStep(candidateStep, index);
      return;
    }
    target = null;
    index += direction;
  }
  if (!target) {
    if (direction > 0) {
      const pageTitle = tour.title;
      endTutorial(false);
      showToast(
        `${pageTitle} tutorial stopped because no visible step was available.`,
        true,
      );
    }
    return;
  }

  tutorialState.target?.removeAttribute("data-tutorial-active-target");
  tutorialState.index = index;
  tutorialState.target = target;
  target.setAttribute("data-tutorial-active-target", "true");
  const step = tour.steps[index];
  elements.tutorialPage.textContent = tour.title;
  elements.tutorialProgress.textContent = `Step ${index + 1} of ${tour.steps.length}`;
  elements.tutorialTitle.textContent = step.title;
  elements.tutorialInstruction.textContent = step.instruction;
  elements.tutorialBack.disabled = index === 0;
  elements.tutorialNext.textContent =
    index === tour.steps.length - 1 ? "Finish" : "Next";
  elements.tutorialLayer.classList.remove("is-fallback");
  elements.tutorialSpotlight.hidden = false;
  setupTutorialGate(step);

  const reducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  ).matches;
  target.scrollIntoView({
    behavior: reducedMotion ? "auto" : "smooth",
    block: "center",
    inline: "nearest",
  });
  clearTimeout(tutorialState.settleTimer);
  tutorialState.settleTimer = window.setTimeout(
    scheduleTutorialPosition,
    reducedMotion ? 0 : 220,
  );
  scheduleTutorialPosition();
  observeTutorialTarget();
  focusTutorialStepTarget(target);
}

function tutorialFocusables() {
  const dialogControls = [
    ...elements.tutorialDialog.querySelectorAll(
      "button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled])",
    ),
  ].filter((control) => !control.hidden && !control.closest("[hidden]"));
  const target = tutorialState.target;
  if (
    target?.matches(
      "button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href]",
    )
  )
    dialogControls.push(target);
  return [...new Set(dialogControls)];
}

function tutorialEditableHasFocus() {
  const active = document.activeElement;
  return Boolean(
    active?.matches(
      'input, select, textarea, [contenteditable]:not([contenteditable="false"]), [contenteditable="true"]',
    ),
  );
}

function advanceTutorial(direction = 1) {
  if (!tutorialState.active || tutorialState.index < 0) return false;
  const tour = tutorialTours[tutorialState.page];
  const step = tutorialStep();
  if (
    direction > 0 &&
    step?.gate &&
    !step.optional &&
    !tutorialState.gateComplete
  ) {
    tutorialUi.gateStatus.textContent =
      "Complete the highlighted step before continuing";
    tutorialInteractionTarget(step)?.focus?.({ preventScroll: true });
    return false;
  }
  if (direction > 0 && tutorialState.index >= tour.steps.length - 1) {
    if (
      !tutorialState.gateComplete &&
      !tutorialVisibleRect(tutorialState.target)
    )
      return false;
    invalidateTutorialGate();
    endTutorial(true);
    return true;
  }
  let nextIndex = tutorialState.index + (direction < 0 ? -1 : 1);
  if (
    direction > 0 &&
    step?.request?.method === "GET" &&
    step.request.bind &&
    tutorialBindingIsOwned(step.request.bind)
  ) {
    const record = tutorialRecordFor(
      step.request.bind,
      tutorialState.session.bindings[step.request.bind],
    );
    if (record?.cleanupState === "archived") {
      const cleanupIndex = tour.steps.findIndex(
        (candidate) => candidate.request?.cleanupBinding === step.request.bind,
      );
      if (cleanupIndex > tutorialState.index) nextIndex = cleanupIndex;
    }
  }
  if (nextIndex < 0 || nextIndex >= tour.steps.length) return false;
  invalidateTutorialGate();
  showTutorialStep(nextIndex, direction < 0 ? -1 : 1);
  return true;
}

function onTutorialKeydown(event) {
  if (!tutorialState.active) return;
  if (document.querySelector("dialog[data-app-confirmation][open]")) return;
  if (event.key === "Tab") {
    const controls = tutorialFocusables();
    if (!controls.length) return;
    const current = controls.indexOf(document.activeElement);
    const next =
      current < 0
        ? event.shiftKey
          ? controls.length - 1
          : 0
        : (current + (event.shiftKey ? -1 : 1) + controls.length) %
          controls.length;
    event.preventDefault();
    controls[next].focus({ preventScroll: true });
    event.stopImmediatePropagation();
    return;
  }
  if (event.key === "Escape") {
    event.preventDefault();
    event.stopImmediatePropagation();
    endTutorial(false);
    return;
  }
  if (tutorialEditableHasFocus()) return;
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    event.stopImmediatePropagation();
    advanceTutorial(-1);
    return;
  }
  if (event.key === "ArrowRight") {
    event.preventDefault();
    event.stopImmediatePropagation();
    advanceTutorial(1);
    return;
  }
  if (event.key === "Enter" || event.key === " ") {
    const control = document.activeElement;
    if (tutorialEventHitsTarget({ target: control }, tutorialState.target))
      return;
    if (
      control instanceof HTMLButtonElement &&
      elements.tutorialDialog.contains(control) &&
      control !== elements.tutorialNext &&
      control !== elements.tutorialBack
    )
      return;
    event.preventDefault();
    event.stopImmediatePropagation();
    if (control === elements.tutorialNext) advanceTutorial(1);
    else if (control === elements.tutorialBack) advanceTutorial(-1);
    return;
  }
  event.stopImmediatePropagation();
}

function onTutorialNavigation() {
  if (tutorialState.active && activeTab !== tutorialState.page)
    endTutorial(false);
}

function onTutorialNextCapture(event) {
  if (!tutorialState.active) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  if (event.isTrusted) advanceTutorial(1);
}

function onTutorialBackCapture(event) {
  if (!tutorialState.active) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  if (event.isTrusted) advanceTutorial(-1);
}

function observeTutorialPage() {
  tutorialState.pageObserver?.disconnect();
  tutorialState.pageObserver = null;
  const panel = document.querySelector(
    `[data-tab-panel="${tutorialState.page}"]`,
  );
  if (!panel || !("MutationObserver" in window)) return;
  tutorialState.pageObserver = new MutationObserver(onTutorialNavigation);
  tutorialState.pageObserver.observe(panel, {
    attributes: true,
    attributeFilter: ["class", "hidden", "aria-hidden"],
  });
}

function addTutorialRuntimeListeners() {
  document.addEventListener("keydown", onTutorialKeydown, true);
  for (const type of ["input", "change", "click", "submit"])
    document.addEventListener(type, onTutorialInteraction, true);
  elements.tutorialNext.addEventListener("click", onTutorialNextCapture, true);
  elements.tutorialBack.addEventListener("click", onTutorialBackCapture, true);
  window.addEventListener("resize", scheduleTutorialPosition, {
    passive: true,
  });
  window.addEventListener("orientationchange", scheduleTutorialPosition, {
    passive: true,
  });
  window.addEventListener("scroll", scheduleTutorialPosition, true);
  window.addEventListener("hashchange", onTutorialNavigation);
  window.visualViewport?.addEventListener("resize", scheduleTutorialPosition, {
    passive: true,
  });
  window.visualViewport?.addEventListener("scroll", scheduleTutorialPosition, {
    passive: true,
  });
  observeTutorialPage();
}

function removeTutorialRuntimeListeners() {
  document.removeEventListener("keydown", onTutorialKeydown, true);
  for (const type of ["input", "change", "click", "submit"])
    document.removeEventListener(type, onTutorialInteraction, true);
  elements.tutorialNext.removeEventListener(
    "click",
    onTutorialNextCapture,
    true,
  );
  elements.tutorialBack.removeEventListener(
    "click",
    onTutorialBackCapture,
    true,
  );
  window.removeEventListener("resize", scheduleTutorialPosition);
  window.removeEventListener("orientationchange", scheduleTutorialPosition);
  window.removeEventListener("scroll", scheduleTutorialPosition, true);
  window.removeEventListener("hashchange", onTutorialNavigation);
  window.visualViewport?.removeEventListener(
    "resize",
    scheduleTutorialPosition,
  );
  window.visualViewport?.removeEventListener(
    "scroll",
    scheduleTutorialPosition,
  );
  tutorialState.pageObserver?.disconnect();
  tutorialState.pageObserver = null;
}

function startTutorial(page, launcher) {
  const tour = tutorialTours[page];
  if (!tour || tutorialState.active) return;
  tutorialState.active = true;
  tutorialState.page = page;
  tutorialState.index = -1;
  tutorialState.target = null;
  tutorialState.launcher = launcher;
  tutorialState.originalFocus = document.activeElement;
  tutorialState.originalTab = activeTab;
  tutorialState.originalCollectionView = document.querySelector(
    "[data-data-tab][aria-selected=true]",
  )?.dataset.dataTab;
  tutorialState.scrollX = window.scrollX;
  tutorialState.scrollY = window.scrollY;
  tutorialState.openedDetails = new Map();
  tutorialState.revealed = new Map();
  tutorialState.recovering = false;
  const savedSession = loadTutorialSession(page);
  tutorialState.session = savedSession || newTutorialSession(page);
  tutorialState.sessionGeneration = tutorialUniqueId("run");
  tutorialState.session.sessionGeneration = tutorialState.sessionGeneration;
  tutorialState.gateGeneration = 0;
  tutorialState.ownedVerified = new Map();
  ensureInteractiveTutorialControls();
  if (activeTab !== page) activateTab(page);
  document.body.classList.add("has-active-tutorial");
  SkynetDialog.openGuide(elements.tutorialLayer);
  addTutorialRuntimeListeners();
  if (savedSession) showTutorialResumeChoice();
  else showTutorialStep(0, 1);
}

function endTutorial(completed = false) {
  if (!tutorialState.active) return;
  const page = tutorialState.page;
  const originalTab = tutorialState.originalTab;
  const launcher = tutorialState.launcher;
  const originalFocus = tutorialState.originalFocus;
  const scrollX = tutorialState.scrollX;
  const scrollY = tutorialState.scrollY;
  if (tutorialState.session) {
    tutorialState.session.finished = completed;
    if (tutorialState.index >= 0)
      tutorialState.session.stepId =
        tutorialStep()?.id || tutorialState.session.stepId;
    persistTutorialSession();
  }
  invalidateTutorialGate();
  tutorialState.active = false;
  tutorialState.target?.removeAttribute("data-tutorial-active-target");
  tutorialState.resizeObserver?.disconnect();
  tutorialState.resizeObserver = null;
  tutorialState.renderObserver?.disconnect();
  tutorialState.renderObserver = null;
  if (tutorialState.animationFrame)
    window.cancelAnimationFrame(tutorialState.animationFrame);
  tutorialState.animationFrame = 0;
  clearTimeout(tutorialState.settleTimer);
  removeTutorialRuntimeListeners();
  [...tutorialState.revealed.keys()].reverse().forEach((section) => {
    if (section.isConnected)
      hideRevealedPanel(section, null, { restoreFocus: false });
  });
  [...tutorialState.openedDetails.keys()].reverse().forEach((disclosure) => {
    if (disclosure.isConnected) disclosure.open = false;
  });
  tutorialState.revealed.clear();
  tutorialState.openedDetails.clear();
  document.body.classList.remove("has-active-tutorial");
  SkynetDialog.closeGuide(elements.tutorialLayer);
  elements.tutorialLayer.classList.remove("is-fallback");
  tutorialUi.confirmPanel.hidden = true;
  tutorialUi.resumePanel.hidden = true;
  tutorialUi.useValue.hidden = true;
  if (completed) {
    try {
      window.localStorage.setItem(
        workspaceStorageKey(tutorialStorageKey(page)),
        "1",
      );
    } catch {
      // Completion persistence is optional.
    }
    updateTutorialLaunchStatus(page);
  }
  if (
    tutorialState.originalCollectionView &&
    typeof setCollectionView === "function"
  )
    setCollectionView(tutorialState.originalCollectionView, { persist: false });
  if (activeTab !== originalTab) activateTab(originalTab);
  window.scrollTo({ left: scrollX, top: scrollY, behavior: "auto" });
  window.requestAnimationFrame(() => {
    const fallback = workspaceHeading(page)?.querySelector("h1");
    const focusTarget = launcher?.isConnected
      ? launcher
      : originalFocus?.isConnected
        ? originalFocus
        : fallback;
    if (focusTarget instanceof HTMLElement)
      focusTarget.focus({ preventScroll: true });
  });
  tutorialState.page = null;
  tutorialState.index = -1;
  tutorialState.target = null;
  tutorialState.launcher = null;
  tutorialState.originalFocus = null;
  tutorialState.session = null;
  tutorialState.pendingAttempt = null;
  tutorialState.sessionGeneration = null;
  tutorialState.ownedVerified = new Map();
}

function workspaceHeading(page) {
  const workspace = ["collection", "datasets"].includes(page)
    ? "data"
    : ["experiments", "adapters", "runs"].includes(page)
      ? "experiment-workspace"
      : page;
  return document.querySelector(`#${workspace} .page-heading`);
}

function initializeTutorials() {
  initializeTutorialRecordRows();
  for (const [page, tour] of Object.entries(tutorialTours)) {
    const heading = workspaceHeading(page);
    if (!heading) continue;
    let actions = heading.querySelector(":scope > .page-actions");
    if (!actions) {
      actions = document.createElement("div");
      actions.className = "page-actions";
      [...heading.children]
        .filter((child) => child.tagName === "BUTTON")
        .forEach((button) => actions.append(button));
      heading.append(actions);
    }
    const button = document.createElement("button");
    button.id = `${page}-tutorial-button`;
    button.type = "button";
    button.className = "button button-outline tutorial-launch";
    button.dataset.tutorialPage = page;
    button.textContent = "Tutorial";
    button.setAttribute("aria-label", `Start ${tour.title} tutorial`);
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      startTutorial(page, button);
    });
    actions.insertBefore(button, actions.firstChild);
    updateTutorialLaunchStatus(page);
  }

  elements.tutorialExit.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    endTutorial(false);
  });
  elements.tutorialBack.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    showTutorialStep(tutorialState.index - 1, -1);
  });
  elements.tutorialNext.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    const steps = tutorialTours[tutorialState.page]?.steps || [];
    if (tutorialState.index >= steps.length - 1) endTutorial(true);
    else showTutorialStep(tutorialState.index + 1, 1);
  });
  elements.tutorialLayer.addEventListener("click", (event) => {
    if (elements.tutorialDialog.contains(event.target)) return;
    event.preventDefault();
    event.stopPropagation();
  });
}

function isSensitiveKey(key) {
  return /(token|secret|password|credential|private.?key)/i.test(key);
}

function trackingProviderLabel(provider) {
  if (provider === "wandb") return "Weights & Biases";
  if (provider === "mlflow") return "MLflow";
  return String(provider || "Tracking");
}

function selectedTrackingProviders() {
  return [
    elements.wandbEnabled.checked ? "wandb" : null,
    elements.mlflowEnabled.checked ? "mlflow" : null,
  ].filter(Boolean);
}

function validTrackingUrl(value) {
  if (!value || typeof value !== "string") return null;
  try {
    const parsed = new URL(value);
    return parsed.protocol === "https:" || parsed.protocol === "http:"
      ? parsed.href
      : null;
  } catch (_error) {
    return null;
  }
}

function normalizedTrackingItems(entity) {
  if (!entity || typeof entity !== "object") return [];
  const rawLinks = Array.isArray(entity.tracking_links)
    ? entity.tracking_links
    : Array.isArray(entity.tracking?.links)
      ? entity.tracking.links
      : Array.isArray(entity.latest_revision?.tracking_links)
        ? entity.latest_revision.tracking_links
        : [];
  const rawStatuses = Array.isArray(entity.tracking_statuses)
    ? entity.tracking_statuses
    : Array.isArray(entity.tracking?.providers)
      ? entity.tracking.providers
      : [];
  const items = [...rawLinks, ...rawStatuses].filter(
    (item) => item && typeof item === "object",
  );
  const legacy = [
    ["wandb", entity.wandb_url || entity.wandb_run_url, entity.wandb_run_id],
    [
      "mlflow",
      entity.mlflow_url || entity.mlflow_run_url,
      entity.mlflow_run_id,
    ],
  ];
  legacy.forEach(([provider, url, remoteId]) => {
    if (url || remoteId)
      items.push({
        provider,
        url,
        remote_id: remoteId,
        status: url ? "connected" : "queued",
      });
  });
  const deduped = new Map();
  items.forEach((item) => {
    const provider = String(item.provider || "").toLowerCase();
    if (!provider) return;
    const key = `${provider}:${item.scope || ""}:${item.remote_id || item.url || item.status || ""}`;
    if (!deduped.has(key)) deduped.set(key, { ...item, provider });
  });
  return [...deduped.values()];
}

function trackingLinksHtml(entity, { compact = false } = {}) {
  const items = normalizedTrackingItems(entity);
  if (!items.length) return '<span class="secondary">-</span>';
  return `<span class="tracking-links${compact ? " is-compact" : ""}">${items
    .map((item) => {
      const provider = trackingProviderLabel(item.provider);
      const url = validTrackingUrl(item.url);
      const label = item.provider === "wandb" ? "W&B" : item.label || provider;
      const identity = url
        ? `<a class="tracking-link" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer" title="Open ${escapeHtml(provider)}">${escapeHtml(label)}</a>`
        : `<strong>${escapeHtml(provider)}</strong>`;
      const lastError = String(item.last_error || item.error || "").trim();
      return `<span class="tracking-provider-status">${identity}${lastError ? `<span class="tracking-provider-error" title="${escapeHtml(lastError)}">${escapeHtml(lastError)}</span>` : ""}</span>`;
    })
    .join("")}</span>`;
}

function renderTrackingDetail(target, entity) {
  const items = normalizedTrackingItems(entity);
  target.hidden = items.length === 0;
  target.innerHTML = items.length
    ? `<strong>Tracking</strong>${trackingLinksHtml(entity)}`
    : "";
}

function normalizeTrackingConnections(payload) {
  const raw = payload?.connections;
  if (Array.isArray(raw)) return raw;
  if (raw && typeof raw === "object") {
    return Object.entries(raw).map(([provider, connection]) => ({
      provider,
      ...(connection || {}),
    }));
  }
  throw new Error(
    'Tracking connection response did not contain "connections".',
  );
}

function trackingConnectionElements(provider) {
  return provider === "wandb"
    ? {
        form: elements.wandbConnectionForm,
        status: elements.wandbConnectionStatus,
        detail: elements.wandbConnectionDetail,
        disconnect: elements.disconnectWandb,
        secrets: [elements.wandbApiKey],
      }
    : {
        form: elements.mlflowConnectionForm,
        status: elements.mlflowConnectionStatus,
        detail: elements.mlflowConnectionDetail,
        disconnect: elements.disconnectMlflow,
        secrets: [
          elements.mlflowConnectionPassword,
          elements.mlflowConnectionToken,
        ],
      };
}

function renderTrackingConnection(provider, connection) {
  const ui = trackingConnectionElements(provider);
  const normalized =
    connection && typeof connection === "object"
      ? { provider, ...connection }
      : {
          provider,
          configured: false,
          connected: false,
          status: "not_configured",
        };
  trackingConnections.set(provider, normalized);
  const connected = Boolean(normalized.connected);
  ui.status.className = `state-pill ${connected ? "is-running" : normalized.last_error ? "is-failed" : "is-other"}`;
  ui.status.textContent = connected ? "Connected" : "Not connected";
  ui.detail.textContent = normalized.last_error || "";
  ui.detail.hidden = !normalized.last_error;
  ui.detail.classList.toggle("is-error", Boolean(normalized.last_error));
  renderConnectionControls(ui.form, {
    connected,
    busy: ui.form.dataset.busy === "true",
  });
  ui.disconnect.textContent = "Disconnect";
  if (provider === "wandb") {
    if (!elements.wandbEntity.dataset.dirty)
      elements.wandbEntity.value = normalized.entity || "";
    elements.wandbExperimentConnection.textContent = normalized.connected
      ? `Connected workspace: ${normalized.entity || "missing verified entity"}`
      : "Not connected; configure in Settings";
  } else {
    const uri = normalized.tracking_uri || normalized.base_url || "";
    if (!elements.mlflowConnectionUri.dataset.dirty)
      elements.mlflowConnectionUri.value = uri;
    if (!elements.mlflowConnectionUsername.dataset.dirty)
      elements.mlflowConnectionUsername.value = normalized.username || "";
    elements.mlflowUri.value = uri;
    elements.mlflowExperimentConnection.textContent = normalized.connected
      ? `Connected: ${uri || "endpoint verified"}`
      : "Not connected; configure in Settings";
  }
  renderTrackingNamePreview();
  updateExperimentSubmitState();
}

function renderTrackingConnections(payload) {
  const rows = normalizeTrackingConnections(payload);
  const byProvider = new Map(
    rows.map((row) => [String(row.provider || "").toLowerCase(), row]),
  );
  ["wandb", "mlflow"].forEach((provider) =>
    renderTrackingConnection(provider, byProvider.get(provider)),
  );
}

function renderTrackingConnectionsUnavailable(message) {
  ["wandb", "mlflow"].forEach((provider) => {
    const ui = trackingConnectionElements(provider);
    trackingConnections.set(provider, {
      ...(trackingConnections.get(provider) || {}),
      status: "unavailable",
      connected: false,
      last_error: message,
    });
    ui.status.className = "state-pill is-failed";
    ui.status.textContent = "Unavailable";
    ui.detail.textContent = message;
    ui.detail.hidden = false;
    ui.detail.classList.add("is-error");
    renderConnectionControls(ui.form, { connected: false, loaded: false });
    const experimentStatus =
      provider === "wandb"
        ? elements.wandbExperimentConnection
        : elements.mlflowExperimentConnection;
    experimentStatus.textContent = "Connection API unavailable";
  });
  renderTrackingNamePreview();
  updateExperimentSubmitState();
}

async function loadTrackingConnections(force = false) {
  if (trackingConnectionsLoaded && !force) return;
  try {
    const payload = await api("/api/tracking/connections");
    renderTrackingConnections(payload);
    trackingConnectionsLoaded = true;
  } catch (error) {
    trackingConnectionsLoaded = false;
    renderTrackingConnectionsUnavailable(error.message);
    throw error;
  }
}

function trackingConnectionPayload(provider) {
  if (provider === "wandb") {
    const payload = {};
    if (elements.wandbApiKey.value)
      payload.api_key = elements.wandbApiKey.value;
    if (trackingConnections.get(provider)?.base_url)
      payload.base_url = trackingConnections.get(provider).base_url;
    if (elements.wandbEntity.value.trim())
      payload.entity = elements.wandbEntity.value.trim();
    payload.remember = true;
    return payload;
  }
  const payload = {
    tracking_uri: elements.mlflowConnectionUri.value.trim(),
    verify_tls: true,
  };
  if (elements.mlflowConnectionUsername.value.trim())
    payload.username = elements.mlflowConnectionUsername.value.trim();
  if (elements.mlflowConnectionPassword.value)
    payload.password = elements.mlflowConnectionPassword.value;
  if (elements.mlflowConnectionToken.value)
    payload.token = elements.mlflowConnectionToken.value;
  payload.remember = true;
  return payload;
}

function setTrackingConnectionBusy(provider, busy) {
  const ui = trackingConnectionElements(provider);
  renderConnectionControls(ui.form, {
    connected: Boolean(trackingConnections.get(provider)?.connected),
    busy,
  });
}

function clearTrackingSecrets(provider) {
  trackingConnectionElements(provider).secrets.forEach((input) => {
    input.value = "";
  });
}

async function submitTrackingConnection(provider, action) {
  const ui = trackingConnectionElements(provider);
  const notificationScope = `tracking:${provider}`;
  if (
    ui.form.dataset.busy === "true" ||
    (action === "connect" && trackingConnections.get(provider)?.connected)
  )
    return;
  const payload =
    action === "connect" ? trackingConnectionPayload(provider) : null;
  if (action === "connect") {
    if (
      provider === "wandb" &&
      !payload.api_key &&
      !trackingConnections.get(provider)?.configured
    ) {
      showToast("Enter a W&B API key to connect.", true, {
        scope: notificationScope,
      });
      elements.wandbApiKey.focus();
      return;
    }
    if (provider === "mlflow" && !payload.tracking_uri) {
      showToast("Enter the MLflow tracking URI.", true, {
        scope: notificationScope,
      });
      elements.mlflowConnectionUri.focus();
      return;
    }
  }
  setTrackingConnectionBusy(provider, true);
  ui.detail.classList.remove("is-error");
  ui.detail.hidden = false;
  ui.detail.textContent =
    action === "disconnect" ? "Disconnecting…" : "Connecting…";
  try {
    const path =
      action === "connect"
        ? `/api/tracking/connections/${provider}/connect`
        : `/api/tracking/connections/${provider}/${action}`;
    const result = await api(path, {
      method: "POST",
      ...(action === "connect" ? { body: JSON.stringify(payload) } : {}),
    });
    if (!result?.connection)
      throw new Error('Tracking response did not contain "connection".');
    ui.form
      .querySelectorAll("[data-dirty]")
      .forEach((input) => delete input.dataset.dirty);
    renderTrackingConnection(provider, result.connection);
    trackingConnectionsLoaded = true;
    clearTrackingSecrets(provider);
    await refreshResolvedSettings();
    showToast(
      `${trackingProviderLabel(provider)} ${action === "disconnect" ? "disconnected" : "connected"}.`,
      false,
      { scope: notificationScope },
    );
  } catch (error) {
    // Failed validation changes the canonical status; every view must see it.
    await loadTrackingConnections(true).catch(() => {});
    await refreshResolvedSettings();
    ui.detail.textContent = error.message;
    ui.detail.hidden = false;
    ui.detail.classList.add("is-error");
    showToast(
      `${trackingProviderLabel(provider)} ${action} failed: ${error.message}`,
      true,
      { scope: notificationScope },
    );
  } finally {
    setTrackingConnectionBusy(provider, false);
  }
}

function derivedTrackingName(value) {
  return String(value || "").trim();
}

function renderTrackingNamePreview() {
  const experimentName = derivedTrackingName(elements.experimentName.value);
  const previews = [];
  if (elements.wandbEnabled.checked) {
    const connection = trackingConnections.get("wandb") || {};
    const project =
      derivedTrackingName(elements.wandbProject.value) ||
      experimentName ||
      "Enter an experiment name";
    previews.push(
      `<div><strong>W&amp;B</strong><span>${escapeHtml(connection.entity || "No verified workspace")} / ${escapeHtml(project)}</span><small>${escapeHtml(elements.wandbRunName.value.trim() || "No run name template")}</small></div>`,
    );
  }
  if (elements.mlflowEnabled.checked) {
    const experiment =
      derivedTrackingName(elements.mlflowExperiment.value) ||
      experimentName ||
      "Enter an experiment name";
    previews.push(
      `<div><strong>MLflow</strong><span>${escapeHtml(experiment)}</span><small>${escapeHtml(elements.mlflowRunName.value.trim() || "No run name template")}</small></div>`,
    );
  }
  elements.wandbProject.placeholder = experimentName
    ? `Automatic: ${experimentName}`
    : "Defaults to the Skynet experiment name";
  elements.mlflowExperiment.placeholder = experimentName
    ? `Automatic: ${experimentName}`
    : "Defaults to the Skynet experiment name";
  elements.trackingNamePreview.innerHTML = previews.length
    ? previews.join("")
    : '<span class="secondary">No tracking provider selected.</span>';
}

function flattenSettings(value, prefix = "", depth = 0) {
  if (!value || typeof value !== "object") {
    return [[prefix || "value", value]];
  }
  const entries = [];
  for (const [key, nested] of Object.entries(value)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (isSensitiveKey(path)) {
      entries.push([path, nested ? "configured" : "not configured"]);
    } else if (nested && typeof nested === "object") {
      entries.push(...flattenSettings(nested, path, depth + 1));
    } else {
      entries.push([path, nested]);
    }
  }
  return entries;
}

function renderSettings(payload) {
  const settings =
    payload && typeof payload === "object" && !Array.isArray(payload)
      ? payload.settings && typeof payload.settings === "object"
        ? payload.settings
        : payload
      : null;
  if (!settings) throw new Error("Settings API response must be an object.");
  if (typeof renderWorkspaceStorage === "function")
    renderWorkspaceStorage(settings.storage);
  const rows = flattenSettings(settings);
  elements.settingsBody.innerHTML = rows.length
    ? rows
        .map(
          ([key, value]) => `
      <tr>
        <td><code>${escapeHtml(key)}</code></td>
        <td class="wrap-cell">${escapeHtml(value ?? "-")}</td>
      </tr>`,
        )
        .join("")
    : emptyRow(2, "No settings were returned.");
}

async function refreshResolvedSettings() {
  try {
    renderSettings(await api("/api/settings"));
    clearNotificationScope("settings:resolved");
  } catch (error) {
    elements.settingsBody.innerHTML = emptyRow(
      2,
      "Settings could not be loaded.",
    );
    showNotice(elements.settingsError, `Settings: ${error.message}`, {
      scope: "settings:resolved",
    });
  }
}

async function loadSettings(force = false) {
  if (loadedTabs.has("settings") && !force) return;
  elements.refreshSettings.disabled = true;
  try {
    const [settingsResult, connectionsResult] = await Promise.allSettled([
      api("/api/settings"),
      loadTrackingConnections(force),
      ...(typeof loadSlackSettings === "function" ? [loadSlackSettings()] : []),
    ]);
    if (settingsResult.status === "fulfilled") {
      renderSettings(settingsResult.value);
      clearNotificationScope("settings:resolved");
    } else {
      elements.settingsBody.innerHTML = emptyRow(
        2,
        "Settings could not be loaded.",
      );
      showNotice(
        elements.settingsError,
        `Settings: ${settingsResult.reason.message}`,
        { scope: "settings:resolved" },
      );
    }
    if (connectionsResult.status === "rejected")
      showNotice(
        elements.settingsError,
        `Tracking: ${connectionsResult.reason.message}`,
        { scope: "settings:connections" },
      );
    else clearNotificationScope("settings:connections");
    loadedTabs.add("settings");
  } catch (error) {
    elements.settingsBody.innerHTML = emptyRow(
      2,
      "Settings could not be loaded.",
    );
    showNotice(
      elements.settingsError,
      `Settings API unavailable: ${error.message}`,
      { scope: "settings:resolved" },
    );
  } finally {
    elements.refreshSettings.disabled = false;
  }
}

function updateExperimentFields() {
  const manualGpu = elements.gpuMode.value === "manual";
  elements.experimentGpuCount.disabled = !manualGpu;
  elements.experimentGpuCount.required = manualGpu;

  const checkpoint = elements.checkpointMode.value !== "none";
  elements.checkpointPathField.hidden = !checkpoint;
  elements.checkpointPath.disabled = !checkpoint;
  elements.checkpointPath.required = checkpoint;

  document.querySelectorAll(".tracked-field-wandb input").forEach((input) => {
    input.disabled = !elements.wandbEnabled.checked;
  });
  document.querySelectorAll(".tracked-field-mlflow input").forEach((input) => {
    input.disabled = !elements.mlflowEnabled.checked;
  });
  elements.mlflowUri.readOnly = true;
  renderTrackingNamePreview();

  const evaluate = elements.evaluationEnabled.checked;
  elements.experimentEvaluationSuites.disabled = !evaluate;
  updateExperimentSubmitState();
}

function loadActiveTab(tab, force = false) {
  if (tab === "cluster") return refreshCluster({ force });
  if (["experiments", "presets"].includes(tab)) return loadExperiments(force);
  if (tab === "collection") {
    window.loadLiveXR?.();
    return loadCollection(force);
  }
  if (tab === "datasets") return loadDataRegistry(force);
  if (tab === "runs") return loadRuns(true);
  if (tab === "evaluations") return loadEvaluations(force);
  if (tab === "evaluation-runs") return loadEvaluations(force);
  if (tab === "evaluation-suites") return loadEvaluationCatalog(force);
  if (tab === "adapters") return loadAdapters(force);
  if (tab === "settings") return loadSettings(force);
  if (tab === "hands") return window.loadHands?.(force);
  return Promise.resolve();
}

function activateTab(tab, updateHash = true, requestedView = null) {
  if (window.SkynetStorageConfigured === false) {
    tab = "settings";
    requestedView = null;
  }
  const allowed = [
    "cluster",
    "experiments",
    "collection",
    "datasets",
    "runs",
    "evaluations",
    "adapters",
    "settings",
    "hands",
  ];
  const isData = ["data", "collection", "datasets"].includes(tab);
  const isExperiments = ["experiments", "adapters", "runs"].includes(tab);
  const isEvaluations = [
    "evaluations",
    "evaluation-suites",
    "evaluation-runs",
  ].includes(tab);
  const navigation = isData
    ? dataNavigation
    : isExperiments
      ? experimentNavigation
      : isEvaluations
        ? evaluationNavigation
        : tab === "settings"
          ? window.settingsNavigation
          : tab === "cluster"
            ? window.clusterNavigation
            : null;
  const view = navigation
    ? window.SkynetStorageConfigured === false
      ? "storage"
      : requestedView || navigation.viewForTab(tab)
    : null;
  const next = isData
    ? ["registry", "files"].includes(view)
      ? "datasets"
      : "collection"
    : isExperiments
      ? view === "submit"
        ? "experiments"
        : view
      : isEvaluations
        ? view === "submit"
          ? "evaluations"
          : `evaluation-${view}`
        : allowed.includes(tab)
          ? tab
          : "cluster";
  if (next !== activeTab) {
    elements.toast.replaceChildren();
    elements.toast.hidden = true;
    stopClusterAutoRefresh();
    closeActiveDisclosure({ restoreFocus: false });
    if (activeRunAttemptDisclosure) resetRunAttemptContext();
  }
  activeTab = next;
  document.querySelectorAll("[data-tab-panel]").forEach((panel) => {
    panel.hidden =
      panel.dataset.tabPanel === "data"
        ? !isData
        : panel.dataset.tabPanel === "experiment-workspace"
          ? !isExperiments
          : panel.dataset.tabPanel === "evaluation-workspace"
            ? !isEvaluations
            : panel.dataset.tabPanel !== next;
  });
  navigation?.render(view);
  document.querySelectorAll("[data-tab-target]").forEach((link) => {
    const selected =
      link.dataset.tabGroup === "data"
        ? isData
        : link.dataset.tabGroup === "experiments"
          ? isExperiments
          : link.dataset.tabGroup === "evaluations"
            ? isEvaluations
            : link.dataset.tabTarget === next;
    link.classList.toggle("is-active", selected);
    if (link.getAttribute("role") === "tab") {
      link.setAttribute("aria-selected", String(selected));
      link.tabIndex = selected ? 0 : -1;
    }
  });
  const destination = navigation
    ? navigation.url(view).href
    : new URL(`#${next}`, location.href).href;
  if (location.href !== destination) {
    if (updateHash) history.pushState(null, "", destination);
    else if (navigation) history.replaceState(null, "", destination);
  }
  const loading = loadActiveTab(next);
  window.scrollTo({ top: 0, behavior: "instant" });
  return loading;
}

document.querySelectorAll("[data-tab-target]").forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    activateTab(link.dataset.tabTarget);
  });
});

function restoreNavigation() {
  activateTab(location.hash.slice(1), false);
}
window.addEventListener("hashchange", restoreNavigation);
window.addEventListener("popstate", restoreNavigation);

function restoreSshGatewayPreference() {
  let savedGateway;
  try {
    savedGateway = window.localStorage.getItem(SSH_GATEWAY_PREFERENCE_KEY);
  } catch {
    return false;
  }
  if (savedGateway === null) return false;
  if (
    ![...elements.gateway.options].some(
      (option) => option.value === savedGateway,
    )
  )
    return false;
  elements.gateway.value = savedGateway;
  return true;
}

function persistSshGatewayPreference() {
  try {
    window.localStorage.setItem(
      SSH_GATEWAY_PREFERENCE_KEY,
      elements.gateway.value,
    );
  } catch {
    // Keep gateway selection usable when browser storage is unavailable.
  }
}

restoreSshGatewayPreference();

elements.gateway.addEventListener("change", () => {
  persistSshGatewayPreference();
  invalidateExperimentPreview();
  if (activeTab === "evaluations") scheduleEvaluationTargetValidation();
  if (activeTab === "cluster")
    refreshCluster({ force: true, background: true });
  if (elements.experimentRevision.value) inspectRepositoryRuntime();
});
document.querySelector("#jobs-previous").addEventListener("click", () => {
  jobPage = Math.max(0, jobPage - 1);
  renderJobs(visibleQueueJobs, queueTotal);
});
document.querySelector("#jobs-next").addEventListener("click", () => {
  jobPage += 1;
  renderJobs(visibleQueueJobs, queueTotal);
});
elements.refreshButton.addEventListener("click", () =>
  refreshCluster({ force: true }),
);

elements.refreshExperiments.addEventListener("click", () =>
  loadExperiments(true),
);
elements.experimentPreviewButton.addEventListener("click", previewExperiment);
elements.newExperimentPreset.addEventListener("click", openExperimentPreset);
elements.experimentPresetDialog.addEventListener(
  "close",
  restoreExperimentEditor,
);
elements.saveExperimentButton.addEventListener("click", () => {
  if (!elements.experimentPresetDialog.open) createExperiment(false);
});
elements.experimentForm.addEventListener("submit", (event) => {
  event.preventDefault();
  createExperiment(true);
});
function experimentFormChanged(event) {
  if (
    event?.target?.dataset.adapterInputPath !== "native.config.training_preset"
  )
    updateTrainingPresetLabel();
  invalidateExperimentPreview();
  elements.toast.querySelectorAll(".is-error").forEach(dismissToast);
  updateExperimentSubmitState();
}
elements.experimentForm.addEventListener("input", experimentFormChanged);
elements.experimentForm.addEventListener("change", experimentFormChanged);
document
  .querySelector("#experiment-preview-variant")
  .addEventListener("change", renderSelectedExperimentScript);
document
  .querySelector("#experiment-preview-full")
  .addEventListener("change", renderSelectedExperimentScript);
document
  .querySelector("#experiment-preview-download")
  .addEventListener("click", () => {
    const index = Number(
      document.querySelector("#experiment-preview-variant").value,
    );
    const script = experimentPreviewScripts[index];
    if (!script) return;
    const url = URL.createObjectURL(new Blob([script], { type: "text/plain" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `variant-${index + 1}.sbatch`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
elements.experimentAdapter.addEventListener("change", () => {
  const selectedId = elements.experimentAdapter.value;
  loadedExperimentCanonicalContext = null;
  populateExperimentAdapters(selectedId);
  elements.experimentAdapter.setCustomValidity("");
  elements.experimentAdapter.removeAttribute("aria-invalid");
  applySelectedAdapter();
});
elements.experimentSource.addEventListener("input", scheduleSourceBranchLoad);
elements.experimentSource.addEventListener("change", () => {
  renderAdapterDeclaredFields();
  loadSourceBranches();
});
elements.sourceRefRefresh.addEventListener("click", () => {
  const pinnedCommit =
    elements.experimentBranch.dataset.pinnedCommit === "true";
  delete elements.experimentBranch.dataset.pinnedCommit;
  loadSourceBranches(!pinnedCommit, true);
});
elements.experimentBranch.addEventListener("change", () => {
  const repository = elements.experimentSource.value.trim();
  const branch = elements.experimentBranch.value;
  const branchTip =
    sourceBranches.find(
      (branch) => branch.name === elements.experimentBranch.value,
    )?.sha || "";
  try {
    const savedCommit =
      savedSourceSelection(repository).commits?.[branch] || "";
    saveSourceSelection(repository, branch, savedCommit || branchTip);
    loadSourceCommits(savedCommit || branchTip);
  } catch (error) {
    setSourceRefStatus(error.message, "error");
  }
});
elements.experimentRevision.addEventListener("change", () => {
  updateSourceCommitMeta();
  try {
    saveSourceSelection(
      elements.experimentSource.value.trim(),
      elements.experimentBranch.value,
      elements.experimentRevision.value,
    );
  } catch (error) {
    setSourceRefStatus(error.message, "error");
    return;
  }
  inspectRepositoryRuntime();
  renderAdapterDeclaredFields();
});
elements.experimentWorkdir.addEventListener("change", () =>
  inspectRepositoryRuntime(),
);
elements.adapterDeclaredFieldsGrid.addEventListener("input", (event) => {
  const control = event.target.closest("[data-adapter-input-path]");
  if (!control) return;
  const values = adapterDeclaredScopeValues(renderedAdapterDeclaredScope, true);
  if (control.dataset.adapterInputSensitive === "true")
    values?.delete(control.dataset.adapterInputPath);
  else
    values?.set(control.dataset.adapterInputPath, {
      kind: control.dataset.adapterInputKind,
      raw: adapterFieldRawValue(control),
      touched: true,
    });
  validateAdapterDeclaredFields({ focus: false, notify: false });
  renderCommonHyperparameterDefaults();
});
elements.adapterDeclaredFieldsGrid.addEventListener("change", (event) => {
  const control = event.target.closest("[data-adapter-input-path]");
  if (!control) return;
  if (control.dataset.adapterInputPath === "native.config.training_preset") {
    if (control.value) applyTrainingPreset(JSON.parse(control.value));
    return;
  }
  const values = adapterDeclaredScopeValues(renderedAdapterDeclaredScope, true);
  if (control.dataset.adapterInputSensitive === "true")
    values?.delete(control.dataset.adapterInputPath);
  else
    values?.set(control.dataset.adapterInputPath, {
      kind: control.dataset.adapterInputKind,
      raw: adapterFieldRawValue(control),
      touched: true,
    });
  if (
    adapterInputFields().some(
      (f) =>
        f.data_binding?.contract_selector === control.dataset.adapterInputPath,
    )
  )
    renderAdapterDeclaredFields();
  validateAdapterDeclaredFields({ focus: false, notify: false });
  renderCommonHyperparameterDefaults();
});
elements.experimentDataBundle.addEventListener("change", () => {
  elements.experimentDataBundle.setCustomValidity("");
  elements.experimentDataBundle.removeAttribute("aria-invalid");
  populateExperimentDataBundles();
  renderAdapterDeclaredFields();
  validateAdapterDeclaredFields({ focus: false, notify: false });
});
elements.experimentRuntime.addEventListener("change", () =>
  applyRuntimeSelection(true),
);
elements.experimentRuntimeProfileSelect.addEventListener("change", () =>
  applyRuntimeProfileSelection(true),
);
elements.gpuMode.addEventListener("change", updateExperimentFields);
elements.checkpointMode.addEventListener("change", updateExperimentFields);
elements.wandbEnabled.addEventListener("change", updateExperimentFields);
elements.mlflowEnabled.addEventListener("change", updateExperimentFields);
elements.experimentName.addEventListener("input", renderTrackingNamePreview);
elements.wandbProject.addEventListener("input", renderTrackingNamePreview);
elements.wandbRunName.addEventListener("input", renderTrackingNamePreview);
elements.mlflowExperiment.addEventListener("input", renderTrackingNamePreview);
elements.mlflowRunName.addEventListener("input", renderTrackingNamePreview);
elements.evaluationEnabled.addEventListener("change", updateExperimentFields);
elements.experimentsBody.addEventListener("click", (event) => {
  const button = event.target.closest("[data-experiment-action]");
  if (!button) return;
  if (button.dataset.experimentAction === "load") {
    activateTab("experiments", true, "submit").then(() =>
      loadExperimentConfiguration(
        button.dataset.id,
        button.dataset.revisionNumber,
        button,
      ),
    );
    return;
  }
  if (button.dataset.experimentAction === "view")
    viewExperiment(button.dataset.id, button);
  if (button.dataset.experimentAction === "submit")
    submitDraftExperiment(button.dataset.id, button);
});

elements.refreshRuns.addEventListener("click", refreshRunsPage);
elements.runSearch.addEventListener("input", renderRuns);
elements.runStatusFilter.addEventListener("change", renderRuns);
elements.runsBody.addEventListener("click", (event) => {
  const button = event.target.closest("[data-run-action]");
  if (button?.dataset.runAction === "view") viewRun(button.dataset.id, button);
});
elements.attemptsBody.addEventListener("click", (event) => {
  const button = event.target.closest("[data-attempt-action='view']");
  if (button) toggleRunAttemptDetail(button.dataset.attemptKey, button);
});
elements.runDetailActions.addEventListener("click", (event) => {
  const button = event.target.closest("[data-run-action]");
  if (!button) return;
  if (button.dataset.runAction === "evaluate") {
    if (button.disabled || button.dataset.runActionEnabled !== "true") return;
    startEvaluationForRun(button.dataset.id);
    return;
  }
  if (button.disabled || button.dataset.runActionEnabled !== "true") return;
  if (button.dataset.runAction === "attach-tracking") {
    attachRunTracking(
      button.dataset.id,
      button.dataset.trackingProvider,
      button.dataset.trackingActionPath,
      button,
    );
    return;
  }
  if (button.dataset.runAction === "recover_submission")
    recoverRunSubmission(button.dataset.id, button);
  if (button.dataset.runAction === "resume")
    resumeRun(button.dataset.id, button.dataset.runActionReason || "");
  if (button.dataset.runAction === "rerun")
    rerunRun(button.dataset.id, button.dataset.runActionReason || "");
});
elements.runDetailActions.addEventListener("click", handleCancellationAction);
document.addEventListener("visibilitychange", () => {
  stopClusterAutoRefresh();
  if (document.visibilityState !== "visible") {
    stopRunDetailPolling({ clearStatus: false });
    stopEvaluationListPolling();
    window.clearTimeout(evaluationDetailPollTimer);
    evaluationDetailPollTimer = null;
    return;
  }
  if (activeTab === "runs" && activeRunDetailId && !elements.runDetail.hidden) {
    startRunDetailPolling(activeRunDetailId, null, { initialDelay: 0 });
  }
  if (
    activeTab === "evaluation-runs" &&
    activeEvaluationDetailId &&
    !elements.evaluationDetail.hidden
  ) {
    scheduleEvaluationDetailPolling(activeEvaluationDetailId);
  }
  if (activeTab === "evaluation-runs") scheduleEvaluationListPolling(0);
  if (activeTab === "cluster") refreshCluster({ force: true });
});

elements.refreshEvaluations.addEventListener("click", () =>
  loadActiveTab(activeTab, true),
);
document
  .querySelector("#evaluation-suite-search")
  .addEventListener("input", renderEvaluationCatalog);
document
  .getElementById("evaluation-suites-body")
  .addEventListener("click", (event) => {
    const button = event.target.closest("[data-suite-view]");
    if (button) openEvaluationSuite(button.dataset.suiteView, button);
  });
elements.evaluationSuite.addEventListener("change", () => {
  elements.evaluationSuite.setCustomValidity("");
  elements.evaluationSuite.removeAttribute("aria-invalid");
  delete elements.evaluationSuite.dataset.taskSelectionError;
  pendingEvaluationSuiteId = "";
  updateEvaluationEnvironmentFromSuite();
  scheduleEvaluationTargetValidation();
});
elements.evaluationRunId.addEventListener("input", () => {
  const runId = elements.evaluationRunId.value.trim();
  elements.evaluationCheckpoint.value = "";
  pendingEvaluationSuiteId = elements.evaluationSuite.value;
  clearTimeout(evaluationSuiteReloadTimer);
  elements.evaluationSuite.innerHTML = `<option value="">${runId ? "Loading compatible suites..." : "Loading suites..."}</option>`;
  elements.evaluationSuite.value = "";
  elements.evaluationSuiteStatus.textContent = runId
    ? `Loading suites declared for training run ${runId}...`
    : "Loading registered evaluation suites...";
  updateEvaluationEnvironmentFromSuite();
  scheduleEvaluationTargetValidation();
  evaluationSuiteReloadTimer = window.setTimeout(async () => {
    if (elements.evaluationRunId.value.trim() !== runId) return;
    await loadEvaluationSuites(false, runId);
    if (elements.evaluationRunId.value.trim() === runId)
      pendingEvaluationSuiteId = "";
  }, 350);
});
elements.evaluationCheckpoint.addEventListener("input", () =>
  scheduleEvaluationTargetValidation(),
);
elements.evaluationTasksFilter.addEventListener("click", (event) => {
  if (
    elements.evaluationTasksFilter.getAttribute("aria-disabled") === "true" &&
    event.target.closest("summary")
  ) {
    event.preventDefault();
  }
});
elements.evaluationTasksOptions.addEventListener("change", () => {
  updateEvaluationTaskLabel();
  scheduleEvaluationTargetValidation();
});
elements.evaluationTasksAll.addEventListener("click", () => {
  elements.evaluationTasksOptions.querySelectorAll("input").forEach((input) => {
    input.checked = true;
  });
  updateEvaluationTaskLabel();
  scheduleEvaluationTargetValidation();
});
elements.evaluationTasksClear.addEventListener("click", () => {
  const defaults =
    selectedEvaluationSuite()?.default_tasks ||
    selectedEvaluationSuite()?.config_json?.default_tasks ||
    [];
  elements.evaluationTasksOptions.querySelectorAll("input").forEach((input) => {
    input.checked = defaults.includes(input.value);
  });
  applyEvaluationTaskSelectionPolicy();
  updateEvaluationTaskLabel();
  scheduleEvaluationTargetValidation();
});
elements.evaluationArgv.addEventListener("input", () =>
  scheduleEvaluationTargetValidation(),
);
elements.evaluationResumeArgv.addEventListener("input", () =>
  scheduleEvaluationTargetValidation(),
);
elements.evaluationForm.addEventListener("submit", createEvaluation);
elements.evaluationsBody.addEventListener("click", (event) => {
  const button = event.target.closest("[data-evaluation-action]");
  if (button?.dataset.evaluationAction === "view")
    viewEvaluation(button.dataset.id, button);
});
elements.evaluationsBody.addEventListener("click", handleCancellationAction);
elements.evaluationDetailActions.addEventListener(
  "click",
  handleCancellationAction,
);

elements.refreshDataRegistry.addEventListener("click", () =>
  loadDataRegistry(true),
);
document
  .getElementById("data-resource-search")
  .addEventListener("input", () => {
    delete document.getElementById("data-resource-search").dataset.recordingId;
    delete document.getElementById("data-resource-search").dataset.resourceIds;
    renderDataResources();
  });
document
  .querySelector("#data-show-archived")
  .addEventListener("change", refreshDataResourceTables);
elements.dataResourceForm.addEventListener("submit", createDataResource);
if (elements.showDataResourceForm) {
  elements.showDataResourceForm.addEventListener("click", (event) => {
    resetDataResourceEditor(false);
    revealPanel(elements.dataResourceForm, {
      focusTarget: elements.dataResourceProvider,
      launcher: event.currentTarget,
    });
  });
}
elements.dataResourcesBody.addEventListener("click", (event) => {
  const button = event.target.closest("[data-resource-action]");
  if (!button) return;
  if (button.dataset.resourceAction === "recordings")
    showDatasetRecordings(button.dataset.id);
  if (button.dataset.resourceAction === "dataset")
    window.openPreparedDataset?.(button.dataset.id);
  if (button.dataset.resourceAction === "import-detail")
    openDataImportDetail(button.dataset.id, button);
  if (button.dataset.resourceAction === "import")
    selectDataResourceForImport(button.dataset.id, button);
  if (button.dataset.resourceAction === "version")
    selectDataResourceForVersion(button.dataset.id, button);
  if (button.dataset.resourceAction === "edit")
    openDataResourceEditor(button.dataset.id, button);
  if (button.dataset.resourceAction === "archive")
    archiveDataResource(button.dataset.id);
  if (button.dataset.resourceAction === "restore")
    restoreDataResource(button.dataset.id);
});
elements.dataVersionForm.addEventListener("submit", createDataVersion);
elements.dataImportForm.addEventListener("submit", submitDataImport);
elements.dataImportForm.addEventListener("input", renderDataImportBudget);
elements.closeDataImportForm.addEventListener("click", () =>
  hideRevealedPanel(elements.dataImportForm),
);
elements.dataImportsBody.addEventListener("click", (event) => {
  const button = event.target.closest("[data-import-action]");
  if (button?.dataset.importAction === "detail")
    openDataImportDetail(button.dataset.id, button);
  if (button?.dataset.importAction === "cancel")
    cancelDataImport(button.dataset.id, button);
});
document
  .getElementById("data-import-detail-content")
  .addEventListener("click", (event) => {
    const button = event.target.closest("[data-import-action]");
    if (button?.dataset.importAction === "logs")
      loadDataImportLogs(button.dataset.id);
    if (button?.dataset.importAction === "cancel")
      cancelDataImport(button.dataset.id, button);
  });
document
  .getElementById("data-import-detail-dialog")
  .addEventListener("close", () => {
    selectedDataImportId = null;
  });

elements.closeDataVersionForm.addEventListener("click", () => {
  hideRevealedPanel(elements.dataVersionForm);
});
elements.dataDerivationForm.addEventListener("submit", createDataDerivation);
if (elements.showDataDerivationForm) {
  elements.showDataDerivationForm.addEventListener("click", (event) => {
    revealPanel(elements.dataDerivationForm, {
      focusTarget: elements.dataDerivationOutput,
      launcher: event.currentTarget,
    });
  });
}
document
  .querySelector("#show-collection-session-form")
  .addEventListener("click", (event) => {
    populateCollectionAdapterSelect();
    elements.collectionGateway.value = elements.gateway.value;
    updateCollectionCapabilityEvidence(true);
    revealPanel(elements.collectionSessionForm, {
      focusTarget: elements.collectionSessionAdapter,
      launcher: event.currentTarget,
    });
  });
elements.evaluationForm.addEventListener("input", () => {
  updateEvaluationSubmitState();
  scheduleEvaluationTargetValidation();
});
elements.evaluationForm.addEventListener("change", () => {
  updateEvaluationSubmitState();
  scheduleEvaluationTargetValidation();
});
document
  .querySelector("#evaluation-search")
  .addEventListener("input", () => renderEvaluations());
document
  .querySelector("#evaluation-state-filter")
  .addEventListener("change", () => renderEvaluations());
elements.refreshCollection.addEventListener("click", () => {
  window.loadLiveXR?.();
  loadCollection(true);
  document.dispatchEvent(
    new CustomEvent("dataset-preparation-refresh-requested"),
  );
});
elements.addCollectionAdapter.addEventListener("click", (event) =>
  fillCollectionAdapterForm(null, event.currentTarget),
);
elements.closeCollectionAdapter.addEventListener("click", () => {
  hideRevealedPanel(
    elements.collectionAdapterForm,
    elements.addCollectionAdapter,
  );
});
elements.collectionAdapterForm.addEventListener(
  "submit",
  saveCollectionAdapter,
);
elements.collectionAdaptersBody.addEventListener("click", (event) => {
  const button = event.target.closest("[data-collection-adapter-action]");
  if (!button) return;
  const id = button.dataset.id;
  if (button.dataset.collectionAdapterAction === "edit")
    openCollectionAdapter(id, button);
  if (button.dataset.collectionAdapterAction === "template")
    reviewCollectionTemplate(id, button);
  if (button.dataset.collectionAdapterAction === "session") {
    populateCollectionAdapterSelect(id);
    applyCollectionAdapterDefaults();
    revealPanel(elements.collectionSessionForm, {
      focusTarget: elements.collectionSessionName,
      launcher: button,
    });
  }
  if (button.dataset.collectionAdapterAction === "archive")
    setCollectionAdapterArchived(id, true);
  if (button.dataset.collectionAdapterAction === "restore")
    setCollectionAdapterArchived(id, false);
});
elements.collectionRegisterOutput.addEventListener(
  "change",
  updateCollectionRegistrationFields,
);
elements.collectionSessionAdapter.addEventListener("change", () => {
  elements.collectionSessionAdapter.setCustomValidity("");
  elements.collectionSessionAdapter.removeAttribute("aria-invalid");
  applyCollectionAdapterDefaults();
});
elements.collectionSessionForm.addEventListener(
  "submit",
  createCollectionSession,
);
elements.collectionSessionsBody.addEventListener("click", (event) => {
  const button = event.target.closest("[data-collection-session-action]");
  if (button?.dataset.collectionSessionAction === "view")
    viewCollectionSession(button.dataset.id, button);
});
elements.collectionSessionActions.addEventListener("click", (event) => {
  const button = event.target.closest("[data-collection-lifecycle-action]");
  if (button)
    runCollectionLifecycleAction(button.dataset.collectionLifecycleAction);
});
elements.collectionLoadStdout.addEventListener("click", () =>
  loadCollectionLog("stdout"),
);
elements.collectionLoadStderr.addEventListener("click", () =>
  loadCollectionLog("stderr"),
);
elements.collectionCompleteForm.addEventListener(
  "submit",
  completeCollectionSession,
);

elements.refreshSettings.addEventListener("click", () => loadSettings(true));
elements.wandbConnectionForm.addEventListener("submit", (event) => {
  event.preventDefault();
  submitTrackingConnection("wandb", "connect");
});
elements.mlflowConnectionForm.addEventListener("submit", (event) => {
  event.preventDefault();
  submitTrackingConnection("mlflow", "connect");
});
elements.disconnectWandb.addEventListener("click", () =>
  submitTrackingConnection("wandb", "disconnect"),
);
elements.disconnectMlflow.addEventListener("click", () =>
  submitTrackingConnection("mlflow", "disconnect"),
);
["wandb", "mlflow"].forEach((provider) => {
  trackingConnectionElements(provider).form.addEventListener(
    "input",
    (event) => {
      if (!event.target.matches("input")) return;
      event.target.dataset.dirty = "true";
    },
  );
});

elements.refreshAdapters.addEventListener("click", () => loadAdapters(true));
elements.addAdapter.addEventListener("click", (event) =>
  startAdapterCreate(event.currentTarget),
);
elements.adaptersBody.addEventListener("click", (event) => {
  const button = event.target.closest("[data-adapter-action]");
  if (!button) return;
  const id = button.dataset.id;
  if (button.dataset.adapterAction === "view") openAdapter(id, false, button);
  if (button.dataset.adapterAction === "edit") openAdapter(id, true, button);
  if (button.dataset.adapterAction === "clone") cloneAdapter(id);
  if (button.dataset.adapterAction === "archive") setAdapterArchived(id, true);
  if (button.dataset.adapterAction === "restore") setAdapterArchived(id, false);
});
elements.adapterEditor.addEventListener("submit", saveAdapter);
elements.closeAdapterEditor.addEventListener("click", () => {
  hideRevealedPanel(elements.adapterEditor, elements.addAdapter);
});
elements.editAdapter.addEventListener("click", () =>
  setAdapterEditorMode("edit"),
);
elements.validateAdapter.addEventListener("click", validateAdapterManifest);

installStructuredEvaluationTaskOptions();
installViewportFilterMenus();
installLatestApiReadGuard();
installDisclosureBehavior();
initializeTutorials();
dataNavigation.mountTutorial();
updateExperimentFields();
updateCollectionRegistrationFields();
updateCollectionCapabilityEvidence(true);
resetRuntimeInspection();
const initialHash = location.hash.slice(1);
activateTab(
  [
    "data",
    "cluster",
    "experiments",
    "collection",
    "datasets",
    "runs",
    "evaluations",
    "adapters",
    "settings",
    "hands",
  ].includes(initialHash)
    ? initialHash
    : "cluster",
  false,
);

window.useRegisteredDataset = async (versionId) => {
  void activateTab("experiments", true, "submit");
  await loadTrainingInputs();
  const dataset = trainingDatasetRows.find((row) => row.id === versionId);
  if (!dataset)
    throw new Error("These files are no longer available. Refresh Datasets.");
  elements.experimentDataBundle.value = versionId;
  populateExperimentDataBundles();
  renderAdapterDeclaredFields();
  invalidateExperimentPreview();
  refreshExperimentModelIO();
  showToast(
    "Dataset selected. Choose a compatible adapter and review the experiment settings.",
  );
};

window.usePreparedDataset = async (job) => {
  void activateTab("experiments", true, "submit");
  await loadTrainingInputs();
  const setup = job.training_setup;
  const adapter = adapterRows.find(
    (a) => adapterManifest(a).slug === setup?.adapter && !adapterArchived(a),
  );
  if (!adapter)
    throw new Error(
      "The training adapter is unavailable. Restore it in Adapters.",
    );
  loadedExperimentCanonicalContext = null;
  loadedExperimentAdapterSnapshot = null;
  populateExperimentAdapters(adapterOptionId(adapter));
  elements.experimentAdapter.value = adapterOptionId(adapter);
  applySelectedAdapter({ loadSource: false });
  elements.experimentName.value =
    (job.name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "")
      .slice(0, 70) || "recorded-dataset") +
    "-" +
    String(job.format || setup.adapter).replace(/[^a-z0-9-]+/g, "-");
  renderTrackingNamePreview();
  elements.experimentDataBundle.value = job.version_id;
  populateExperimentDataBundles();
  renderAdapterDeclaredFields();
  if (setup.preset) applyTrainingPreset(setup.preset);
  invalidateExperimentPreview();
  elements.experimentForm.scrollIntoView({ block: "start" });
  installPinnedSourceRevision(setup, "dataset preparation");
  await inspectRepositoryRuntime();
  const profile = runtimeProfiles.find((p) => p.id === setup.runtime_profile);
  if (profile) {
    elements.experimentRuntime.value = profile.backend;
    applyRuntimeSelection();
    elements.experimentRuntimeProfileSelect.value = profile.id;
    applyRuntimeProfileSelection();
  }
  showToast(
    profile
      ? "Dataset, policy, pinned source and runtime selected. Preview the training run when ready."
      : "Dataset and pinned policy selected. Choose an available training runtime, then preview.",
  );
};

// I/O uses the same resolver in adapter inspection, submission, and saved attempts.
elements.experimentForm.addEventListener("input", (event) => {
  if (event.target.matches("[data-adapter-input-path], #native-overrides"))
    refreshExperimentModelIO();
});
elements.experimentForm.addEventListener("change", (event) => {
  if (
    event.target.matches(
      "[data-adapter-input-path], #experiment-adapter, #experiment-data-bundle",
    )
  )
    refreshExperimentModelIO();
});
elements.adapterManifest.addEventListener("input", () => {
  refreshAdapterModelIO();
});

document.getElementById("experiment-search").addEventListener("input", () => {
  const search = document.getElementById("experiment-search");
  delete search.dataset.datasetId;
  delete search.dataset.presetIds;
  renderExperiments();
});
