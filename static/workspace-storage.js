/* Personal base path. Existing runs retain their recorded locations. */
const storageForm = document.querySelector("#workspace-storage-form");
const storagePath = document.querySelector("#workspace-base-path");
const storageStatus = document.querySelector("#workspace-storage-status");
const storageDetail = document.querySelector("#workspace-storage-detail");
let storageSaved = null;
let storageBusy = false;

function storageControls() {
  const dirty = storagePath.value.trim() !== storageSaved?.work_root;
  storagePath.disabled = storageBusy || !storageSaved;
  document.querySelector("#save-workspace-storage").disabled =
    storageBusy || !storageSaved || !dirty;
  document.querySelector("#init-workspace").disabled =
    storageBusy || !storageSaved || dirty;
}

function renderWorkspaceStorage(settings) {
  if (!settings?.work_root) return;
  storageSaved = settings;
  if (!storageForm.dataset.dirty) storagePath.value = settings.work_root;
  storageStatus.textContent = storageForm.dataset.dirty
    ? "Unsaved changes"
    : "Saved";
  if (!storageBusy) {
    storageDetail.classList.remove("is-error");
    storageDetail.textContent = storageForm.dataset.dirty
      ? `Saved path: ${settings.work_root}. Save your changes before initializing directories.`
      : `Your base path: ${settings.work_root}. Save changes first, then initialize directories to check cluster access.`;
  }
  storageControls();
}

async function storageAction(action) {
  if (storageBusy || !storageSaved) return;
  storageBusy = true;
  storageForm.setAttribute("aria-busy", "true");
  storageControls();
  storageDetail.classList.remove("is-error");
  try {
    await action();
  } catch (error) {
    storageDetail.textContent = error.message;
    storageDetail.classList.add("is-error");
    storageStatus.textContent = "Needs attention";
  } finally {
    storageBusy = false;
    storageForm.removeAttribute("aria-busy");
    storageControls();
  }
}

storagePath.addEventListener("input", () => {
  if (storagePath.value.trim() === storageSaved?.work_root)
    delete storageForm.dataset.dirty;
  else storageForm.dataset.dirty = "true";
  storageStatus.textContent = storageForm.dataset.dirty
    ? "Unsaved changes"
    : "Saved";
  storageDetail.classList.remove("is-error");
  storageDetail.textContent =
    "Save path applies to new training runs. Existing files are not moved.";
  storageControls();
});

storageForm.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!storageSaved || !storageForm.reportValidity()) return;
  const payload = {
    work_root: storagePath.value.trim(),
    expected_work_root: storageSaved.work_root,
  };
  void storageAction(async () => {
    storageDetail.textContent = "Saving path…";
    const saved = await api("/api/workspace/storage", {
      method: "PUT",
      body: JSON.stringify(payload),
    });
    delete storageForm.dataset.dirty;
    renderWorkspaceStorage(saved);
    storageDetail.textContent = `Saved ${saved.work_root}. Initialize directories to check cluster access.`;
    // Other settings and new experiment previews must use the saved root too.
    if (typeof refreshResolvedSettings === "function")
      await refreshResolvedSettings();
    if (typeof loadedTabs !== "undefined") loadedTabs.delete("experiments");
  });
});

async function initializeWorkspaceStorage() {
  if (!storageSaved || storageForm.dataset.dirty) return;
  const root = storageSaved.work_root;
  return storageAction(async () => {
    storageDetail.textContent = `Checking and initializing ${root}…`;
    const gateway = encodeURIComponent(
      document.querySelector("#gateway").value,
    );
    const result = await api(`/api/workspace/init?gateway=${gateway}`, {
      method: "POST",
    });
    storageStatus.textContent = "Ready";
    storageDetail.textContent = `Directories ready at ${result.work_root} through ${result.gateway}.`;
  });
}
