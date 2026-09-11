/* A personal path is registered only after cluster validation and initialization. */
const storageForm = document.querySelector("#workspace-storage-form");
const storagePath = document.querySelector("#workspace-base-path");
const storageStatus = document.querySelector("#workspace-storage-status");
const storageDetail = document.querySelector("#workspace-storage-detail");
let storageSaved = null;
let storageBusy = false;

function storageControls() {
  const dirty = storagePath.value.trim() !== (storageSaved?.work_root || "");
  storagePath.disabled = storageBusy || !storageSaved;
  document.querySelector("#save-workspace-storage").disabled =
    storageBusy || !storageSaved || !storagePath.value.trim() || !dirty;
}

function renderWorkspaceStorage(settings) {
  if (!settings || !("work_root" in settings)) return;
  storageSaved = settings;
  window.SkynetStorageConfigured = settings.configured;
  if (!storageForm.dataset.dirty) storagePath.value = settings.work_root || "";
  storageStatus.textContent = storageForm.dataset.dirty ? "Unsaved changes"
    : settings.configured ? "Ready" : "Setup required";
  if (!storageBusy) {
    storageDetail.classList.remove("is-error");
    storageDetail.textContent = settings.configured
      ? `Base path: ${settings.work_root}`
      : "Set a base path to continue. It must be new or empty.";
  }
  storageControls();
}

storagePath.addEventListener("input", () => {
  if (storagePath.value.trim() === (storageSaved?.work_root || ""))
    delete storageForm.dataset.dirty;
  else storageForm.dataset.dirty = "true";
  storageStatus.textContent = storageForm.dataset.dirty ? "Unsaved changes"
    : storageSaved?.configured ? "Ready" : "Setup required";
  storageDetail.classList.remove("is-error");
  storageDetail.textContent = "The path is saved after validation and initialization succeed.";
  storageControls();
});

storageForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (storageBusy || !storageSaved || !storageForm.reportValidity()) return;
  storageBusy = true;
  storageForm.setAttribute("aria-busy", "true");
  storageControls();
  storageDetail.classList.remove("is-error");
  storageStatus.textContent = "Initializing";
  storageDetail.textContent = "Checking path, permissions and empty directory…";
  try {
    const gateway = encodeURIComponent(document.querySelector("#gateway").value);
    const saved = await api(`/api/workspace/storage?gateway=${gateway}`, {
      method: "PUT",
      body: JSON.stringify({
        work_root: storagePath.value.trim(),
        expected_work_root: storageSaved.work_root,
      }),
    });
    delete storageForm.dataset.dirty;
    renderWorkspaceStorage(saved);
    storageDetail.textContent = `Directories ready at ${saved.work_root} through ${saved.gateway}.`;
    if (typeof loadedTabs !== "undefined") loadedTabs.delete("experiments");
    if (typeof refreshResolvedSettings === "function") await refreshResolvedSettings();
  } catch (error) {
    storageDetail.textContent = error.message;
    storageDetail.classList.add("is-error");
    storageStatus.textContent = "Needs attention";
  } finally {
    storageBusy = false;
    storageForm.removeAttribute("aria-busy");
    storageControls();
  }
});
