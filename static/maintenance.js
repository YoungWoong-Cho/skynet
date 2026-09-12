/* Shared deletion entry point: use data-delete-kind / data-delete-id for every resource. */
(() => {
  const dialog = document.querySelector("#maintenance-dialog");
  const title = document.querySelector("#maintenance-title");
  const message = document.querySelector("#maintenance-message");
  const content = document.querySelector("#maintenance-content");
  const confirm = document.querySelector("#maintenance-confirm");
  let sequence = 0, busy = false, current = null;
  const gateway = () => document.querySelector("#gateway").value;
  const bytes = value => value == null ? "—" : `${(value / 1024 / 1024).toFixed(2)} MB`;
  const labels = {experiment: "experiment", run: "training run", evaluation: "evaluation and rollouts", adapter: "adapter", suite: "evaluation suite"};
  function setBusy(value) {
    busy = value;
    dialog.setAttribute("aria-busy", String(value));
    dialog.querySelector("[data-dialog-close]").disabled = dialog.dataset.blockClose === "true";
    confirm.disabled = value || !current || (current.blockers?.length > 0);
    content.querySelectorAll("input").forEach(input => { input.disabled = value || input.dataset.protected === "true"; });
  }
  function table(headers, rows) {
    return `<table><thead><tr>${headers.map(text => `<th>${escapeHtml(text)}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table>`;
  }
  async function open(kind, id) {
    if (busy) return;
    const request = ++sequence;
    current = null;
    title.textContent = kind === "storage" ? "Inspect cluster files" : `Delete ${labels[kind]}`;
    message.textContent = "Checking dependencies and files…";
    message.classList.remove("is-error");
    content.replaceChildren();
    confirm.textContent = "Delete";
    if (!dialog.open) SkynetDialog.open(dialog);
    setBusy(true);
    try {
      const result = await api(kind === "storage"
        ? `/api/maintenance/storage?gateway=${encodeURIComponent(gateway())}`
        : `/api/maintenance/history/${kind}/${encodeURIComponent(id)}?gateway=${encodeURIComponent(gateway())}`);
      if (request !== sequence) return;
      current = {...result, kind, id};
      if (kind === "storage") {
        message.textContent = `Base path: ${result.root}. Select files to permanently remove.`;
        content.innerHTML = table(["Select", "Path", "Size", "Reason"], result.items.map(item =>
          `<tr><td><input type="checkbox" data-path="${escapeHtml(item.path)}" aria-label="Select ${escapeHtml(item.path)}" data-protected="${!item.selectable}" ${item.selectable ? "" : "disabled"}></td><td>${escapeHtml(item.path)}</td><td>${bytes(item.size_bytes)}</td><td>${escapeHtml(item.reason)}</td></tr>`));
        if (!result.items.length) message.textContent = "No unreferenced files found in the inspected output directories.";
        for (const item of result.pending_deletions || []) {
          const button = document.createElement("button");
          button.type = "button"; button.className = "button button-outline";
          button.dataset.deleteKind = item.kind; button.dataset.deleteId = item.id;
          button.textContent = `Finish deleting ${labels[item.kind]} ${item.id}`;
          content.append(button);
        }
        confirm.textContent = "Delete selected files";
      } else if (result.blockers.length) {
        message.textContent = "Delete these dependencies first, or finish the active work:";
        content.innerHTML = table(["Item", "Required action"], result.blockers.map(item =>
          `<tr><td>${escapeHtml(item.label)}</td><td>${escapeHtml(item.reason)} ${item.id && labels[item.kind] ? `<button type="button" class="button button-outline" data-delete-kind="${item.kind}" data-delete-id="${escapeHtml(item.id)}">Review deletion</button>` : ""}</td></tr>`));
      } else {
        message.textContent = `${result.retry ? "Retry deletion of" : "Permanently delete"} ${result.label}? Skynet records and listed files will be removed. External tracking services keep their own history.`;
        const counts = Object.entries(result.counts).map(([name, count]) => `${count} ${({adapters: "adapter versions", evaluation_suites: "suite versions"}[name] || name.replaceAll("_", " "))}`).join(" · ");
        const paragraph = document.createElement("p"); paragraph.textContent = counts; content.append(paragraph);
        content.insertAdjacentHTML("beforeend", table(["File or directory", "Size"], result.files.map(item =>
          `<tr><td>${escapeHtml(item.path)}${item.exists ? "" : " (already absent)"}</td><td>${bytes(item.size_bytes)}</td></tr>`)));
        confirm.textContent = result.retry ? "Retry deletion" : "Delete permanently";
      }
      for (const notice of result.notices || []) {
        const paragraph = document.createElement("p");
        paragraph.className = "inline-alert is-warning dialog-notice";
        paragraph.textContent = notice;
        content.prepend(paragraph);
      }
    } catch (error) {
      message.textContent = error.message; message.classList.add("is-error");
    } finally {
      if (request === sequence) { setBusy(false); updateSelection(); }
    }
  }
  function updateSelection() {
    if (current?.kind === "storage") confirm.disabled = busy || !content.querySelector("input:checked:not(:disabled)");
  }
  content.addEventListener("change", updateSelection);
  dialog.addEventListener("cancel", event => { if (dialog.dataset.blockClose === "true") event.preventDefault(); });
  dialog.addEventListener("close", () => { sequence++; current = null; });
  document.addEventListener("click", event => {
    const launcher = event.target.closest("[data-delete-kind]");
    if (launcher) open(launcher.dataset.deleteKind, launcher.dataset.deleteId);
  });
  document.querySelector("#inspect-cluster-storage").addEventListener("click", () => open("storage"));
  confirm.addEventListener("click", async () => {
    if (busy || !current || current.blockers?.length) return;
    const plan = current;
    const paths = [...content.querySelectorAll("input:checked")].map(input => input.dataset.path);
    if (plan.kind === "storage" && !paths.length) return;
    dialog.dataset.blockClose = "true";
    setBusy(true);
    message.textContent = "Deleting…";
    message.classList.remove("is-error");
    try {
      await api(plan.kind === "storage" ? "/api/maintenance/storage/cleanup" : `/api/maintenance/history/${plan.kind}/${encodeURIComponent(plan.id)}`, {
        method: plan.kind === "storage" ? "POST" : "DELETE",
        body: JSON.stringify({token: plan.token, gateway: gateway(), ...(plan.kind === "storage" ? {paths} : {})}),
      });
      let refreshError = null;
      if (plan.kind !== "storage") {
        closeActiveDisclosure({restoreFocus: false});
        try { await refreshAfterDeletion(plan.kind); }
        catch (error) { refreshError = error; }
      }
      delete dialog.dataset.blockClose;
      SkynetDialog.close(dialog);
      showToast(refreshError ? `Deleted. Refresh the page to update the lists: ${refreshError.message}` : "Deletion completed.", Boolean(refreshError));
    } catch (error) {
      current = null;
      message.textContent = `${error.message} Close and review deletion again to retry.`;
      message.classList.add("is-error");
    } finally { delete dialog.dataset.blockClose; setBusy(false); }
  });
})();
