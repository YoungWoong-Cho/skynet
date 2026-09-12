/* Shared shell and lifecycle for all application dialogs.
 * Static surfaces declare data-dialog-panel and title/close bindings in HTML.
 * Runtime confirmations use create(); neither path supplies its own modal chrome.
 */
window.SkynetDialog = (() => {
  // Every surface uses one header, close control and scrolling body. Content
  // panels stay in place so form ownership and feature-specific listeners survive.
  function mount(surface) {
    const panel = surface.querySelector(":scope > [data-dialog-panel]");
    if (!panel || panel.classList.contains("dialog-shell")) return;
    const layout = document
      .getElementById("app-dialog-layout")
      .content.cloneNode(true);
    const title = layout.querySelector("[data-dialog-title]");
    title.id = surface.dataset.dialogTitleId || `${surface.id}-title`;
    title.textContent = surface.dataset.dialogTitle || "Details";
    surface.setAttribute("aria-labelledby", title.id);
    const close = layout.querySelector("[data-dialog-close]");
    if (surface.dataset.dialogCloseId) close.id = surface.dataset.dialogCloseId;
    const actions = layout.querySelector("[data-dialog-actions-slot]");
    if (surface.dataset.dialogActionsId)
      actions.id = surface.dataset.dialogActionsId;
    for (const slot of ["heading", "actions"]) {
      const source = panel.querySelector(
        `:scope > template[data-dialog-${slot}]`,
      );
      if (source) {
        layout
          .querySelector(`[data-dialog-${slot}-slot]`)
          .append(source.content);
        source.remove();
      }
    }
    layout.querySelector("[data-dialog-body]").append(...panel.childNodes);
    panel.classList.add("dialog-shell");
    panel.append(layout);
  }
  for (const surface of document.querySelectorAll(".app-dialog"))
    mount(surface);

  let sequence = 0;
  function create({ title, content, compact = false }) {
    const dialog = document.createElement("dialog");
    dialog.id = `app-dialog-${++sequence}`;
    dialog.className = `app-dialog${compact ? " app-dialog-compact" : ""}`;
    dialog.dataset.dialogTitle = title;
    const panel = document.createElement("section");
    panel.setAttribute("data-dialog-panel", "");
    panel.append(content);
    dialog.append(panel);
    mount(dialog);
    return dialog;
  }
  const launchers = new WeakMap();
  const initialized = new WeakSet();
  const openDialogs = [];
  let fullscreenExitAt = -Infinity;
  let wasFullscreen = false;
  function isFullscreen() {
    return Boolean(
      document.fullscreenElement ||
        document.webkitFullscreenElement ||
        [...document.querySelectorAll("video")].some(
          (video) => video.webkitDisplayingFullscreen,
        ),
    );
  }
  function fullscreenOwnsEscape() {
    return isFullscreen() || performance.now() - fullscreenExitAt < 300;
  }
  function fullscreenChanged() {
    const active = isFullscreen();
    if (wasFullscreen && !active) fullscreenExitAt = performance.now();
    wasFullscreen = active;
  }
  function requestFullscreenExit() {
    let owner = document;
    let exit;
    if (
      document.fullscreenElement &&
      typeof document.exitFullscreen === "function"
    )
      exit = document.exitFullscreen;
    else if (
      document.webkitFullscreenElement &&
      typeof document.webkitExitFullscreen === "function"
    )
      exit = document.webkitExitFullscreen;
    else {
      owner = [...document.querySelectorAll("video")].find(
        (video) => video.webkitDisplayingFullscreen,
      );
      exit = owner?.webkitExitFullscreen;
    }
    if (typeof exit !== "function") return;
    wasFullscreen = true;
    try {
      const result = exit.call(owner);
      fullscreenChanged();
      Promise.resolve(result).then(fullscreenChanged, () => {});
    } catch {
      // Native Escape remains available if the browser refuses an API exit.
    }
  }
  document.addEventListener("fullscreenchange", fullscreenChanged, true);
  document.addEventListener("webkitfullscreenchange", fullscreenChanged, true);
  document.addEventListener(
    "webkitbeginfullscreen",
    () => {
      wasFullscreen = true;
    },
    true,
  );
  document.addEventListener(
    "webkitendfullscreen",
    () => {
      wasFullscreen = false;
      fullscreenExitAt = performance.now();
    },
    true,
  );
  document.addEventListener(
    "keydown",
    (event) => {
      if (event.key !== "Escape") return;
      if (fullscreenOwnsEscape()) {
        // Request exit when the key reaches the page, while preserving native fallback.
        if (isFullscreen()) requestFullscreenExit();
        else event.preventDefault();
        event.stopImmediatePropagation();
        return;
      }
      const dialog = openDialogs.findLast((candidate) => candidate.open);
      if (
        document.body.classList.contains("has-active-tutorial") &&
        !dialog?.hasAttribute("data-app-confirmation")
      )
        return;
      if (!dialog) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      close(dialog);
    },
    true,
  );
  function open(dialog, { launcher = document.activeElement } = {}) {
    if (dialog.open) return;
    mount(dialog);
    const panel = dialog.querySelector(":scope > [data-dialog-panel]");
    if (panel) {
      panel.hidden = false;
      panel.querySelector("[data-dialog-body]").scrollTop = 0;
    }
    if (!initialized.has(dialog)) {
      dialog.addEventListener("cancel", (event) => {
        event.stopPropagation();
        if (fullscreenOwnsEscape() || dialog.dataset.blockClose === "true")
          event.preventDefault();
      });
      dialog.addEventListener("close", () => {
        if (dialog.open) return;
        const index = openDialogs.indexOf(dialog);
        if (index >= 0) openDialogs.splice(index, 1);
        const guide = dialog.querySelector("#tutorial-layer");
        if (guide) document.body.append(guide);
        const launcher = launchers.get(dialog);
        launchers.delete(dialog);
        if (
          launcher?.isConnected &&
          !launcher.closest("[hidden], dialog:not([open])")
        )
          launcher.focus({ preventScroll: true });
      });
      initialized.add(dialog);
    }
    launchers.set(dialog, launcher);
    dialog.querySelector(".dialog-notice")?.remove();
    const guide = document.getElementById("tutorial-layer");
    if (guide && !guide.hidden && dialog.hasAttribute("data-panel-dialog"))
      dialog.append(guide);
    dialog.returnValue = "";
    dialog.showModal();
    openDialogs.push(dialog);
  }
  function close(dialog, value = "") {
    if (!dialog || dialog.dataset.blockClose === "true") return;
    const guide = dialog.querySelector("#tutorial-layer");
    if (guide) document.body.append(guide);
    if (dialog.open) dialog.close(value);
  }
  // Guided tutorials leave the highlighted control interactive and manage their
  // own focus range. They share the dialog surface without making that control inert.
  function openGuide(layer) {
    layer.hidden = false;
    layer.setAttribute("aria-hidden", "false");
  }
  function placeGuide(layer, target) {
    const modal = target.closest("dialog[data-panel-dialog]");
    const previous = layer.closest("dialog[data-panel-dialog]");
    if (previous && previous !== modal) {
      document.body.append(layer);
      close(previous);
    }
    if (modal) modal.append(layer);
  }
  function closeGuide(layer) {
    document.body.append(layer);
    layer.hidden = true;
    layer.setAttribute("aria-hidden", "true");
  }
  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-dialog-close]");
    if (button) close(button.closest("dialog"));
  });
  return {
    open,
    close,
    create,
    openGuide,
    closeGuide,
    placeGuide,
    fullscreenOwnsEscape,
  };
})();

function askUserDialog(message, defaultValue = null) {
  return new Promise((resolve) => {
    const form = document.createElement("form");
    form.method = "dialog";
    const label = document.createElement("label");
    label.className = "dialog-message";
    label.textContent = message;
    form.append(label);
    let input = null;
    if (defaultValue !== null) {
      input = document.createElement("input");
      input.value = defaultValue;
      input.setAttribute("aria-label", message);
      label.append(input);
    }
    const actions = document.createElement("div");
    actions.className = "form-actions";
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "button button-outline";
    cancel.textContent = "Cancel";
    cancel.addEventListener("click", () =>
      SkynetDialog.close(dialog, "cancel"),
    );
    const confirm = document.createElement("button");
    confirm.type = "submit";
    confirm.className = "button button-accent";
    confirm.textContent = "Continue";
    actions.append(cancel, confirm);
    form.append(actions);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      SkynetDialog.close(dialog, "confirm");
    });
    const dialog = SkynetDialog.create({
      title: defaultValue === null ? "Confirm action" : "Enter details",
      content: form,
      compact: true,
    });
    dialog.setAttribute("data-app-confirmation", "");
    document.body.append(dialog);
    SkynetDialog.open(dialog);
    document.dispatchEvent(
      new CustomEvent("skynet:confirmation-open", { detail: { dialog } }),
    );
    dialog.addEventListener(
      "close",
      () => {
        const confirmation = new CustomEvent("skynet:confirmation-close", {
          detail: { dialog, accepted: dialog.returnValue === "confirm" },
          cancelable: true,
        });
        document.dispatchEvent(confirmation);
        const accepted =
          dialog.returnValue === "confirm" && !confirmation.defaultPrevented;
        const result = input ? (accepted ? input.value : null) : accepted;
        dialog.remove();
        resolve(result);
      },
      { once: true },
    );
    (input || cancel).focus();
  });
}
