/* The recording review dialog is the shared surface for application dialogs. */
window.SkynetDialog = (() => {
  const launchers = new WeakMap();
  const initialized = new WeakSet();
  function open(dialog, { launcher = document.activeElement } = {}) {
    if (dialog.open) return;
    if (!initialized.has(dialog)) {
      dialog.addEventListener("close", () => {
        if (dialog.open) return;
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
  }
  function close(dialog, value = "") {
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
  return { open, close, openGuide, closeGuide, placeGuide };
})();

function askUserDialog(message, defaultValue = null) {
  return new Promise((resolve) => {
    const dialog = document.createElement("dialog");
    dialog.className = "app-dialog app-dialog-compact";
    const heading = document.createElement("div");
    heading.className = "panel-heading";
    const title = document.createElement("h3");
    title.textContent =
      defaultValue === null ? "Confirm action" : "Enter details";
    dialog.setAttribute("aria-label", title.textContent);
    heading.append(title);
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
    dialog.append(heading, form);
    document.body.append(dialog);
    SkynetDialog.open(dialog);
    dialog.addEventListener(
      "close",
      () => {
        const accepted = dialog.returnValue === "confirm";
        const result = input ? (accepted ? input.value : null) : accepted;
        dialog.remove();
        resolve(result);
      },
      { once: true },
    );
    (input || cancel).focus();
  });
}
