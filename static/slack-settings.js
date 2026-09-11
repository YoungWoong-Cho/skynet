/* Personal Slack settings. Secrets are never persisted or prefilled in the browser. */
const slackForm = document.querySelector("#slack-notifications-form");
const slackField = (name) => slackForm.elements.namedItem(name);
const slackStatus = document.querySelector("#slack-notification-status");
const slackDetail = document.querySelector("#slack-notification-detail");
let slackSaved = null;
let slackBusy = false;

function renderSlackSettings(settings) {
  slackSaved = settings;
  if (!slackForm.dataset.dirty) {
    slackField("enabled").checked = settings.enabled;
    slackField("app_url").value = settings.app_url || "";
    slackForm.querySelectorAll('[name="events"]').forEach((input) => {
      input.checked = settings.events.includes(input.value);
    });
  }
  slackStatus.textContent =
    settings.last_error || settings.failed_count
      ? "Needs attention"
      : settings.enabled
        ? "Enabled"
        : "Disabled";
  slackStatus.className = `state-pill ${settings.last_error ? "is-failed" : settings.enabled ? "is-running" : "is-other"}`;
  slackDetail.textContent =
    settings.last_error ||
    [
      settings.configured
        ? "Webhook saved in your workspace’s secure store on the Skynet server."
        : "Add a Slack incoming webhook to connect.",
      `${settings.pending_count} pending · ${settings.failed_count} failed notifications.`,
      settings.last_sent_at
        ? `Last delivered: ${new Date(settings.last_sent_at).toLocaleString()}.`
        : "",
    ]
      .filter(Boolean)
      .join(" ");
  document.querySelector("#slack-send-test").disabled =
    slackBusy || !settings.configured;
  document.querySelector("#slack-disconnect").disabled =
    slackBusy || !settings.configured;
  document.querySelector("#slack-retry-failed").disabled =
    slackBusy ||
    !settings.configured ||
    !settings.enabled ||
    !settings.failed_count;
}

async function loadSlackSettings() {
  try {
    renderSlackSettings(await api("/api/notifications/slack"));
  } catch (error) {
    slackStatus.textContent = "Unavailable";
    slackDetail.textContent = error.message;
    throw error;
  }
}

async function slackAction(action) {
  if (slackBusy) return;
  slackBusy = true;
  slackForm.setAttribute("aria-busy", "true");
  slackDetail.textContent = "Working…";
  slackForm.querySelectorAll("input,button").forEach((input) => {
    input.disabled = true;
  });
  try {
    await action();
  } catch (error) {
    slackDetail.textContent = error.message;
    slackStatus.textContent = "Needs attention";
  } finally {
    slackBusy = false;
    slackForm.removeAttribute("aria-busy");
    slackForm.querySelectorAll("input,button").forEach((input) => {
      input.disabled = false;
    });
    document.querySelector("#slack-send-test").disabled =
      !slackSaved?.configured;
    document.querySelector("#slack-disconnect").disabled =
      !slackSaved?.configured;
    document.querySelector("#slack-retry-failed").disabled =
      !slackSaved?.configured ||
      !slackSaved?.enabled ||
      !slackSaved?.failed_count;
  }
}

slackForm.addEventListener("input", () => {
  slackForm.dataset.dirty = "true";
  slackDetail.textContent =
    "Unsaved changes. Save settings applies them. Send test message uses the saved destination.";
});
slackForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const payload = {
    webhook_url: slackField("webhook_url").value.trim() || null,
    enabled: slackField("enabled").checked,
    events: [...slackForm.querySelectorAll('[name="events"]:checked')].map(
      (input) => input.value,
    ),
    app_url: slackField("app_url").value.trim(),
  };
  void slackAction(async () => {
    const settings = await api("/api/notifications/slack", {
      method: "PUT",
      body: JSON.stringify(payload),
    });
    slackField("webhook_url").value = "";
    delete slackForm.dataset.dirty;
    renderSlackSettings(settings);
  });
});
document.querySelector("#slack-send-test").addEventListener("click", () => {
  void slackAction(async () => {
    const result = await api("/api/notifications/slack/test", {
      method: "POST",
    });
    slackDetail.textContent = result.message;
  });
});
document.querySelector("#slack-disconnect").addEventListener("click", () => {
  void slackAction(async () => {
    const settings = await api("/api/notifications/slack", {
      method: "DELETE",
    });
    slackField("webhook_url").value = "";
    delete slackForm.dataset.dirty;
    renderSlackSettings(settings);
  });
});

document.querySelector("#slack-retry-failed").addEventListener("click", () => {
  void slackAction(async () => {
    renderSlackSettings(
      await api("/api/notifications/slack/retry", { method: "POST" }),
    );
  });
});
