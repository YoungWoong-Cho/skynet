/* Slack validates the webhook before enabling all job notifications. */
const slackForm = document.querySelector("#slack-notifications-form");
const slackWebhook = document.querySelector("#slack-webhook");
const slackStatus = document.querySelector("#slack-notification-status");
const slackDetail = document.querySelector("#slack-notification-detail");
let slackSaved = null;
let slackBusy = false;

function renderSlackSettings(settings) {
  slackSaved = settings;
  const connected = Boolean(settings.configured && settings.enabled);
  slackStatus.textContent = connected ? "Connected" : "Not connected";
  slackStatus.className = `state-pill ${connected ? "is-running" : "is-other"}`;
  slackDetail.textContent = settings.last_error || "";
  slackDetail.hidden = !settings.last_error;
  slackDetail.classList.toggle("is-error", Boolean(settings.last_error));
  renderConnectionControls(slackForm, { connected, busy: slackBusy });
}

async function loadSlackSettings() {
  try {
    renderSlackSettings(await api("/api/notifications/slack"));
  } catch (error) {
    slackStatus.textContent = "Unavailable";
    slackDetail.textContent = error.message;
    slackDetail.hidden = false;
    slackDetail.classList.add("is-error");
    renderConnectionControls(slackForm, { connected: false, loaded: false });
    throw error;
  }
}

async function slackAction(action) {
  if (slackBusy || !slackSaved) return;
  const connected = Boolean(slackSaved.configured && slackSaved.enabled);
  if (action === "connect" && connected) return;
  const payload =
    action === "connect" ? { webhook_url: slackWebhook.value.trim() } : null;
  if (action === "connect" && !slackForm.reportValidity()) return;
  slackBusy = true;
  renderConnectionControls(slackForm, { connected, busy: true });
  slackDetail.hidden = false;
  slackDetail.classList.remove("is-error");
  slackDetail.textContent =
    action === "connect" ? "Connecting…" : "Disconnecting…";
  try {
    const settings = await api(
      action === "connect"
        ? "/api/notifications/slack/connect"
        : "/api/notifications/slack",
      {
        method: action === "connect" ? "POST" : "DELETE",
        ...(payload ? { body: JSON.stringify(payload) } : {}),
      },
    );
    slackWebhook.value = "";
    renderSlackSettings(settings);
  } catch (error) {
    slackDetail.textContent = error.message;
    slackDetail.hidden = false;
    slackDetail.classList.add("is-error");
  } finally {
    slackBusy = false;
    renderConnectionControls(slackForm, {
      connected: Boolean(slackSaved?.configured && slackSaved?.enabled),
    });
  }
}

slackForm.addEventListener("submit", (event) => {
  event.preventDefault();
  void slackAction("connect");
});
document.querySelector("#slack-disconnect").addEventListener("click", () => {
  void slackAction("disconnect");
});
