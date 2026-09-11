import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { JSDOM } from "jsdom";
const window = new JSDOM(
  await readFile(new URL("../static/index.html", import.meta.url), "utf8"),
  { runScripts: "outside-only", url: "http://skynet:8080/#settings" },
).window;
const el = (id) => window.document.getElementById(id);
const flush = async () => {
  for (let i = 0; i < 4; i++)
    await new Promise((resolve) => setImmediate(resolve));
};
const calls = [];
let handler = async () => ({
  configured: false,
  enabled: false,
  events: ["submitted", "running", "cancelled", "failed", "completed"],
  app_url: "",
  pending_count: 0,
  failed_count: 0,
});
window.api = async (path, options = {}) => {
  calls.push({ path, ...options });
  return handler(path, options);
};
window.eval(
  await readFile(
    new URL("../static/slack-settings.js", import.meta.url),
    "utf8",
  ),
);
try {
  await window.loadSlackSettings();
  assert.equal(el("slack-send-test").disabled, true);
  const form = el("slack-notifications-form");
  form.elements.enabled.checked = true;
  el("slack-webhook").value =
    "https://hooks.slack.com/services/T/B/private-webhook";
  el("slack-webhook").dispatchEvent(
    new window.Event("input", { bubbles: true }),
  );
  let resolveSave;
  handler = () =>
    new Promise((resolve) => {
      resolveSave = resolve;
    });
  form.dispatchEvent(new window.Event("submit", { cancelable: true }));
  const count = calls.length;
  form.dispatchEvent(new window.Event("submit", { cancelable: true }));
  assert.equal(calls.length, count);
  assert.equal(el("slack-webhook").disabled, true);
  assert.equal(JSON.parse(calls.at(-1).body).enabled, true);
  resolveSave({
    configured: true,
    enabled: true,
    events: ["submitted", "running", "failed"],
    app_url: "http://skynet:8080",
    pending_count: 0,
    failed_count: 0,
  });
  await flush();
  assert.equal(el("slack-webhook").value, "");
  assert.equal(el("slack-send-test").disabled, false);
  assert.equal(window.localStorage.length, 0);
  el("slack-app-url").value = "http://unsaved:8080";
  el("slack-app-url").dispatchEvent(
    new window.Event("input", { bubbles: true }),
  );
  handler = async () => ({
    configured: true,
    enabled: true,
    events: ["submitted"],
    app_url: "http://saved:8080",
    pending_count: 1,
    failed_count: 0,
  });
  await window.loadSlackSettings();
  assert.equal(el("slack-app-url").value, "http://unsaved:8080");
  handler = async () => ({ ok: true, message: "Test message sent." });
  el("slack-send-test").click();
  await flush();
  assert.equal(calls.at(-1).path, "/api/notifications/slack/test");
  assert.equal(calls.at(-1).body, undefined);
  assert.match(
    el("slack-notification-detail").textContent,
    /Test message sent/,
  );
  handler = async () => {
    throw new Error("Network unavailable");
  };
  el("slack-send-test").click();
  await flush();
  assert.equal(el("slack-send-test").disabled, false);
  assert.match(
    el("slack-notification-detail").textContent,
    /Network unavailable/,
  );
  handler = async () => ({
    configured: false,
    enabled: false,
    events: [],
    app_url: "",
    pending_count: 0,
    failed_count: 0,
  });
  el("slack-disconnect").click();
  await flush();
  assert.equal(calls.at(-1).method, "DELETE");
  assert.equal(el("slack-send-test").disabled, true);
  assert.equal(form.dataset.dirty, undefined);
  console.log(
    "Slack settings: save, secret clearing, duplicate clicks, drafts, test, errors, disconnect passed.",
  );
} finally {
  window.close();
}
