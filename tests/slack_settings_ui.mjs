import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { JSDOM } from "jsdom";
const window = new JSDOM(
  await readFile(new URL("../static/index.html", import.meta.url), "utf8"),
  {
    runScripts: "outside-only",
    url: "http://skynet:8080/#settings",
  },
).window;
const el = (id) => window.document.getElementById(id);
const flush = async () => {
  for (let i = 0; i < 4; i++)
    await new Promise((resolve) => setImmediate(resolve));
};
const calls = [];
const disconnected = { configured: false, enabled: false, last_error: null };
const connected = { configured: true, enabled: true, last_error: null };
let handler = async () => disconnected;
window.api = async (path, options = {}) => {
  calls.push({ path, ...options });
  return handler(path, options);
};
for (const file of ["connection-settings.js", "slack-settings.js"])
  window.eval(
    await readFile(new URL("../static/" + file, import.meta.url), "utf8"),
  );
try {
  const form = el("slack-notifications-form");
  const connect = form.querySelector('button[type="submit"]');
  await window.loadSlackSettings();
  assert.equal(connect.hidden, false);
  assert.equal(el("slack-disconnect").hidden, true);
  assert.equal(el("slack-send-test"), null);
  assert.equal(el("slack-retry-failed"), null);
  assert.equal(form.querySelector('input[type="checkbox"]'), null);
  el("slack-webhook").value =
    "https://hooks.slack.com/services/T/B/private-webhook";
  let resolveConnect;
  handler = () =>
    new Promise((resolve) => {
      resolveConnect = resolve;
    });
  form.dispatchEvent(new window.Event("submit", { cancelable: true }));
  const count = calls.length;
  form.dispatchEvent(new window.Event("submit", { cancelable: true }));
  assert.equal(calls.length, count);
  assert.equal(calls.at(-1).path, "/api/notifications/slack/connect");
  assert.equal(el("slack-webhook").disabled, true);
  resolveConnect(connected);
  await flush();
  assert.equal(el("slack-webhook").value, "");
  assert.equal(el("slack-webhook").disabled, true);
  assert.equal(connect.hidden, true);
  assert.equal(el("slack-disconnect").hidden, false);
  assert.equal(window.localStorage.length, 0);
  handler = async () => {
    throw Error("Connection unavailable");
  };
  el("slack-disconnect").click();
  await flush();
  assert.match(
    el("slack-notification-detail").textContent,
    /Connection unavailable/,
  );
  assert.equal(el("slack-disconnect").disabled, false);
  assert.equal(connect.hidden, true);
  handler = async () => disconnected;
  el("slack-disconnect").click();
  await flush();
  assert.equal(connect.hidden, false);
  assert.equal(el("slack-disconnect").hidden, true);
  assert.equal(el("slack-webhook").disabled, false);
  el("slack-webhook").value = "https://hooks.slack.com/services/T/B/rejected";
  handler = async () => {
    throw Error("Webhook revoked");
  };
  connect.click();
  await flush();
  assert.equal(connect.hidden, false);
  assert.equal(connect.disabled, false);
  assert.match(el("slack-notification-detail").textContent, /Webhook revoked/);
  assert.equal(el("slack-notification-detail").hidden, false);
  assert.equal(el("slack-disconnect").hidden, true);
  console.log(
    "Slack: Connect/Disconnect, validation errors, single action, duplicate clicks and secret clearing passed.",
  );
} finally {
  window.close();
}
