import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { JSDOM } from "jsdom";

const dom = new JSDOM(
  `
  <section id="collection"><div id="collection-view-live"></div></section>
  <form id="live-xr-start-form"><button id="live-xr-start"></button></form>
  <input type="checkbox" id="live-xr-consent">
  <div id="live-xr-consent-field"></div><div id="live-xr-consent-status"></div>
  <div id="live-xr-message"></div><div id="live-xr-error"></div>
  <div id="live-xr-target"></div>
  <table><tbody id="live-xr-sessions"></tbody></table>
  <button id="live-xr-refresh"></button>
`,
  { runScripts: "outside-only", pretendToBeVisual: true },
);
const { window } = dom;
const requests = [];
window.fetch = (path, options) =>
  new Promise((resolve) => {
    requests.push({
      path,
      options,
      resolve: (data) => resolve({ ok: true, json: async () => data }),
    });
  });
const flush = () => new Promise((resolve) => setImmediate(resolve));
const get = (id) => window.document.getElementById(id);
const job = {
  id: "test-session",
  state: "PENDING",
  job_id: "123",
  server_ready: false,
};
try {
  window.eval(
    await readFile(new URL("../static/live-xr.js", import.meta.url), "utf8"),
  );
  assert.equal(requests.length, 1, "initial refresh is pending");
  get("live-xr-start-form").dispatchEvent(
    new window.Event("submit", { cancelable: true }),
  );
  assert.equal(get("live-xr-start").textContent, "Starting…");
  requests[1].resolve(job);
  await flush();
  assert.match(get("live-xr-sessions").textContent, /test-ses/);
  requests[0].resolve({ sessions: [], license: { accepted: true } });
  await flush();
  assert.match(
    get("live-xr-sessions").textContent,
    /test-ses/,
    "stale refresh must not erase the accepted session",
  );
  assert.equal(get("live-xr-start").disabled, true);

  get("live-xr-refresh").click();
  const stop = [...get("live-xr-sessions").querySelectorAll("button")].find(
    (b) => b.textContent === "Stop session",
  );
  stop.click();
  assert.equal(stop.textContent, "Stopping…");
  requests[3].resolve({ ...job, stop_requested: true, server_ready: false });
  await flush();
  requests[2].resolve({ sessions: [job], license: { accepted: true } });
  await flush();
  const stopping = [...get("live-xr-sessions").querySelectorAll("button")].find(
    (b) => b.textContent === "Stopping…",
  );
  assert.ok(
    stopping?.disabled,
    "stale refresh must not re-enable a completed stop request",
  );
  get("live-xr-refresh").click();
  requests[4].resolve({
    sessions: [
      {
        ...job,
        state: "STOPPED",
        scheduler_final: true,
        profile: { execution: "workstation" },
        gateway: "test-workstation",
      },
    ],
    license: { accepted: true },
    target: {
      execution: "workstation",
      host: "test-workstation",
      duration_minutes: 30,
    },
  });
  await flush();
  assert.match(
    get("live-xr-target").textContent,
    /directly on test-workstation/,
  );
  assert.match(
    get("live-xr-sessions").textContent,
    /Workstation · test-workstation/,
  );
  assert.doesNotMatch(get("live-xr-sessions").textContent, /GPU job/);
  console.log(
    "Live UI regression passed: pending refreshes preserve start and stop responses.",
  );
} finally {
  window.close();
}
