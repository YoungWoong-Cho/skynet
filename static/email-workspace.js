/* Email identifies a trusted-team workspace; it is deliberately not verified. */
(() => {
  const gate = document.getElementById("email-workspace-gate");
  const content = document.getElementById("email-workspace-content");
  const form = document.getElementById("email-workspace-form");
  const email = document.getElementById("workspace-email");
  const message = document.getElementById("workspace-message");
  const button = form.querySelector("button[type=submit]");
  const retry = document.createElement("button");
  retry.id = "workspace-retry";
  retry.type = "button";
  retry.className = "button";
  retry.textContent = "Retry connection";
  retry.hidden = true;
  message.after(retry);
  const originalFetch = window.fetch.bind(window);
  let workspace = null;
  let leaving = false;
  let applicationStarted = false;
  const requests = new Set();
  const sessionChannel = typeof BroadcastChannel === "undefined"
    ? null : new BroadcastChannel("skynet-workspace-session");
  if (sessionChannel) {
    sessionChannel.onmessage = event => {
      if (event.data?.type === "session-changed") leave("Workspace session changed in another tab.");
    };
    window.addEventListener("pagehide", () => sessionChannel.close(), { once: true });
  }
  window.addEventListener("pageshow", event => {
    if (!event.persisted || leaving) return;
    // A restored page has stopped streams and may hold an expired workspace.
    // Reload once to revalidate the cookie and create fresh subscriptions.
    leaving = true;
    window.stopSkynetLiveRefresh?.();
    content.hidden = true;
    gate.hidden = false;
    message.textContent = "Reconnecting…";
    for (const controller of requests) controller.abort();
    location.reload();
  });

  function leave(messageText = "") {
    if (leaving) return;
    leaving = true;
    window.stopSkynetLiveRefresh?.();
    content.hidden = true;
    gate.hidden = false;
    message.textContent = "Switching workspace…";
    for (const controller of requests) controller.abort();
    // A full navigation discards forms, pending callbacks, dialogs and caches.
    const url = new URL(location.href);
    url.search = "";
    url.hash = "";
    if (messageText) url.searchParams.set("workspace_notice", messageText);
    location.replace(url.href);
  }

  async function sessionRequest(method = "GET", payload) {
    let response;
    try {
      response = await originalFetch("/api/workspace/session", {
        method,
        credentials: "same-origin",
        cache: "no-store",
        headers: {
          "Content-Type": "application/json",
          ...(workspace ? { "X-Skynet-Workspace": workspace.id } : {}),
        },
        ...(payload ? { body: JSON.stringify(payload) } : {}),
      });
    } catch (error) {
      if (error.name === "AbortError") throw error;
      throw new Error(
        "Cannot connect to Skynet. Check your network or VPN connection and try again.",
      );
    }
    const result = await response.json().catch(() => ({}));
    if (!response.ok)
      throw new Error(
        typeof result.detail === "string"
          ? result.detail
          : response.status >= 500
            ? "The server is temporarily unavailable. Try again shortly."
            : "Unable to open workspace. Try again.",
      );
    return result.workspace;
  }

  function installWorkspaceFetch() {
    window.fetch = async (input, options = {}) => {
      const url = new URL(
        input instanceof Request ? input.url : String(input),
        location.href,
      );
      if (url.origin !== location.origin || !url.pathname.startsWith("/api/")) {
        return originalFetch(input, options);
      }
      if (leaving) throw new DOMException("Workspace changed", "AbortError");
      const controller = new AbortController();
      const signal =
        options.signal || (typeof input !== "string" ? input.signal : null);
      // Keep forwarding deadlines after headers arrive, while the response body loads.
      const requestSignal = signal
        ? AbortSignal.any([controller.signal, signal])
        : controller.signal;
      const headers = new Headers(
        options.headers ||
          (typeof input !== "string" ? input.headers : undefined),
      );
      headers.set("X-Skynet-Workspace", workspace.id);
      requests.add(controller);
      try {
        const response = await originalFetch(input, {
          ...options,
          headers,
          signal: requestSignal,
        });
        if (response.status === 401 || response.status === 409) {
          const result = await response
            .clone()
            .json()
            .catch(() => ({}));
          if (
            ["workspace_required", "workspace_changed"].includes(result.code)
          ) {
            leave(
              result.code === "workspace_changed"
                ? "Workspace changed in another tab."
                : "Enter your email to continue.",
            );
            throw new DOMException("Workspace changed", "AbortError");
          }
        }
        return response;
      } finally {
        requests.delete(controller);
      }
    };
  }

  async function loadApplication() {
    applicationStarted = true;
    document.getElementById("workspace-current-email").textContent =
      workspace.email;
    document.getElementById("workspace-current-email").title = workspace.email;
    window.SkynetWorkspace = Object.freeze({
      id: workspace.id,
      email: workspace.email,
      storageKey: (key) => `${key}:workspace:${workspace.id}`,
    });
    if (workspace.id === "legacy") {
      // Only the existing owner inherits preferences from the single-user app.
      try {
        const keys = Object.keys(localStorage).filter(
          (key) =>
            !key.includes(":workspace:") &&
            (key === "skynet:ssh-gateway" ||
              key.startsWith("skynet.tutorial.")),
        );
        for (const key of keys) {
          const scopedKey = window.SkynetWorkspace.storageKey(key);
          if (localStorage.getItem(scopedKey) === null) {
            localStorage.setItem(scopedKey, localStorage.getItem(key));
          }
          localStorage.removeItem(key);
        }
      } catch {
        /* Browser preference storage is optional. */
      }
    }
    window.SkynetStorageConfigured = workspace.storage_configured;
    installWorkspaceFetch();
    for (const placeholder of document.querySelectorAll(
      "script[data-workspace-src]",
    )) {
      await new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = placeholder.dataset.workspaceSrc;
        script.async = false;
        script.onload = resolve;
        script.onerror = () =>
          reject(
            new Error("The app could not load. Reload this page to retry."),
          );
        document.head.append(script);
      });
    }
    gate.hidden = true;
    content.hidden = false;
    // Measure only the sticky offset. The header uses a separate responsive
    // minimum so a temporary wrap cannot lock it at its largest observed height.
    if (typeof ResizeObserver !== "undefined") {
      new ResizeObserver((entries) => {
        const height = entries[0].target.getBoundingClientRect().height;
        if (height)
          document.documentElement.style.setProperty(
            "--topbar-height",
            `${height}px`,
          );
      }).observe(content.querySelector(".topbar"));
    }
    if (window.SkynetStorageConfigured === false) {
      document
        .getElementById("workspace-storage-form")
        .scrollIntoView({ block: "center" });
      document
        .getElementById("workspace-base-path")
        .focus({ preventScroll: true });
    }
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (button.disabled || !form.reportValidity()) return;
    button.disabled = true;
    message.textContent = "Opening workspace…";
    retry.hidden = true;
    retry.disabled = true;
    try {
      await sessionRequest("POST", { email: email.value.trim() });
      sessionChannel?.postMessage({ type: "session-changed" });
      leave();
    } catch (error) {
      message.textContent = error.message;
      button.disabled = false;
      email.focus();
    } finally {
      retry.disabled = false;
    }
  });

  document
    .getElementById("workspace-sign-out")
    .addEventListener("click", async (event) => {
      const signOutButton = event.currentTarget;
      signOutButton.disabled = true;
      try {
        await sessionRequest("DELETE");
        sessionChannel?.postMessage({ type: "session-changed" });
        leave();
      } catch (error) {
        signOutButton.disabled = false;
        const notice = document.getElementById("workspace-sign-out-error");
        notice.textContent = error.message;
      }
    });

  async function start() {
    message.textContent = "Loading workspace…";
    button.disabled = true;
    retry.disabled = true;
    retry.hidden = true;
    try {
      workspace = await sessionRequest();
      if (workspace) return await loadApplication();
      message.textContent = new URL(location.href).searchParams.has(
        "workspace_notice",
      )
        ? "Your workspace session changed. Enter your email to continue."
        : "";
      button.disabled = false;
      email.focus();
    } catch (error) {
      message.textContent = error.message;
      button.disabled = false;
      retry.hidden = false;
    } finally {
      retry.disabled = false;
    }
  }
  retry.addEventListener("click", () => {
    if (retry.disabled || button.disabled) return;
    // A partially loaded app must restart without installing scripts twice.
    if (applicationStarted) location.reload();
    else start();
  });
  start();
})();
