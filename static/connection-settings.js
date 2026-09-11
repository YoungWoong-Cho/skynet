/* Shared connection controls for Settings integrations. */
function renderConnectionControls(
  form,
  { connected, busy = false, loaded = true },
) {
  form.dataset.connected = String(connected);
  form.dataset.busy = String(busy);
  form.setAttribute("aria-busy", String(busy));
  form.querySelectorAll("input").forEach((input) => {
    input.disabled = busy || connected || !loaded;
  });
  const connect = form.querySelector('button[type="submit"]');
  const disconnect = form.querySelector("[data-connection-disconnect]");
  connect.hidden = connected;
  disconnect.hidden = !connected;
  connect.disabled = disconnect.disabled = busy || !loaded;
}
