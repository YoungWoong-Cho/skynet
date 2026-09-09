/* One Data navigation row; legacy collection links still open their saved view. */
window.dataNavigation = (() => {
  const aliases = { live: "collect", recordings: "recording", cycles: "setup" };
  const views = ["setup", "collect", "recording", "registry"];
  const normalize = (view) =>
    views.includes(view) ? view : aliases[view] || "collect";
  let current = "collect";
  function viewForTab(tab) {
    const params = new URLSearchParams(location.search);
    if (tab === "datasets") return "registry";
    if (tab === "collection")
      return normalize(
        params.get("collection_view") ||
          (current === "registry" ? "collect" : current),
      );
    return normalize(
      params.get("data_view") || params.get("collection_view") || current,
    );
  }
  function url(view) {
    const next = new URL(location.href);
    next.searchParams.delete("collection_view");
    next.searchParams.set("data_view", normalize(view));
    next.hash = "data";
    return next;
  }
  function render(view) {
    current = normalize(view);
    const collectionView = {
      collect: "live",
      recording: "recordings",
      setup: "setup",
    }[current];
    document.querySelectorAll("[data-collection-view]").forEach((panel) => {
      panel.hidden = panel.dataset.collectionView !== collectionView;
    });
    document.querySelectorAll("[data-data-tab]").forEach((button) => {
      const selected = button.dataset.dataTab === current;
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
    const registryTutorial = document.getElementById(
      "datasets-tutorial-button",
    );
    if (registryTutorial) registryTutorial.hidden = current !== "registry";
    document.getElementById("refresh-collection").hidden =
      current === "registry";
    document.getElementById("refresh-data-registry").hidden =
      current !== "registry";
  }
  function select(view, { persist = true, focus = false } = {}) {
    view = normalize(view);
    // activateTab owns panel loading and history, including browser Back/Forward.
    activateTab("data", persist, view);
    if (focus)
      document
        .querySelector(`[data-data-tab="${view}"]`)
        .focus({ preventScroll: true });
  }
  document.querySelectorAll("[data-data-tab]").forEach((button) => {
    button.addEventListener("click", () => select(button.dataset.dataTab));
    button.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key))
        return;
      event.preventDefault();
      const index = views.indexOf(button.dataset.dataTab);
      const next =
        event.key === "Home"
          ? 0
          : event.key === "End"
            ? views.length - 1
            : (index + (event.key === "ArrowRight" ? 1 : -1) + views.length) %
              views.length;
      select(views[next], { focus: true });
    });
  });
  document.querySelectorAll("[data-collection-go]").forEach((button) =>
    button.addEventListener("click", () => {
      select(button.dataset.collectionGo, { focus: true });
      const guide = button.dataset.collectionGuide === "record";
      document
        .querySelector(guide ? "#vision-pro-guide" : "#collection-view-setup")
        .scrollIntoView({ block: "start" });
      if (guide)
        document
          .querySelector("#vision-pro-guide-title")
          .focus({ preventScroll: true });
    }),
  );
  function mountTutorial() {
    const tutorial = document.getElementById("collection-tutorial-button");
    if (tutorial) {
      document
        .getElementById("collection-adapter-tutorial-slot")
        .append(tutorial);
      tutorial.textContent = "Adapter tutorial";
      tutorial.setAttribute(
        "aria-label",
        "Start advanced collection adapter tutorial",
      );
      tutorial.title =
        "Advanced tutorial: configure collection adapters. Headset recording instructions are above.";
    }
  }
  return { viewForTab, render, url, select, mountTutorial };
})();
function setCollectionView(view, options) {
  dataNavigation.select(view, options);
}
