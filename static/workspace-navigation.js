/* One keyboard convention for primary and workspace tabs. */
function installTabKeyboardNavigation(buttons, select) {
  buttons.forEach((button, index) => button.addEventListener("keydown", (event) => {
    if (event.key === " ") {
      event.preventDefault();
      select(button);
      return;
    }
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1
      : (index + (event.key === "ArrowRight" ? 1 : -1) + buttons.length) % buttons.length;
    select(buttons[next]);
    buttons[next].focus({ preventScroll: true });
    buttons[next].scrollIntoView({ block: "nearest", inline: "nearest" });
  }));
}
installTabKeyboardNavigation(
  [...document.querySelectorAll('.tab-nav [role="tab"]')],
  (button) => activateTab(button.dataset.tabTarget),
);

/* Shared workspace tabs, history and keyboard navigation for Data and Experiments. */
function createWorkspaceNavigation({
  page,
  parameter,
  selector,
  views,
  initial,
  aliases = {},
  legacyParameters = [],
  legacyView = () => null,
  renderView,
}) {
  const buttons = [...document.querySelectorAll(selector)];
  const normalize = (view) =>
    views.includes(view) ? view : aliases[view] || initial;
  const buttonView = (button) => button.getAttribute(selector.slice(1, -1));
  let current = initial;
  function viewForTab(tab) {
    const params = new URLSearchParams(location.search);
    return normalize(
      legacyView(tab, current, params) || params.get(parameter) || current,
    );
  }
  function url(view) {
    const next = new URL(location.href);
    legacyParameters.forEach((key) => next.searchParams.delete(key));
    next.searchParams.set(parameter, normalize(view));
    next.hash = page;
    return next;
  }
  function render(view) {
    current = normalize(view);
    buttons.forEach((button) => {
      const selected = buttonView(button) === current;
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
    renderView(current);
  }
  function select(view, { persist = true, focus = false } = {}) {
    view = normalize(view);
    // activateTab owns panel loading and browser Back/Forward history.
    activateTab(page, persist, view);
    if (focus)
      buttons
        .find((button) => buttonView(button) === view)
        ?.focus({ preventScroll: true });
  }
  buttons.forEach((button) => {
    button.addEventListener("click", () => select(buttonView(button)));
  });
  installTabKeyboardNavigation(buttons, (button) => select(buttonView(button)));

  return { viewForTab, url, render, select };
}

window.dataNavigation = createWorkspaceNavigation({
  page: "data",
  parameter: "data_view",
  selector: "[data-data-tab]",
  views: ["setup", "collect", "recording", "registry"],
  initial: "collect",
  aliases: { live: "collect", recordings: "recording", cycles: "setup" },
  legacyParameters: ["collection_view"],
  legacyView(tab, current, params) {
    if (tab === "datasets") return "registry";
    if (tab === "collection")
      return (
        params.get("collection_view") ||
        (current === "registry" ? "collect" : current)
      );
    return params.get("data_view") || params.get("collection_view");
  },
  renderView(view) {
    const collectionView = {
      collect: "live",
      recording: "recordings",
      setup: "setup",
    }[view];
    document.querySelectorAll("[data-collection-view]").forEach((panel) => {
      panel.hidden = panel.dataset.collectionView !== collectionView;
    });
    const tutorial = document.getElementById("datasets-tutorial-button");
    if (tutorial) tutorial.hidden = view !== "registry";
    document.getElementById("refresh-collection").hidden = view === "registry";
    document.getElementById("refresh-data-registry").hidden =
      view !== "registry";
  },
});

window.experimentNavigation = createWorkspaceNavigation({
  page: "experiments",
  parameter: "experiment_view",
  selector: "[data-experiment-tab]",
  views: ["submit", "adapters"],
  initial: "submit",
  legacyView: (tab) => (tab === "adapters" ? "adapters" : null),
  renderView(view) {
    for (const [page, selected] of [
      ["experiments", "submit"],
      ["adapters", "adapters"],
    ]) {
      document.getElementById(`refresh-${page}`).hidden = view !== selected;
      const tutorial = document.getElementById(`${page}-tutorial-button`);
      if (tutorial) tutorial.hidden = view !== selected;
    }
  },
});

function setCollectionView(view, options) {
  dataNavigation.select(view, options);
}

document.querySelectorAll("[data-collection-go]").forEach((button) =>
  button.addEventListener("click", () => {
    dataNavigation.select(button.dataset.collectionGo, { focus: true });
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

dataNavigation.mountTutorial = () => {
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
};
