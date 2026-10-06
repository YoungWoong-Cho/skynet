/* One keyboard convention for primary and workspace tabs. */
function installTabKeyboardNavigation(buttons, select) {
  buttons.forEach((button, index) =>
    button.addEventListener("keydown", (event) => {
      if (event.key === " ") {
        event.preventDefault();
        select(button);
        return;
      }
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key))
        return;
      event.preventDefault();
      const next =
        event.key === "Home"
          ? 0
          : event.key === "End"
            ? buttons.length - 1
            : (index + (event.key === "ArrowRight" ? 1 : -1) + buttons.length) %
              buttons.length;
      select(buttons[next]);
      buttons[next].focus({ preventScroll: true });
      buttons[next].scrollIntoView({ block: "nearest", inline: "nearest" });
    }),
  );
}
installTabKeyboardNavigation(
  [...document.querySelectorAll('.tab-nav [role="tab"]')],
  (button) => activateTab(button.dataset.tabTarget),
);

/* Shared workspace tabs, history and keyboard navigation. */
function createWorkspaceNavigation({
  page,
  parameter,
  selector,
  views,
  initial,
  aliases = {},
  pageView = () => null,
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
      pageView(tab, current) || params.get(parameter) || current,
    );
  }
  function url(view) {
    const next = new URL(location.href);
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
  views: ["collect", "recording", "registry", "files"],
  initial: "collect",
  aliases: {
    live: "collect",
    recordings: "recording",
    setup: "collect",
  },
  pageView(tab, current) {
    if (tab === "datasets") return "registry";
    if (tab === "collection") return current === "registry" ? "collect" : current;
    return null;
  },
  renderView(view) {
    if (typeof selectDataCatalog === "function") selectDataCatalog(view);
    const collectionView = {
      collect: "live",
      recording: "recordings",
    }[view];
    document.querySelectorAll("[data-collection-view]").forEach((panel) => {
      panel.hidden = panel.dataset.collectionView !== collectionView;
    });
    const tutorial = document.getElementById("datasets-tutorial-button");
    if (tutorial) tutorial.hidden = !["registry", "files"].includes(view);
    const heading = document.getElementById("data-catalog-title");
    if (heading) heading.textContent = view === "files" ? "Files" : "Datasets";
    if (typeof refreshDataResourceTables === "function")
      refreshDataResourceTables();
    document.getElementById("refresh-collection").hidden = [
      "registry", "files",
    ].includes(view);
    document.getElementById("refresh-data-registry").hidden = ![
      "registry", "files",
    ].includes(view);
  },
});

window.experimentNavigation = createWorkspaceNavigation({
  page: "experiments",
  parameter: "experiment_view",
  selector: "[data-experiment-tab]",
  views: ["presets", "submit", "adapters", "runs", "notes"],
  initial: "submit",
  pageView: (tab) => (["adapters", "runs", "notes"].includes(tab) ? tab : null),
  renderView(view) {
    for (const [page, selected] of [
      ["experiments", "submit"],
      ["adapters", "adapters"],
      ["runs", "runs"],
    ]) {
      document.getElementById(`refresh-${page}`).hidden =
        view !== selected && !(page === "experiments" && ["presets", "notes"].includes(view));
      const tutorial = document.getElementById(`${page}-tutorial-button`);
      if (tutorial) tutorial.hidden = view !== selected;
    }
  },
});

window.evaluationNavigation = createWorkspaceNavigation({
  page: "evaluations",
  parameter: "evaluation_view",
  selector: "[data-evaluation-tab]",
  views: ["submit", "suites", "runs"],
  initial: "submit",
  pageView: (tab) =>
    ({ "evaluation-suites": "suites", "evaluation-runs": "runs" })[tab],
  renderView(view) {
    const tutorial = document.getElementById("evaluations-tutorial-button");
    if (tutorial) tutorial.hidden = view !== "submit";
  },
});

function setCollectionView(view, options) {
  dataNavigation.select(view, options);
}

document.querySelectorAll("[data-collection-go]").forEach((button) =>
  button.addEventListener("click", () => {
    dataNavigation.select(button.dataset.collectionGo, { focus: true });
    const setup = document.getElementById("collection-setup-details");
    if (setup) setup.open = true;
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

for (const [page, views] of [
  ["settings", ["connections", "storage", "notifications"]],
  ["cluster", ["gpu", "jobs"]],
]) {
  window[page + "Navigation"] = createWorkspaceNavigation({
    page,
    parameter: page + "_view",
    selector: "[data-" + page + "-tab]",
    views,
    initial: views[0],
    renderView(view) {
      document.querySelectorAll("[data-" + page + "-view]").forEach((panel) => {
        panel.hidden = panel.getAttribute("data-" + page + "-view") !== view;
      });
    },
  });
}
