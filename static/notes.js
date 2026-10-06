/* Notes reuse the workspace navigation, table renderer and dialog lifecycle. */
(() => {
  const el = id => document.getElementById(id);
  const search = el("notes-search");
  const table = el("notes-table");
  const dialog = el("workspace-note-dialog");
  const content = el("workspace-note-content");
  const breadcrumb = el("notes-path");
  const editor = el("note-editor-dialog");
  const form = el("note-editor-form");
  const fileInput = el("note-editor-files");
  let notes = [], folders = [], generation = 0, listGeneration = 0;
  // The note in the viewer, and the saved note the editor changes (null for a new note).
  let shown = null, editing = null, removed = new Set(), busy = false;
  let activeFolder = new URL(location.href).searchParams.get("note_folder") || "";
  const isActive = () => !el("experiment-notes").hidden;
  const folderName = id => folders.find(folder => folder.id === id)?.name || "Notes";
  const noteUrl = id => `/api/notes/${encodeURIComponent(id)}`;
  const attachmentUrl = (id, name) => `${noteUrl(id)}/attachments/${encodeURIComponent(name)}`;
  // The server's attachment rule (types from the file input's accept list), checked before any write.
  const attachmentTypes = fileInput.accept.split(",");
  const attachable = name => /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(name) &&
    attachmentTypes.some(type => name.toLowerCase().endsWith(type));

  // A search spans every folder; an empty search lists the current folder.
  // An index note (title starting "00 ") stays on top. The rest keep the server's newest-first order.
  const pinned = note => note.title.startsWith("00 ");
  function visibleNotes(query) {
    const shown = notes.filter(note => query ? note.title.toLowerCase().includes(query)
      : (note.folder_id || "") === activeFolder);
    return [...shown.filter(pinned), ...shown.filter(note => !pinned(note))];
  }

  function chooseFolder(id) {
    activeFolder = id;
    search.value = "";
    const url = new URL(location.href);
    if (!id) url.searchParams.delete("note_folder");
    else url.searchParams.set("note_folder", id);
    history.replaceState(null, "", url);
    render();
  }

  function acceptListing(result) {
    notes = result.notes;
    folders = result.folders || [];
    if (activeFolder && !folders.some(folder => folder.id === activeFolder)) chooseFolder("");
    render();
  }

  function render() {
    const query = search.value.trim().toLowerCase();
    const counts = new Map();
    for (const note of notes) counts.set(note.folder_id, (counts.get(note.folder_id) || 0) + 1);
    breadcrumb.innerHTML = activeFolder
      ? `<button type="button" class="entity-link" data-note-folder="">Notes</button><span aria-hidden="true">/</span><strong aria-current="page">${escapeHtml(folderName(activeFolder))}</strong>`
      : '<strong aria-current="page">Notes</strong>';
    el("notes-location").hidden = !activeFolder;
    // Folders are one level deep: new folders are made from the top level.
    el("new-folder").hidden = Boolean(activeFolder);
    const folderRows = activeFolder ? [] : folders.filter(folder => folder.name.toLowerCase().includes(query)).map(folder => ({
      id:folder.id,
      cells:[
        {className:"wrap-cell", html:`<button type="button" class="entity-link node-name" data-note-folder="${escapeHtml(folder.id)}"><span aria-hidden="true">📁</span> ${escapeHtml(folder.name)}</button>`},
        `Folder · ${counts.get(folder.id) || 0} notes`,
        '',
        '—',
        '—',
        {className:"row-actions", html:`<button type="button" data-note-folder="${escapeHtml(folder.id)}">Open</button>`},
      ],
    }));
    SkynetJobHistory.render(table, {
      columns: ["Name", "Kind", "Status", "Created", "Updated", "Actions"],
      empty: query ? "No notes match your filter." : "This folder is empty.",
      rows: [...folderRows, ...visibleNotes(query).map(note => ({
        id: note.id,
        cells: [
          {className: "wrap-cell", html: `<button type="button" class="entity-link node-name" data-open-note="${escapeHtml(note.id)}">${escapeHtml(note.title)}</button>`},
          // A search spans folders, so its results name the folder each note is in.
          query ? `Markdown · ${escapeHtml(folderName(note.folder_id))}` : 'Markdown',
          // The status comes from the note's "상태:" header line. Notes without one leave it blank.
          escapeHtml(note.status || ""),
          escapeHtml(formatDate(note.created_at)),
          escapeHtml(formatDate(note.updated_at)),
          {className: "row-actions", html: `<button type="button" data-open-note="${escapeHtml(note.id)}">View</button>`},
        ],
      }))],
    });
  }

  // The server sends math as escaped TeX; KaTeX is fetched once, only for notes that use it.
  let katexLoading = null;
  function loadKatex() {
    if (window.katex) return Promise.resolve(window.katex);
    const base = "/static/vendor/katex-0.16.22/";
    const load = node => new Promise((resolve, reject) => {
      node.onload = resolve;
      node.onerror = () => reject(new Error("KaTeX could not load"));
      document.head.append(node);
    });
    katexLoading ||= Promise.all([
      load(Object.assign(document.createElement("link"), {rel: "stylesheet", href: base + "katex.min.css"})),
      load(Object.assign(document.createElement("script"), {src: base + "katex.min.js"})),
    ]).then(() => window.katex, error => {katexLoading = null; throw error;});
    return katexLoading;
  }

  async function renderMath(root) {
    const nodes = [...root.querySelectorAll(".math")];
    if (!nodes.length) return;
    try {
      const katex = await loadKatex();
      for (const node of nodes) katex.render(node.textContent, node, {
        displayMode: node.classList.contains("math-display"), throwOnError: false,
      });
    } catch {
      // Keep the TeX source readable when KaTeX is unavailable.
    }
  }

  const noteActions = hidden => {
    for (const action of ["download", "edit", "delete"]) el(`workspace-note-${action}`).hidden = hidden;
  };

  async function openNote(id, launcher) {
    const ticket = ++generation;
    shown = null;
    content.textContent = "Loading note…";
    dialog.querySelector("[data-dialog-title]").textContent = "Note";
    noteActions(true);
    el("workspace-note-date").textContent = "";
    clearNotice(el("workspace-note-error"));
    if (!dialog.open) SkynetDialog.open(dialog, {launcher});
    try {
      const {note} = await api(noteUrl(id));
      if (ticket !== generation || !dialog.open) return;
      shown = note;
      dialog.querySelector("[data-dialog-title]").textContent = note.title;
      el("workspace-note-date").textContent = `Created ${formatDate(note.created_at)} · Updated ${formatDate(note.updated_at)}`;
      content.innerHTML = note.html; // Server renderer disables raw HTML and unsafe links.
      void renderMath(content);
      el("workspace-note-download").href = `${noteUrl(note.id)}/download`;
      noteActions(false);
      const url = new URL(location.href);
      url.searchParams.set("note", note.id);
      history.replaceState(null, "", url);
    } catch (error) {
      if (ticket === generation && dialog.open) content.textContent = error.message;
    }
  }

  function setBusy(value) {
    busy = value;
    for (const surface of [dialog, editor]) {
      surface.dataset.blockClose = String(value);
      surface.setAttribute("aria-busy", String(value));
    }
    for (const control of [...form.elements, ...dialog.querySelectorAll("button")]) control.disabled = value;
  }

  function renderAttachments() {
    SkynetJobHistory.render(el("note-editor-attachments"), {
      columns: ["Name", "Size", "Actions"],
      empty: "No attachments.",
      rows: (editing?.attachments || []).filter(item => !removed.has(item.name)).map(item => ({
        id: item.name,
        cells: [
          {className: "wrap-cell", html: `<a href="${escapeHtml(item.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(item.name)}</a>`},
          escapeHtml(formatDataBytes(item.size_bytes)),
          {className: "row-actions", html: `<button type="button" data-remove-attachment="${escapeHtml(item.name)}" aria-label="Remove ${escapeHtml(item.name)}">Remove</button>`},
        ],
      })),
    });
  }

  function openEditor(note, launcher) {
    if (busy) return;
    editing = note;
    removed = new Set();
    editor.querySelector("[data-dialog-title]").textContent = note ? "Edit note" : "New note";
    el("note-editor-title").value = note?.title || "";
    el("note-editor-folder").replaceChildren(new Option("No folder", ""),
      ...folders.map(folder => new Option(folder.name, folder.id)));
    el("note-editor-folder").value = note ? note.folder_id || "" : activeFolder;
    el("note-editor-markdown").value = note?.markdown || "";
    fileInput.value = "";
    clearNotice(el("note-editor-error"));
    renderAttachments();
    SkynetDialog.open(editor, {launcher});
    el("note-editor-title").focus({preventScroll: true});
  }

  // Content first, so a stale version is refused before any attachment changes.
  form.addEventListener("submit", async event => {
    event.preventDefault();
    if (busy || !form.reportValidity()) return;
    const body = {
      title: el("note-editor-title").value.trim(),
      markdown: el("note-editor-markdown").value,
      folder_id: el("note-editor-folder").value || null,
    };
    const uploads = [...fileInput.files];
    const refused = uploads.map(file => file.name).filter(name => !attachable(name));
    clearNotice(el("note-editor-error"));
    if (refused.length) {
      showNotice(el("note-editor-error"), `${refused.join(", ")} cannot be attached, so nothing was saved. Attachment names use letters, numbers, dots, underscores or hyphens, start with a letter or number and end in ${attachmentTypes.join(", ")}.`, {scope: "note-editor"});
      return;
    }
    // Each attachment change is tried once the content is saved; failures are grouped by outcome.
    const failed = new Map();
    const attempt = (name, outcome, request) => request.catch(error => {
      const key = `${outcome}: ${error.message}`;
      failed.set(key, [...failed.get(key) || [], name]);
    });
    let saved = null;
    setBusy(true);
    try {
      ({note: saved} = await api(editing ? noteUrl(editing.id) : "/api/notes", {
        method: editing ? "PUT" : "POST",
        body: JSON.stringify(editing ? {...body, expected_updated_at: editing.updated_at} : body),
      }));
      for (const {name} of saved.attachments.filter(item => removed.has(item.name)))
        await attempt(name, "removed", api(attachmentUrl(saved.id, name), {method: "DELETE"}));
      for (const file of uploads)
        await attempt(file.name, "attached", api(attachmentUrl(saved.id, file.name), {
          method: "PUT", body: file, headers: {"Content-Type": file.type || "application/octet-stream"},
        }));
      if (failed.size) {
        // The note exists now: a retry updates its current version instead of creating another,
        // and sends only the files chosen again.
        editing = await api(noteUrl(saved.id)).then(result => result.note, () => saved);
        editor.querySelector("[data-dialog-title]").textContent = "Edit note";
        fileInput.value = "";
        renderAttachments();
        const outcomes = [...failed].map(([outcome, names]) => `${names.join(", ")} ${names.length > 1 ? "were" : "was"} not ${outcome}`);
        showNotice(el("note-editor-error"), `Saved the note, but ${outcomes.join("; ")}`, {scope: "note-editor"});
        return;
      }
      setBusy(false);
      SkynetDialog.close(editor);
      void openNote(saved.id);
    } catch (error) {
      showNotice(el("note-editor-error"), error.message, {scope: "note-editor"});
    } finally {
      setBusy(false);
    }
  });

  window.loadNotes = async () => {
    const ticket = ++listGeneration;
    const button = el("refresh-experiments");
    button.disabled = true;
    try {
      const result = await api("/api/notes");
      if (ticket !== listGeneration) return;
      acceptListing(result);
      clearNotificationScope("notes");
      // Open a linked note, or reload the shown note after it changed elsewhere; never under
      // the editor, whose close runs this again.
      const selected = new URL(location.href).searchParams.get("note");
      const changed = shown?.id === selected &&
        notes.find(note => note.id === selected)?.updated_at !== shown.updated_at;
      if (isActive() && selected && !editor.open && (!dialog.open || changed)) await openNote(selected);
    } catch (error) {
      showNotice(el("notes-error"), error.message, {scope: "notes"});
    } finally {
      button.disabled = false;
    }
  };

  search.addEventListener("input", render);
  breadcrumb.addEventListener("click", event => {
    const button = event.target.closest("[data-note-folder]");
    if (button) chooseFolder(button.dataset.noteFolder);
  });
  table.addEventListener("click", event => {
    const folder = event.target.closest("[data-note-folder]");
    if (folder) {chooseFolder(folder.dataset.noteFolder); return;}
    const button = event.target.closest("[data-open-note]");
    if (button) void openNote(button.dataset.openNote, button);
  });
  content.addEventListener("click", event => {
    const link = event.target.closest("a");
    if (!link || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const url = new URL(link.href, location.href);
    if (url.origin !== location.origin) return;
    const note = url.searchParams.get("note"), folder = url.searchParams.get("note_folder");
    if (note) {
      event.preventDefault();
      void openNote(note);
    } else if (folder) {
      event.preventDefault();
      SkynetDialog.close(dialog);
      chooseFolder(folder);
    }
  });
  el("new-note").addEventListener("click", event => openEditor(null, event.currentTarget));
  el("new-folder").addEventListener("click", async event => {
    const button = event.currentTarget;
    const name = (await askUserDialog("Folder name", ""))?.trim();
    if (!name) return;
    clearNotice(el("notes-error"));
    button.disabled = true;
    try {
      const {folder} = await api("/api/notes/folders", {method: "POST", body: JSON.stringify({name})});
      // The write's list refresh may already hold the folder; it also brings the server's order.
      folders = [...folders.filter(item => item.id !== folder.id), folder];
      if (isActive() && !activeFolder) {
        chooseFolder(folder.id);
        // New folder hides inside a folder; a new note now starts in this one.
        el("new-note").focus({preventScroll: true});
      } else {
        render();
      }
    } catch (error) {
      showNotice(el("notes-error"), error.message, {scope: "notes"});
    } finally {
      button.disabled = false;
    }
  });
  el("workspace-note-edit").addEventListener("click", event => openEditor(shown, event.currentTarget));
  el("workspace-note-delete").addEventListener("click", async () => {
    const note = shown;
    if (busy || !note || !(await askUserDialog(`Delete “${note.title}” and its attachments? This cannot be undone.`))) return;
    clearNotice(el("workspace-note-error"));
    setBusy(true);
    try {
      await api(noteUrl(note.id), {method: "DELETE"});
      setBusy(false);
      SkynetDialog.close(dialog);
    } catch (error) {
      showNotice(el("workspace-note-error"), error.message, {scope: "note"});
    } finally {
      setBusy(false);
    }
  });
  el("note-editor-attachments").addEventListener("click", event => {
    const button = event.target.closest("[data-remove-attachment]");
    if (!button || busy) return;
    removed.add(button.dataset.removeAttachment);
    renderAttachments();
  });
  dialog.addEventListener("close", () => {
    generation++;
    shown = null;
    const url = new URL(location.href);
    url.searchParams.delete("note");
    history.replaceState(null, "", url);
  });
  editor.addEventListener("close", () => { if (isActive()) void loadNotes(); });
  // Every successful note write also invalidates "notes", which reloads the list.
  window.SkynetRefresh?.register("notes", ["notes"], isActive, loadNotes);
  render();
  if (isActive()) void loadNotes();
})();
