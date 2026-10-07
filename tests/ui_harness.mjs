// Shared jsdom bootstrap for the browser tests: the application page in a window,
// the browser APIs jsdom lacks, dialog polyfills, script loading and app.js slices.
// Each helper is opt-in so a test keeps exactly the shims it relied on.
import {readFile} from 'node:fs/promises';
import {JSDOM} from 'jsdom';
import {indexHtml} from './index_page.cjs';

const STATIC = new URL('../static/', import.meta.url);
export const readStatic = (file) => readFile(new URL(file, STATIC), 'utf8');

// The scripts a full page needs, in the order index.html loads them.
export const APP_SCRIPTS = ['dialogs.js', 'workspace-navigation.js', 'connection-settings.js', 'app.js'];

// The application page as the server sends it, in a window whose scripts the test evaluates.
export function pageWindow({html = indexHtml(), url = 'http://localhost:8080/', pretendToBeVisual = true, ...options} = {}) {
  return new JSDOM(html, {runScripts: 'outside-only', pretendToBeVisual, url, ...options}).window;
}

export const byId = (w) => (id) => w.document.getElementById(id);

// Settles `ticks` macrotask turns so awaited requests and their renders drain.
export const flusher = (ticks = 4) => async () => {
  for (let i = 0; i < ticks; i++) await new Promise((resolve) => setImmediate(resolve));
};

// Records every MutationObserver the scripts create so a test can drive them.
export function spyMutationObservers(w) {
  const observers = [], Native = w.MutationObserver;
  w.MutationObserver = class extends Native {
    constructor(callback) { super(callback); observers.push(this); }
  };
  return observers;
}

// jsdom has none of these; the defaults keep a page from reaching the network or layout.
export function stubBrowserApis(w, {fetch = true, scroll = true, matchMedia = true, cssEscape = false} = {}) {
  if (fetch) w.fetch = () => new Promise(() => {});
  if (scroll) w.scrollTo = w.HTMLElement.prototype.scrollIntoView = () => {};
  if (matchMedia) w.matchMedia = () => ({matches: false, addEventListener() {}, removeEventListener() {}});
  if (cssEscape) w.CSS = {escape: (value) => value};
}

// <dialog> in jsdom has no showModal/close. `closeEvent` dispatches 'close' like a browser,
// `guarded` ignores a close on an already closed dialog, `returnValue` records the close value.
export function polyfillDialogs(w, {closeEvent = true, guarded = true, returnValue = true} = {}) {
  w.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
  w.HTMLDialogElement.prototype.close = function (value = '') {
    if (guarded && !this.open) return;
    if (returnValue) this.returnValue = value;
    this.open = false;
    if (closeEvent) this.dispatchEvent(new w.Event('close'));
  };
}

// Evaluates static/ scripts in the window; `append` adds a test hook after the named file.
export async function loadScripts(w, files = APP_SCRIPTS, append = {}) {
  for (const file of files) {
    const extra = append[file];
    w.eval((await readStatic(file)) + (extra ? '\n' + extra : ''));
  }
}

// One top-level function of app.js, by name, through its closing brace.
export function appFunction(source, name) {
  const start = source.search(new RegExp(`^(?:async )?function ${name}\\(`, 'm'));
  if (start < 0) throw new Error(`app.js has no function ${name}`);
  return source.slice(start, source.indexOf('\n}\n', start) + 3);
}

// The UI_TIMING block app.js functions read. A const from one eval is invisible to the
// next, so callers evaluate it in the same string as the functions that use it.
export function appTimingBlock(source) {
  const start = source.indexOf('const UI_TIMING = ');
  return source.slice(start, source.indexOf('\n});\n', start) + 5);
}

// What app.js's api() rejects with for an HTTP error: the detail as the message plus the status.
export const apiError = (message, status = 400) => Object.assign(new Error(message), {status});
