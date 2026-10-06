// The application page as the server sends it. index() fills the gateway, queue,
// GPU and default-value placeholders from the cluster configuration; browser
// tests take the same fragments from skynet_app.page_markup.
const { execFileSync } = require('node:child_process');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');

const ROOT = join(__dirname, '..');
let markup = null;

function clusterMarkup() {
  markup ||= JSON.parse(execFileSync(join(ROOT, '.venv/bin/python'), ['-m', 'skynet_app.page_markup'], {
    cwd: ROOT, encoding: 'utf8',
  }));
  return markup;
}

function indexHtml() {
  let html = readFileSync(join(ROOT, 'static/index.html'), 'utf8');
  for (const [placeholder, fragment] of Object.entries(clusterMarkup())) html = html.replaceAll(placeholder, fragment);
  return html;
}

module.exports = { indexHtml, clusterMarkup };
