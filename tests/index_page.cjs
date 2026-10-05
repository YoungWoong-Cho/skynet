// The application page as the server sends it. index() fills every gateway
// <select> from the configured gateways; browser tests use this fixed list.
const { readFileSync } = require('node:fs');
const { join } = require('node:path');

const GATEWAYS = ['sky1', 'sky2'];
const GATEWAY_OPTIONS = `<option value="auto">Auto: ${GATEWAYS.join(', then ')}</option>` +
  GATEWAYS.map(host => `<option value="${host}">Prefer ${host}</option>`).join('');

function indexHtml() {
  return readFileSync(join(__dirname, '../static/index.html'), 'utf8')
    .replaceAll('<!-- gateway-options -->', GATEWAY_OPTIONS);
}

module.exports = { indexHtml };
