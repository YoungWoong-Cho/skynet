import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {JSDOM} from 'jsdom';
const w = new JSDOM(await readFile(new URL('../static/index.html', import.meta.url), 'utf8'), {runScripts:'outside-only',pretendToBeVisual:true}).window;
w.HTMLDialogElement.prototype.showModal = function(){this.open=true;};
w.HTMLDialogElement.prototype.close = function(value=''){if(this.open){this.returnValue=value;this.open=false;this.dispatchEvent(new w.Event('close'));}};
w.eval(await readFile(new URL('../static/dialogs.js',import.meta.url),'utf8'));
try {
  for(const surface of w.document.querySelectorAll('dialog, [role="dialog"]')) assert.ok(surface.classList.contains('app-dialog'),surface.id);
  const launcher=w.document.createElement('button'); w.document.body.append(launcher); launcher.focus();
  const review=w.document.getElementById('live-review-dialog');
  w.SkynetDialog.open(review); w.document.getElementById('live-review-close').click();
  assert.equal(review.open,false); assert.equal(w.document.activeElement,launcher);
  const canceled=w.askUserDialog('Proceed?');
  w.document.querySelector('dialog[open] .button-outline').click();
  assert.equal(await canceled,false); assert.equal(w.document.activeElement,launcher);
  const entered=w.askUserDialog('Name','Before');
  const prompt=w.document.querySelector('dialog[open]');
  prompt.querySelector('input').value='After';
  prompt.querySelector('form').dispatchEvent(new w.Event('submit',{cancelable:true}));
  assert.equal(await entered,'After');
  const guide=w.document.getElementById('tutorial-layer'); w.SkynetDialog.openGuide(guide);
  assert.equal(guide.hidden,false); assert.equal(w.document.querySelectorAll('[inert]').length,0);
  w.SkynetDialog.closeGuide(guide); assert.equal(guide.hidden,true);
  console.log('Shared dialogs: surfaces, close actions, focus restoration, confirmation, prompt and interactive guide passed.');
} finally {w.close();}
