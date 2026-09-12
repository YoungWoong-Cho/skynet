import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {JSDOM} from 'jsdom';
const w = new JSDOM(await readFile(new URL('../static/index.html', import.meta.url), 'utf8'), {runScripts:'outside-only',pretendToBeVisual:true}).window;
w.HTMLDialogElement.prototype.showModal = function(){this.open=true;};
w.HTMLDialogElement.prototype.close = function(value=''){if(this.open){this.returnValue=value;this.open=false;this.dispatchEvent(new w.Event('close'));}};
w.eval(await readFile(new URL('../static/dialogs.js',import.meta.url),'utf8'));
try {
  for(const surface of w.document.querySelectorAll('dialog, [role="dialog"]')) assert.ok(surface.classList.contains('app-dialog'),surface.id);
  const surfaces=[...w.document.querySelectorAll('dialog, [role="dialog"]')];
  assert.equal(surfaces.length,20,'inventory covers every static dialog and the guided tutorial');
  for(const surface of surfaces) {
    const shell=surface.querySelector(':scope > .dialog-shell');
    assert.ok(shell,`${surface.id}: shared shell`);
    assert.equal(shell.querySelectorAll(':scope > .dialog-heading').length,1,`${surface.id}: one shared header`);
    assert.equal(shell.querySelectorAll(':scope > [data-dialog-body]').length,1,`${surface.id}: one scroll body`);
    const close=shell.querySelector(':scope > .dialog-heading [data-dialog-close]');
    assert.equal(close.textContent.trim(),'Close',`${surface.id}: consistent Close`);
    assert.equal(close.type,'button','Close cannot submit a form');
    assert.ok(w.document.getElementById(surface.getAttribute('aria-labelledby')),`${surface.id}: accessible title`);
    assert.equal(shell.querySelector(':scope > .dialog-heading h3').id,surface.getAttribute('aria-labelledby'));
  }
  const ids=[...w.document.querySelectorAll('[id]')].map(n=>n.id);
  assert.equal(ids.length,new Set(ids).size,'hydration preserves unique ids');
  for(const form of w.document.querySelectorAll('form[data-dialog-panel]')) {
    for(const submit of form.querySelectorAll('button[type="submit"]'))assert.equal(submit.form,form,'shared body preserves form submission ownership');
  }
  const launcher=w.document.createElement('button'); w.document.body.append(launcher); launcher.focus();
  const review=w.document.getElementById('live-review-dialog');
  w.SkynetDialog.open(review); w.document.getElementById('live-review-close').click();
  assert.equal(review.open,false); assert.equal(w.document.activeElement,launcher);
  const canceled=w.askUserDialog('Proceed?');
  w.document.querySelector('dialog[open] .button-outline').click();
  assert.equal(await canceled,false); assert.equal(w.document.activeElement,launcher);
  const entered=w.askUserDialog('Name','Before');
  const prompt=w.document.querySelector('dialog[open]');
  assert.ok(prompt.querySelector(':scope > .dialog-shell > .dialog-heading'), 'dynamic prompt uses the same component');
  prompt.querySelector('input').value='After';
  prompt.querySelector('form').dispatchEvent(new w.Event('submit',{cancelable:true}));
  assert.equal(await entered,'After');
  w.SkynetDialog.open(review);
  const reviewBody=review.querySelector('[data-dialog-body]');reviewBody.scrollTop=200;
  const nested=w.document.getElementById('run-attempt-detail-dialog');
  w.SkynetDialog.open(nested);
  nested.dataset.blockClose='true';
  const nativeCancel=new w.Event('cancel',{cancelable:true});nested.dispatchEvent(nativeCancel);
  assert.equal(nativeCancel.defaultPrevented,true,'native Escape obeys close blocking');
  nested.querySelector('[data-dialog-close]').click();assert.equal(nested.open,true);
  delete nested.dataset.blockClose;
  w.document.dispatchEvent(new w.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true}));
  assert.equal(nested.open,false,'Escape only closes the topmost dialog');assert.equal(review.open,true);
  assert.equal(reviewBody.scrollTop,200,'closing a nested modal preserves the parent position');
  w.SkynetDialog.close(review);w.SkynetDialog.open(review);
  assert.equal(reviewBody.scrollTop,0,'reopening starts at the top');w.SkynetDialog.close(review);
  const guide=w.document.getElementById('tutorial-layer'); w.SkynetDialog.openGuide(guide);
  assert.equal(guide.hidden,false); assert.equal(w.document.querySelectorAll('[inert]').length,0);
  w.SkynetDialog.closeGuide(guide); assert.equal(guide.hidden,true);
  console.log('Shared dialogs: surfaces, close actions, focus restoration, confirmation, prompt and interactive guide passed.');
} finally {w.close();}
