import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {JSDOM, VirtualConsole} from 'jsdom';

const html = await readFile(new URL('../static/index.html', import.meta.url), 'utf8');
const bootstrap = await readFile(new URL('../static/email-workspace.js', import.meta.url), 'utf8');
const sources = new Map();
for (const match of html.matchAll(/data-workspace-src="([^"?]+)/g)) {
  sources.set(match[1], await readFile(new URL('..'+match[1], import.meta.url), 'utf8'));
}
const flush = async () => {for (let i=0; i<5; i++) await new Promise(resolve=>setImmediate(resolve));};
function setup(session=null, preferences={}) {
  const errors=[];
  const virtualConsole=new VirtualConsole();
  virtualConsole.on('jsdomError', error => {if (!error.message.includes('navigation')) errors.push(error);});
  const dom=new JSDOM(html, {url:'http://localhost:8080/?data_view=recording#runs', runScripts:'outside-only', pretendToBeVisual:true, virtualConsole});
  const w=dom.window;
  const observers=[]; const Observer=w.MutationObserver;
  w.MutationObserver=class extends Observer {constructor(callback){super(callback);observers.push(this);}};
  for (const key of ['Headers','Request','Response']) w[key]=globalThis[key];
  w.matchMedia=()=>({matches:false,addEventListener(){},removeEventListener(){}});
  w.scrollTo=w.HTMLElement.prototype.scrollIntoView=()=>{};
  const calls=[]; const loaded=[];
  let handler=async (input, options={})=>{
    const raw=input instanceof Request ? input.url : String(input);
    const path=new URL(raw, w.location.href).pathname;
    calls.push({path, options});
    if (path==='/api/workspace/session') return Response.json({workspace:session});
    if (path==='/api/fixture') return Response.json({ok:true});
    return new Promise(()=>{});
  };
  w.fetch=(...args)=>handler(...args);
  const append=w.document.head.append.bind(w.document.head);
  w.document.head.append=(node)=>{
    append(node);
    if(node.tagName==='SCRIPT') {
      loaded.push(new URL(node.src).pathname);
      try {Object.defineProperty(w.document, "currentScript", {configurable:true, value:node}); w.eval(sources.get(loaded.at(-1))); node.onload();}
      catch(error){errors.push(error); node.onerror();}
    }
  };
  for(const [key,value] of Object.entries(preferences)) w.localStorage.setItem(key,value);
  w.eval(bootstrap);
  return {w, calls, loaded, errors, setHandler(value){handler=value;}, close(){for(const observer of observers) observer.disconnect(); w.close();}, el:id=>w.document.getElementById(id)};
}

// No application code or private API requests execute before email selection.
{
  const f=setup(); await flush();
  assert.equal(f.el('email-workspace-content').hidden,true);
  assert.equal(f.el('email-workspace-gate').hidden,false);
  assert.equal(f.loaded.length,0);
  assert.deepEqual(f.calls.map(c=>c.path), ['/api/workspace/session']);
  const form=f.el('email-workspace-form'), button=form.querySelector('button');
  f.el('workspace-email').value='not-an-email';
  form.dispatchEvent(new f.w.Event('submit', {cancelable:true})); await flush();
  assert.equal(f.calls.length,1);
  let resolve;
  f.setHandler(()=>new Promise(r=>{resolve=r;}));
  f.el('workspace-email').value='alice@example.com';
  form.dispatchEvent(new f.w.Event('submit', {cancelable:true}));
  assert.equal(button.disabled,true);
  let duplicate=false; f.setHandler(()=>{duplicate=true;});
  form.dispatchEvent(new f.w.Event('submit', {cancelable:true}));
  assert.equal(duplicate,false);
  resolve(Response.json({detail:'Try again'}, {status:503})); await flush();
  assert.equal(button.disabled,false);
  assert.match(f.el('workspace-message').textContent,/Try again/);
  assert.equal(f.loaded.length,0);
  await flush(); assert.deepEqual(f.errors,[]); f.close();
}
// Run the actual application scripts in their production order after session load.
{
  const f=setup({id:'alice', email:'alice@example.com'}); await flush();
  assert.deepEqual(f.errors,[]);
  assert.deepEqual(f.loaded,[...sources.keys()]);
  assert.equal(f.el('email-workspace-content').hidden,false);
  assert.equal(f.el('email-workspace-gate').hidden,true);
  assert.equal(f.el('workspace-current-email').textContent,'alice@example.com');
  assert.equal(f.w.SkynetWorkspace.storageKey('preference'),'preference:workspace:alice');
  assert.ok(f.calls.filter(c=>c.path!=='/api/workspace/session').every(c=>new Headers(c.options.headers).get('X-Skynet-Workspace')==='alice'));
  await f.w.fetch(new f.w.URL('/api/fixture', f.w.location.href));
  // A failed sign-out must re-enable the exact button after the async handler.
  f.setHandler(async()=>Response.json({detail:'Network unavailable'}, {status:503}));
  f.el('workspace-sign-out').click(); await flush();
  assert.equal(f.el('workspace-sign-out').disabled,false);
  assert.match(f.el('workspace-sign-out-error').textContent,/Network unavailable/);
  // A tab with the old workspace cannot use the replacement cookie silently.
  f.setHandler(async()=>Response.json({code:'workspace_changed'}, {status:409}));
  await assert.rejects(f.w.fetch('/api/fixture'), error=>error.name==='AbortError');
  assert.equal(f.el('email-workspace-content').hidden,true);
  assert.equal(f.el('email-workspace-gate').hidden,false);
  await flush(); assert.deepEqual(f.errors,[]); f.close();
}
{
  const f=setup({id:'legacy',email:'ycho420@gatech.edu'},{'skynet:ssh-gateway':'sky2','skynet.tutorial.completed.runs':'1'});
  await flush();
  assert.equal(f.el('gateway').value,'sky2');
  assert.equal(f.w.localStorage.getItem('skynet:ssh-gateway:workspace:legacy'),'sky2');
  assert.equal(f.w.localStorage.getItem('skynet.tutorial.completed.runs:workspace:legacy'),'1');
  assert.equal(f.w.localStorage.getItem('skynet:ssh-gateway'),null);
  assert.deepEqual(f.errors,[]); f.close();
}
{
  const f=setup({id:'new',email:'new@example.com',storage_configured:false});
  await flush();
  assert.equal(f.w.SkynetStorageConfigured,false);
  assert.equal(f.el('workspace-storage-form').closest('[data-tab-panel]').hidden,false);
  f.w.activateTab('experiments');
  assert.equal(f.el('workspace-storage-form').closest('[data-tab-panel]').hidden,false);
  assert.deepEqual(f.errors,[]); f.close();
}
console.log('Email workspace UI: gate, real script initialization, validation, switching and stale tabs passed');
