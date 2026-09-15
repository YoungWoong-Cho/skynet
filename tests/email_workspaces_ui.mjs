import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {JSDOM, VirtualConsole} from 'jsdom';

const html = await readFile(new URL('../static/index.html', import.meta.url), 'utf8');
assert.doesNotMatch(html, /For your trusted team\. Anyone entering this email/);
assert.doesNotMatch(html, /Enter your email to open your configurations/);
const bootstrap = await readFile(new URL('../static/email-workspace.js', import.meta.url), 'utf8');
const sources = new Map();
for (const match of html.matchAll(/data-workspace-src="([^"?]+)/g)) {
  sources.set(match[1], await readFile(new URL('..'+match[1], import.meta.url), 'utf8'));
}
const flush = async () => {for (let i=0; i<5; i++) await new Promise(resolve=>setImmediate(resolve));};
function setup(session=null, preferences={}, sessionResponse=null, Channel=null) {
  const errors=[],navigations=[];
  const virtualConsole=new VirtualConsole();
  virtualConsole.on('jsdomError', error => {
    if(error.message.includes('navigation')) navigations.push(error);
    else errors.push(error);
  });
  const dom=new JSDOM(html, {url:'http://localhost:8080/?data_view=recording#runs', runScripts:'outside-only', pretendToBeVisual:true, virtualConsole});
  const w=dom.window;
  if (Channel) w.BroadcastChannel=Channel;
  const observers=[]; const Observer=w.MutationObserver;
  w.MutationObserver=class extends Observer {constructor(callback){super(callback);observers.push(this);}};
  for (const key of ['Headers','Request','Response','AbortController','AbortSignal']) w[key]=globalThis[key];
  w.matchMedia=()=>({matches:false,addEventListener(){},removeEventListener(){}});
  w.scrollTo=w.HTMLElement.prototype.scrollIntoView=()=>{};
  const calls=[]; const loaded=[];
  let handler=async (input, options={})=>{
    const raw=input instanceof Request ? input.url : String(input);
    const path=new URL(raw, w.location.href).pathname;
    calls.push({path, options});
    if (path==='/api/workspace/session') return sessionResponse ? sessionResponse() : Response.json({workspace:session});
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
  return {w, calls, loaded, errors, navigations, setHandler(value){handler=value;}, close(){for(const observer of observers) observer.disconnect(); w.close();}, el:id=>w.document.getElementById(id)};
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
  f.setHandler(async()=>new Response('Internal Server Error',{status:500}));
  form.dispatchEvent(new f.w.Event('submit',{cancelable:true}));await flush();
  assert.match(f.el('workspace-message').textContent,/server is temporarily unavailable/);
  assert.doesNotMatch(f.el('workspace-message').textContent,/Unexpected token/);
  await flush(); assert.deepEqual(f.errors,[]); f.close();
}
// An unavailable cluster keeps the gate usable and resumes the existing session.
{
  const unavailable=()=>Response.json({
    code:'cluster_unavailable',
    detail:'Cannot access the Skynet cluster. Check your network or VPN connection and try again.',
  }, {status:503});
  const f=setup(null, {}, unavailable); await flush();
  const retry=f.el('workspace-retry');
  assert.equal(f.el('email-workspace-content').hidden,true);
  assert.equal(f.el('email-workspace-gate').hidden,false);
  assert.match(f.el('workspace-message').textContent,/Cannot access the Skynet cluster/);
  assert.match(f.el('workspace-message').textContent,/network or VPN/);
  assert.equal(retry.hidden,false);
  assert.equal(retry.type,'button');
  assert.equal(f.loaded.length,0);
  let resolve;
  f.setHandler(()=>new Promise(r=>{resolve=r;}));
  retry.click();
  assert.equal(retry.disabled,true);
  retry.click();
  resolve(unavailable()); await flush();
  assert.equal(retry.disabled,false);
  assert.equal(retry.hidden,false);
  let sessionRequests=0;
  f.setHandler(async(input)=>{
    if(String(input)==='/api/workspace/session') {
      sessionRequests++;
      return Response.json({workspace:{id:'alice',email:'alice@example.com'}});
    }
    return new Promise(()=>{});
  });
  retry.click(); await flush();
  assert.equal(sessionRequests,1);
  assert.equal(f.el('email-workspace-content').hidden,false);
  assert.equal(f.el('email-workspace-gate').hidden,true);
  assert.deepEqual(f.loaded,[...sources.keys()]);
  assert.equal(f.el('workspace-current-email').textContent,'alice@example.com');
  assert.deepEqual(f.errors,[]); f.close();
}
// Browser-level connection failures offer the same actionable retry path.
{
  const f=setup(null, {}, ()=>{throw new TypeError('Failed to fetch');}); await flush();
  assert.match(f.el('workspace-message').textContent,/Cannot connect to Skynet/);
  assert.match(f.el('workspace-message').textContent,/network or VPN/);
  assert.equal(f.el('workspace-retry').hidden,false);
  f.setHandler(async()=>Response.json({workspace:null}));
  f.el('workspace-retry').click(); await flush();
  assert.equal(f.el('workspace-retry').hidden,true);
  assert.equal(f.el('email-workspace-form').querySelector('button[type=submit]').disabled,false);
  assert.equal(f.loaded.length,0);
  assert.deepEqual(f.errors,[]); f.close();
}
// Run the actual application scripts in their production order after session load.
{
  const f=setup({id:'alice', email:'alice@example.com'}); await flush();
  assert.deepEqual(f.errors,[]);
  assert.deepEqual(f.loaded,[...sources.keys()]);
  assert.equal(f.el('email-workspace-content').hidden,false);
  assert.equal(f.el('email-workspace-gate').hidden,true);
  assert.equal(f.el('workspace-current-email').textContent,'alice@example.com');
  const header=f.el('workspace-current-email').closest('header');
  assert.ok(header);
  assert.equal(f.el('workspace-sign-out').textContent.trim(),'sign out');
  assert.ok(f.el('workspace-current-email').compareDocumentPosition(f.el('gateway')) & f.w.Node.DOCUMENT_POSITION_FOLLOWING);
  assert.doesNotMatch(header.textContent,/Workspace:|Switch email/);
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
// A conversion dialog can share a catalog read started by an earlier poll.
{
  const f=setup({id:'alice', email:'alice@example.com'}); await flush();
  try {
    let networkReads=0;
    f.setHandler((_input, options)=>{
      networkReads++;
      return new Promise((_resolve,reject)=>options.signal.addEventListener(
        'abort', ()=>reject(options.signal.reason), {once:true},
      ));
    });
    const results=await Promise.allSettled([
      f.w.apiRequest('/api/fixture', {timeoutMs:5}),
      f.w.apiRequest('/api/fixture', {timeoutMs:100}),
    ]);
    assert.equal(networkReads, 1, 'Concurrent catalog reads share one server request');
    for (const result of results) {
      assert.equal(result.status, 'rejected');
      assert.match(result.reason.message, /Request timed out/, 'A later reader receives the shared deadline reason');
      assert.doesNotMatch(result.reason.message, /aborted without reason/);
    }
  } finally { f.close(); }
}
// An explicit caller cancellation retains its reason instead of becoming a timeout.
{
  const f=setup({id:'alice', email:'alice@example.com'}); await flush();
  try {
    f.setHandler((_input, options)=>new Promise((_resolve,reject)=>options.signal.addEventListener(
      'abort', ()=>reject(options.signal.reason), {once:true},
    )));
    const controller=new f.w.AbortController();
    const reason=new f.w.DOMException('Caller cancelled', 'AbortError');
    const request=f.w.apiRequest('/api/fixture', {timeoutMs:100, signal:controller.signal});
    controller.abort(reason);
    await assert.rejects(request, error=>error===reason);
  } finally { f.close(); }
}
// Switching workspace still cancels reads that are waiting for response headers.
{
  const f=setup({id:'alice', email:'alice@example.com'}); await flush();
  try {
    f.setHandler((input, options)=>String(input)==='/api/fixture'
      ? new Promise((_resolve,reject)=>options.signal.addEventListener('abort', ()=>reject(options.signal.reason), {once:true}))
      : Promise.resolve(Response.json({code:'workspace_changed'}, {status:409})));
    const pending=f.w.apiRequest('/api/fixture', {timeoutMs:100});
    const interrupted=assert.rejects(pending, error=>error.name==='AbortError' && !/timed out/.test(error.message));
    await assert.rejects(f.w.apiRequest('/api/workspace-check'), error=>error.name==='AbortError');
    await interrupted;
    assert.equal(f.el('email-workspace-content').hidden,true);
    assert.equal(f.el('email-workspace-gate').hidden,false);
  } finally { f.close(); }
}
// Read deadlines remain understandable through the workspace fetch wrapper.
for (const phase of ['headers', 'body']) {
  const f=setup({id:'alice', email:'alice@example.com'}); await flush();
  try {
    f.setHandler(async(_input, options)=>{
      const interrupt=(reject)=>{
        const abort=()=>reject(options.signal.reason);
        if (options.signal.aborted) abort();
        else options.signal.addEventListener('abort', abort, {once:true});
      };
      return phase === 'headers'
        ? new Promise((_resolve,reject)=>interrupt(reject))
        : new Response(new ReadableStream({start(stream){interrupt(reason=>stream.error(reason));}}));
    });
    const result=await Promise.race([
      f.w.apiRequest('/api/fixture', {timeoutMs:5}).then(
        ()=>({completed:true}), error=>({error}),
      ),
      new Promise(resolve=>setTimeout(()=>resolve({stalled:true}),50)),
    ]);
    assert.equal(result.stalled, undefined, `A stalled response ${phase} must honor the read deadline`);
    assert.match(result.error?.message || '', /Request timed out/);
    assert.doesNotMatch(result.error?.message || '', /aborted without reason/);
  } finally { f.close(); }
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
// Back/Forward-cache restoration reloads once, revalidating the session and
// rebuilding streams/channels that pagehide stopped. Ordinary pageshow is inert.
{
  const f=setup({id:'alice',email:'alice@example.com'});await flush();
  try {
    f.w.dispatchEvent(new f.w.PageTransitionEvent('pageshow',{persisted:false}));
    assert.equal(f.navigations.length,0);
    assert.equal(f.el('email-workspace-content').hidden,false);
    f.setHandler((_input,options)=>new Promise((_resolve,reject)=>
      options.signal.addEventListener('abort',()=>reject(options.signal.reason),{once:true})));
    const pending=f.w.apiRequest('/api/fixture',{timeoutMs:1000});
    const interrupted=assert.rejects(pending,error=>error.name==='AbortError');
    f.w.dispatchEvent(new f.w.PageTransitionEvent('pagehide',{persisted:true}));
    f.w.dispatchEvent(new f.w.PageTransitionEvent('pageshow',{persisted:true}));
    await interrupted;
    assert.equal(f.navigations.length,1);
    assert.equal(f.el('email-workspace-content').hidden,true);
    assert.equal(f.el('workspace-message').textContent,'Reconnecting…');
    assert.equal(f.w.SkynetRefresh.stopped,true);
    await assert.rejects(f.w.fetch('/api/fixture'),error=>error.name==='AbortError');
    f.w.dispatchEvent(new f.w.PageTransitionEvent('pageshow',{persisted:true}));
    assert.equal(f.navigations.length,1,'restoration requests only one full reload');
    assert.deepEqual(f.errors,[]);
  } finally {f.close();}
}
// A sign-out in one tab immediately stops private refreshes in every old tab.
{
  const channels=[],messages=[];
  class Channel {
    constructor(name){this.name=name;channels.push(this);}
    postMessage(value){
      messages.push(value);
      for(const peer of channels)if(peer!==this && !peer.closed && peer.name===this.name)
        peer.onmessage?.({data:value});
    }
    close(){this.closed=true;}
  }
  const first=setup({id:'alice',email:'alice@example.com'}, {}, null, Channel);
  const second=setup({id:'alice',email:'alice@example.com'}, {}, null, Channel);
  try {
    await flush();
    second.setHandler((_input,options)=>new Promise((_resolve,reject)=>
      options.signal.addEventListener('abort',()=>reject(options.signal.reason),{once:true})));
    const pending=second.w.apiRequest('/api/fixture',{timeoutMs:1000});
    const interrupted=assert.rejects(pending,error=>error.name==='AbortError');
    first.setHandler(async()=>Response.json({workspace:null}));
    first.el('workspace-sign-out').click();await flush();await interrupted;
    assert.equal(JSON.stringify(messages),JSON.stringify([{type:'session-changed'}]),'Only a control signal is broadcast');
    for(const tab of [first,second]){
      assert.equal(tab.w.SkynetRefresh.stopped,true);
      assert.equal(tab.el('email-workspace-content').hidden,true);
      await assert.rejects(tab.w.fetch('/api/fixture'),error=>error.name==='AbortError');
      tab.w.dispatchEvent(new tab.w.Event('pagehide'));
    }
    assert.ok(channels.every(channel=>channel.closed));
    assert.deepEqual(first.errors,[]);assert.deepEqual(second.errors,[]);
  } finally {first.close();second.close();}
}
console.log('Email workspace UI: gate, real script initialization, validation, switching, stale tabs and cross-tab sign-out passed');
