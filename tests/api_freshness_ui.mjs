import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {JSDOM} from 'jsdom';
const html = await readFile(new URL('../static/index.html', import.meta.url), 'utf8');
const source = await readFile(new URL('../static/app.js', import.meta.url), 'utf8');
const flush = async () => { for (let i=0;i<8;i++) await new Promise(resolve=>setImmediate(resolve)); };
const deferred = () => { let resolve; const promise = new Promise(r=>{resolve=r;}); return {promise,resolve}; };
const w = new JSDOM(html,{url:'http://localhost:8080/#cluster',runScripts:'outside-only',pretendToBeVisual:true}).window;
const observers=[], Observer=w.MutationObserver;
w.MutationObserver=class extends Observer {constructor(fn){super(fn);observers.push(this);}};
for(const key of ['Headers','Request','Response','AbortController','AbortSignal'])w[key]=globalThis[key];
w.scrollTo=w.HTMLElement.prototype.scrollIntoView=()=>{};
w.matchMedia=()=>({matches:false,addEventListener(){},removeEventListener(){}});
w.CSS={escape:value=>value};
w.SkynetWorkspace={id:'test-workspace',storageKey:key=>'test:'+key};
const streams=[];
w.EventSource=class {
 constructor(url){this.url=url;this.listeners=new Map();streams.push(this);}
 addEventListener(type,callback){this.listeners.set(type,callback);}
 emit(type,value={}){this.listeners.get(type)?.({data:JSON.stringify(value)});}
 close(){this.closed=true;}
};
let handle=async()=>Response.json({jobs:[],account_usage:[],gateway:'sky2'});
let fetchCalls=[];
w.fetch=(path,options={})=>{fetchCalls.push([String(path),options.method||'GET']);return handle(String(path),options);};
const el=id=>w.document.getElementById(id);
try {
 for(const file of ['dialogs.js','workspace-navigation.js','connection-settings.js'])w.eval(await readFile(new URL('../static/'+file,import.meta.url),'utf8'));
 w.eval(source+`\nwindow.testPage=page=>{activeTab=page;};window.testDataState=()=>trainingDatasetRows.map(row=>row.id);`);
 await flush();
 assert.equal(streams.length,1);
 assert.equal(streams[0].url,'/api/changes?expected_workspace=test-workspace');
 // A post-write reader cannot join a pre-write request; the earlier caller also
 // receives the committed snapshot instead of reviving deleted rows.
 const old=deferred();let reads=0;
 handle=async(path,options)=>{
  if(options.method==='POST')return Response.json({resource:{id:'new'}});
  if(path==='/api/data/resources')return ++reads===1?old.promise:Response.json({resources:[{id:'new'}]});
  return Response.json({});
 };
 const before=w.apiRequest('/api/data/resources');await flush();
 await w.apiRequest('/api/data/resources',{method:'POST',body:'{}'});
 const after=w.apiRequest('/api/data/resources');await flush();
 old.resolve(Response.json({resources:[{id:'deleted'}]}));
 assert.equal((await before).resources[0].id,'new');
 assert.equal((await after).resources[0].id,'new');
 assert.equal(reads,2,'one old request and one shared post-commit request');
 // The invalidation barrier also covers a response body arriving after headers.
 let body,bodyReads=0;
 handle=async(path,options)=>{
  if(options.method==='POST')return Response.json({});
  if(++bodyReads===1)return new Response(new ReadableStream({start(controller){body=controller;}}));
  return Response.json({name:'committed'});
 };
 const bodyRead=w.apiRequest('/api/data/body');await flush();
 await w.apiRequest('/api/data/body',{method:'POST',body:'{}'});
 body.enqueue(new TextEncoder().encode('{"name":"stale"}'));body.close();
 assert.equal((await bodyRead).name,'committed');assert.equal(bodyReads,2);
 // One in-flight refresh plus one trailing refresh for many changes, with no TTL.
 const coordinator=w.createRefreshCoordinator();let runs=0;const slow=deferred();
 coordinator.register('test',['data'],()=>true,async()=>{if(++runs===1)await slow.promise;});
 coordinator.invalidate(['data']);await flush();
 coordinator.invalidate(['data']);coordinator.invalidate(['data']);await flush();
 assert.equal(runs,1);slow.resolve();await coordinator.flush();assert.equal(runs,2);
 // Keep the gate and callback in the coordinator's realm: cross-realm promises
 // change microtask order and can hide an invalidation just before finalization.
 const tailRace=w.eval(`(() => {
   const c=createRefreshCoordinator();let resolve,reads=0;
   const gate=new Promise(done=>{resolve=done;});
   c.register('tail',['data'],()=>true,()=>{reads++;return gate;});
   const first=c.flush();
   gate.then(()=>c.invalidate(['data']));
   return {finish:async()=>{resolve();await first;},reads:()=>reads};
 })()`);
 await tailRace.finish();await flush();
 assert.equal(tailRace.reads(),2,'an invalidation before running clears gets a trailing read');
 const failed=w.createRefreshCoordinator();let failures=0;
 failed.register('failure',['data'],()=>true,async()=>{failures++;throw new Error('offline');});
 failed.invalidate(['data']);await flush();assert.equal(failures,1,'failed reads do not spin');
 failed.invalidate(['data']);await flush();assert.equal(failures,2,'a later event retries failed reads');
 let visible=false,hiddenReads=0;
 coordinator.register('hidden',['settings'],()=>visible,async()=>{hiddenReads++;});
 coordinator.invalidate(['settings']);await flush();assert.equal(hiddenReads,0);
 visible=true;await coordinator.flush();assert.equal(hiddenReads,1);
 const closing=w.createRefreshCoordinator(),closingRead=deferred();let closingRuns=0;
 closing.register('closing',['data'],()=>true,async()=>{closingRuns++;await closingRead.promise;});
 closing.invalidate(['data']);await flush();closing.invalidate(['data']);closing.stop();
 closingRead.resolve();await flush();assert.equal(closingRuns,1,'workspace leave stops trailing refreshes');
 // New prepared data reaches an already-open training form and preserves edits.
 const dataset=id=>({id,name:id,assignments:[],selection:{version_id:id}});
 let datasets=[dataset('old')],selectionReads=0,registryReads=0;
 handle=async(path)=>{
  if(path==='/api/data/selections'){selectionReads++;return Response.json({datasets});}
  if(path.startsWith('/api/data/resources?')){registryReads++;assert.match(path,/include_versions=true/);return Response.json({resources:[{id:'r',category:'dataset',kind:'dataset',display_name:'Registered',source_key:'registered',versions:[]}],resource_types:{}});}
  if(path==='/api/data/imports')return Response.json({imports:[]});
  if(path==='/api/data/derivations')return Response.json({derivations:[]});
  if(path==='/api/adapters?include_archived=true')return Response.json({adapters:[]});
  return Response.json({});
 };
 w.testPage('experiments');await w.loadDataBundles(true);
 el('experiment-data-bundle').value='old';el('experiment-name').value='My unsaved experiment';
 datasets=[dataset('old'),dataset('newly-prepared')];
 streams[0].emit('change',{v:1,topics:['data']});await flush();
 assert.deepEqual([...el('experiment-data-bundle').options].map(o=>o.value),['','old','newly-prepared']);
 assert.equal(el('experiment-data-bundle').value,'old');assert.equal(el('experiment-name').value,'My unsaved experiment');
 assert.equal(registryReads,0,'hidden registry is only marked dirty');
 assert.equal(selectionReads,2);
 // On navigation the dirty registry needs exactly its 3 list requests, no N GETs.
 w.testPage('datasets');fetchCalls=[];await w.SkynetRefresh.flush();
 assert.equal(registryReads,1);
 assert.equal(fetchCalls.filter(([path])=>path.startsWith('/api/data/')).length,3);
 assert.equal(fetchCalls.some(([path])=>/^\/api\/data\/resources\//.test(path)),false);
 // Changes while the browser is hidden make no request, then focus catches up.
 Object.defineProperty(w.document,'visibilityState',{configurable:true,value:'hidden'});
 streams[0].emit('change',{v:1,topics:['data']});await flush();assert.equal(registryReads,1);
 Object.defineProperty(w.document,'visibilityState',{configurable:true,value:'visible'});
 w.document.dispatchEvent(new w.Event('visibilitychange'));await flush();assert.equal(registryReads,2);
 streams[0].emit('change',{v:2,topics:['future-domain']});await flush();
 assert.equal(registryReads,3,'unknown event version resynchronizes visible state');
 // Experiment rows are independent of a slow form catalog.
 w.testPage('experiments');const inputs=deferred();let listRequested=false;
 w.loadTrainingInputs=()=>inputs.promise;w.loadEvaluationSuites=async()=>[];w.loadTrackingConnections=async()=>{};
 w.api=async path=>{assert.equal(path,'/api/experiments');listRequested=true;return {experiments:[]};};
 await w.loadExperiments(true);assert.equal(listRequested,true);inputs.resolve();
 w.testPage('cluster');
 // A closed initial stream is explicitly retried, and unavailable closes it.
 const timers=[];const oldSet=w.setTimeout.bind(w),oldClear=w.clearTimeout.bind(w);
 w.setTimeout=(callback,delay)=>{if(delay===1000){timers.push(callback);return -99;}return oldSet(callback,delay);};
 w.clearTimeout=id=>{if(id!==-99)oldClear(id);};
 streams[0].emit('error');assert.equal(streams[0].closed,true);assert.equal(timers.length,1);
 timers.shift()();assert.equal(streams.length,2);
 streams[1].emit('resync');assert.equal(w.SkynetRefresh.connected,true);
 streams[1].emit('unavailable');assert.equal(streams[1].closed,true);
 w.dispatchEvent(new w.Event('pagehide'));await flush();
 console.log('API freshness: mutation/header/body races, trailing invalidation, prepared selections, preserved edits, bounded registry reads, focus and SSE reconnect passed.');
} finally {for(const observer of observers)observer.disconnect();w.close();}
