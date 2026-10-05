import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const app = fs.readFileSync('static/app.js', 'utf8');
const source = app.slice(app.indexOf('function connectWorkspaceChanges('), app.indexOf('function initializeLiveRefresh('));
const channels = new Map(), queues = new Map(), owners = new Set(), streams = [];
class Channel {
  constructor(key) { this.key=key; const peers=channels.get(key)||new Set();peers.add(this);channels.set(key,peers); }
  postMessage(data) { for(const peer of channels.get(this.key)||[]) if(peer!==this) queueMicrotask(()=>peer.onmessage?.({data})); }
  close() { channels.get(this.key).delete(this);this.onmessage=null; }
}
function drain(key) {
  if(owners.has(key))return;
  const entry=queues.get(key)?.shift();if(!entry)return;
  owners.add(key);entry.active=true;
  Promise.resolve().then(entry.callback).then(entry.resolve,entry.reject).finally(()=>{owners.delete(key);drain(key);});
}
const locks={request(key,{signal},callback){return new Promise((resolve,reject)=>{
  const entry={callback,resolve,reject,active:false};const queue=queues.get(key)||[];queue.push(entry);queues.set(key,queue);
  signal.addEventListener('abort',()=>{if(!entry.active){const i=queue.indexOf(entry);if(i>=0)queue.splice(i,1);reject(new DOMException('stopped','AbortError'));}},{once:true});drain(key);
});}};
class Stream {
  constructor(url){this.url=url;this.events={};this.closed=false;streams.push(this);}
  addEventListener(name,handler){this.events[name]=handler;}
  emit(name,data){this.events[name]?.({data:JSON.stringify(data)});}
  close(){this.closed=true;}
}
const window={EventSource:Stream,BroadcastChannel:Channel,navigator:{locks}};
const context=vm.createContext({window,AbortController,setTimeout,clearTimeout,encodeURIComponent});vm.runInContext(source,context);
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const received=Array.from({length:6},()=>[]);
const stops=received.map(list=>context.connectWorkspaceChanges('workspace-A',{
  resync:()=>list.push('resync'),change:topics=>list.push([...topics]),connection:value=>list.push(value)
}));
await tick();assert.equal(streams.length,1,'six open tabs must share one connection');
streams[0].emit('resync');streams[0].emit('change',{v:1,topics:['data']});await tick();
for(const list of received){assert.ok(list.includes('resync'));assert.ok(list.some(v=>Array.isArray(v)&&v[0]==='data'));}
const late=[];const stopLate=context.connectWorkspaceChanges('workspace-A',{resync:()=>{},change:()=>{},connection:v=>late.push(v)});await tick();assert.ok(late.includes(true),'a later tab inherits connection readiness');stopLate();
stops[2]();await tick();assert.equal(streams.length,1,'closing a follower must not replace the leader');
stops[0]();await tick();await tick();assert.equal(streams.length,2,'closing the leader transfers the stream to another tab');assert.ok(streams[0].closed);
const oldLength=received[2].length;streams[1].emit('change',{v:1,topics:['settings']});await tick();assert.equal(received[2].length,oldLength,'closed tabs receive nothing');
const other=[];const stopOther=context.connectWorkspaceChanges('workspace-B',{resync:()=>other.push('resync'),change:t=>other.push(t),connection:()=>{}});await tick();
assert.equal(streams.length,3,'different workspaces are isolated');streams[1].emit('change',{v:1,topics:['data']});await tick();assert.deepEqual(other,[]);
for(const stop of stops)stop();stopOther();await tick();assert.ok(streams.every(s=>s.closed));
console.log('Shared live updates: six tabs, delivery, workspace isolation, leader failover and cleanup passed.');
