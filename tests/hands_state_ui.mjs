import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {JSDOM} from 'jsdom';
const w = new JSDOM(await readFile(new URL('../static/index.html', import.meta.url), 'utf8'), {runScripts:'outside-only',pretendToBeVisual:true,url:'http://localhost:8080/#cluster'}).window;
const el = id => w.document.getElementById(id);
const flush = async () => { for(let i=0;i<5;i++) await new Promise(resolve=>setImmediate(resolve)); };
const records = [];
let saved = [];
let snapshot = {};
let failPoseList = false;
let modelLoads = 0;
const model = {revision:'a'.repeat(40),urdf_url:'/model.urdf',license_url:'/LICENSE',mesh_count:1,simulation_robot:'floating_shadow_left',joints:[{name:'bend',type:'revolute',lower:-Math.PI/9,upper:Math.PI/18,mimic:null}]};
const catalog = {hands:[{key:'shadow',name:'Shadow Hand',source_kind:'URDF',notes:'Model',repository:'test/test',revision:model.revision,sides:{right:'right.urdf',left:'left.urdf'},variants:{right:{state:'READY'},left:{state:'READY'}}}]};
w.AbortSignal.timeout = () => undefined;
w.fetch = async (url, options={}) => {
  records.push({url,options});
  let payload;
  if (url === '/api/hands') payload=catalog;
  else if (url.endsWith('/model')) payload=model;
  else if (url.endsWith('/poses')) {
    if (options.method === 'POST') {
      const body=JSON.parse(options.body);
      if(saved.some(p=>p.name===body.name)) return {ok:false,json:async()=>({detail:'A pose with this name already exists.'})};
      const pose={...body,id:String(saved.length+1).padStart(32,'0'),created_at:1};
      saved.push(pose); payload=pose;
    } else {
      if(failPoseList) return {ok:false,json:async()=>({detail:'Pose list temporarily unavailable'})};
      payload={poses:saved};
    }
  } else if (options.method === 'PATCH') { const pose=saved.find(p=>url.endsWith(p.id)); pose.name=JSON.parse(options.body).name; payload=pose; }
  else if (options.method === 'DELETE') { saved=saved.filter(p=>!url.endsWith(p.id)); payload={deleted:true}; }
  else throw new Error('Unexpected request '+url);
  return {ok:true,json:async()=>structuredClone(payload)};
};
w.TestHandsViewer=class {clear(){}render(){}fit(){}async load(){modelLoads++;return true;}setJoints(values){snapshot={...values};}};
w.HTMLDialogElement.prototype.showModal=function(){this.open=true;};
w.HTMLDialogElement.prototype.close=function(value=''){if(this.open){this.returnValue=value;this.open=false;this.dispatchEvent(new w.Event('close'));}};
const script=w.document.createElement("script");
script.src=w.document.querySelector('script[data-workspace-src*="hands-ui"]').dataset.workspaceSrc;
Object.defineProperty(w.document,'currentScript',{value:script});
w.eval(await readFile(new URL('../static/dialogs.js',import.meta.url),'utf8'));
w.eval((await readFile(new URL('../static/hands-ui.js',import.meta.url),'utf8')).replace('await import("/static/hands-viewer.js" + version)','({ HandsViewer: window.TestHandsViewer })'));
const changeAngle=(value,type='change')=>{el('hand-value').value=value;el('hand-value').dispatchEvent(new w.Event(type,{bubbles:true}));};
const submit=async()=>{el('hand-pose-form').dispatchEvent(new w.Event('submit',{bubbles:true,cancelable:true}));await flush();};
try {
  await w.loadHands();
  // A previous side's saved pose must never remain actionable during a new read.
  el('hand-pose-list').add(new w.Option('Previous-side pose','previous'));
  el('hand-pose-list').value='previous';el('hand-pose-list').dispatchEvent(new w.Event('change'));
  failPoseList=true;
  el('hand-side').value='left';el('hand-side').dispatchEvent(new w.Event('change'));await flush();
  assert.equal(el('hand-canvas').hidden,false,'A secondary pose-list failure preserves the successfully loaded canvas');
  assert.equal(el('hand-controls').hidden,false);
  assert.match(el('hands-error').textContent,/Saved poses could not be loaded/);
  assert.equal(el('hand-pose-list').value,'');
  assert.equal(el('hand-pose-load').disabled,true,'Old-side poses cannot be loaded while the new list is unavailable');
  const loadsBeforeRecovery=modelLoads;
  failPoseList=false;await w.loadHands(true);
  assert.equal(modelLoads,loadsBeforeRecovery,'Retry restores only saved poses, retaining the valid model');
  assert.equal(el('hand-canvas').hidden,false);
  assert.equal(el('hands-error').hidden,true);
  assert.match(el('hand-status').textContent,/adjustable joints/);
  const fetchNormally=w.fetch;
  let rejectOldPoseRead,holdOldPoseRead=true;
  w.fetch=(url,options={})=>{
    if(holdOldPoseRead&&url.endsWith('/poses')&&!options.method){
      holdOldPoseRead=false;return new Promise((_,reject)=>{rejectOldPoseRead=reject;});
    }
    return fetchNormally(url,options);
  };
  const oldRefresh=w.loadHands(true);await flush();
  el('hand-side').value='right';el('hand-side').dispatchEvent(new w.Event('change'));await flush();
  rejectOldPoseRead(new Error('Outdated left-side read failure'));await oldRefresh;
  assert.equal(el('hands-error').hidden,true,'A late refresh failure cannot replace the current side’s healthy state');
  assert.equal(el('hand-canvas').hidden,false);
  w.fetch=fetchNormally;
  el('hand-side').value='left';el('hand-side').dispatchEvent(new w.Event('change'));await flush();
  changeAngle('7.25');el('hand-pose-name').value='Unsaved grip';
  const before=snapshot.bend;
  const loadedModels=modelLoads;
  await w.loadHands(true);
  assert.equal(modelLoads,loadedModels,'Refresh does not replace a loaded pinned model or reset its camera');
  failPoseList=true;await w.loadHands(true);
  assert.equal(snapshot.bend,before,'A failed refresh preserves edits');
  failPoseList=false;await w.loadHands(true);
  assert.equal(el('hands-error').hidden,true,'A successful retry clears the obsolete error');
  assert.equal(el('hand-side').value,'left');
  assert.equal(snapshot.bend,before,'Refresh preserves exact applied joint value');
  assert.equal(el('hand-pose-name').value,'Unsaved grip');
  const selected=el('hands-catalog').querySelector('button');
  assert.equal(selected.disabled,true);selected.click();
  assert.equal(el('hand-side').value,'left');assert.equal(snapshot.bend,before);
  changeAngle('');el('hand-value').dispatchEvent(new w.Event('blur'));
  assert.equal(Number(el('hand-value').value),7.25,'Blank angle restores applied value on blur');
  assert.equal(el('hand-angle').step,'any');
  el('hand-angle').value=el('hand-angle').max;el('hand-angle').dispatchEvent(new w.Event('input'));
  assert.equal(snapshot.bend,model.joints[0].upper,'Slider can reach exact declared endpoint');
  assert.equal(Number(el('hand-angle').value),Number(el('hand-value').value),'Slider and number agree');
  el('hand-pose-name').value='   ';await submit();
  assert.match(el('hand-pose-error').textContent,/name/);assert.equal(w.document.activeElement,el('hand-pose-name'));
  assert.equal(saved.length,0);
  el('hand-pose-name').value=' Grip ';await submit();
  assert.equal(saved.length,1);assert.equal(saved[0].name,'Grip');
  assert.equal(el('hand-pose-error').hidden,true);assert.match(el('hand-pose-message').textContent,/Saved/);
  assert.match(el('hand-pose-list').selectedOptions[0].textContent,/Grip · .* · 00000000/);
  el('hand-pose-name').value='Grip';await submit();
  assert.equal(saved.length,1);assert.match(el('hand-pose-error').textContent,/already exists/);
  el('hand-pose-load').click();assert.equal(el('hand-pose-error').hidden,true);
  w.askUserDialog=async()=> 'Renamed';el('hand-pose-rename').click();await flush();
  assert.equal(saved[0].name,'Renamed');assert.match(el('hand-pose-list').selectedOptions[0].textContent,/Renamed/);
  w.askUserDialog=async()=>false;el('hand-pose-delete').click();await flush();assert.equal(saved.length,1);
  w.askUserDialog=async()=>true;el('hand-pose-delete').click();await flush();assert.equal(saved.length,0);
  console.log('Hands state: refresh/selection preservation, exact endpoint, blank input, inline errors, save/load, rename/delete passed.');
} finally {w.close();}
