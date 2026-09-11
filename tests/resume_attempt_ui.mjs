import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {JSDOM} from 'jsdom';
const w=new JSDOM(await readFile(new URL('../static/index.html',import.meta.url),'utf8'),{runScripts:'outside-only',pretendToBeVisual:true,url:'http://localhost:8080/#runs'}).window;
const observers=[];const Observer=w.MutationObserver;
w.MutationObserver=class extends Observer{constructor(cb){super(cb);observers.push(this);}};
w.fetch=()=>new Promise(()=>{});
w.scrollTo=w.HTMLElement.prototype.scrollIntoView=()=>{};
w.matchMedia=()=>({matches:false,addEventListener(){},removeEventListener(){}});
w.CSS={escape:s=>s};
w.HTMLDialogElement.prototype.showModal=function(){this.open=true;};
w.HTMLDialogElement.prototype.close=function(){this.open=false;this.dispatchEvent(new w.Event('close'));};
const el=id=>w.document.getElementById(id);
try {
 for(const file of ['dialogs.js','workspace-navigation.js','connection-settings.js','app.js'])w.eval(await readFile(new URL('../static/'+file,import.meta.url),'utf8')+(file==='app.js'?'\nactiveTab="runs";':''));
 const first={id:'first',attempt_number:1,status:'FAILED',slurm_job_id:'old-job'};
 const second={id:'second',attempt_number:2,status:'SUBMITTING'};
 let submitted=false,posts=0,release;
 const run=()=>({id:'run',status:submitted?'SUBMITTING':'FAILED',attempts:submitted?[first,second]:[first],manual_actions:{resume:{enabled:!submitted}}});
 w.askUserDialog=async()=>true;
 w.api=async(path,options={})=>{
  if(options.method==='POST'){posts++;await new Promise(resolve=>{release=resolve;});submitted=true;return {run_id:'run',status:'SUBMITTING'};}
  if(path==='/api/runs')return {runs:[run()]};
  if(path==='/api/runs/run')return {run:run()};
  return {content:'log'};
 };
 await w.loadRuns(true);
 await w.viewRun('run',el('runs-body').querySelector('button'));
 const button=el('attempts-body').querySelector('button');
 await w.toggleRunAttemptDetail(button.dataset.attemptKey,button);
 assert.equal(el('run-attempt-detail-dialog').open,true);
 el('run-search').value='old-job';el('run-status-filter').value='failed';
 const request=w.resumeRun('run');
 await new Promise(resolve=>setImmediate(resolve));
 assert.match(el('run-detail-actions').textContent,/Submitting new attempt/);
 assert.equal(el('run-detail-actions').querySelector('[data-run-action="resume"]').disabled,true);
 await w.resumeRun('run');assert.equal(posts,1,'double clicking cannot create duplicate attempts');
 release();await request;
 assert.equal(el('run-search').value,'');assert.equal(el('run-status-filter').value,'all');
 assert.equal(el('attempts-body').rows.length,2,'new attempt is rendered before resume resolves, without a poll');
 assert.match(el('attempts-body').textContent,/SUBMITTING/);
 assert.match(el('run-attempt-detail-title').textContent,/2/,'open detail switches to the new attempt');
 assert.equal(el('run-attempt-detail-dialog').open,true);
 console.log('Resume: duplicate guard, immediate attempt rendering, cleared stale filters and selected new detail passed.');
}finally{for(const observer of observers)observer.disconnect();w.close();}
