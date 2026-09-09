import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {JSDOM} from 'jsdom';
const w = new JSDOM(await readFile(new URL('../static/index.html', import.meta.url),'utf8'), {runScripts:'outside-only', pretendToBeVisual:true, url:'http://localhost:8080/#runs'}).window;
const observers=[];const NativeObserver=w.MutationObserver;
w.MutationObserver=class extends NativeObserver{constructor(callback){super(callback);observers.push(this);}};
const el=id=>w.document.getElementById(id);
w.fetch=()=>new Promise(()=>{});
w.scrollTo=w.HTMLElement.prototype.scrollIntoView=()=>{};
w.matchMedia=()=>({matches:false,addEventListener(){},removeEventListener(){}});
w.CSS={escape:s=>s};
w.HTMLDialogElement.prototype.showModal=function(){this.open=true;};
w.HTMLDialogElement.prototype.close=function(value=''){if(this.open){this.open=false;this.returnValue=value;this.dispatchEvent(new w.Event('close'));}};
const flush=async()=>{for(let i=0;i<4;i++)await new Promise(r=>setImmediate(r));};
try{
 for(const file of ['dialogs.js','workspace-navigation.js','app.js'])w.eval((await readFile(new URL('../static/'+file,import.meta.url),'utf8')) + (file==='app.js' ? '\nwindow.setupAttemptTest=()=>{activeRunDetailId="test-run";elements.runDetail.hidden=false;};' : ''));
 w.api=async()=>({content:'sample log'});
 w.setupAttemptTest();
 const payload={run:{id:'test-run',status:'RUNNING',stages:[
   {id:'train-stage',stage_type:'TRAIN'},
   {id:'evaluation-stage',stage_type:'EVALUATE'},
 ],attempts:[
   {id:'attempt-1',stage_id:'train-stage',attempt_number:1,status:'RUNNING',slurm_job_id:'123'},
   {id:'evaluation-attempt',stage_id:'evaluation-stage',attempt_number:1,status:'SUCCEEDED',slurm_job_id:'456'},
 ]}};
 w.renderRunDetailContent(payload,'test-run');
 assert.equal(el('attempts-body').rows.length,1,'training detail excludes evaluation attempts');
 assert.doesNotMatch(el('attempts-body').textContent,/456/);
 assert.equal(w.runAttemptCount(payload.run),1,'attempt count reflects training only');
 assert.equal(w.runAttemptRecords({attempts:[{id:'legacy'}]}).length,1,'legacy training attempts remain visible');
 assert.equal(w.runAttemptRecords({stages:[{id:'utility',stage_type:'UTILITY'}],attempts:[{id:'utility-attempt',stage_id:'utility'}]}).length,1,'utility runs keep their own attempts');
 const launch=el('attempts-body').querySelector('button');
 launch.click();await flush();
 assert.equal(el('run-attempt-detail-dialog').open,true);
 assert.equal(el('run-attempt-detail').closest('tr'),null);
 assert.equal(launch.textContent,'Detail');
 w.renderRunDetailContent(payload,'test-run',{preserveAttempt:true});
 w.remountActiveRunAttemptDisclosure();
 assert.equal(el('run-attempt-detail-dialog').open,true,'refresh keeps modal open');
 el('run-attempt-detail-dialog').querySelector('[data-dialog-close]').click();
 assert.equal(el('run-attempt-detail-dialog').open,false);
 assert.equal(el('run-attempt-detail').hidden,true);
 w.renderRunDetailContent(payload,'test-run',{preserveAttempt:true});
 assert.equal(el('run-attempt-detail-dialog').open,false,'refresh does not reopen dismissed details');
 el('attempts-body').querySelector('button').click();await flush();
 w.document.dispatchEvent(new w.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true}));
 assert.equal(el('run-attempt-detail-dialog').open,false);
 console.log('Attempt modal: opening, close, Escape, stable table and refresh behavior passed.');
}finally{for(const observer of observers)observer.disconnect();await flush();w.close();}
