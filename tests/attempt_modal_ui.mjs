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
 for(const file of ['dialogs.js','workspace-navigation.js','connection-settings.js','app.js'])w.eval((await readFile(new URL('../static/'+file,import.meta.url),'utf8')) + (file==='app.js' ? '\nwindow.setupAttemptTest=()=>{activeRunDetailId="test-run";elements.runDetail.hidden=false;};window.setupRunHistoryTest=(rows)=>{activeTab="runs";runRows=rows;renderRuns();};window.runModalContext=()=>({id:activeRunDetailId,poll:runDetailPollRunId});' : ''));
 w.api=async()=>({content:'sample log'});
 w.setupAttemptTest();
 const payload={run:{id:'test-run',status:'RUNNING',adapter_name:'egoverse-act',adapter_version:7,stages:[
   {id:'train-stage',stage_type:'TRAIN'},
   {id:'evaluation-stage',stage_type:'EVALUATE'},
 ],attempts:[
   {id:'attempt-1',stage_id:'train-stage',attempt_number:1,status:'RUNNING',slurm_job_id:'123',
    model_io:{entries:[['Input · RGB · scene_front','1 × 256 × 256 × 3'],['Output · Joint commands','100 × 28']],note:'Per sample.'},
    adapter_settings:{'native.config.epochs':2000,'native.config.validation_every':200,'native.config.reject_outliers':false,'native.config.train_batches':0,'native.config.model_overrides':{encoder:'<img src=x onerror=alert(1)>'}},
    execution_snapshot_json:{adapter:{slug:'egoverse-act',version:8,manifest:{train:{input_fields:[
      {path:'native.config.epochs',label:'Epochs'},
      {path:'native.config.validation_every',label:'Validate every epochs'},
      {path:'native.config.reject_outliers',label:'Filter training outliers'},
      {path:'native.config.train_batches',label:'Training batches per epoch'},
      {path:'native.config.model_overrides',label:'Model overrides'},
    ]}}}},
   },
   {id:'evaluation-attempt',stage_id:'evaluation-stage',attempt_number:1,status:'SUCCEEDED',slurm_job_id:'456'},
 ]}};
 w.renderRunDetailContent(payload,'test-run');
 assert.match(el('run-detail-meta').textContent,/egoverse-act · v7/);
 assert.match(w.runRowDescriptor(payload.run).cells[1].html,/egoverse-act · v7/);
 assert.equal(w.trainingAdapterLabel({adapter_name:'old-adapter',adapter_version:1},payload.run.attempts[0]),'egoverse-act · v8','attempt identity uses its pinned version');
 assert.equal(w.trainingAdapterLabel({}), 'Not recorded');
 assert.equal(w.trainingAdapterLabel({adapter_name:'legacy'}), 'legacy');
 assert.equal(el('attempts-body').rows.length,1,'training detail excludes evaluation attempts');
 assert.ok(el('run-detail-tracking').compareDocumentPosition(el('run-checkpoints')) & w.Node.DOCUMENT_POSITION_FOLLOWING, 'tracking appears above checkpoints');
 assert.equal(el('run-checkpoints').className, el('run-attempts').className, 'checkpoints and attempts share one table section');
 assert.ok(el('run-detail-actions').compareDocumentPosition(el('attempts-body')) & w.Node.DOCUMENT_POSITION_FOLLOWING, 'run actions appear above the attempts table');
 assert.equal(el('attempts-body').closest('.table-frame'), null, 'attempts have no extra table frame');

 assert.doesNotMatch(el('attempts-body').textContent,/456/);
 assert.equal(w.runAttemptCount(payload.run),1,'attempt count reflects training only');
 assert.equal(w.runAttemptRecords({attempts:[{id:'legacy'}]}).length,1,'legacy training attempts remain visible');
 assert.equal(w.runAttemptRecords({stages:[{id:'utility',stage_type:'UTILITY'}],attempts:[{id:'utility-attempt',stage_id:'utility'}]}).length,1,'utility runs keep their own attempts');
 const launch=el('attempts-body').querySelector('button');
 launch.click();await flush();
 assert.equal(el('run-attempt-detail-dialog').open,true);
 assert.match(el('run-attempt-detail-meta').textContent,/egoverse-act · v8/);
 assert.equal(el('run-attempt-detail').closest('tr'),null);
 assert.equal(launch.textContent,'Detail');
 assert.match(el('run-attempt-model-io').textContent,/256 × 256 × 3/);
 assert.match(el('run-attempt-model-io').textContent,/100 × 28/);
 assert.equal(el('run-attempt-adapter-section').hidden,false);
 const settings=Object.fromEntries([...el('run-attempt-adapter-settings').children].map(row=>[row.querySelector('span').textContent,row.querySelector('strong').textContent]));
 assert.deepEqual(settings,{'Epochs':'2000','Validate every epochs':'200','Filter training outliers':'false','Training batches per epoch':'0','Model overrides':'{"encoder":"<img src=x onerror=alert(1)>"}'});
 assert.equal(el('run-attempt-adapter-settings').querySelector('img'),null,'configuration is escaped');
 assert.doesNotMatch(el('run-attempt-hyperparameters').textContent,/Filter training outliers/,'adapter configuration has its own section');
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
 w.renderRunAttemptMetadata({attempt:{adapter_settings:{'native.config.epochs':5}},attemptNumber:2});
 assert.match(el('run-attempt-adapter-settings').textContent,/5/);
 assert.doesNotMatch(el('run-attempt-adapter-settings').textContent,/2000|Filter training outliers/,'switching attempts clears previous settings');
 w.renderRunAttemptMetadata({attempt:{},attemptNumber:3});
 assert.equal(el('run-attempt-adapter-section').hidden,true);
 assert.equal(el('run-attempt-adapter-settings').textContent,'');
 assert.doesNotMatch(el('run-attempt-model-io').textContent,/100 × 28/);
 assert.match(el('run-attempt-model-io').textContent,/Sizes were not recorded/);
 const commonKeys=['learning_rate','batch_size','batch_semantics','gradient_accumulation','num_workers','precision','max_steps'];
 w.renderRunAttemptMetadata({attempt:{adapter_settings:{'native.config.epochs':10},common_hyperparameters:Object.fromEntries(commonKeys.map(key=>[key,null])),common_hyperparameter_provenance:Object.fromEntries(commonKeys.map(key=>[key,{status:'not_applicable',source:'not_applicable'}]))},attemptNumber:4});
 assert.equal(el('run-attempt-hyperparameters').closest('section').hidden,true);
 assert.equal(el('run-attempt-adapter-section').hidden,false,'adapter settings remain visible without common hyperparameters');
 // Exercise the actual history launcher, shared parent dialog, and nested attempt dialog.
 w.closeRunAttemptDisclosure({restoreFocus:false});
 w.setupRunHistoryTest([payload.run]);
 let releaseDetail;
 w.api=path=>path==='/api/runs/test-run'
   ? new Promise(resolve=>{releaseDetail=resolve;})
   : Promise.resolve({content:'sample log'});
 const historyLaunch=el('runs-body').querySelector('[data-run-action="view"]');
 historyLaunch.focus();historyLaunch.click();await flush();
 const historyDialog=el('run-detail-dialog');
 assert.equal(historyDialog.open,true,'View opens the shared dialog while loading');
 assert.equal(historyDialog.classList.contains('app-dialog'),true);
 assert.equal(historyLaunch.getAttribute('aria-haspopup'),'dialog');
 assert.equal(historyLaunch.getAttribute('aria-controls'),historyDialog.id);
 assert.equal(historyLaunch.hasAttribute('aria-expanded'),false);
 assert.equal(historyLaunch.textContent,'View');
 assert.match(el('attempts-body').textContent,/Loading attempts/);
 assert.equal(el('run-detail').closest('dialog'),historyDialog,'run details stay inside the shared modal');
 assert.equal(el('runs-body').querySelector('.row-disclosure-companion'),null,'no inline detail row remains');
 releaseDetail(payload);await flush();
 assert.equal(el('attempts-body').rows.length,1);
 assert.equal(w.runModalContext().poll,'test-run','active run polling continues in the modal');
 assert.match(el('run-detail-actions').textContent,/Start evaluation/);
 el('attempts-body').querySelector('[data-attempt-action="view"]').click();await flush();
 assert.equal(historyDialog.open,true);
 assert.equal(el('run-attempt-detail-dialog').open,true);
 w.document.dispatchEvent(new w.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true}));
 assert.equal(el('run-attempt-detail-dialog').open,false,'Escape closes only the topmost attempt dialog');
 assert.equal(historyDialog.open,true,'parent attempt list remains open');
 w.renderRuns({background:true});
 w.renderRunDetail({...payload,run:{...payload.run,updated_at:'2026-09-12T01:00:00Z'}},'test-run',{preserveAttempt:true,background:true});
 await flush();
 assert.equal(historyDialog.open,true,'background updates keep the run dialog open');
 assert.equal(el('run-detail').closest('dialog'),historyDialog);
 assert.equal(el('run-attempt-detail-dialog').open,false,'background updates do not reopen the dismissed attempt');
 el('close-run-detail').click();await flush();
 assert.equal(historyDialog.open,false);
 assert.equal(el('run-detail').hidden,true);
 assert.equal(w.runModalContext().id,null);
 assert.equal(w.runModalContext().poll,null,'closing the run dialog stops its polling');
 await new Promise(resolve=>w.requestAnimationFrame(resolve));
 assert.equal(w.document.activeElement,historyLaunch,'closing restores focus to View');
 // A response arriving after dismissal must not reopen either dialog.
 historyLaunch.click();await flush();
 assert.equal(historyDialog.open,true);
 w.document.dispatchEvent(new w.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true}));
 releaseDetail(payload);await flush();
 assert.equal(historyDialog.open,false,'late detail responses cannot reopen a closed run');
 assert.equal(el('run-detail').hidden,true);
 assert.equal(w.runModalContext().poll,null);
 console.log('Run and attempt modals: history launcher, nested dialogs, polling, refresh, focus, stale responses and adapter details passed.');
}finally{for(const observer of observers)observer.disconnect();await flush();w.close();}
