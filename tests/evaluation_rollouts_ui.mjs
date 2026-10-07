import assert from 'node:assert/strict';
import {pageWindow, byId, flusher, spyMutationObservers, stubBrowserApis, polyfillDialogs, readStatic} from './ui_harness.mjs';
const w = pageWindow({url:'http://localhost:8080/#evaluations'});
const observers = spyMutationObservers(w);
const el = byId(w);
stubBrowserApis(w, {cssEscape:true});
w.HTMLMediaElement.prototype.pause=function(){};
w.HTMLMediaElement.prototype.load=function(){this.loads=(this.loads||0)+1;};
polyfillDialogs(w, {returnValue:false});
const flush = flusher(5);
const base={id:'eval',run_id:'run',status:'RUNNING',suite_name:'dexverse_recorded'};
const episodes=[{id:'ep-a',task:'cube',episode_index:0,seed:0,status:'SUCCEEDED',success:false,video_path:'/videos/a.mp4'}, {id:'ep-b',task:'stick',episode_index:0,seed:1,status:'PENDING',success:null}];
try {
 for(const file of ['dialogs.js','workspace-navigation.js','connection-settings.js','app.js']) w.eval(await readStatic(file)+(file==='app.js'?'\nwindow.setupEvaluationTest=rows=>{activeTab="evaluation-runs";evaluationRows=rows;renderEvaluations();};window.prepareEvaluationSubmission=()=>{evaluationTargetValidationState={pending:false,valid:true,signature:evaluationTargetSignature()};};':''));
 let resolveDetail; const logReplies=[];
 w.api=url=>url.includes('/logs') ? new Promise(resolve=>logReplies.push(resolve)) : new Promise(resolve=>{resolveDetail=resolve;});
 w.setupEvaluationTest([base]);
 const button=el('evaluations-body').querySelector('[data-evaluation-action]');
 button.click();
 assert.equal(el('evaluation-detail').hidden,false,'panel opens before network returns');
 assert.match(el('evaluation-rollouts-body').textContent,/Loading/);
 assert.match(el('evaluation-rollout-count').textContent,/Loading/);
 resolveDetail({evaluation:{...base,episodes,attempts:[{id:'a',slurm_job_id:'123',status:'RUNNING',attempt_number:1}]}});
 await flush();
 assert.equal(el('evaluation-rollouts-body').rows.length,2,'all episodes appear including pending');
 assert.match(el('evaluation-rollouts-body').rows[0].textContent,/0%/);
 assert.equal(logReplies.length,0,'opening results never waits for or downloads logs');
 // Lifecycle actions refresh even while the Results disclosure preserves its launcher.
 const evaluationRow=()=>el('evaluations-body').querySelector('[data-evaluation-id="eval"]');
 for(const status of ['CREATED','SUBMITTING','PENDING','PENDING_SLURM','RUNNING','RETRY_PENDING']) {
   w.updateEvaluationRow({...base,status});
   assert.equal(evaluationRow().querySelector('[data-cancel-action]').textContent,'Cancel');
   assert.equal(evaluationRow().querySelector('[data-delete-kind]'),null);
 }
 // History refresh must not reload submission catalogs or training suggestions.
 const oldSuites=w.loadEvaluationSuites, oldHistoryApi=w.api;
 const historyCalls=[];
 w.loadEvaluationSuites=()=>{throw new Error('History loaded submission catalogs');};
 w.api=async url=>{historyCalls.push(url);assert.equal(url,'/api/evaluations');return {evaluations:[base]};};
 await w.loadEvaluations(true);
 assert.deepEqual(historyCalls,['/api/evaluations']);
 w.loadEvaluationSuites=oldSuites;w.api=oldHistoryApi;
 const originalApi=w.api, originalConfirm=w.askUserDialog;
 const cancelCalls=[];
 w.api=(url,options)=>{cancelCalls.push({url,options});return Promise.reject(new Error('offline'));};
 w.askUserDialog=async()=>false;
 evaluationRow().querySelector('[data-cancel-action]').click();await flush();
 assert.equal(cancelCalls.length,0,'dismissing confirmation sends no cancellation');
 w.askUserDialog=async()=>true;
 evaluationRow().querySelector('[data-cancel-action]').click();await flush();
 assert.equal(cancelCalls.length,1);
 assert.equal(cancelCalls[0].url,'/api/evaluations/eval/cancel');
 assert.equal(cancelCalls[0].options.method,'POST','row Cancel uses the existing cancellation endpoint');
 assert.equal(evaluationRow().querySelector('[data-cancel-action]').disabled,false,'failed cancellation restores the row button');
 w.api=originalApi;w.askUserDialog=originalConfirm;
 w.updateEvaluationRow({...base,status:'CANCELLING'});
 assert.equal(evaluationRow().querySelector('[data-cancel-action]').textContent,'Cancelling…');
 assert.equal(evaluationRow().querySelector('[data-cancel-action]').disabled,true);
 assert.equal(evaluationRow().querySelector('[data-delete-kind]'),null);
 for(const status of ['CANCELLED','SUCCEEDED','FAILED']) {
   w.updateEvaluationRow({...base,status});
   assert.equal(evaluationRow().querySelector('[data-cancel-action]'),null);
   assert.equal(evaluationRow().querySelector('[data-delete-kind]').textContent,'Delete');
   assert.equal(evaluationRow().querySelector('[data-evaluation-action]'),button,'status changes preserve the open Results launcher');
   assert.equal(el('evaluation-detail').hidden,false);
 }
 w.updateEvaluationRow(base);

 const launch=el('evaluation-rollouts-body').querySelector('button'); launch.click();
 assert.equal(el('evaluation-rollout-dialog').open,true,'modal opens before log requests complete');
 assert.match(el('evaluation-rollout-video').src,/ep-a\/video$/);
 assert.ok(el('evaluation-result-summary').querySelector('table'));
 assert.ok(el('evaluation-attempt-meta').querySelector('table'));
 const loads=el('evaluation-rollout-video').loads;
 w.renderEvaluationRollouts({...base,episodes,attempts:[]});
 assert.equal(el('evaluation-rollout-video').loads,loads,'polling keeps current playback');
 el('evaluation-rollout-dialog').querySelector('[data-dialog-close]').click();
 assert.equal(el('evaluation-rollout-dialog').open,false);
 for(const resolve of logReplies) resolve('stale first-episode logs');
 await flush();
 assert.doesNotMatch(el('evaluation-stdout-log').textContent,/stale/,'closed modal rejects stale logs');
 el('evaluation-rollouts-body').querySelectorAll('button')[1].click();
 assert.equal(el('evaluation-rollout-dialog').open,true);
 assert.equal(el('evaluation-rollout-video').hidden,true);
 assert.match(el('evaluation-rollout-title').textContent,/stick/);
 w.document.dispatchEvent(new w.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true}));
 assert.equal(el('evaluation-rollout-dialog').open,false,'Escape closes shared modal');
 const other={...base,id:'other-eval'};
 w.setupEvaluationTest([{...base,episodes},other]);
 const otherButton=el('evaluations-body').querySelector('[data-id="other-eval"]');
 // Browsers deliver mutation observers between capture and bubble listeners.
 // The new disclosure must survive the previous row's queued hidden mutation.
 w.toggleDisclosure(el('evaluation-detail'),'evaluation:other-eval',otherButton,{rowOwned:true});
 await flush();
 void w.viewEvaluation('other-eval',otherButton);
 assert.equal(el('evaluation-detail-dialog').open,true,'switching evaluations opens the common modal');
 assert.equal(el('evaluation-detail').hidden,false,'one click switches to another evaluation');
 assert.equal(el('evaluation-detail-title').textContent,w.evaluationName(other),'the newly clicked evaluation opens immediately');
 assert.equal(el('evaluation-detail').closest('tbody'),null,'result content stays outside the scrolling table');
 assert.equal(el('evaluation-detail').closest('dialog'),el('evaluation-detail-dialog'));
 assert.equal(otherButton.textContent,'Results','the modal launcher keeps its action label');
 resolveDetail({evaluation:{...other,episodes:[]}});
 await flush();
 assert.match(el('evaluation-rollouts-body').textContent,/No rollouts recorded/,'empty results replace the loading state');
 assert.equal(el('evaluation-rollout-count').textContent,'0 episodes');
 assert.equal(el('evaluation-detail').hidden,false,'switch remains open after the response');
 assert.ok(el('evaluation-search').closest('.panel-heading-tools .toolbar'),'evaluation filters reuse the training toolbar');
 const parallel=el('evaluation-parallelism').closest('.field');
 assert.equal(parallel.nextElementSibling.querySelector('input').id,'evaluation-max-attempts');
 // A different training run must resolve its own checkpoint.
 el('evaluation-checkpoint').value='/runs/previous/checkpoints/last.ckpt';
 el('evaluation-run-id').value='new-run';
 el('evaluation-run-id').dispatchEvent(new w.Event('input',{bubbles:true}));
 assert.equal(el('evaluation-checkpoint').value,'','switching runs clears the previous checkpoint');
 const prediction={...episodes[0],success:null,metrics_json:{joint_mse:0.125}};
 w.renderEvaluationRollouts({...base,episodes:[prediction],attempts:[]});
 assert.match(el('evaluation-rollouts-body').textContent,/joint_mse: 0.125/,'prediction results display their measured error');
 el('evaluation-rollouts-body').querySelector('button').click();
 assert.match(el('evaluation-result-summary').textContent,/joint_mse/);
 el('evaluation-rollout-dialog').querySelector('[data-dialog-close]').click();
 // Submission must use the same row disclosure as a Results click, including
 // replacing an already-open evaluation and clearing filters hiding the new row.
 w.activateTab=async()=>{};
 w.scheduleEvaluationTargetValidation=()=>{};
 w.evaluationTargetSignature=()=> 'submission-test';
 w.evaluationPayload=()=>({run_id:'run',seeds:[0]});
 el('evaluation-form').reportValidity=()=>true;
 for(const id of ['submitted-with-open-row','submitted-with-closed-row']) {
   const submitted={...base,id};
   el('evaluation-search').value='unrelated run';
   el('evaluation-state-filter').value='FAILED';
   w.prepareEvaluationSubmission();
   w.loadEvaluations=async()=>{throw new Error('A successful submission must not depend on history loading');};
   w.api=(url,options)=>options?.method==='POST'
     ? Promise.resolve({evaluation:submitted})
     : new Promise(resolve=>{resolveDetail=resolve;});
   await w.createEvaluation(new w.Event('submit',{cancelable:true}));
   await flush();
   assert.equal(el('evaluation-search').value,'');
   assert.equal(el('evaluation-state-filter').value,'all');
   const row=el('evaluations-body').querySelector(`[data-evaluation-id="${id}"]`);
   assert.equal(el('evaluation-detail').closest('dialog'),el('evaluation-detail-dialog'),'submitted details use the common modal');
   assert.equal(el('evaluation-detail').hidden,false,'submission opens before detail response');
   assert.equal(el('evaluation-detail-dialog').open,true);
   resolveDetail({evaluation:{...submitted,episodes:[]}});
   await flush();
   w.setupEvaluationTest([submitted,base,other]);
   assert.equal(el('evaluation-detail').closest('dialog'),el('evaluation-detail-dialog'),'refresh preserves the modal placement');
   row.querySelector('[data-evaluation-action]').click();
   assert.equal(el('evaluation-detail').hidden,true,'the submitted row closes with one click');
 }
 // Runtime failures reuse the existing canonical retry action and evaluation ID.
 const failed={...base,status:'FAILED',manual_actions:{retry_submission:{enabled:true,label:'Retry evaluation'}},episodes:[]};
 w.setupEvaluationTest([failed]);
 w.api=async()=>({evaluation:failed});
 el('evaluations-body').querySelector('[data-evaluation-action]').click();
 await flush();
 const retry=el('evaluation-detail-actions').querySelector('[data-evaluation-retry-submission]');
 assert.equal(retry.textContent,'Retry evaluation');
 let acceptRetry; const retryCalls=[];
 w.api=(url,options)=>{retryCalls.push({url,options});return options?.method==='POST'
   ? new Promise(resolve=>{acceptRetry=resolve;})
   : Promise.resolve({evaluation:{...failed,status:'PENDING',manual_actions:{retry_submission:{enabled:false}},episodes:[]}});};
 retry.click();retry.click();
 assert.equal(retryCalls.length,1,'disabled action prevents duplicate submission');
 assert.equal(retryCalls[0].url,'/api/evaluations/eval/retry-submission');
 assert.equal(retryCalls[0].options.method,'POST');
 acceptRetry({evaluation:{...failed,status:'PENDING'}});await flush();
 assert.equal(el('evaluation-detail-actions').querySelector('[data-evaluation-retry-submission]'),null,'active retry cannot be submitted again');
 // A succeeded job whose result read failed re-reads result.json instead of rerunning episodes.
 const unread={...base,id:'unread',status:'FAILED',manual_actions:{retry_submission:{enabled:false},reread_result:{enabled:true,label:'Re-read result'}},episodes:[]};
 w.setupEvaluationTest([unread]);
 w.api=async()=>({evaluation:unread});
 el('evaluations-body').querySelector('[data-evaluation-action]').click();
 await flush();
 assert.equal(el('evaluation-detail-actions').querySelector('[data-evaluation-retry-submission]'),null,'no rerun is offered for a succeeded job');
 const reread=el('evaluation-detail-actions').querySelector('[data-evaluation-reread-result]');
 assert.equal(reread.textContent,'Re-read result');
 let acceptReread; const rereadCalls=[];
 w.api=(url,options)=>{rereadCalls.push({url,options});return options?.method==='POST'
   ? new Promise(resolve=>{acceptReread=resolve;})
   : Promise.resolve({evaluation:{...unread,status:'SUCCEEDED',manual_actions:{reread_result:{enabled:false}},episodes:[]}});};
 reread.click();reread.click();
 assert.equal(rereadCalls.length,1,'disabled action prevents a duplicate read');
 assert.equal(rereadCalls[0].url,'/api/evaluations/unread/reread-result');
 assert.equal(rereadCalls[0].options.method,'POST');
 acceptReread({evaluation:{...unread,status:'SUCCEEDED'}});await flush();
 assert.equal(el('evaluation-detail-actions').querySelector('[data-evaluation-reread-result]'),null,'a recovered result cannot be read again');
 console.log('Evaluation UI: immediate panel, rollout rows, modal, playback, row switching, submission placement and field order passed.');
} finally {for(const observer of observers)observer.disconnect();await flush();w.close();}
