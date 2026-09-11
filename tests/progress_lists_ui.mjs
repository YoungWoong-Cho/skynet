import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {JSDOM} from 'jsdom';

const w = new JSDOM(await readFile(new URL('../static/index.html', import.meta.url), 'utf8'), {
  runScripts:'outside-only', pretendToBeVisual:true, url:'http://localhost:8080/#runs',
}).window;
const observers=[];const Observer=w.MutationObserver;
w.MutationObserver=class extends Observer {constructor(callback){super(callback);observers.push(this);}};
w.fetch=()=>new Promise(()=>{});
w.scrollTo=w.HTMLElement.prototype.scrollIntoView=()=>{};
w.matchMedia=()=>({matches:false,addEventListener(){},removeEventListener(){}});
w.CSS={escape:value=>value};
const el=id=>w.document.getElementById(id);
const flush=async()=>{for(let i=0;i<5;i++)await new Promise(resolve=>setImmediate(resolve));};
try {
  for(const file of ['dialogs.js','workspace-navigation.js','app.js']) {
    w.eval(await readFile(new URL('../static/'+file,import.meta.url),'utf8')+(file==='app.js'
      ? 'window.progressPage=page=>{activeTab=page;loadedTabs.clear();};' : ''));
  }
  const timers=new Map();let timerId=-1;
  const originalSet=w.setTimeout.bind(w),originalClear=w.clearTimeout.bind(w);
  w.setTimeout=(callback,delay,...args)=>{
    if(delay!==1500&&delay!==5000)return originalSet(callback,delay,...args);
    const id=timerId--;timers.set(id,{callback,delay});return id;
  };
  w.clearTimeout=id=>{if(!timers.delete(id))originalClear(id);};
  const runTimer=async delay=>{
    const entry=[...timers].find(([,timer])=>timer.delay===delay);
    assert.ok(entry,'Pending remote progress schedules an automatic list update');
    timers.delete(entry[0]);await entry[1].callback();await flush();
  };
  let runReads=0;
  w.api=async path=>{
    assert.equal(path,runReads===0?'/api/runs':'/api/runs?refresh_progress=false');runReads++;
    return {progress_refresh_pending:runReads===1,runs:[{id:'run',experiment_name:runReads===1?'Stored row':'Enriched row',status:'FAILED'}]};
  };
  w.progressPage('runs');
  await w.loadRuns(true);
  assert.match(el('runs-body').textContent,/Stored row/,'Stored rows render before remote enrichment completes');
  await runTimer(1500);
  assert.match(el('runs-body').textContent,/Enriched row/,'The enrichment result becomes visible without another user click');
  assert.equal(runReads,2);
  assert.equal([...timers.values()].some(timer=>timer.delay===1500),false,'Completed enrichment stops its extra polling');
  runReads=0;await w.loadRuns(true);w.progressPage('settings');
  await runTimer(1500);assert.equal(runReads,1,'Leaving the workspace prevents a pending enrichment poll');

  let evaluationReads=0;
  w.loadEvaluationSuites=async()=>{};
  w.api=async path=>{
    if(path==='/api/runs')return {runs:[]};
    assert.equal(path,evaluationReads===0?'/api/evaluations':'/api/evaluations?refresh_progress=false');evaluationReads++;
    return {progress_refresh_pending:evaluationReads===1,evaluations:[{id:'evaluation',run_id:'run',
      suite_name:evaluationReads===1?'Stored evaluation':'Enriched evaluation',status:'FAILED',task_selection_json:[],seeds_json:[]}]};
  };
  w.progressPage('evaluations');await w.loadEvaluations(true);
  assert.match(el('evaluations-body').textContent,/Stored evaluation/);
  await runTimer(5000);
  assert.match(el('evaluations-body').textContent,/Enriched evaluation/);
  assert.equal(evaluationReads,2);
  assert.equal([...timers.values()].some(timer=>timer.delay===5000),false,'Terminal-only lists stop polling when enrichment settles');
  console.log('Progress lists: immediate stored rows, automatic enrichment, settled polling and workspace exit passed.');
} finally {
  for(const observer of observers)observer.disconnect();
  await flush();w.close();
}
