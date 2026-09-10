import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {JSDOM} from "jsdom";
const w = new JSDOM('<section id="experiment-model-io"></section><section id="adapter-model-io"></section>',{runScripts:"outside-only"}).window;
const app=await readFile(new URL('../static/app.js',import.meta.url),'utf8');
for (const name of ['escapeHtml','keyValueHtml']) {
 const start=app.indexOf(`function ${name}(`);
 w.eval(app.slice(start,app.indexOf('\nfunction ',start+1)));
}
w.eval(app.slice(app.indexOf('const modelIORequests ='), app.indexOf('function trainingAdapterLabel(')));
const el=id=>w.document.getElementById(id);
const pending=[],calls=[];
w.api=(url,options)=>{calls.push(JSON.parse(options.body));return new Promise(resolve=>pending.push(resolve));};
w.selectedAdapter=()=>({});
w.adapterManifest=()=>({slug:'test',train:{input_fields:[{path:'secret',default:'do-not-send',sensitive:true}]}});
w.declaredAdapterInputFields=()=>[];
w.elements={nativeOverrides:{value:''},adapterManifest:{value:'{}'}};
w.selectedExperimentDataBundle=()=>({id:'dataset'});
w.experimentBundleCompatibility=()=>({compatible:true});
const tick=()=>new Promise(r=>setTimeout(r,220));
try {
 w.refreshExperimentModelIO(); await tick();
 assert.equal(calls.at(-1).bundle_id,'dataset');
 assert.deepEqual(calls.at(-1).manifest.train.input_fields,[],'secret defaults never sent');
 w.refreshExperimentModelIO();await tick();
 pending[1]({entries:[['New shape','25 × 28']],note:'Current dataset'});
 await tick();
 pending[0]({entries:[['Old shape','100 × 28']],note:'Stale dataset'});
 await tick();
 assert.match(el('experiment-model-io').textContent,/25 × 28/);
 assert.doesNotMatch(el('experiment-model-io').textContent,/100 × 28/);
 w.refreshExperimentModelIO();await tick();
 w.experimentBundleCompatibility=()=>({compatible:false});
 w.refreshExperimentModelIO();
 pending[2]({entries:[['Stale shape','100 × 28']]});await tick();
 assert.match(el('experiment-model-io').textContent,/Choose a compatible dataset/);
 assert.doesNotMatch(el('experiment-model-io').textContent,/100 × 28/);
 w.renderModelIO(el('adapter-model-io'),{entries:[['<img>','<script>']],note:'<b>'});
 assert.equal(el('adapter-model-io').querySelector('script,img,b'),null);
 w.refreshAdapterModelIO();await tick();
 w.elements.adapterManifest.value='{';w.refreshAdapterModelIO();
 pending.at(-1)({entries:[['Stale editor','20']]});await tick();
 assert.match(el('adapter-model-io').textContent,/Enter a valid manifest/);
 console.log('Model I/O UI: debounce, stale responses, invalid input, compatibility and escaping passed.');
} finally {w.close();}
