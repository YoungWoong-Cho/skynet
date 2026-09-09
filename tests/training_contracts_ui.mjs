import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {JSDOM} from 'jsdom';
const w = new JSDOM(await readFile(new URL('../static/index.html', import.meta.url),'utf8'), {runScripts:'outside-only', pretendToBeVisual:true, url:'http://localhost:8080/#experiments'}).window;
const observers=[]; const Observer=w.MutationObserver;
w.MutationObserver=class extends Observer {constructor(callback){super(callback);observers.push(this);}};
w.fetch=()=>new Promise(()=>{});
w.scrollTo=w.HTMLElement.prototype.scrollIntoView=()=>{};
w.matchMedia=()=>({matches:false,addEventListener(){},removeEventListener(){}});
w.CSS={escape:s=>s};
const el=id=>w.document.getElementById(id);
const input=key=>el('adapter-field-native-config-'+key);
const manifest={schema_version:'skynet.adapter/v1',slug:'policy-test',display_name:'Test policy',runtime:{allowed_backends:['existing']},capabilities:{supports_resume:false}, defaults:{hyperparameters:{batch_size:256}},train:{
 supported_canonical_fields:['train.batch.value'], default_preset:'state/v1',
 presets:[{id:'state/v1',name:'State preset',values:{'train.batch.value':256,'native.config.observation_mode':'state'}},{id:'rgb/v1',name:'RGB preset',values:{'train.batch.value':8,'native.config.observation_mode':'rgb'}}],
 input_fields:[
 {path:'native.config.training_preset',label:'Preset',kind:'string',choices:['state/v1','rgb/v1'],default:'state/v1'},
 {path:'native.config.observation_mode',label:'Observations',kind:'string',choices:['state','rgb'],default:'state'},
 {path:'native.config.dataset',label:'Dataset',kind:'string',required:true,data_binding:{role:'training_data',position:0,formats:['test-zarr/v1'],contracts:['state/v1','rgb/v1'],contract_selector:'native.config.observation_mode',contract_choices:{state:['state/v1','rgb/v1'],rgb:['rgb/v1']},value_path:'version.path'}},
 {path:'native.config.steps',label:'Steps',kind:'integer',minimum:1,maximum:100,default:20}
 ]}};
try {
 for(const name of ['dialogs.js','collection-ui.js','app.js']) {
  let source=await readFile(new URL('../static/'+name,import.meta.url),'utf8');
  if(name==='app.js')source+=`\nwindow.configureContractTest=(manifest)=>{
    adapterRows=[{id:'adapter-test',name:'Test policy',latest_version:{id:'version-test',version_number:1,manifest}}];
    dataBundleRows=[{id:'bundle-test',name:'Imported source',version:'1',assignments:[{role:'training_data',position:0,version:{status:'READY',path:'/cluster/test',format:'test-zarr/v1',metadata:{contract:'state/v1',validation:{status:'PASSED'}}}}]}];
    populateExperimentAdapters(); applySelectedAdapter({loadSource:false}); populateExperimentDataBundles();
    elements.experimentDataBundle.value='bundle-test'; renderAdapterDeclaredFields();
  };`;
  w.eval(source);
 }
 w.configureContractTest(manifest);
 assert.equal(input('dataset').value,'/cluster/test');
 input('training_preset').value=JSON.stringify('rgb/v1');
 input('training_preset').dispatchEvent(new w.Event('change',{bubbles:true}));
 assert.equal(el('hp-batch-size').value,'8');
 assert.match(el('hp-batch-size-default').textContent,/Default: 8/);
 assert.equal(input('observation_mode').value,JSON.stringify('rgb'));
 assert.equal(input('dataset').value,'');
 assert.equal(w.validateAdapterDeclaredFields({focus:false,notify:false}),false);
 input('training_preset').value=JSON.stringify('state/v1');
 input('training_preset').dispatchEvent(new w.Event('change',{bubbles:true}));
 assert.equal(el('hp-batch-size').value,'256');
 assert.equal(input('dataset').value,'/cluster/test');
 el('hp-batch-size').value='128';
 el('hp-batch-size').dispatchEvent(new w.Event('input',{bubbles:true}));
 assert.equal(input('training_preset').value,JSON.stringify('custom'));
 assert.equal(input('training_preset').selectedOptions[0].textContent,'Custom');
 assert.equal(input('observation_mode').value,JSON.stringify('state'));
 assert.equal(el('hp-batch-size').value,'128');
 // Rerendering must preserve custom values; reselecting a preset restores it.
 w.renderAdapterDeclaredFields();
 assert.equal(input('training_preset').value,JSON.stringify('custom'));
 input('training_preset').value=JSON.stringify('rgb/v1');
 input('training_preset').dispatchEvent(new w.Event('change',{bubbles:true}));
 assert.equal(el('hp-batch-size').value,'8');
 input('observation_mode').value=JSON.stringify('state');
 input('observation_mode').dispatchEvent(new w.Event('change',{bubbles:true}));
 assert.equal(input('training_preset').value,JSON.stringify('custom'));
 assert.equal(el('hp-batch-size').value,'8');
 assert.equal(input('dataset').value,'/cluster/test');
 assert.equal(el('experiment-node-mode'),null);
 // Old loaded snapshots can retain unsupported defaults in disabled fields.
 el('hp-max-steps').value='1000';
 assert.equal(el('hp-max-steps').disabled,true);
 assert.equal(w.experimentPayload().hyperparameters.max_steps,undefined);
 assert.equal(w.experimentPayload().hyperparameters.batch_size,8);

 input('steps').value='101';
 assert.equal(w.validateAdapterDeclaredFields({focus:false,notify:false}),false);
 assert.match(input('steps').validationMessage,/maximum is 100/);
 console.log('Training contracts UI: generic preset switching, defaults, imported dataset compatibility and input bounds passed.');
} finally {for(const observer of observers)observer.disconnect();w.close();}
