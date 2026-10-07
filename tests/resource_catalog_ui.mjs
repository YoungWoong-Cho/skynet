import assert from 'node:assert/strict';
import {pageWindow, byId, flusher, spyMutationObservers, stubBrowserApis, polyfillDialogs, loadScripts, readStatic} from './ui_harness.mjs';
const w = pageWindow({url:'http://localhost:8080/?data_view=registry#data'});
const observers = spyMutationObservers(w);
stubBrowserApis(w);
polyfillDialogs(w, {guarded:false, returnValue:false});
const el = byId(w), flush = flusher(5);
try{
  await loadScripts(w);el('email-workspace-content').hidden=false;
  w.eval('window.seedPresets = rows => { experimentRows=rows; renderExperiments(); };');
  const historyFixture=w.document.createElement('div');
  historyFixture.textContent='Loading…';
  w.SkynetJobHistory.render(historyFixture,{columns:['State'],rows:[{id:'attempt',cells:['Ready']}],empty:'No attempts.'});
  assert.doesNotMatch(historyFixture.textContent,/Loading/,'Initial shared table render replaces stale loading content');
  assert.equal(historyFixture.querySelectorAll('table').length,1);
  const retainedRow=historyFixture.querySelector('tbody tr');
  w.SkynetJobHistory.render(historyFixture,{columns:['State'],rows:[{id:'attempt',cells:['Running']}],empty:'No attempts.'});
  assert.equal(historyFixture.querySelector('tbody tr'),retainedRow,'Refresh reuses the existing history row');
  const types={dataset:{dataset:'Dataset',demonstrations:'Demonstrations'},file:{model:'Model',simulation_assets:'Simulation assets'}};
  const resources=[
    {id:'file',category:'file',kind:'model',provider:'huggingface',display_name:'Robot model',source_key:'Robot model',versions:[]},
    {id:'unclassified',kind:'demonstrations',provider:'test',display_name:'Unknown',source_key:'Unknown',versions:[]},
  ];
  const datasets=[
    {id:'cube',resource_id:'cube-source',category:'dataset',kind:'demonstrations',provider:'collection',display_name:'Cube · UniDex',recording_ids:['one','two'],metadata:{adapter:{name:'UniDex'},num_episodes:51},created_at:'2026-09-19'},
    {id:'cube-hpt',resource_id:'cube-source',category:'dataset',kind:'demonstrations',provider:'collection',display_name:'Cube · HPT',recording_ids:['one','two'],metadata:{adapter:{name:'HPT'},num_episodes:51},created_at:'2026-09-18'},
    {id:'external',resource_id:'external-source',category:'dataset',kind:'dataset',provider:'huggingface',display_name:'External',recording_ids:[],metadata:{num_episodes:20},created_at:'2026-09-17'},
  ];
  w.SkynetDatasetUI={adapterLabel:row=>row.metadata?.adapter?.name||'—',inputModalities:()=>'<ul class="adapter-input-modalities"><li>State</li></ul>'};
  let imports=[{id:'failed-import',resource_id:'unpublished-source',state:'FAILED',error:'Download interrupted',request:{subset:'new/demo'}},{id:'pending-import',resource_id:'pending-source',state:'RUNNING',slurm_job_id:'123',request:{subset:'pending/demo'}}];
  let readConversionHistory=async()=>({exports:[]});
  w.api=async path=>path==='/api/data/exports/jobs'?readConversionHistory():path.startsWith('/api/data/datasets?')?{datasets}:path.startsWith('/api/data/resources?')?{resources,resource_types:types}:path==='/api/data/imports'?{imports}:{items:[]};
  await w.activateTab('data',true,'registry'); await w.loadDataRegistry(true);
  const row=id=>el('data-resources-body').querySelector(`[data-resource-id="${id}"]`);
  const headers=()=>[...el('data-resources-body').closest('table').querySelectorAll('thead th')].filter(x=>!x.hidden).map(x=>x.textContent.trim());
  assert.deepEqual(headers(),['Name','Recordings','Adapter','Input modalities','Episodes','Experiment presets','Created at','Actions']);
  assert.equal(row('unclassified'),null,'Missing category must not silently classify as a dataset');
  assert.equal(row('file'),null);
  assert.equal(row('cube').cells[2].textContent,'UniDex');
  assert.equal(row('cube-hpt').cells[2].textContent,'HPT');
  assert.equal(row('external').cells[2].textContent,'—');
  assert.equal(el('data-resource-count').hidden,true);
  assert.equal(el('show-data-import-history').hidden,false);
  assert.equal(row('unpublished-source'),null,'Failed imports have no synthetic dataset or source-group row');
  el('show-data-import-history').click();await flush();
  assert.equal(el('data-import-history-dialog').open,true);
  assert.equal(el('data-imports-body').rows.length,2);
  el('data-imports-body').querySelector('[data-import-action=detail][data-id=failed-import]').click();await flush();
  assert.equal(el('data-import-detail-dialog').open,true);
  assert.match(el('data-import-detail-content').textContent,/Download interrupted/);
  w.SkynetDialog.close(el('data-import-detail-dialog'));
  assert.ok(el('data-imports-body').querySelector('[data-import-action=cancel][data-id=pending-import]'),'Active imports retain cancellation without a dataset row');
  const importApi=w.api;let cancelled;
  w.askUserDialog=async()=>true;
  w.api=async(path,options={})=>{if(options.method==='POST'&&path==='/api/data/imports/pending-import/cancel'){cancelled=path;return {};}return importApi(path,options);};
  el('data-imports-body').querySelector('[data-import-action=cancel][data-id=pending-import]').click();await flush();
  assert.equal(cancelled,'/api/data/imports/pending-import/cancel');
  w.api=importApi;
  w.SkynetDialog.close(el('data-import-history-dialog'));

  // Exercise the actual conversion controller with the shared catalog navigation.
  w.eval(await readStatic('policy-exports.js'));
  await flush();
  const historyButton=el('show-data-conversion-history');
  assert.equal(historyButton.previousElementSibling,el('show-data-import-history'));
  assert.equal(el('policy-export-history'),null,'No separate conversion panel remains');
  assert.equal(historyButton.hidden,false,'Datasets can open even an empty history');
  historyButton.click();await flush();
  assert.equal(el('prepared-dataset-dialog').open,true);
  assert.match(el('prepared-dataset-content').textContent,/No conversion attempts/);
  el('close-prepared-dataset').click();await flush();
  assert.equal(historyButton.hidden,false,'Closing history cannot hide its launcher');
  assert.equal(w.document.activeElement,historyButton,'Shared dialog restores launcher focus');
  let resolveLateHistory;
  readConversionHistory=()=>new Promise(resolve=>{resolveLateHistory=resolve;});
  historyButton.click();await flush();
  el('close-prepared-dataset').click();
  await w.activateTab('data',true,'files');
  assert.equal(historyButton.hidden,true,'Files never exposes conversion history');
  resolveLateHistory({exports:[]});await flush();
  assert.equal(historyButton.hidden,true,'Late history completion cannot change the selected view');
  assert.equal(el('prepared-dataset-dialog').open,false,'Late response cannot reopen a closed dialog');
  readConversionHistory=async()=>{throw Error('History unavailable');};
  await w.refreshPreparedDatasets();
  assert.equal(historyButton.hidden,true,'Background failure cannot expose conversion history on Files');
  await w.activateTab('data',true,'registry');
  assert.equal(historyButton.hidden,false,'Datasets restores its launcher without a history reload');
  historyButton.click();await flush();
  assert.match(el('prepared-dataset-error').textContent,/History unavailable/);
  el('close-prepared-dataset').click();await flush();
  assert.equal(historyButton.hidden,false,'Failure and close preserve the launcher');
  readConversionHistory=async()=>({exports:[]});
  historyButton.click();await flush();
  assert.equal(el('prepared-dataset-error').hidden,true);
  el('close-prepared-dataset').click();

  let opened;w.openPreparedDataset=async id=>{opened=id;};
  row('cube').querySelector('[data-resource-action=dataset]').click();await flush();assert.equal(opened,'cube');
  w.document.dispatchEvent(new w.CustomEvent('dataset-preparation-changed',{detail:[
    {id:'failed',resource_id:'cube',state:'FAILED',error:'bad new config'},
    {id:'pending',resource_id:'cube',state:'QUEUED'},
    {id:'finished',resource_id:'cube',state:'READY'}
  ]}));
  assert.doesNotMatch(row('cube').cells[0].textContent,/Conversion failed|Conversion: queued/,'Conversion attempts do not appear as dataset rows');
  assert.equal(row('cube').cells[2].textContent,'UniDex','Failure does not replace a successful dataset');
  el('show-data-resource-form').click();
  assert.equal(el('data-resource-category').value,'dataset');
  assert.deepEqual([...el('data-resource-kind').options].map(x=>x.value),Object.keys(types.dataset));
  // New still accepts external datasets, but its temporary source group is never a main row.
  el('data-resource-provider').value='huggingface';el('data-resource-namespace').value='org';
  el('data-resource-name').value='New import';el('data-resource-source-key').value='new-import';
  el('data-resource-kind').value='demonstrations';
  const catalogApi=w.api, creationCalls=[];
  const importedResource={id:'new-source',provider:'huggingface',namespace:'org',source_key:'new-import',display_name:'New import',category:'dataset',kind:'demonstrations'};
  w.api=async(path,options={})=>{
    if(options.method==='POST'){
      creationCalls.push({path,body:JSON.parse(options.body)});
      if(path==='/api/data/resources') return {resource:importedResource};
      if(path==='/api/data/resources/new-source/imports') return {import:{id:'import-new',slurm_job_id:'123'}};
    }
    return catalogApi(path,options);
  };
  await w.createDataResource({preventDefault(){}});
  assert.equal(creationCalls[0].body.category,'dataset');
  assert.equal(el('data-import-form-dialog').open,true,'Registration immediately opens import for the returned source');
  assert.equal(el('data-import-resource-id').value,'new-source');
  assert.equal(row('new-source'),null,'Unpublished source groups remain internal');
  el('data-import-revision').value='a'.repeat(40);el('data-import-subset').value='demo';
  el('data-import-format').value='test-format/v1';el('data-import-role').value='training_data';
  await w.submitDataImport({preventDefault(){}});
  assert.equal(creationCalls[1].path,'/api/data/resources/new-source/imports');
  assert.equal(creationCalls[1].body.revision,'a'.repeat(40));
  assert.equal(el('data-import-form-dialog').open,false);
  w.api=catalogApi;
  // Cancelling after source registration must remain resumable without exposing a group row.
  const localSource={id:'local-source',provider:'local',namespace:'org',source_key:'local-data',display_name:'Original source title',description:'Original description',category:'dataset',kind:'demonstrations'};
  const fillSource=provider=>{
    el('show-data-resource-form').click();
    el('data-resource-provider').value=provider;el('data-resource-namespace').value='org';
    el('data-resource-source-key').value=provider==='local'?'local-data':'new-import';
    el('data-resource-name').value='A different draft name';el('data-resource-kind').value='demonstrations';
  };
  fillSource('local');
  w.api=async(path,options={})=>path==='/api/data/resources'&&options.method==='POST'?{resource:localSource}:catalogApi(path,options);
  await w.createDataResource({preventDefault(){}});
  assert.equal(el('data-version-form-dialog').open,true);
  el('close-data-version-form').click();
  for(const source of [localSource,importedResource]){
    fillSource(source.provider);
    const resumeCalls=[];
    w.api=async(path,options={})=>{
      resumeCalls.push({path,...options});
      if(path==='/api/data/resources'&&options.method==='POST') throw Object.assign(Error('Duplicate source'),{status:409});
      if(path.startsWith('/api/data/resources?')&&new URL(path,'http://localhost').searchParams.has('provider')) return {resources:[source]};
      return catalogApi(path,options);
    };
    await w.createDataResource({preventDefault(){}});
    const lookup=resumeCalls.find(call=>call.path.includes('provider='));
    const params=new URL(lookup.path,'http://localhost').searchParams;
    assert.equal(params.get('provider'),source.provider);assert.equal(params.get('namespace'),'org');
    assert.equal(params.get('category'),'dataset');assert.equal(params.has('include_versions'),false);
    assert.equal(resumeCalls.some(call=>call.method==='PATCH'),false,'Reusing registration never overwrites its metadata');
    const form=source.provider==='local'?'data-version-form':'data-import-form';
    assert.equal(el(form+'-dialog').open,true);
    assert.equal(el(source.provider==='local'?'data-version-resource-id':'data-import-resource-id').value,source.id);
    el('close-'+form).click();
  }
  assert.equal(localSource.display_name,'Original source title');
  assert.equal(localSource.description,'Original description');
  fillSource('local');
  const unavailableCalls=[];
  w.api=async(path,options={})=>{unavailableCalls.push(path);throw Object.assign(Error('Service unavailable'),{status:503});};
  await w.createDataResource({preventDefault(){}});
  assert.deepEqual(unavailableCalls,['/api/data/resources'],'Non-conflict failures never attempt source reuse');
  assert.equal(el('data-resource-form-dialog').open,true);
  el('close-data-resource-form').click();
  w.api=catalogApi;
  await w.activateTab('data',true,'files');
  assert.deepEqual(headers(),['Name','Type','Formats','Source','Updated','Actions']);
  assert.equal(row('cube'),null);assert.equal(row('file').cells[1].textContent,'Model');
  el('show-data-resource-form').click();
  assert.equal(el('data-resource-category').value,'file');
  assert.deepEqual([...el('data-resource-kind').options].map(x=>x.value),Object.keys(types.file));
  el('data-resource-provider').value='huggingface';el('data-resource-namespace').value='org';el('data-resource-name').value='Robot';el('data-resource-source-key').value='robot-v1';el('data-resource-kind').value='model';
  let submitted;
  w.api=async (path,request={})=>{if(request.method==='POST'){submitted=JSON.parse(request.body);throw Error('Stop after capturing request');}return {};};
  await w.createDataResource({preventDefault(){}});
  assert.equal(submitted.category,'file');assert.equal(submitted.kind,'model');
  assert.equal(submitted.display_name,'Robot');assert.equal(submitted.source_key,'robot-v1');assert.equal('name' in submitted,false);
  // Reciprocal links carry exact IDs, while typing returns to ordinary search.
  const presets=[{id:'preset',name:'Cube preset',dataset_ids:['cube']},{id:'other',name:'Other preset',dataset_ids:['external']}];
  w.api=async path=>path.startsWith('/api/data/datasets?')?{datasets}:path.startsWith('/api/data/resources?')?{resources,resource_types:types}:path==='/api/data/imports'?{imports}:path==='/api/experiments'?{experiments:presets}:{};
  w.seedPresets(presets);
  await w.activateTab('experiments',true,'presets');
  assert.equal(el('experiments-body').closest('[data-tab-panel]').id,'presets');
  assert.ok(!el('presets').hidden);
  el('experiments-body').querySelector('[data-preset-datasets="preset"]').click();await flush();
  assert.equal(new URL(w.location.href).searchParams.get('data_view'),'registry');
  assert.ok(row('cube'));assert.equal(row('external'),null);
  assert.ok(row('cube').querySelector('[data-delete-kind="dataset"]'));
  el('data-resource-search').value='External';el('data-resource-search').dispatchEvent(new w.Event('input'));
  assert.equal(row('cube'),null);assert.ok(row('external'));
  const link=w.document.createElement('button');
  link.dataset.datasetPresets='cube';link.dataset.datasetLabel='Cube';link.dataset.presetIds='["preset"]';
  w.document.body.append(link);link.click();await flush();
  assert.equal(new URL(w.location.href).searchParams.get('experiment_view'),'presets');
  assert.equal(el('experiments-body').querySelectorAll('tr').length,1);
  assert.match(el('experiments-body').textContent,/Cube preset/);
  el('experiment-search').value='Other';el('experiment-search').dispatchEvent(new w.Event('input'));
  assert.match(el('experiments-body').textContent,/Other preset/);
  assert.doesNotMatch(el('experiments-body').textContent,/Cube preset/);
  el('data-resource-search').dataset.resourceIds='["cube"]';
  await w.openConvertedDataset({resource_id:'external-source',version_id:'external'});
  assert.ok(row('external'));assert.equal(row('cube'),null,'A newly converted dataset clears the previous preset filter');
  console.log('Catalog: flat results, explicit categories, exact preset links, separate attempts, modal navigation and external registration/import passed.');
}finally{for(const observer of observers)observer.disconnect();await flush();w.close();}
