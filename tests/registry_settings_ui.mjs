import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {JSDOM} from 'jsdom';
const w = new JSDOM(await readFile(new URL('../static/index.html', import.meta.url), 'utf8'), {runScripts:'outside-only', pretendToBeVisual:true, url:'http://localhost:8080/#settings'}).window;
const observers=[];const NativeObserver=w.MutationObserver;w.MutationObserver=class extends NativeObserver{constructor(cb){super(cb);observers.push(this);}};
const el = id => w.document.getElementById(id);
const flush = async () => {for (let i=0;i<3;i++) await new Promise(r=>setImmediate(r));};
w.fetch=()=>new Promise(()=>{});
w.scrollTo=w.HTMLElement.prototype.scrollIntoView=()=>{};
w.matchMedia=()=>({matches:false,addEventListener(){},removeEventListener(){}});
w.HTMLDialogElement.prototype.showModal=function(){this.open=true;};
w.HTMLDialogElement.prototype.close=function(value=''){if(this.open){this.returnValue=value;this.open=false;this.dispatchEvent(new w.Event('close'));}};
try {
  for (const name of ['dialogs.js','workspace-navigation.js','connection-settings.js','app.js']) w.eval((await readFile(new URL('../static/'+name,import.meta.url),'utf8')) + (name === 'app.js' ? `window.registryFixture=(versions,resources=[])=>{dataVersionRows=versions;dataResourceRows=resources;renderDataVersions();renderDataResources();};
    window.trackingFixture=provider=>trackingConnections.get(provider);
    window.missingTutorialFixture=()=>{tutorialState.active=true;tutorialState.page='datasets';tutorialState.session=newTutorialSession('datasets');tutorialState.sessionGeneration=tutorialState.session.sessionGeneration;showTutorialUnavailableStep(tutorialTours.datasets.steps[7],7);};
    window.tutorialBackFixture=()=>{
      const steps=tutorialTours.datasets.steps; const readIndex=steps.findIndex(step=>step.id==='read');
      tutorialState.index=steps.length-1;tutorialState.gateComplete=true;
      tutorialState.session.completedStepIds=[steps.at(-1).id,'read'];
      showTutorialUnavailableStep(steps[readIndex],readIndex);
      return {index:tutorialState.index,expected:readIndex};
    };
    window.tutorialPendingFixture=(confirmationWindowOpen=true)=>{
      resetTutorialAttempt();tutorialState.active=true;tutorialState.page='datasets';
      tutorialState.index=tutorialTours.datasets.steps.length-1;
      tutorialState.session=newTutorialSession('datasets');tutorialState.session.bindings={resourceId:'fixture'};
      tutorialState.sessionGeneration=tutorialState.session.sessionGeneration;
      const attempt={sessionGeneration:tutorialState.sessionGeneration,gateGeneration:tutorialState.gateGeneration,
        stepId:tutorialStep().id,attemptId:'fixture',expiresAt:Date.now()+8000,confirmationDeadline:Date.now()+120000,
        confirmationWindowOpen,claimed:false,request:{method:'DELETE',bind:'resourceId'},
        expectedPath:'/api/data/resources/fixture',bindingId:'fixture'};
      tutorialState.pendingAttempt=attempt;scheduleTutorialAttemptExpiry(attempt);return attempt;
    };
    window.pendingTutorial=()=>tutorialState.pendingAttempt;
    window.tutorialActionFixture=(stepId='archive',owned=true)=>{
      invalidateTutorialGate();tutorialState.active=true;tutorialState.page='collection';
      tutorialState.session=newTutorialSession('collection');
      tutorialState.session.token='tutorial-20260910T041936Z-fixture';
      tutorialState.sessionGeneration=tutorialState.session.sessionGeneration;
      tutorialState.index=tutorialTours.collection.steps.findIndex(step=>step.id===stepId);
      tutorialState.session.bindings={collectionAdapterId:'fixture'};
      const expectedIdentity=tutorialCollectionAdapterIdentity(tutorialContext());
      const record={binding:'collectionAdapterId',id:'fixture',kind:'collection-adapter',cleanup:'archive',
        cleanupState:'cleanup pending',ownerToken:tutorialState.session.token,expectedIdentity};
      tutorialState.session.ownedRecords=[record];tutorialState.ownedVerified=new Map();
      if(owned)tutorialState.ownedVerified.set(record.binding,{id:record.id,ownerToken:record.ownerToken,expectedIdentity});
      document.querySelectorAll('[data-fixture-tutorial-row]').forEach(row=>row.remove());
      const row=document.createElement('div');row.dataset.fixtureTutorialRow='';row.dataset.collectionAdapterId='fixture';
      const button=document.createElement('button');button.dataset.collectionAdapterAction=stepId==='archive'?'archive':'edit';
      row.append(button);document.body.append(row);tutorialState.target=button;
      tutorialState.gateComplete=false;tutorialState.confirmationArmed=true;
      onTutorialInteraction({type:'click',isTrusted:true,target:button,preventDefault(){},stopImmediatePropagation(){}});
      return {attempt:tutorialState.pendingAttempt,record,button,token:tutorialState.session.token};
    };
    window.tutorialResultState=()=>({index:tutorialState.index,gateComplete:tutorialState.gateComplete,stepId:tutorialStep().id,
      record:tutorialState.session.ownedRecords[0],completed:[...tutorialState.session.completedStepIds]});
    window.hideCurrentTutorialControl=()=>{tutorialState.target.remove();showTutorialUnavailableStep(tutorialStep(),tutorialState.index);};
    window.completedTutorialFixture=()=>{tutorialState.index=tutorialTours.datasets.steps.length-1;tutorialState.target=document.createElement('button');completeTutorialGate('Archived');return advanceTutorial();};` : ''));

  // Bundle authoring has been removed from the user-facing registry.
  assert.equal(el('data-bundle-version-picker'),null);
  assert.equal(el('data-bundle-assignments'),null);
  w.loadDataRegistry=async()=>{};
  w.activateTab('datasets');el('data-show-archived').checked=true;
  w.registryFixture([], [{id:'archived',category:'dataset',archived_at:'today',display_name:'Example',source_key:'example-source'}]);
  assert.ok(el('data-resources-body').querySelector('[data-resource-action="restore"]'));
  assert.equal(el('data-resources-body').querySelector('[data-resource-action="version"]'),null);
  // Each integration has one visible action; validation happens in Connect.
  const connection={provider:'mlflow',configured:true,connected:true,status:'connected',tracking_uri:'https://fixture.test',username:'alice',verify_tls:true};
  w.renderTrackingConnection('mlflow',connection);
  assert.equal(el('mlflow-connection-username').value,'alice');
  assert.equal(el('mlflow-connection-username').disabled,true);
  assert.equal(el('disconnect-mlflow').hidden,false);
  assert.equal(el('mlflow-connection-form').querySelector('button[type="submit"]').hidden,true);
  assert.equal(el('test-mlflow-connection'),null);
  assert.equal(el('mlflow-connection-form').querySelector('.tracking-draft-notice'),null);
  w.renderTrackingConnectionsUnavailable('Timed out');assert.equal(w.trackingFixture('mlflow').status,'unavailable');
  let fail=true;
  w.api=async(path)=>{
    if(path.endsWith('/connect')){if(fail)throw Error('Unauthorized');return {connection};}
    if(path==='/api/tracking/connections')return {connections:{mlflow:{...connection,connected:false,status:'error',last_error:'Unauthorized'}}};
    if(path==='/api/settings')return {tracking:{connections:{mlflow:{...connection,connected:!fail,status:fail?'error':'connected'}}}};
    throw Error(path);
  };
  await w.submitTrackingConnection('mlflow','connect');
  assert.equal(el('mlflow-connection-status').textContent,'Not connected');
  assert.match(el('settings-body').textContent,/error/);
  assert.equal(el('mlflow-connection-form').querySelector('button[type="submit"]').hidden,false);
  w.showNotice(el('settings-error'),'Old settings fetch error',{scope:'settings:resolved'});fail=false;await w.submitTrackingConnection('mlflow','connect');
  assert.equal(el('settings-error').hidden,true,el('settings-error').textContent);
  assert.equal(el('mlflow-connection-status').textContent,'Connected');
  assert.equal(el('toast').querySelectorAll('.is-error[data-notification-scope="tracking:mlflow"]').length,0,'Successful Connect clears its obsolete alert');
  // Recovering one provider never dismisses another provider or an unrelated issue.
  w.showToast('W&B connection needs attention',true,{scope:'tracking:wandb'});
  w.showNotice(el('settings-error'),'Unrelated configuration issue',{scope:'settings:unrelated'});
  w.renderTrackingConnection('mlflow',{...connection,connected:false});
  fail=true;await w.submitTrackingConnection('mlflow','connect');
  await w.submitTrackingConnection('mlflow','connect');
  assert.equal(el('toast').querySelectorAll('.is-error[data-notification-scope="tracking:mlflow"]').length,1,'Repeated failure replaces its own alert');
  assert.match(el('toast').textContent,/MLflow connect failed: Unauthorized/);
  fail=false;await w.submitTrackingConnection('mlflow','connect');
  assert.equal(el('mlflow-connection-status').textContent,'Connected');
  assert.doesNotMatch(el('toast').textContent,/MLflow connect failed/);
  assert.match(el('toast').textContent,/W&B connection needs attention/);
  assert.match(el('settings-error').textContent,/Unrelated configuration issue/);
  assert.equal(el('settings-error').hidden,false);
  const scopedDialog=w.SkynetDialog.create({title:'Fixture',content:w.document.createElement('div')});
  scopedDialog.dataset.panelDialog='fixture';w.document.body.append(scopedDialog);
  w.SkynetDialog.open(scopedDialog);
  w.showToast('Dialog action failed',true,{scope:'fixture:dialog-action'});
  w.showNotice(scopedDialog.querySelector('.dialog-notice'),'Other dialog issue',{scope:'fixture:other'});
  w.showToast('Dialog action recovered',false,{scope:'fixture:dialog-action'});
  assert.doesNotMatch(scopedDialog.textContent,/Dialog action failed/);
  assert.match(scopedDialog.textContent,/Other dialog issue/);
  w.SkynetDialog.close(scopedDialog);scopedDialog.remove();
  w.activateTab('datasets');w.missingTutorialFixture();await flush();
  assert.equal(el('tutorial-layer').hidden,false);
  assert.equal(el('tutorial-layer').closest('dialog:not([open])'),null);
  assert.ok([...el('tutorial-layer').querySelectorAll('button')].some(button=>/restart/i.test(button.textContent)&&!button.hidden));
  const back=w.tutorialBackFixture();assert.equal(back.index,back.expected,'Back to a missing earlier completed step changes the index');
  assert.equal(w.completedTutorialFixture(),true,'successful cleanup can finish without its removed button');
  // Native confirmation reading does not consume the ordinary eight-second claim window.
  const originalNow=w.Date.now;let now=1000;w.Date.now=()=>now;
  let attempt=w.tutorialPendingFixture();
  w.document.body.classList.add('has-active-tutorial');
  let answer=w.askUserDialog('Archive this fixture?');
  let confirmation=w.document.querySelector('dialog[data-app-confirmation][open]');
  assert.equal(attempt.confirmationDialog,confirmation);
  for (const key of ['Tab','Enter',' ']) {
    const keyboard=new w.KeyboardEvent('keydown',{key,bubbles:true,cancelable:true});
    w.onTutorialKeydown(keyboard);assert.equal(keyboard.defaultPrevented,false,'Native confirmation owns '+key);
  }
  now+=20000;
  assert.equal(w.tutorialObserveApiStart('/api/data/resources/fixture',{method:'DELETE'}),null,'No claim while confirmation remains open');
  confirmation.querySelector('form').dispatchEvent(new w.Event('submit',{cancelable:true}));
  assert.equal(await answer,true);
  assert.ok(attempt.expiresAt>now);
  assert.equal(w.tutorialObserveApiStart('/api/data/resources/other',{method:'DELETE'}),null,'Record binding remains exact');
  assert.equal(w.tutorialObserveApiStart('/api/data/resources/fixture',{method:'POST'}),null,'Method binding remains exact');
  assert.ok(w.tutorialObserveApiStart('/api/data/resources/fixture',{method:'DELETE'}));
  attempt=w.tutorialPendingFixture();answer=w.askUserDialog('Cancel this fixture?');
  confirmation=w.document.querySelector('dialog[data-app-confirmation][open]');
  confirmation.dispatchEvent(new w.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true}));
  assert.equal(await answer,false);assert.equal(w.pendingTutorial(),null,'Escape cancels the attempt');
  attempt=w.tutorialPendingFixture();answer=w.askUserDialog('Expired fixture?');
  confirmation=w.document.querySelector('dialog[data-app-confirmation][open]');
  now+=120001;
  confirmation.querySelector('form').dispatchEvent(new w.Event('submit',{cancelable:true}));
  assert.equal(await answer,false,'Late Continue cannot execute an expired tutorial action');
  assert.equal(w.pendingTutorial(),null);
  attempt=w.tutorialPendingFixture(false);const expiry=attempt.expiresAt;
  answer=w.askUserDialog('Unrelated later confirmation');
  confirmation=w.document.querySelector('dialog[data-app-confirmation][open]');
  assert.equal(attempt.expiresAt,expiry,'An unrelated later dialog cannot extend authorization');
  w.SkynetDialog.close(confirmation,'cancel');assert.equal(await answer,false);
  // Native dispatch may checkpoint microtasks between document capture and the
  // delegated action handler. Its confirmation still belongs to the same task.
  let action=w.tutorialActionFixture();
  await Promise.resolve();
  assert.equal(action.attempt.confirmationWindowOpen,true,'The capture-listener microtask checkpoint must not close the action window');
  answer=w.askUserDialog('Archive the bound collection adapter?');
  confirmation=w.document.querySelector('dialog[data-app-confirmation][open]');
  assert.equal(action.attempt.confirmationDialog,confirmation);
  now+=50000;
  const continueButton=confirmation.querySelector('button[type="submit"]');continueButton.focus();
  const enter=new w.KeyboardEvent('keydown',{key:'Enter',bubbles:true,cancelable:true});
  w.onTutorialKeydown(enter);assert.equal(enter.defaultPrevented,false);
  confirmation.querySelector('form').dispatchEvent(new w.Event('submit',{cancelable:true}));
  assert.equal(await answer,true);
  let claim=w.tutorialObserveApiStart('/api/collection/adapters/fixture',{method:'DELETE'});
  assert.ok(claim,'Delayed keyboard confirmation retains the exact DELETE claim');
  w.hideCurrentTutorialControl();
  w.tutorialObserveApiSuccess(claim,{adapter:{id:'fixture',archived_at:'2026-09-10T00:00:00Z'}});
  assert.equal(w.tutorialResultState().record.cleanupState,'archived');
  assert.equal(w.tutorialResultState().gateComplete,true,'A row disappearing before the response settles does not invalidate the claim');
  assert.equal(el('tutorial-next').disabled,false);
  const finishCollectionTutorial=()=>{
    const heading=el('data').querySelector('.page-heading h1');
    const ordinaryRect=heading.getBoundingClientRect;
    heading.getBoundingClientRect=()=>({x:0,y:0,left:0,top:0,right:300,bottom:40,width:300,height:40});
    assert.equal(w.advanceTutorial(),true);
    assert.equal(w.tutorialResultState().stepId,'session-boundary','With the optional session form unavailable, cleanup still reaches the shared Data heading');
    assert.equal(el('tutorial-next').textContent,'Finish');
    assert.equal(el('tutorial-next').disabled,false);
    assert.equal(w.advanceTutorial(),true);
    assert.equal(JSON.parse(w.localStorage.getItem('skynet.tutorial.progress.v2.collection')).finished,true,'Finish persists actual tutorial completion');
    assert.equal(heading.hasAttribute('data-tutorial-active-target'),false,'Finishing removes the same heading marker that fallback recovery applied');
    assert.doesNotMatch(el('toast').textContent,/Collection tutorial stopped because no visible step/);
    heading.getBoundingClientRect=ordinaryRect;
  };
  finishCollectionTutorial();

  // Recovery is a user-triggered exact GET; identity and bound ID still govern it.
  const archivedPayload=fixture=>({adapter:{id:'fixture',archived_at:'2026-09-10T00:00:00Z',
    adapter_key:fixture.token.toLowerCase()+'-collection-adapter',manifest:{metadata:{tutorial_token:fixture.token}}}});
  action=w.tutorialActionFixture('read',false);
  claim=w.tutorialObserveApiStart('/api/collection/adapters/fixture');
  w.tutorialObserveApiSuccess(claim,archivedPayload(action));
  assert.equal(w.tutorialResultState().record.cleanupState,'archived');
  assert.ok(w.tutorialResultState().completed.includes('archive'));
  assert.match(el('tutorial-gate-status').textContent,/Already archived/);
  assert.equal(w.advanceTutorial(),true);
  assert.equal(w.tutorialResultState().stepId,'archive','Already archived recovery skips invalid update steps');
  assert.equal(el('tutorial-next').disabled,false,'Missing archived-row action is presented as complete');
  w.endTutorial(false);
  assert.equal(el('data').querySelector('.page-heading h1').hasAttribute('data-tutorial-active-target'),false,'Closing directly from unavailable cleanup also removes its target marker');
  action=w.tutorialActionFixture('read',false);claim=w.tutorialObserveApiStart('/api/collection/adapters/fixture');
  w.tutorialObserveApiSuccess(claim,archivedPayload(action));
  w.advanceTutorial();
  finishCollectionTutorial();
  action=w.tutorialActionFixture('read',false);claim=w.tutorialObserveApiStart('/api/collection/adapters/fixture');
  const mismatched=archivedPayload(action);mismatched.adapter.manifest.metadata.tutorial_token='another-owner';
  w.tutorialObserveApiSuccess(claim,mismatched);
  assert.notEqual(w.tutorialResultState().record.cleanupState,'archived','An archived but differently owned record is not accepted');
  assert.equal(w.tutorialResultState().gateComplete,false);
  action=w.tutorialActionFixture('read',false);claim=w.tutorialObserveApiStart('/api/collection/adapters/fixture');
  const wrongId=archivedPayload(action);wrongId.adapter.id='other';w.tutorialObserveApiSuccess(claim,wrongId);
  assert.notEqual(w.tutorialResultState().record.cleanupState,'archived','A different archived record cannot settle the bound GET');
  assert.equal(w.tutorialResultState().gateComplete,false);
  action=w.tutorialActionFixture('read',false);claim=w.tutorialObserveApiStart('/api/collection/adapters/fixture');
  const activePayload=archivedPayload(action);delete activePayload.adapter.archived_at;
  w.tutorialObserveApiSuccess(claim,activePayload);
  assert.equal(w.tutorialResultState().gateComplete,true);
  assert.equal(w.beginTutorialRecovery(w.tutorialResultState().record),true);
  assert.equal(w.tutorialResultState().gateComplete,false,'Recovery always requires a fresh exact Read even when an older Read completed');
  w.resetTutorialAttempt();w.Date.now=originalNow;w.document.body.classList.remove('has-active-tutorial');
  console.log('Registry/Settings regressions: retired bundle controls, restore, Connect/Disconnect validation, shared modal alerts and tutorial confirmations/recovery passed.');
} catch(error) {console.error(error);process.exitCode=1;} finally {for(const observer of observers)observer.disconnect();await flush();w.close();}
