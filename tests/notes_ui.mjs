import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {JSDOM} from 'jsdom';
import {indexHtml} from './index_page.cjs';
const w = new JSDOM(indexHtml(), {
  runScripts:'outside-only', pretendToBeVisual:true, url:'http://localhost:8080/?data_view=registry#data',
}).window;
const el=id=>w.document.getElementById(id), visible=id=>!el(id).closest('[hidden]');
const observers=[],Observer=w.MutationObserver;
w.MutationObserver=class extends Observer {constructor(callback){super(callback);observers.push(this);}};
for(const key of ['Headers','Request','Response','AbortController','AbortSignal'])w[key]=globalThis[key];
const streams=[];
w.SkynetWorkspace={id:'test-workspace',storageKey:key=>'test:'+key};
w.EventSource=class {
  constructor(){this.listeners=new Map();streams.push(this);}
  addEventListener(type,callback){this.listeners.set(type,callback);}
  emit(type,value={}){this.listeners.get(type)?.({data:JSON.stringify(value)});}
  close(){}
};
// The fake Notes API answers "METHOD path" routes; every other request stays pending.
const fetched=[],requests=[];let routes={};
const json=(value,status=200)=>Response.json(value,{status});
w.fetch=(input,init={})=>{
  const path=String(input),method=init.method||'GET';
  fetched.push(path);
  if(!path.startsWith('/api/notes'))return new Promise(()=>{});
  requests.push({path,method,body:init.body,type:new Headers(init.headers).get('Content-Type')});
  const route=routes[`${method} ${path}`];
  const value=typeof route==='function'?route(init):route;
  return Promise.resolve(value===undefined?json({detail:'Note not found'},404):value instanceof Response?value:json(value));
};
const since=count=>requests.slice(count).map(request=>`${request.method} ${request.path}`);
const writes=count=>requests.slice(count).filter(request=>request.method!=='GET');
w.scrollTo=w.HTMLElement.prototype.scrollIntoView=()=>{};
w.matchMedia=()=>({matches:false,addEventListener(){},removeEventListener(){}});
w.HTMLDialogElement.prototype.showModal=function(){this.setAttribute('open','');};
w.HTMLDialogElement.prototype.close=function(value){if(value!==undefined)this.returnValue=value;this.removeAttribute('open');this.dispatchEvent(new w.Event('close'));};
const settle=()=>new Promise(resolve=>setTimeout(resolve,20));
const rows=()=>[...el('notes-table').querySelectorAll('tbody tr[data-history-id]')].map(row=>row.dataset.historyId);
const filter=async value=>{el('notes-search').value=value;el('notes-search').dispatchEvent(new w.Event('input'));};
// jsdom cannot choose files: the chosen list is faked, and clearing the input's value empties it.
const setFiles=chosen=>{
  Object.defineProperty(el('note-editor-files'),'files',{configurable:true,get:()=>chosen});
  Object.defineProperty(el('note-editor-files'),'value',{configurable:true,get:()=>chosen.length?chosen[0].name:'',set:value=>{if(value==='')chosen=[];}});
};
const clearFiles=()=>{delete el('note-editor-files').files;delete el('note-editor-files').value;};
const editorDialog=()=>el('note-editor-dialog'),viewer=()=>el('workspace-note-dialog');
try {
  for(const file of ['dialogs.js','workspace-navigation.js','connection-settings.js','app.js'])
    w.eval(await readFile(new URL('../static/'+file,import.meta.url),'utf8'));
  const note={id:'latest',title:'UniDex latest',folder_id:null,status:'진행',created_at:'2026-09-18T12:00:00Z',updated_at:'2026-09-20T18:00:00Z'};
  const full=(value,extra={})=>({note:{markdown:'',html:'',attachments:[],...value,...extra}});
  routes={'GET /api/notes':{notes:[note],folders:[]},'GET /api/notes/latest':full(note,{html:'<p>1,000 steps · latest 3 · final 1</p>'})};
  const mark=fetched.length;
  w.activateTab('experiments',true,'notes');
  w.eval(await readFile(new URL('../static/notes.js',import.meta.url),'utf8'));
  await settle();
  el('email-workspace-content').hidden=false;
  assert.deepEqual(since(0),['GET /api/notes'],'A Notes deep link loads before the signed-in shell becomes visible');
  assert.ok(visible('experiment-notes'));assert.equal(visible('datasets'),false);assert.equal(visible('collection'),false);
  assert.equal(el('experiment-tab-notes').getAttribute('aria-selected'),'true');
  assert.deepEqual(fetched.slice(mark).filter(path=>/^\/api\/(data|collection)/.test(path)),[],'Notes does not fetch collection or registry');
  assert.equal(el('notes-table').querySelectorAll('tbody tr').length,1);
  assert.deepEqual([...el('notes-table').querySelectorAll('thead th')].map(cell=>cell.textContent.trim()),['Name','Kind','Status','Created','Updated','Actions']);
  assert.equal(el('notes-table').querySelectorAll('tbody td')[2].textContent.trim(),'진행','Status shows the header status');
  // A status comes from note text, so it is shown as text.
  const markup='<img/src/onerror=alert&#40;1&#41;>';
  routes['GET /api/notes']={notes:[{...note,status:markup}],folders:[]};
  await w.loadNotes();
  assert.equal(el('notes-table').querySelectorAll('tbody td')[2].textContent,markup,'A status is shown as text');
  assert.equal(el('notes-table').querySelector('tbody img'),null);
  routes['GET /api/notes']={notes:[note],folders:[]};
  await w.loadNotes();
  assert.deepEqual([...el('notes-table').querySelectorAll('tbody td')].slice(3,5).map(cell=>cell.textContent.trim()),
    [w.formatDate(note.created_at),w.formatDate(note.updated_at)],'Created and Updated show their own times');
  assert.ok(visible('refresh-experiments'));assert.equal(visible('refresh-collection'),false);
  assert.ok(visible('new-note'));
  await filter('no match');
  assert.match(el('notes-table').textContent,/No notes match/);
  await filter('');
  el('notes-table').querySelector('button').click();await settle();
  assert.ok(viewer().open);
  assert.equal(el('workspace-note-date').textContent,`Created ${w.formatDate(note.created_at)} · Updated ${w.formatDate(note.updated_at)}`);
  assert.ok(viewer().querySelector('.dialog-shell'));
  assert.match(el('workspace-note-content').textContent,/1,000 steps/);
  assert.match(el('workspace-note-download').href,/\/api\/notes\/latest\/download$/);
  for(const id of ['workspace-note-download','workspace-note-edit','workspace-note-delete'])assert.ok(visible(id),id+' is offered');
  viewer().querySelector('[data-dialog-close]').click();await settle();
  assert.equal(viewer().open,false);
  assert.equal(new URL(w.location.href).searchParams.get('note'),null);
  let count=requests.length;el('refresh-experiments').click();await settle();
  assert.deepEqual(since(count),['GET /api/notes']);
  el('data-tab-files').click();await settle();assert.ok(visible('datasets'));assert.equal(visible('experiment-notes'),false);
  el('experiment-tab-notes').click();await settle();assert.ok(visible('experiment-notes'));assert.match(el('notes-table').textContent,/UniDex latest/);

  const folders=[{id:'folder-0',name:'egoisim'},{id:'folder-1',name:'warp-extension'}];
  routes['GET /api/notes']={notes:[{...note,folder_id:'folder-0'}],folders};
  routes['GET /api/notes/latest']=full({...note,folder_id:'folder-0'},{html:'<p>Original note</p>'});
  await w.loadNotes();
  assert.equal(el('notes-table').querySelectorAll('[data-note-folder].entity-link').length,2);
  assert.equal(el('notes-table').querySelector('tbody tr').children.length,6,'Folder rows fill every column');
  assert.equal(el('notes-table').querySelectorAll('[data-open-note]').length,0);
  el('notes-table').querySelector('[data-note-folder="folder-0"]').click();
  assert.match(el('notes-path').textContent,/egoisim/);
  assert.equal(new URL(w.location.href).searchParams.get('note_folder'),'folder-0');
  assert.ok(el('notes-table').querySelector('[data-open-note]'));
  assert.equal(el('notes-table').querySelector('input[type="checkbox"], [data-select-note]'),null,'Rows have no selection controls');
  el('notes-table').querySelector('[data-open-note]').click();await settle();
  assert.match(el('workspace-note-content').textContent,/Original note/);
  viewer().querySelector('[data-dialog-close]').click();await settle();
  assert.equal(new URL(w.location.href).searchParams.get('note_folder'),'folder-0');
  el('notes-path').querySelector('[data-note-folder=""]').click();
  assert.equal(new URL(w.location.href).searchParams.get('note_folder'),null);
  el('notes-table').querySelector('[data-note-folder="folder-1"]').click();await w.loadNotes();
  assert.match(el('notes-path').textContent,/warp-extension/);
  assert.match(el('notes-table').textContent,/This folder is empty/);
  // New folders are made at the top level with the shared prompt, then opened.
  assert.equal(visible('new-folder'),false,'Folders are one level deep');
  el('notes-path').querySelector('[data-note-folder=""]').click();
  assert.ok(visible('new-folder'));
  const prompt=()=>w.document.querySelector('dialog[data-app-confirmation]');
  count=requests.length;
  el('new-folder').click();await settle();
  prompt().querySelector('button[type="button"]:not([data-dialog-close])').click();await settle();
  assert.deepEqual(writes(count),[],'A cancelled prompt creates nothing');
  routes['POST /api/notes/folders']=()=>json({detail:'A folder named warp-extension already exists'},422);
  el('new-folder').click();await settle();
  prompt().querySelector('input').value='warp-extension';
  prompt().querySelector('button[type="submit"]').click();await settle();
  assert.match(el('notes-error').textContent,/already exists/);
  assert.equal(new URL(w.location.href).searchParams.get('note_folder'),null);
  const archive={id:'folder-2',name:'warp-archive'};
  // A response whose body arrives only when released.
  const heldResponse=value=>{let release;const held=new Promise(resolve=>release=resolve);
    return {release,response:()=>new Response(new ReadableStream({start(controller){held.then(()=>{
      controller.enqueue(new TextEncoder().encode(JSON.stringify(value)));controller.close();});}}),
      {headers:{'Content-Type':'application/json'}})};};
  routes['POST /api/notes/folders']=init=>({folder:{...archive,name:JSON.parse(init.body).name}});
  const refresh=heldResponse({notes:[{...note,folder_id:'folder-0'}],folders:[...folders,archive]});
  routes['GET /api/notes']=refresh.response;
  count=requests.length;
  el('new-folder').click();await settle();
  prompt().querySelector('input').value='  warp-archive ';
  prompt().querySelector('button[type="submit"]').click();await settle();
  assert.deepEqual(writes(count).map(request=>[request.method,request.path,JSON.parse(request.body)]),
    [['POST','/api/notes/folders',{name:'warp-archive'}]]);
  assert.equal(new URL(w.location.href).searchParams.get('note_folder'),'folder-2');
  assert.match(el('notes-path').textContent,/warp-archive/,'The new folder is named before the list refresh');
  assert.equal(visible('new-folder'),false);
  assert.equal(w.document.activeElement,el('new-note'),'Focus moves off the hidden New folder button');
  refresh.release();await settle();
  assert.equal(new URL(w.location.href).searchParams.get('note_folder'),'folder-2');
  routes['GET /api/notes']={notes:[{...note,folder_id:'folder-0'}],folders:[...folders,archive]};
  el('new-note').click();await settle();
  assert.equal(el('note-editor-folder').value,'folder-2','A new note starts in the opened folder');
  editorDialog().querySelector('[data-dialog-close]').click();await settle();
  // When the list refresh lands before the create response's body, the folder still shows once.
  el('notes-path').querySelector('[data-note-folder=""]').click();
  const second={id:'folder-3',name:'warp-second'};
  const createdFolder=heldResponse({folder:second});
  routes['POST /api/notes/folders']=createdFolder.response;
  routes['GET /api/notes']={notes:[{...note,folder_id:'folder-0'}],folders:[...folders,archive,second]};
  el('new-folder').click();await settle();
  prompt().querySelector('input').value='warp-second';
  prompt().querySelector('button[type="submit"]').click();await settle();
  await w.loadNotes();
  createdFolder.release();await settle();
  el('notes-path').querySelector('[data-note-folder=""]').click();
  assert.deepEqual(rows().filter(id=>id==='folder-3'),['folder-3'],'A new folder is listed once');
  routes['GET /api/notes']={notes:[{...note,folder_id:'folder-0'}],folders};
  await w.loadNotes();
  el('notes-table').querySelector('[data-note-folder="folder-1"]').click();await w.loadNotes();
  for(const id of ['notes-new-folder','notes-up-folder','notes-rename-folder','notes-selection',
    'notes-select-all','notes-selection-count','notes-move','notes-folder-dialog','notes-move-dialog'])
    assert.equal(el(id),null, id+' is removed entirely');
  assert.doesNotMatch(el('experiment-notes').textContent,/All notes|Unfiled/);
  assert.equal(el('notes-folders'),null,'No custom sidebar');
  assert.equal(el('note-editor-dialog').querySelector('[id*="folder-name"], [id*="new-folder"]'),null,'The editor only chooses existing folders');
  const older={id:'older',title:'Edited later',created_at:'2026-09-01T00:00:00Z',updated_at:'2026-09-30T00:00:00Z',folder_id:'folder-1'};
  routes['GET /api/notes']={notes:[{...note,folder_id:'folder-1'},older],folders};
  await w.loadNotes();
  assert.deepEqual(rows(),['latest','older'],'Rows keep the server order: newest created first, even when an older note was edited later');
  const index={id:'index',title:'00 목차 · WARP++',created_at:'2026-08-15T00:00:00Z',updated_at:'2026-08-15T00:00:00Z',folder_id:'folder-1'};
  routes['GET /api/notes']={notes:[{...note,folder_id:'folder-1'},index,older],folders};
  await w.loadNotes();
  assert.deepEqual(rows(),['index','latest','older'],'An index note titled "00 ..." stays on top, the rest keep the server order');
  routes['GET /api/notes']={notes:[{...note,folder_id:'folder-1'},older],folders};
  await w.loadNotes();

  // A search matches titles in every folder, and folder names at the root.
  const egoNote={id:'ego-note',title:'Egoisim latest results',folder_id:'folder-0',created_at:'2026-08-01T00:00:00Z',updated_at:'2026-08-01T00:00:00Z'};
  const rootNote={id:'root-note',title:'Root plan',folder_id:null,created_at:'2026-07-01T00:00:00Z',updated_at:'2026-07-01T00:00:00Z'};
  routes['GET /api/notes']={notes:[{...note,folder_id:'folder-1'},older,egoNote,rootNote],folders};
  await w.loadNotes();
  assert.deepEqual(rows(),['latest','older']);
  await filter('LATEST');assert.deepEqual(rows(),['latest','ego-note'],'Search inside a folder spans all folders');
  assert.deepEqual([...el('notes-table').querySelectorAll('tbody tr')].map(row=>row.children[1].textContent),
    ['Markdown · warp-extension','Markdown · egoisim'],'Search results name the folder of each note');
  await filter('');assert.deepEqual(rows(),['latest','older'],'An empty search shows the current folder');
  el('notes-path').querySelector('[data-note-folder=""]').click();
  assert.deepEqual(rows(),['folder-0','folder-1','root-note']);
  await filter('ego');assert.deepEqual(rows(),['folder-0','ego-note'],'Root search matches folder names and every note title');
  await filter('plan');assert.deepEqual(rows(),['root-note']);
  assert.equal(el('notes-table').querySelector('tbody tr').children[1].textContent,'Markdown · Notes');
  await filter('');
  assert.equal(el('notes-table').querySelector('[data-history-id="root-note"]').children[1].textContent,'Markdown');

  // The viewer shows the server rendering as sent; attachment figures come from the server.
  const figure='<p><a class="note-figure" href="/api/notes/root-note/attachments/plot.svg"><img src="/api/notes/root-note/attachments/plot.svg" alt="Plot" loading="eager" decoding="async"></a></p>'+
    '<p><a href="/static/old-plot.svg">Old plot — 그래프 열기</a> <a href="/?experiment_view=notes&amp;note=ego-note#experiments">Ego</a></p>';
  routes['GET /api/notes/root-note']=full(rootNote,{html:figure});
  routes['GET /api/notes/ego-note']=full(egoNote,{html:'<p><a href="/?experiment_view=notes&amp;note_folder=folder-0#experiments">Back to egoisim</a></p>'});
  el('notes-table').querySelector('[data-open-note="root-note"]').click();await settle();
  const content=el('workspace-note-content');
  assert.deepEqual([...content.querySelectorAll('img')].map(image=>image.getAttribute('src')),['/api/notes/root-note/attachments/plot.svg']);
  assert.equal(content.querySelector('a[href="/static/old-plot.svg"]').textContent,'Old plot — 그래프 열기','Links are not rewritten into figures');
  let prevented=null;
  w.addEventListener('click',event=>{prevented=event.defaultPrevented;event.preventDefault();},{once:true});
  content.querySelector('a[href*="note=ego-note"]').dispatchEvent(new w.MouseEvent('click',{bubbles:true,cancelable:true,metaKey:true}));await settle();
  assert.equal(prevented,false,'A modified click on a note link is left to the browser (new tab)');
  assert.equal(viewer().querySelector('[data-dialog-title]').textContent,rootNote.title);
  content.querySelector('a[href*="note=ego-note"]').click();await settle();
  assert.ok(viewer().open);
  assert.equal(viewer().querySelector('[data-dialog-title]').textContent,egoNote.title,'A note link opens that note in place');
  assert.equal(new URL(w.location.href).searchParams.get('note'),'ego-note');
  content.querySelector('a').click();await settle();
  assert.equal(viewer().open,false,'A folder link closes the note');
  assert.equal(new URL(w.location.href).searchParams.get('note'),null);
  assert.equal(new URL(w.location.href).searchParams.get('note_folder'),'folder-0');
  assert.deepEqual(rows(),['ego-note']);

  w.katex={render(tex,node,options){node.textContent=`${options.displayMode?'display':'inline'}:${tex}`;}};
  routes['GET /api/notes/ego-note']=full(egoNote,{html:'<p><span class="math math-inline">a_i</span></p><div class="math math-display">\\frac{1}{k}</div>'});
  el('notes-table').querySelector('[data-open-note]').click();await settle();
  assert.deepEqual([...content.querySelectorAll('.math')].map(node=>node.textContent),
    ['inline:a_i','display:\\frac{1}{k}'],'Server TeX is typeset in inline and display mode');

  // A notes change reloads the visible list, and the open note only when it changed.
  count=requests.length;
  streams[0].emit('change',{v:1,topics:['notes']});await settle();
  assert.deepEqual(since(count),['GET /api/notes'],'An unchanged open note is not fetched again');
  const edited={...egoNote,updated_at:'2026-10-01T00:00:00Z'};
  routes['GET /api/notes']={notes:[{...note,folder_id:'folder-1'},older,edited,rootNote],folders};
  routes['GET /api/notes/ego-note']=full(edited,{html:'<p>Edited elsewhere</p>'});
  count=requests.length;
  streams[0].emit('change',{v:1,topics:['notes']});await settle();
  assert.deepEqual(since(count),['GET /api/notes','GET /api/notes/ego-note']);
  assert.match(content.textContent,/Edited elsewhere/);
  viewer().querySelector('[data-dialog-close]').click();await settle();
  count=requests.length;
  streams[0].emit('change',{v:1,topics:['data']});await settle();
  assert.deepEqual(since(count),[],'Other topics do not reload Notes');
  el('data-tab-files').click();await settle();
  count=requests.length;
  streams[0].emit('change',{v:1,topics:['notes']});await settle();
  assert.deepEqual(since(count),[],'Hidden Notes ignore change topics');
  el('experiment-tab-notes').click();await settle();
  assert.deepEqual(since(count),['GET /api/notes'],'Returning to Notes reloads once');

  // New note: POST the fields, then upload each chosen file as the raw request body.
  const created={id:'2026-10-05-new-plan',created_at:'2026-10-05T09:00:00Z',updated_at:'2026-10-05T09:00:00Z'};
  routes['POST /api/notes']=init=>full({...created,...JSON.parse(init.body)});
  routes[`PUT /api/notes/${created.id}/attachments/plot.svg`]={attachment:{name:'plot.svg'}};
  routes[`PUT /api/notes/${created.id}/attachments/loss-curve.csv`]={attachment:{name:'loss-curve.csv'}};
  routes[`GET /api/notes/${created.id}`]=full({...created,title:'New plan',folder_id:'folder-1'},{html:'<p>New plan</p>'});
  el('new-note').click();
  assert.ok(editorDialog().open);
  assert.equal(w.document.activeElement,el('note-editor-title'),'The editor starts on its Title field');
  assert.equal(editorDialog().querySelector('[data-dialog-title]').textContent,'New note');
  // The picker offers, and the editor accepts, exactly the server's attachment types.
  const serverTypes=(await readFile(new URL('../skynet_app/notes.py',import.meta.url),'utf8')).match(/MEDIA_TYPES = \{([^}]*)\}/)[1];
  assert.deepEqual(el('note-editor-files').accept.split(',').sort(),[...serverTypes.matchAll(/"(\w+)":/g)].map(match=>'.'+match[1]).sort());
  // A name the server would refuse is reported before anything is written.
  el('note-editor-title').value='Refused';
  setFiles([new w.File(['x'],'plot.svg'),new w.File(['x'],'Screenshot 2026-10-05 at 10.00.00.png'),new w.File(['x'],'스크린샷.png'),new w.File(['x'],'paper.pdf')]);
  count=requests.length;
  el('note-editor-save').click();await settle();
  assert.deepEqual(writes(count),[],'An invalid attachment name sends nothing');
  assert.match(el('note-editor-error').textContent,/^Screenshot 2026-10-05 at 10\.00\.00\.png, 스크린샷\.png, paper\.pdf cannot be attached, so nothing was saved/);
  assert.ok(editorDialog().open);
  assert.deepEqual([...el('note-editor-folder').options].map(option=>[option.value,option.textContent]),
    [['','No folder'],['folder-0','egoisim'],['folder-1','warp-extension']]);
  assert.equal(el('note-editor-folder').value,'folder-0','A new note starts in the current folder');
  assert.match(el('note-editor-attachments').textContent,/No attachments/);
  el('note-editor-title').value='New plan';
  el('note-editor-markdown').value='# New plan\n\n![Plot](plot.svg)\n';
  el('note-editor-folder').value='folder-1';
  const files=[new w.File(['<svg/>'],'plot.svg',{type:'image/svg+xml'}),new w.File(['step,loss\n'],'loss-curve.csv')];
  setFiles(files);
  count=requests.length;
  el('note-editor-save').click();await settle();await settle();
  const sent=writes(count);
  assert.deepEqual(sent.map(request=>`${request.method} ${request.path}`),
    ['POST /api/notes',`PUT /api/notes/${created.id}/attachments/plot.svg`,`PUT /api/notes/${created.id}/attachments/loss-curve.csv`]);
  assert.deepEqual(JSON.parse(sent[0].body),{title:'New plan',markdown:'# New plan\n\n![Plot](plot.svg)\n',folder_id:'folder-1'});
  assert.equal(sent[1].body,files[0],'The File itself is the request body');
  assert.equal(sent[1].type,'image/svg+xml');
  assert.equal(sent[2].body,files[1]);
  assert.equal(sent[2].type,'application/octet-stream');
  assert.equal(editorDialog().open,false);
  assert.ok(viewer().open);
  assert.equal(viewer().querySelector('[data-dialog-title]').textContent,'New plan','The saved note reopens');
  assert.ok(since(count).includes(`GET /api/notes/${created.id}`));
  assert.ok(since(count).includes('GET /api/notes'),'The list reloads');
  viewer().querySelector('[data-dialog-close]').click();await settle();
  clearFiles();

  // Edit: PUT sends the loaded version; removed attachments are deleted after the content saves.
  el('notes-path').querySelector('[data-note-folder=""]').click();
  el('notes-table').querySelector('[data-note-folder="folder-1"]').click();
  assert.deepEqual(rows(),['latest','older']);
  const attachments=[{name:'old.png',media_type:'image/png',size_bytes:2048,url:'/api/notes/latest/attachments/old.png'},
    {name:'keep.csv',media_type:'text/csv',size_bytes:10,url:'/api/notes/latest/attachments/keep.csv'}];
  const loaded={...note,folder_id:'folder-1',markdown:'# UniDex latest\n',html:'<p>Body</p>',attachments};
  routes['GET /api/notes/latest']={note:loaded};
  el('notes-table').querySelector('[data-open-note="latest"]').click();await settle();
  el('workspace-note-edit').click();
  assert.ok(editorDialog().open);assert.ok(viewer().open,'The editor opens above the note');
  assert.equal(editorDialog().querySelector('[data-dialog-title]').textContent,'Edit note');
  assert.equal(el('note-editor-title').value,'UniDex latest');
  assert.equal(el('note-editor-folder').value,'folder-1');
  assert.equal(el('note-editor-markdown').value,'# UniDex latest\n');
  const attachmentRows=()=>[...el('note-editor-attachments').querySelectorAll('tbody tr[data-history-id]')].map(row=>row.dataset.historyId);
  assert.deepEqual(attachmentRows(),['old.png','keep.csv']);
  assert.match(el('note-editor-attachments').textContent,new RegExp(w.formatDataBytes(2048)));
  assert.equal(el('note-editor-attachments').querySelector('a').getAttribute('href'),'/api/notes/latest/attachments/old.png');
  routes['GET /api/notes']={notes:[{...note,folder_id:'folder-1',updated_at:'2026-10-03T00:00:00Z'},older,edited,rootNote],folders};
  count=requests.length;
  streams[0].emit('change',{v:1,topics:['notes']});await settle();
  assert.deepEqual(since(count),['GET /api/notes'],'A note being edited is not reloaded underneath the editor');
  el('note-editor-attachments').querySelector('[data-remove-attachment="old.png"]').click();
  assert.deepEqual(attachmentRows(),['keep.csv']);
  el('note-editor-markdown').value='# UniDex latest\n\nUpdated.\n';
  routes['PUT /api/notes/latest']=init=>({note:{...loaded,...JSON.parse(init.body),updated_at:'2026-10-04T00:00:00Z'}});
  routes['DELETE /api/notes/latest/attachments/old.png']={deleted:'old.png'};
  count=requests.length;
  el('note-editor-save').click();await settle();await settle();
  assert.deepEqual(writes(count).map(request=>`${request.method} ${request.path}`),['PUT /api/notes/latest','DELETE /api/notes/latest/attachments/old.png']);
  assert.deepEqual(JSON.parse(writes(count)[0].body),{title:'UniDex latest',markdown:'# UniDex latest\n\nUpdated.\n',folder_id:'folder-1',expected_updated_at:note.updated_at});
  assert.equal(editorDialog().open,false);
  assert.ok(since(count).includes('GET /api/notes/latest'),'The saved note reloads in the viewer');

  // A refused save shows the server detail, keeps the editor open and sends nothing else.
  el('workspace-note-edit').click();
  el('note-editor-attachments').querySelector('[data-remove-attachment="keep.csv"]').click();
  routes['PUT /api/notes/latest']=()=>json({detail:'This note changed elsewhere. Reload it before saving.'},409);
  count=requests.length;
  el('note-editor-save').click();await settle();
  assert.deepEqual(writes(count).map(request=>`${request.method} ${request.path}`),['PUT /api/notes/latest']);
  assert.ok(editorDialog().open,'The editor stays open on errors');
  assert.ok(visible('note-editor-error'));
  assert.match(el('note-editor-error').textContent,/changed elsewhere/);
  assert.equal(el('note-editor-save').disabled,false);
  assert.notEqual(editorDialog().dataset.blockClose,'true');
  editorDialog().querySelector('[data-dialog-close]').click();await settle();
  assert.equal(editorDialog().open,false);
  viewer().querySelector('[data-dialog-close]').click();await settle();

  // After a create succeeds, failed uploads leave an edit of that note that names each failed file.
  // A retry never creates a second note, and never sends again a file that was attached.
  const big={id:'2026-10-05-big',title:'Big',folder_id:null,created_at:'2026-10-05T10:00:00Z',updated_at:'2026-10-05T10:00:00Z'};
  const small={name:'small.png',media_type:'image/png',size_bytes:1,url:`/api/notes/${big.id}/attachments/small.png`};
  const tooLarge=()=>json({detail:'An attachment can be at most 64 MiB'},413);
  routes['POST /api/notes']=()=>full(big);
  routes[`PUT /api/notes/${big.id}/attachments/big.png`]=tooLarge;
  routes[`PUT /api/notes/${big.id}/attachments/small.png`]={attachment:small};
  routes[`PUT /api/notes/${big.id}/attachments/huge.png`]=tooLarge;
  routes[`GET /api/notes/${big.id}`]=full({...big,updated_at:'2026-10-05T10:00:01Z'},{attachments:[small]});
  el('new-note').click();
  el('note-editor-title').value='Big';
  setFiles(['big.png','small.png','huge.png'].map(name=>new w.File(['x'],name,{type:'image/png'})));
  count=requests.length;
  el('note-editor-save').click();await settle();await settle();
  assert.deepEqual(writes(count).map(request=>`${request.method} ${request.path}`),['POST /api/notes',
    ...['big.png','small.png','huge.png'].map(name=>`PUT /api/notes/${big.id}/attachments/${name}`)],'Every chosen file is tried once');
  assert.ok(editorDialog().open);
  assert.equal(el('note-editor-error').querySelector('.notification-message').textContent,
    'Saved the note, but big.png, huge.png were not attached: An attachment can be at most 64 MiB');
  assert.equal(editorDialog().querySelector('[data-dialog-title]').textContent,'Edit note');
  assert.deepEqual(attachmentRows(),['small.png'],'The editor lists what was attached');
  assert.equal(el('note-editor-files').files.length,0,'The file input no longer holds the files that were sent');
  assert.equal(el('note-editor-attachments').querySelector('[data-remove-attachment]').getAttribute('aria-label'),'Remove small.png');
  el('note-editor-attachments').querySelector('[data-remove-attachment="small.png"]').click();
  routes[`PUT /api/notes/${big.id}`]=init=>full({...big,...JSON.parse(init.body)},{attachments:[small]});
  routes[`DELETE /api/notes/${big.id}/attachments/small.png`]={deleted:'small.png'};
  routes[`PUT /api/notes/${big.id}/attachments/big.png`]={attachment:{name:'big.png'}};
  setFiles([new w.File(['x'],'big.png',{type:'image/png'})]);
  count=requests.length;
  el('note-editor-save').click();await settle();await settle();
  assert.deepEqual(writes(count).map(request=>`${request.method} ${request.path}`),[`PUT /api/notes/${big.id}`,
    `DELETE /api/notes/${big.id}/attachments/small.png`,`PUT /api/notes/${big.id}/attachments/big.png`],'A removed attachment is not uploaded again');
  assert.equal(JSON.parse(writes(count)[0].body).expected_updated_at,'2026-10-05T10:00:01Z','The retry uses the current version');
  assert.equal(editorDialog().open,false);
  viewer().querySelector('[data-dialog-close]').click();await settle();
  clearFiles();

  // A partly failed save, or a change skipped under the editor, reloads the viewer once the editor closes.
  let current={...loaded,markdown:'old body',html:'<p>old body</p>',attachments:[],updated_at:'2026-10-04T00:00:00Z'};
  routes['GET /api/notes']=()=>({notes:[{...note,folder_id:'folder-1',updated_at:current.updated_at},older,edited,rootNote],folders});
  routes['GET /api/notes/latest']=()=>({note:current});
  routes['PUT /api/notes/latest']=init=>{
    const body=JSON.parse(init.body);
    if(body.expected_updated_at!==current.updated_at)return json({detail:'This note changed elsewhere. Reload it and try again.'},409);
    current={...current,markdown:body.markdown,html:`<p>${body.markdown}</p>`,updated_at:'2026-10-05T11:00:00Z'};
    return {note:current};
  };
  routes['PUT /api/notes/latest/attachments/huge.png']=tooLarge;
  await w.loadNotes();
  el('notes-table').querySelector('[data-open-note="latest"]').click();await settle();
  assert.match(content.textContent,/old body/);
  el('workspace-note-edit').click();
  el('note-editor-markdown').value='NEW body';
  setFiles([new w.File(['x'],'huge.png',{type:'image/png'})]);
  el('note-editor-save').click();await settle();await settle();
  assert.match(el('note-editor-error').textContent,/^Saved the note, but huge\.png was not attached/);
  count=requests.length;
  streams[0].emit('change',{v:1,topics:['notes']});await settle();
  assert.deepEqual(since(count),['GET /api/notes'],'The viewer is not reloaded under the editor');
  assert.match(content.textContent,/old body/);
  editorDialog().querySelector('[data-dialog-close]').click();await settle();
  assert.match(content.textContent,/NEW body/,'Closing the editor shows the saved content');
  assert.match(el('workspace-note-date').textContent,new RegExp(`Updated ${w.formatDate(current.updated_at)}`));
  el('workspace-note-edit').click();
  assert.equal(el('note-editor-markdown').value,'NEW body','Editing again starts from the saved content');
  count=requests.length;
  el('note-editor-save').click();await settle();await settle();
  assert.deepEqual(writes(count).map(request=>`${request.method} ${request.path}`),['PUT /api/notes/latest']);
  assert.equal(editorDialog().open,false,'Saving again is not refused as a change made elsewhere');
  el('workspace-note-edit').click();
  current={...current,markdown:'Other tab',html:'<p>Other tab</p>',updated_at:'2026-10-05T12:00:00Z'};
  streams[0].emit('change',{v:1,topics:['notes']});await settle();
  assert.match(content.textContent,/NEW body/);
  editorDialog().querySelector('[data-dialog-close]').click();await settle();
  assert.match(content.textContent,/Other tab/,'A change made elsewhere shows once the editor closes unsaved');
  viewer().querySelector('[data-dialog-close]').click();await settle();
  clearFiles();

  // A linked note never opens over the editor; it opens once the editor closes.
  const linked=new URL(w.location.href);linked.searchParams.set('note','latest');w.history.replaceState(null,'',linked);
  el('new-note').click();
  await w.loadNotes();
  assert.equal(viewer().open,false,'A linked note waits while the editor is open');
  editorDialog().querySelector('[data-dialog-close]').click();await settle();
  assert.ok(viewer().open);
  assert.equal(viewer().querySelector('[data-dialog-title]').textContent,'UniDex latest');
  viewer().querySelector('[data-dialog-close]').click();await settle();

  // Delete asks first; a cancelled confirmation sends nothing, a failed delete keeps the note open.
  el('notes-table').querySelector('[data-open-note="latest"]').click();await settle();
  const confirmation=()=>w.document.querySelector('dialog[data-app-confirmation]');
  count=requests.length;
  el('workspace-note-delete').click();await settle();
  assert.match(confirmation().textContent,/UniDex latest/);
  confirmation().querySelector('button[type="button"]:not([data-dialog-close])').click();await settle();
  assert.deepEqual(writes(count),[]);
  assert.ok(viewer().open);
  routes['DELETE /api/notes/latest']=()=>json({detail:'Database unavailable'},503);
  el('workspace-note-delete').click();await settle();
  confirmation().querySelector('button[type="submit"]').click();await settle();
  assert.ok(viewer().open);
  assert.match(el('workspace-note-error').textContent,/Database unavailable/);
  assert.equal(el('workspace-note-delete').disabled,false);
  routes['DELETE /api/notes/latest']={deleted:'latest'};
  count=requests.length;
  el('workspace-note-delete').click();await settle();
  confirmation().querySelector('button[type="submit"]').click();await settle();
  assert.deepEqual(writes(count).map(request=>`${request.method} ${request.path}`),['DELETE /api/notes/latest']);
  assert.equal(viewer().open,false);
  assert.ok(since(count).includes('GET /api/notes'),'The list reloads after delete');
  console.log('Notes: shared navigation/table/dialog, isolated loading, search across folders, refresh, download, close and switching passed.');
  console.log('Notes: server rendering, in-content links, live topics, create/edit/upload/delete and error handling passed.');
} finally {for(const observer of observers)observer.disconnect();await settle();w.close();}
