/* Collection views share their forms and data; switching views never submits work. */
function setCollectionView(view, {persist = true, focus = false} = {}) {
  const legacy = view === 'cycles';
  if (legacy) view = 'setup';
  if (!['recordings', 'live', 'setup'].includes(view)) view = 'live';
  document.querySelectorAll('[data-collection-view]').forEach(panel => { panel.hidden = panel.dataset.collectionView !== view; });
  document.querySelectorAll('[data-collection-tab]').forEach(button => {
    const selected = button.dataset.collectionTab === view;
    button.setAttribute('aria-selected', String(selected));
    button.tabIndex = selected ? 0 : -1;
  });
  if (persist) {
    const url = new URL(location.href);
    if (url.searchParams.get('collection_view') !== view) {
      url.searchParams.set('collection_view', view);
      history.pushState(null, '', url);
    }
  }
  document.querySelector('.page-actions [data-collection-guide="record"]').textContent = 'Help';
  if (legacy) document.querySelector('#collection-legacy-cycles').open = true;
  if (view === 'live' || view === 'recordings') window.loadLiveXR?.();
  if (focus) document.querySelector(`[data-collection-tab="${view}"]`).focus({preventScroll: true});
}
function chooseCollectionRecording(digest) {
  const select = document.querySelector('#capture-cycle-recording');
  if (![...select.options].some(option => option.value === digest)) return;
  select.value = digest;
  select.dispatchEvent(new Event('change', {bubbles: true}));
  setCollectionView('cycles');
  select.focus({preventScroll: true});
  document.querySelector('#collection-cycle-composer').scrollIntoView({block: 'start'});
}
document.querySelectorAll('[data-collection-tab]').forEach(button => {
  button.addEventListener('click', () => setCollectionView(button.dataset.collectionTab));
  button.addEventListener('keydown', event => {
    const keys = ['ArrowLeft', 'ArrowRight', 'Home', 'End'];
    if (!keys.includes(event.key)) return;
    event.preventDefault();
    const tabs = [...document.querySelectorAll('[data-collection-tab]')];
    const index = tabs.indexOf(button);
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    setCollectionView(tabs[next].dataset.collectionTab, {focus: true});
  });
});
document.querySelectorAll('[data-collection-go]').forEach(button => button.addEventListener('click', () => {
  if (button.dataset.collectionGuide === 'record' && !document.querySelector('#collection-view-live').hidden) {
    const help = document.querySelector('#live-xr-help');
    help.open = true;
    help.scrollIntoView({block: 'start'});
    help.querySelector('summary').focus({preventScroll: true});
    return;
  }
  setCollectionView(button.dataset.collectionGo, {focus: true});
  const targets = {record: '#collection-view-setup', dexverse: '#collection-dexverse-setup'};
  const target = document.querySelector(targets[button.dataset.collectionGuide] || '#collection-headset-guide');
  if (target.tagName === 'DETAILS') target.open = true;
  for (let parent = target.parentElement; parent; parent = parent.parentElement) {
    if (parent.tagName === 'DETAILS') parent.open = true;
  }
  target.scrollIntoView({block: 'start'});
}));
const collectionTutorial = document.querySelector('#collection-tutorial-button');
if (collectionTutorial) {
  document.querySelector('#collection-adapter-tutorial-slot').append(collectionTutorial);
  collectionTutorial.textContent = 'Adapter tutorial';
  collectionTutorial.setAttribute('aria-label', 'Start advanced collection adapter tutorial');
  collectionTutorial.title = 'Advanced tutorial: configure collection adapters. Headset recording instructions are above.';
}
window.addEventListener('popstate', () => setCollectionView(new URLSearchParams(location.search).get('collection_view'), {persist: false}));
setCollectionView(new URLSearchParams(location.search).get('collection_view'), {persist: false});

document.querySelector('#show-collection-import').addEventListener('click', event => {
  revealPanel(document.querySelector('#collection-import-drawer'), {
    launcher: event.currentTarget, focusTarget: document.querySelector('#local-capture-file'),
  });
});
