(function(){
  const queue = document.getElementById('queue');
  const addBtn = document.getElementById('add-item');
  const saveBtn = document.getElementById('save-batch');
  const startBtn = document.getElementById('start-now');
  const tmpl = document.getElementById('row-template');
  const batchesDiv = document.getElementById('batches');
  const showFinished = document.getElementById('show-finished');
  const clearFinishedBtn = document.getElementById('clear-finished');
  const winStart = document.getElementById('win-start');
  const winEnd = document.getElementById('win-end');
  const savePrefs = document.getElementById('save-prefs');
  const winInfo = document.getElementById('window-info');
  let windowNotice = '';
  const FINISHED_STATUSES = new Set(['complete', 'canceled', 'failed']);
  const REMOVAL_CONFIRM = 'Remove finished batch? CSV links may no longer be shown on this page, but data already exported remains.';

  function renumber(){
    [...queue.children].forEach((row, idx)=>{
      const cell = row.querySelector('.pos');
      if(cell) cell.textContent = idx + 1;
    });
  }

  function addRow(values={}){
    const node = tmpl.content.firstElementChild.cloneNode(true);
    const form = node.querySelector('form');
    Object.entries(values).forEach(([k,v])=>{
      const el = form.elements[k];
      if(el) el.value = v;
    });
    const emailDiv = form.querySelector('select[name="email_checkbox"]')?.closest('div');
    if(emailDiv) emailDiv.style.display = 'none';
    node.querySelector('.rm').addEventListener('click', ()=>{ node.remove(); renumber(); });
    queue.appendChild(node);
    renumber();
  }

  addBtn?.addEventListener('click', ()=>addRow());

  saveBtn?.addEventListener('click', async ()=>{
    const items = [...queue.querySelectorAll('form')].map(f=>{
      const fd = new FormData(f);
      return Object.fromEntries(fd.entries());
    });
    await fetch('/overnight/batches', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({items})});
    queue.innerHTML='';
    renumber();
    loadBatches();
  });

  function renderBatch(b){
    const finished = FINISHED_STATUSES.has(b.status);
    const actions = [];
    if(!finished){
      actions.push(`<button class="start" ${b.status==='running'?'disabled':''}>Start Now</button>`);
      actions.push('<button class="pause">Pause</button>');
      actions.push('<button class="resume">Resume</button>');
      actions.push('<button class="cancel">Cancel</button>');
    }else{
      actions.push('<button class="remove">Remove</button>');
    }
    actions.push(`<a class="csv" href="/overnight/batches/${b.id}/csv">CSV</a>`);
    return `<div class="batch ${b.status}" data-id="${b.id}" data-status="${b.status}">${b.label||b.id} - ${b.status} (${b.items_done||0}/${b.items_total||0}) <span class="win">Runs ${windowNotice}</span> ${actions.join(' ')}</div>`;
  }

  async function loadBatches(forceDefault=false){
    if(forceDefault && showFinished){
      showFinished.checked = false;
    }
    const params = new URLSearchParams();
    if(showFinished?.checked){
      params.set('include_finished', 'true');
    }
    const qs = params.toString();
    const res = await fetch(qs ? `/overnight/batches?${qs}` : '/overnight/batches');
    const data = await res.json();
    batchesDiv.innerHTML = data.batches.map(renderBatch).join('');
    if(clearFinishedBtn){
      clearFinishedBtn.disabled = !data.finished_count;
    }
  }

  batchesDiv.addEventListener('click', async e=>{
    const div = e.target.closest('.batch');
    if(!div) return;
    const id = div.dataset.id;
    if(e.target.classList.contains('start')){
      const running = batchesDiv.querySelector('.batch.running');
      if(running && running!==div){
        if(!confirm('Another batch is running. Queue after current?')) return;
      }
      const resp = await fetch(`/overnight/batches/${id}/start_now`,{method:'POST'});
      const data = await resp.json();
      if(data.status==='queued_after_current') alert('Queued after current batch');
      loadBatches();
    }else if(e.target.classList.contains('pause')){
      await fetch(`/overnight/batches/${id}/pause`,{method:'POST'});
      loadBatches();
    }else if(e.target.classList.contains('resume')){
      await fetch(`/overnight/batches/${id}/resume`,{method:'POST'});
      loadBatches();
    }else if(e.target.classList.contains('cancel')){
      await fetch(`/overnight/batches/${id}/cancel`,{method:'POST'});
      loadBatches();
    }else if(e.target.classList.contains('remove')){
      if(!confirm(REMOVAL_CONFIRM)) return;
      const resp = await fetch(`/overnight/batches/${id}`,{method:'DELETE'});
      if(resp.status===409){
        alert('Batch is still active.');
        return;
      }
      if(!resp.ok){
        alert('Failed to remove batch.');
        return;
      }
      await loadBatches(true);
    }
  });

  startBtn?.addEventListener('click', ()=>{
    const first = batchesDiv.querySelector('.batch');
    if(first) fetch(`/overnight/batches/${first.dataset.id}/start_now`,{method:'POST'}).then(loadBatches);
  });

  showFinished?.addEventListener('change', ()=>{
    loadBatches();
  });

  clearFinishedBtn?.addEventListener('click', async ()=>{
    if(clearFinishedBtn.disabled) return;
    if(!confirm(REMOVAL_CONFIRM)) return;
    const resp = await fetch('/overnight/batches/clear_finished',{method:'POST'});
    if(!resp.ok){
      alert('Failed to clear finished batches.');
      return;
    }
    await loadBatches(true);
  });

  function updateWinInfo(pref){
    if(!winInfo) return;
    const now = pref.now;
    const toMin = t=>{const [h,m]=t.split(':').map(Number);return h*60+m;};
    const n = toMin(now), s = toMin(pref.window_start), e = toMin(pref.window_end);
    const active = s<=e ? (n>=s && n<e) : (n>=s || n<e);
    let msg = 'active';
    if(!active){
      const diff = (s - n + 1440) % 1440;
      msg = `starts in ${Math.floor(diff/60)}h ${diff%60}m`;
    }
    winInfo.textContent = `Window ${pref.window_start}-${pref.window_end} (${msg})`;
  }

  async function init(){
    const pref = await fetch('/overnight/prefs').then(r=>r.json());
    winStart.value = pref.window_start;
    winEnd.value = pref.window_end;
    windowNotice = `${pref.window_start}-${pref.window_end}`;
    updateWinInfo(pref);
    loadBatches();
  }

  savePrefs?.addEventListener('click', async ()=>{
    await fetch('/overnight/prefs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({window_start: winStart.value, window_end: winEnd.value})});
    const pref = await fetch('/overnight/prefs').then(r=>r.json());
    windowNotice = `${pref.window_start}-${pref.window_end}`;
    updateWinInfo(pref);
    loadBatches();
  });

  init();
})();
