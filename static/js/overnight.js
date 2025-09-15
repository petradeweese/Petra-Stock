(function(){
  const tbody = document.getElementById('queue-body');
  const addBtn = document.getElementById('add-item');
  const saveBtn = document.getElementById('save-batch');
  const startBtn = document.getElementById('start-now');
  const batchesDiv = document.getElementById('batches');
  const winStart = document.getElementById('win-start');
  const winEnd = document.getElementById('win-end');
  const savePrefs = document.getElementById('save-prefs');
  const winInfo = document.getElementById('window-info');
  let windowNotice = '';

  function renumber(){
    [...tbody.children].forEach((tr, idx) => {
      const cell = tr.querySelector('td.pos');
      if(cell) cell.textContent = idx + 1;
    });
  }

  function addRow(item={}){
    const tr = document.createElement('tr');
    tr.draggable = true;
    tr.innerHTML = `<td class="pos"></td>
      <td><input value="${item.pattern||''}"></td>
      <td><input value="${item.universe||''}"></td>
      <td><input value='${item.settings?JSON.stringify(item.settings):''}'></td>
      <td><button class="rm">🗑</button></td>`;
    tbody.appendChild(tr);
    renumber();
  }

  addBtn?.addEventListener('click', ()=>addRow());

  tbody.addEventListener('click', e=>{
    if(e.target.classList.contains('rm')){
      e.target.closest('tr').remove();
      renumber();
    }
  });

  let dragEl=null;
  tbody.addEventListener('dragstart', e=>{dragEl=e.target;});
  tbody.addEventListener('dragover', e=>{e.preventDefault();});
  tbody.addEventListener('drop', e=>{
    e.preventDefault();
    const target = e.target.closest('tr');
    if(dragEl && target && dragEl!==target){
      tbody.insertBefore(dragEl, target.nextSibling);
      renumber();
    }
  });

  saveBtn?.addEventListener('click', async ()=>{
    const items=[...tbody.children].map(tr=>{
      const tds=tr.children;
      return {
        pattern: tds[1].firstChild.value,
        universe: tds[2].firstChild.value,
        settings: JSON.parse(tds[3].firstChild.value || '{}')
      };
    });
    await fetch('/overnight/batches',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({items})});
    loadBatches();
  });

  async function loadBatches(){
    const res = await fetch('/overnight/batches');
    const data = await res.json();
    batchesDiv.innerHTML = data.batches.map(b=>`<div class="batch ${b.status}" data-id="${b.id}">${b.label||b.id} - ${b.status} (${b.items_done||0}/${b.items_total||0}) <span class="win">Runs ${windowNotice}</span> <button class="start" ${b.status==='running'||b.status==='complete'?'disabled':''}>Start Now</button> <button class="pause">Pause</button> <button class="resume">Resume</button> <button class="cancel">Cancel</button> <a class="csv" href="/overnight/batches/${b.id}/csv">CSV</a></div>`).join('');
  }

  batchesDiv.addEventListener('click', async e=>{
    const div = e.target.closest('.batch');
    if(!div) return;
    const id=div.dataset.id;
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
      await fetch(`/overnight/batches/${id}/pause`,{method:'POST'}); loadBatches();
    }else if(e.target.classList.contains('resume')){
      await fetch(`/overnight/batches/${id}/resume`,{method:'POST'}); loadBatches();
    }else if(e.target.classList.contains('cancel')){
      await fetch(`/overnight/batches/${id}/cancel`,{method:'POST'}); loadBatches();
    }
  });

  startBtn?.addEventListener('click', ()=>{
    const first = batchesDiv.querySelector('.batch');
    if(first) fetch(`/overnight/batches/${first.dataset.id}/start_now`,{method:'POST'}).then(loadBatches);
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
    await fetch('/overnight/prefs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({window_start:winStart.value, window_end:winEnd.value})});
    const pref = await fetch('/overnight/prefs').then(r=>r.json());
    windowNotice = `${pref.window_start}-${pref.window_end}`;
    updateWinInfo(pref);
    loadBatches();
  });

  init();
})();
