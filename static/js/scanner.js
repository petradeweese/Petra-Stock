(function(){
  const menu = document.getElementById('ctx-menu');
  const toast = document.getElementById('toast');
  const progressFill = document.getElementById('progress-fill');
  const progressText = document.getElementById('progress-text');
  const progressStatus = document.getElementById('progress-status');
  const overlay = document.getElementById('scan-overlay');
  let ctxRow = null; // current row element
  let pollTimer = null;
  let stillTimer = null;
  let maxTimer = null; // legacy timeout (unused but kept for compatibility)
  const HIDE_ARCHIVED_KEY = 'resultsHideArchived';

  function startProgress(){
    progressFill.style.width = '0%';
    progressText.textContent = '0%';
    progressStatus.textContent = '';
    overlay.style.display = 'grid';
    if(stillTimer) clearInterval(stillTimer);
    if(maxTimer) clearTimeout(maxTimer);
  }

  function stopProgress(){
    overlay.style.display = 'none';
    if(pollTimer){ clearTimeout(pollTimer); pollTimer = null; }
    if(stillTimer) clearInterval(stillTimer);
    if(maxTimer){ clearTimeout(maxTimer); maxTimer = null; }
  }

  function showMenu(x,y, tr){
    ctxRow = tr;
    menu.style.left = x+'px';
    menu.style.top  = y+'px';
    menu.hidden = false;
  }
  function hideMenu(){ menu.hidden = true; ctxRow = null; }

  function showToast(msg, ok=true){
    toast.textContent = msg;
    toast.style.borderColor = ok ? '#2e7d32' : '#8b0000';
    toast.style.background  = ok ? '#0f3311' : '#2b0f0f';
    toast.hidden = false;
    setTimeout(()=>toast.hidden=true, 1800);
  }

  function shouldHideArchived(){
    return localStorage.getItem(HIDE_ARCHIVED_KEY) === '1';
  }

  function setHideArchivedState(active){
    localStorage.setItem(HIDE_ARCHIVED_KEY, active ? '1' : '0');
  }

  function updateArchivedToggle(btn, active){
    if(!btn) return;
    btn.textContent = active ? 'Show Archived' : 'Remove Archived';
    btn.classList.toggle('toggle-active', active);
    btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    btn.title = active ? 'Show archived results' : 'Hide archived results';
  }

  function applyArchivedFilter(root, active){
    const table = root?.querySelector('#results-table');
    if(!table) return;
    table.querySelectorAll('tbody tr.row-hover').forEach(tr => {
      const isArchived = (tr.dataset.archived || '').toLowerCase() === 'true';
      tr.style.display = active && isArchived ? 'none' : '';
    });
  }

  function initArchivedToggle(root){
    const btn = root?.querySelector('#btn-remove-archived') || document.getElementById('btn-remove-archived');
    if(!btn) return;
    updateArchivedToggle(btn, shouldHideArchived());
    applyArchivedFilter(root || document, shouldHideArchived());
    if(btn.dataset.removeArchivedBound === '1'){
      return;
    }
    btn.addEventListener('click', () => {
      const next = !shouldHideArchived();
      setHideArchivedState(next);
      updateArchivedToggle(btn, next);
      applyArchivedFilter(root || document, next);
    });
    btn.dataset.removeArchivedBound = '1';
  }

  async function addFavoriteFromRow(tr){
    if(!tr){ showToast('No row selected', false); return; }
    const form = document.getElementById('scan-form');
    const settings = {};
    if(form){
      const fd = new FormData(form);
      for(const [k,v] of fd.entries()) settings[k]=v;
    }
    const payload = {
      ticker: tr.dataset.tkr || '',
      direction: (tr.dataset.dir || 'UP').toUpperCase(),
      rule: tr.dataset.rule || '',
      interval: document.querySelector('select[name="interval"]')?.value || '15m',
      ref_avg_dd: parseFloat(tr.dataset.dd || '0'),
      roi_snapshot: parseFloat(tr.dataset.roi || '0'),
      hit_pct_snapshot: parseFloat(tr.dataset.hit || '0'),
      dd_pct_snapshot: parseFloat(tr.dataset.dd || '0'),
      rule_snapshot: tr.dataset.rule || '',
      settings_json_snapshot: settings
    };
    if(!payload.ticker || !payload.rule){
      showToast('Missing ticker or rule', false);
      return;
    }
    try{
      const res = await fetch('/favorites/add', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify(payload)
      });
      const text = await res.text();
      let data = null;
      if(text){
        try{ data = JSON.parse(text); }
        catch(err){ data = null; }
      }
      if(!res.ok){
        const message = (data && data.error) || (text && text.trim()) || res.statusText || 'Failed to add favorite';
        showToast(message, false);
        return;
      }
      if(data?.ok) showToast(`Added ${payload.ticker} to Favorites`);
      else showToast((data && data.error) || 'Failed to add favorite', false);
    }catch(e){
      const message = (e && e.message) ? e.message : 'Network error adding favorite';
      showToast(message, false);
    }
  }

  // Delegate inside #scan-results so it works after HTMX swaps
  function bindResultsDelegates(){
    const root = document.getElementById('scan-results');
    if(!root) return;

    initArchivedToggle(root);

    // Right-click on any results row
    root.addEventListener('contextmenu', function(e){
      const tr = e.target.closest('tr.row-hover');
      if(!tr) return;
      e.preventDefault();
      showMenu(e.clientX, e.clientY, tr);
    });

    // Left-click on row -> add to favorites (event delegation)
    root.addEventListener('click', async function(e){
      const tr = e.target.closest('tbody tr.row-hover');
      if(tr){
        await addFavoriteFromRow(tr);
        return;
      }

      const btn = e.target.closest('#btn-archive');
      if(!btn) return;

      try {
        const form = document.getElementById('scan-form');
        const params = {};
        if(form){
          const fd = new FormData(form);
          for (const [k,v] of fd.entries()) params[k]=v;
        }

        const rows = [...root.querySelectorAll('tbody tr.row-hover')].map(tr=>{
          const tds = tr.querySelectorAll('td');
          return {
            ticker: tr.dataset.tkr,
            direction: tr.dataset.dir || 'UP',
            avg_roi_pct: parseFloat(tr.dataset.roi || '0'),
            hit_pct: parseFloat(tr.dataset.hit || '0'),
            support: parseInt(tr.dataset.supp || tds[4].textContent || '0', 10),
            avg_dd_pct: parseFloat(tr.dataset.dd || '0'),
            stability: parseFloat(tr.dataset.stab || '0'),
            rule: tr.dataset.rule || ''
          };
        });

        if (rows.length === 0) {
          showToast('No rows to archive', false);
          return;
        }

        const res = await fetch('/archive/save', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({params, rows})
        });
        const data = await res.json();
        if (data?.ok) {
          showToast('Saved to Archive');
          if (data.run_id) setTimeout(()=>{ window.location.href = `/results/${data.run_id}` }, 750);
        } else {
          showToast(data?.error || 'Failed to save', false);
        }
      } catch(e){
        showToast('Network error saving archive', false);
      }
    });
  }

  function startPolling(taskId, silent){
    localStorage.setItem('scanTaskInfo', JSON.stringify({taskId, silent}));
    let last = Date.now();
    const baseDelay = window.SCAN_STATUS_POLL_MS || 2000;
    let delay = baseDelay;
    let lastUpdate = '';
    stillTimer = setInterval(()=>{
      if(Date.now()-last > 10000) progressStatus.textContent = 'still running...';
    }, 5000);

    const poll = async function(){
      try{
        const res = await fetch(`/scanner/status/${taskId}?t=${Date.now()}`, {cache: 'no-store'});
        if(!res.ok) throw new Error('');
        const data = await res.json();
        last = Date.now();
        progressFill.style.width = (data.percent || 0) + '%';
        progressText.textContent = Math.floor(data.percent || 0) + '%';
        progressStatus.textContent = data.message || '';
        if(data.updated_at && data.updated_at !== lastUpdate){
          delay = baseDelay;
          lastUpdate = data.updated_at;
        }
        if(data.state === 'running' || data.state === 'queued'){
          pollTimer = setTimeout(poll, delay);
          delay = Math.min(delay * 1.5, 30000);
        }else if(data.state === 'succeeded'){
          progressFill.style.width = '100%';
          progressText.textContent = '100%';
          try{
            const html = await fetch(`/scanner/results/${taskId}?t=${Date.now()}`, {cache: 'no-store'}).then(r=>{
              if(!r.ok) throw new Error('');
              return r.text();
            });
            const target = document.getElementById('scan-results');
            if(target) target.innerHTML = html;
            bindResultsDelegates();
          }catch(e){
            showToast('Failed to load results', false);
          }
          localStorage.removeItem('scanTaskInfo');
          stopProgress();
        }else{
          localStorage.removeItem('scanTaskInfo');
          stopProgress();
          showToast('Scan failed', false);
        }
      }catch(e){
        pollTimer = setTimeout(poll, delay);
        delay = Math.min(delay * 1.5, 30000);
      }
    };
    poll();
  }

  function runScanner(){
    const form = document.getElementById('scan-form');
    if(!form) return;
    form.addEventListener('submit', async function(ev){
      ev.preventDefault();
      const fd = new FormData(form);
      const silent = fd.get('silent') === 'true';
      if(!silent) startProgress();
      let taskId = '';
      try{
        const res = await fetch(form.action, {method:'POST', body: fd});
        const data = await res.json();
        taskId = data.task_id;
        if(!taskId) throw new Error('no task');
      }catch(e){
        stopProgress();
        showToast('Failed to start scan', false);
        return;
      }
      startPolling(taskId, silent);
    });
  }

  // Global listeners
  document.addEventListener('click', hideMenu);
  window.addEventListener('blur', hideMenu);
  window.addEventListener('resize', hideMenu);
  window.addEventListener('scroll', hideMenu);

  if(document.body){
    document.body.addEventListener('htmx:responseError', function(evt){
      const xhr = evt?.detail?.xhr;
      const responseText = typeof xhr?.responseText === 'string' ? xhr.responseText.trim() : '';
      const statusLabel = xhr?.status ? `${xhr.status}${xhr.statusText ? ' ' + xhr.statusText : ''}`.trim() : '';
      const message = responseText || statusLabel || 'Request failed';
      showToast(message, false);
    });
  }

  document.getElementById('ctx-add-fav').addEventListener('click', async function(ev){
    ev.stopPropagation();
    hideMenu();
    await addFavoriteFromRow(ctxRow);
  });

  document.addEventListener('DOMContentLoaded', function(){
    bindResultsDelegates();
    runScanner();
    const existing = localStorage.getItem('scanTaskInfo');
    if(existing){
      try{
        const info = JSON.parse(existing);
        if(!info.silent){
          startProgress();
          startPolling(info.taskId, false);
        }
      }catch(e){ localStorage.removeItem('scanTaskInfo'); }
    }
  });
})();

