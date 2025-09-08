(function(){
  const overlay = document.getElementById('scan-overlay');
  const menu = document.getElementById('ctx-menu');
  const toast = document.getElementById('toast');
  const progressFill = document.getElementById('progress-fill');
  const progressText = document.getElementById('progress-text');
  let ctxRow = null; // current row element
  let progressTimer = null;

  function startProgress(){
    let pct = 0;
    progressFill.style.width = '0%';
    progressText.textContent = '0%';
    clearInterval(progressTimer);
    progressTimer = setInterval(()=>{
      pct = Math.min(pct + 1, 100);
      progressFill.style.width = pct + '%';
      progressText.textContent = pct + '%';
      if (pct >= 100) clearInterval(progressTimer);
    },50);
  }

  function stopProgress(){
    clearInterval(progressTimer);
    progressFill.style.width = '100%';
    progressText.textContent = '100%';
  }

  function showOverlay(){
    overlay.style.display = 'grid';
    startProgress();
  }
  function hideOverlay(){
    overlay.style.display = 'none';
    stopProgress();
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

  async function addFavoriteFromRow(tr){
    if(!tr){ showToast('No row selected', false); return; }
    const payload = {
      ticker: tr.dataset.tkr || '',
      direction: (tr.dataset.dir || 'UP').toUpperCase(),
      rule: tr.dataset.rule || '',
      interval: document.querySelector('select[name="interval"]')?.value || '15m'
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
      const data = await res.json();
      if (data?.ok) showToast(`Added ${payload.ticker} to Favorites`);
      else showToast(data?.error || 'Failed to add favorite', false);
    }catch(e){
      showToast('Network error adding favorite', false);
    }
  }

  // Delegate inside #scan-results so it works after HTMX swaps
  function bindResultsDelegates(){
    const root = document.getElementById('scan-results');
    if(!root) return;

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

  // Global listeners
  document.addEventListener('click', hideMenu);
  window.addEventListener('blur', hideMenu);
  window.addEventListener('resize', hideMenu);
  window.addEventListener('scroll', hideMenu);

  document.getElementById('ctx-add-fav').addEventListener('click', async function(ev){
    ev.stopPropagation();
    hideMenu();
    await addFavoriteFromRow(ctxRow);
  });

  // HTMX hooks -> overlay + (re)bind delegates
  document.body.addEventListener('htmx:beforeRequest', (evt)=>{
    if (evt.detail.elt?.closest && evt.detail.elt.closest('#scan-form')) showOverlay();
  });
  document.body.addEventListener('htmx:afterSwap', (evt)=>{
    if (evt.detail.target?.id === 'scan-results') {
      bindResultsDelegates();
      hideOverlay();
    }
  });
  document.body.addEventListener('htmx:responseError', hideOverlay);
  document.addEventListener('DOMContentLoaded', bindResultsDelegates);
})();

