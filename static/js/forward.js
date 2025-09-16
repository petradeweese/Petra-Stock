(function(){
  'use strict';

  const table = document.getElementById('forward-table');
  const tbody = document.getElementById('forward-tbody');
  const emptyState = document.getElementById('forward-empty');
  const loadingEl = document.getElementById('forward-loading');
  const errorBox = document.getElementById('forward-error');
  const runBtn = document.getElementById('forward-run');
  const toast = document.getElementById('toast');

  if(!table || !tbody || !runBtn){
    return;
  }

  let favorites = [];
  let loading = false;

  function showToast(message, ok=true){
    if(!toast){ return; }
    toast.textContent = message;
    toast.style.borderColor = ok ? '#2e7d32' : '#8b0000';
    toast.style.background = ok ? '#0f3311' : '#2b0f0f';
    toast.hidden = false;
    setTimeout(()=>{ toast.hidden = true; }, 2000);
  }

  function coerceNumber(value){
    const num = Number(value);
    return Number.isFinite(num) ? num : null;
  }

  function formatPercent(value, decimals=1){
    const num = coerceNumber(value);
    if(num === null){
      return '—';
    }
    return `${num.toFixed(decimals)}%`;
  }

  function formatWindow(value, unit){
    const num = coerceNumber(value);
    const label = (unit || '').trim();
    if(num === null && !label){
      return '—';
    }
    if(num === null){
      return label || '—';
    }
    if(!label){
      return `${num.toFixed(0)}`;
    }
    const compact = label.endsWith('s') ? label.slice(0, -1) : label;
    return `${num.toFixed(num % 1 === 0 ? 0 : 1)} ${compact}`;
  }

  function formatHits(fav){
    const hit = coerceNumber(fav.hit_pct_snapshot ?? fav.hit_pct);
    const supportLabel = fav.support_display || (fav.support_count ?? '—');
    const hitText = hit === null ? '' : `${hit.toFixed(0)}%`;
    if(hitText && supportLabel && supportLabel !== '—'){
      return `${hitText} / ${supportLabel}`;
    }
    if(hitText){
      return hitText;
    }
    return supportLabel && supportLabel !== undefined ? String(supportLabel) : '—';
  }

  function formatRoi(fav){
    const roi = coerceNumber(fav.roi_snapshot ?? fav.avg_roi_pct);
    if(roi === null){
      return '—';
    }
    return `${roi.toFixed(1)}%`;
  }

  function renderFavorites(items){
    favorites = Array.isArray(items) ? items : [];
    loading = false;
    loadingEl.hidden = true;
    errorBox.hidden = true;

    if(!favorites.length){
      table.hidden = true;
      emptyState.hidden = false;
      runBtn.disabled = true;
      tbody.innerHTML = '';
      return;
    }

    emptyState.hidden = true;
    table.hidden = false;
    runBtn.disabled = false;

    tbody.innerHTML = '';
    favorites.forEach((fav) => {
      const tr = document.createElement('tr');
      tr.dataset.favId = String(fav.id ?? '');

      const tickerTd = document.createElement('td');
      const tickerStrong = document.createElement('strong');
      tickerStrong.textContent = (fav.ticker || '').toUpperCase();
      tickerTd.appendChild(tickerStrong);

      const directionTd = document.createElement('td');
      directionTd.textContent = (fav.direction || 'UP').toUpperCase();

      const intervalTd = document.createElement('td');
      intervalTd.textContent = fav.interval || '';

      const lookbackTd = document.createElement('td');
      lookbackTd.textContent = fav.lookback_display || formatPercent(fav.lookback_years, 1).replace('%','y');

      const targetTd = document.createElement('td');
      targetTd.textContent = formatPercent(fav.target_pct ?? fav.target);

      const stopTd = document.createElement('td');
      stopTd.textContent = formatPercent(fav.stop_pct ?? fav.stop);

      const windowTd = document.createElement('td');
      windowTd.textContent = formatWindow(fav.window_value, fav.window_unit);

      const hitsTd = document.createElement('td');
      hitsTd.textContent = formatHits(fav);

      const roiTd = document.createElement('td');
      roiTd.textContent = formatRoi(fav);

      const ruleTd = document.createElement('td');
      ruleTd.className = 'rule-td';
      const code = document.createElement('code');
      code.textContent = fav.rule || fav.rule_snapshot || '';
      ruleTd.appendChild(code);

      const actionsTd = document.createElement('td');
      const form = document.createElement('form');
      form.method = 'post';
      form.action = `/favorites/delete/${fav.id}`;
      form.style.margin = '0';
      const btn = document.createElement('button');
      btn.type = 'submit';
      btn.className = 'del-btn';
      btn.title = 'Remove from favorites';
      btn.setAttribute('aria-label', 'Remove from favorites');
      btn.textContent = '🗑';
      form.appendChild(btn);
      actionsTd.appendChild(form);

      [tickerTd, directionTd, intervalTd, lookbackTd, targetTd, stopTd, windowTd, hitsTd, roiTd, ruleTd, actionsTd].forEach((td) => {
        tr.appendChild(td);
      });

      tbody.appendChild(tr);
    });
  }

  async function loadFavorites(){
    if(loading){
      return;
    }
    loading = true;
    runBtn.disabled = true;
    table.hidden = true;
    emptyState.hidden = true;
    errorBox.hidden = true;
    loadingEl.hidden = false;

    try {
      const res = await fetch('/api/forward/favorites');
      if(!res.ok){
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      renderFavorites(data?.favorites || []);
    } catch (err){
      console.error('Failed to load forward favorites', err);
      loading = false;
      loadingEl.hidden = true;
      runBtn.disabled = true;
      errorBox.hidden = false;
    }
  }

  async function runForwardTests(){
    if(!favorites.length){
      showToast('No favorites to test', false);
      return;
    }
    const ids = favorites
      .map((fav) => {
        const id = Number.parseInt(String(fav.id ?? ''), 10);
        return Number.isFinite(id) ? id : null;
      })
      .filter((id) => id !== null);

    if(!ids.length){
      showToast('No favorites to test', false);
      return;
    }

    const originalLabel = runBtn.textContent;
    runBtn.disabled = true;
    runBtn.textContent = 'Queuing…';

    try {
      const res = await fetch('/api/forward/run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({favorite_ids: ids}),
      });
      let data = null;
      try {
        data = await res.json();
      } catch (_err){
        data = null;
      }
      if(!res.ok || !data || data.ok === false){
        const errorMsg = data?.error || 'Failed to queue forward tests';
        showToast(errorMsg, false);
        return;
      }
      const queued = Number(data?.queued ?? ids.length);
      const message = data?.message || `Queued forward tests for ${queued} favorite${queued === 1 ? '' : 's'}`;
      showToast(message, true);
    } catch (err){
      console.error('Failed to queue forward tests', err);
      showToast('Network error starting forward tests', false);
    } finally {
      runBtn.disabled = favorites.length === 0;
      runBtn.textContent = originalLabel;
    }
  }

  errorBox?.addEventListener('click', (event) => {
    const target = event.target;
    if(target instanceof HTMLElement && target.dataset.action === 'retry'){
      loadFavorites();
    }
  });

  runBtn.addEventListener('click', runForwardTests);

  document.addEventListener('visibilitychange', () => {
    if(document.visibilityState === 'visible' && !loading){
      loadFavorites();
    }
  });

  loadFavorites();
})();
