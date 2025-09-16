(function(){
  'use strict';

  const table = document.getElementById('archive-table');
  const tbody = table ? table.querySelector('tbody') : null;
  const emptyState = document.getElementById('archive-empty');
  const clearAllBtn = document.getElementById('archive-clear-all');
  const toast = document.getElementById('toast');

  if(!table || !tbody){
    return;
  }

  function showToast(message, ok=true){
    if(!toast){ return; }
    toast.textContent = message;
    toast.style.borderColor = ok ? '#2e7d32' : '#8b0000';
    toast.style.background = ok ? '#0f3311' : '#2b0f0f';
    toast.hidden = false;
    setTimeout(()=>{ toast.hidden = true; }, 2000);
  }

  function updateEmptyState(){
    const hasRows = tbody.querySelectorAll('tr').length > 0;
    table.hidden = !hasRows;
    if(emptyState){
      emptyState.hidden = hasRows;
    }
    if(clearAllBtn){
      clearAllBtn.hidden = !hasRows;
      clearAllBtn.disabled = !hasRows;
    }
  }

  async function deleteRun(runId, button){
    if(!Number.isFinite(runId)){
      showToast('Invalid run id', false);
      return;
    }
    const confirmed = window.confirm('Delete this archived run? This removes the row and its results from the Archive. This cannot be undone.');
    if(!confirmed){
      return;
    }
    const originalDisabled = button?.disabled ?? false;
    if(button){
      button.disabled = true;
    }
    try {
      const res = await fetch(`/api/archive/${runId}`, { method: 'DELETE' });
      let data = null;
      try {
        data = await res.json();
      } catch (_err){
        data = null;
      }
      if(!res.ok || (data && data.ok === false)){
        const message = data?.error || 'Failed to delete archived run';
        showToast(message, false);
        return;
      }
      const row = tbody.querySelector(`tr[data-run-id="${runId}"]`);
      if(row){
        row.remove();
      }
      showToast(`Deleted archived run #${runId}`);
      updateEmptyState();
    } catch (err){
      console.error('Failed to delete archived run', err);
      showToast('Network error deleting run', false);
    } finally {
      if(button){
        button.disabled = originalDisabled;
      }
    }
  }

  async function clearAll(){
    const confirmed = window.confirm('Clear all archived runs? This removes every archived run and its results. This cannot be undone.');
    if(!confirmed){
      return;
    }
    const originalLabel = clearAllBtn.textContent;
    clearAllBtn.disabled = true;
    try {
      const res = await fetch('/api/archive/clear', { method: 'POST' });
      let data = null;
      try {
        data = await res.json();
      } catch (_err){
        data = null;
      }
      if(!res.ok || (data && data.ok === false)){
        const message = data?.error || 'Failed to clear archive';
        showToast(message, false);
        clearAllBtn.disabled = false;
        clearAllBtn.textContent = originalLabel;
        return;
      }
      tbody.innerHTML = '';
      showToast('Cleared archived runs');
      updateEmptyState();
    } catch (err){
      console.error('Failed to clear archived runs', err);
      showToast('Network error clearing archive', false);
      clearAllBtn.disabled = false;
      clearAllBtn.textContent = originalLabel;
      return;
    }
    clearAllBtn.textContent = originalLabel;
  }

  table.addEventListener('click', (event) => {
    const target = event.target;
    if(!(target instanceof HTMLElement)){
      return;
    }
    const button = target.closest('.archive-delete');
    if(!button){
      return;
    }
    const runId = Number.parseInt(button.dataset.runId || '', 10);
    deleteRun(runId, button);
  });

  if(clearAllBtn){
    clearAllBtn.addEventListener('click', clearAll);
  }

  updateEmptyState();
})();
