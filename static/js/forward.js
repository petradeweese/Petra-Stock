(function(){
  'use strict';

  const table = document.getElementById('forward-table');
  const tbody = document.getElementById('forward-tbody');
  const emptyState = document.getElementById('forward-empty');
  const loadingEl = document.getElementById('forward-loading');
  const errorBox = document.getElementById('forward-error');
  const runBtn = document.getElementById('forward-run');
  const toast = document.getElementById('toast');
  const historyModal = document.getElementById('forward-history-modal');
  const historyModalTitle = document.getElementById('forward-history-title');
  const historyModalList = document.getElementById('forward-history-list');
  const historyModalStatus = document.getElementById('forward-history-status');
  const historyModalLoadMore = document.getElementById('forward-history-load-more');
  const historyModalExport = document.getElementById('forward-history-export');
  const historyModalDialog = historyModal ? historyModal.querySelector('.forward-history-dialog') : null;
  const historyModalClose = historyModal ? historyModal.querySelector('.forward-history-close') : null;
  const dateFormatter = typeof Intl !== 'undefined' && typeof Intl.DateTimeFormat === 'function'
    ? new Intl.DateTimeFormat(undefined, {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      })
    : null;
  const timeFormatter = typeof Intl !== 'undefined' && typeof Intl.DateTimeFormat === 'function'
    ? new Intl.DateTimeFormat(undefined, {
        hour: '2-digit',
        minute: '2-digit',
        hour12: false,
      })
    : null;

  if(!table || !tbody || !runBtn){
    return;
  }

  let favorites = [];
  let loading = false;
  const HISTORY_LIMIT = 5;
  let activeHistoryPopover = null;
  let historyVirtualizer = null;
  const HISTORY_MODAL_LIMIT = 50;
  const historyModalState = {
    open: false,
    favorite: null,
    offset: 0,
    loading: false,
    done: false,
    runs: [],
  };
  const rootStyle = window.getComputedStyle(document.body || document.documentElement);
  const accentColor = (rootStyle.getPropertyValue('--accent') || '#5ea0ff').trim() || '#5ea0ff';
  const posColor = (rootStyle.getPropertyValue('--pos') || '#3ddc84').trim() || '#3ddc84';
  const negColor = (rootStyle.getPropertyValue('--neg') || '#ff4d4d').trim() || '#ff4d4d';
  const mutedColor = (rootStyle.getPropertyValue('--muted') || '#9aa6b2').trim() || '#9aa6b2';
  const isTestMode = Boolean(window && window.__FORWARD_TEST_MODE__);

  function showToast(message, ok=true, options){
    if(!toast){ return; }
    const settings = typeof options === 'object' && options ? options : {};
    toast.innerHTML = '';
    const text = document.createElement('span');
    text.textContent = message;
    toast.appendChild(text);
    if(settings.actionLabel && typeof settings.onAction === 'function'){
      const actionBtn = document.createElement('button');
      actionBtn.type = 'button';
      actionBtn.className = 'toast-action';
      actionBtn.textContent = settings.actionLabel;
      actionBtn.addEventListener('click', (event) => {
        event.stopPropagation();
        settings.onAction();
      });
      toast.appendChild(actionBtn);
    }
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

  function formatPercentRatio(value, decimals=2){
    const num = coerceNumber(value);
    if(num === null){
      return '—';
    }
    let scaled = num;
    if(Math.abs(scaled) <= 1){
      scaled *= 100;
    }
    if(!Number.isFinite(scaled)){
      return '—';
    }
    return `${scaled.toFixed(decimals)}%`;
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

  function formatEventTime(raw){
    if(!raw){
      return '';
    }
    const date = new Date(raw);
    if(Number.isNaN(date.getTime())){
      return '';
    }
    if(timeFormatter){
      return timeFormatter.format(date);
    }
    return date.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
  }

  function formatSignedPercent(value){
    const num = Number(value);
    if(!Number.isFinite(num)){
      return '';
    }
    let scaled = num;
    if(Math.abs(scaled) <= 1){
      scaled *= 100;
    }
    const sign = scaled > 0 ? '+' : '';
    return `${sign}${scaled.toFixed(2)}%`;
  }

  function formatBars(value){
    if(value === null || value === undefined){
      return '';
    }
    const bars = Number.parseInt(String(value), 10);
    if(!Number.isFinite(bars)){
      return '';
    }
    return `${bars}b`;
  }

  function formatMedianBars(value){
    const num = coerceNumber(value);
    if(num === null){
      return '';
    }
    if(Math.abs(num - Math.round(num)) < 1e-6){
      return `${Math.round(num)}b`;
    }
    return `${num.toFixed(1)}b`;
  }

  function formatRecentEntry(entry){
    if(!entry || typeof entry !== 'object'){
      return '';
    }
    const rawDate = entry.date;
    let dateLabel = '';
    if(rawDate){
      const dt = new Date(rawDate);
      if(!Number.isNaN(dt.getTime())){
        dateLabel = dt.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
      } else if(typeof rawDate === 'string'){
        dateLabel = rawDate.trim();
      }
    }
    const outcome = typeof entry.outcome === 'string' ? entry.outcome.trim().toUpperCase() : '';
    const roiText = formatSignedPercent(entry.roi);
    const barsText = formatBars(entry.tt);
    const pieces = [];
    if(dateLabel){ pieces.push(dateLabel); }
    if(outcome){ pieces.push(outcome); }
    if(roiText){ pieces.push(roiText); }
    if(barsText){ pieces.push(`@${barsText}`); }
    return pieces.join(' ');
  }

  function formatCount(value){
    const num = coerceNumber(value);
    if(num === null){
      return '';
    }
    if(Math.abs(num - Math.round(num)) < 1e-6){
      return String(Math.round(num));
    }
    return num.toFixed(1);
  }

  function formatHistoryTimestamp(raw){
    if(!raw){
      return '';
    }
    const date = new Date(raw);
    if(Number.isNaN(date.getTime())){
      return '';
    }
    const pad = (value) => String(value).padStart(2, '0');
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
  }

  function withCooldown(fn, delay){
    let ready = true;
    return function(...args){
      if(!ready){
        return;
      }
      ready = false;
      try {
        fn.apply(this, args);
      } finally {
        setTimeout(() => {
          ready = true;
        }, delay);
      }
    };
  }

  const SKELETON_COLUMNS = 15;
  const SKELETON_SHAPES = ['short', '', 'tiny', '', '', '', 'short', '', '', '', '', '', '', '', 'tiny'];

  function renderSkeletonRows(count=6){
    if(!tbody){
      return;
    }
    tbody.innerHTML = '';
    for(let i = 0; i < count; i += 1){
      const tr = document.createElement('tr');
      tr.className = 'forward-skeleton-row';
      for(let col = 0; col < SKELETON_COLUMNS; col += 1){
        const td = document.createElement('td');
        const bar = document.createElement('span');
        bar.className = 'forward-skeleton-bar';
        const shape = SKELETON_SHAPES[col] || (col % 4 === 0 ? 'short' : (col % 5 === 0 ? 'tiny' : ''));
        if(shape){
          bar.classList.add(shape);
        }
        td.appendChild(bar);
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
  }

  function closeActiveHistory(){
    if(!activeHistoryPopover){
      return;
    }
    const { button, popover } = activeHistoryPopover;
    popover.hidden = true;
    button.classList.remove('history-chip-active');
    button.setAttribute('aria-expanded', 'false');
    activeHistoryPopover = null;
  }

  function setHistoryModalStatus(message, isError=false){
    if(!historyModalStatus){
      return;
    }
    if(!message){
      historyModalStatus.hidden = true;
      historyModalStatus.textContent = '';
      historyModalStatus.classList.remove('error');
      return;
    }
    historyModalStatus.hidden = false;
    historyModalStatus.textContent = message;
    if(isError){
      historyModalStatus.classList.add('error');
    } else {
      historyModalStatus.classList.remove('error');
    }
  }

  function closeHistoryModal(){
    if(!historyModalState.open || !historyModal){
      return;
    }
    historyModal.hidden = true;
    historyModalState.open = false;
    historyModalState.favorite = null;
    historyModalState.offset = 0;
    historyModalState.loading = false;
    historyModalState.done = false;
    historyModalState.runs = [];
    if(historyVirtualizer){
      historyVirtualizer.reset();
    } else if(historyModalList){
      historyModalList.innerHTML = '';
    }
    setHistoryModalStatus('');
    if(historyModalLoadMore){
      historyModalLoadMore.hidden = false;
      historyModalLoadMore.disabled = false;
      historyModalLoadMore.textContent = 'Load more';
    }
    document.body.classList.remove('modal-open');
  }

  function drawHistorySparkline(canvas, run){
    if(!canvas){
      return;
    }
    const ctx = canvas.getContext('2d');
    if(!ctx){
      return;
    }
    const width = 120;
    const height = 36;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const roiVal = coerceNumber(run?.roi);
    const ddVal = coerceNumber(run?.dd);
    const pathValues = [0];
    if(ddVal !== null && Number.isFinite(ddVal)){
      pathValues.push(-Math.abs(ddVal) * 100);
    }
    if(roiVal !== null && Number.isFinite(roiVal)){
      pathValues.push(roiVal * 100);
    }
    if(pathValues.length < 2){
      pathValues.push(0);
    }

    const minVal = Math.min(...pathValues, 0);
    const maxVal = Math.max(...pathValues, 0);
    const range = Math.max(1e-6, maxVal - minVal);
    const points = pathValues.map((value, index) => {
      const x = pathValues.length === 1 ? width : (width * index) / (pathValues.length - 1);
      const y = height - ((value - minVal) / range) * height;
      return { x, y };
    });

    if(minVal < 0 && maxVal > 0){
      const zeroY = height - ((0 - minVal) / range) * height;
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(154,166,178,0.35)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.moveTo(0, zeroY);
      ctx.lineTo(width, zeroY);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.beginPath();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = accentColor || '#5ea0ff';
    points.forEach((point, idx) => {
      if(idx === 0){
        ctx.moveTo(point.x, point.y);
      } else {
        ctx.lineTo(point.x, point.y);
      }
    });
    ctx.stroke();

    if(points.length){
      const start = points[0];
      ctx.beginPath();
      ctx.fillStyle = accentColor || '#5ea0ff';
      ctx.arc(start.x, start.y, 3, 0, Math.PI * 2);
      ctx.fill();

      const end = points[points.length - 1];
      const hasExit = roiVal !== null && Number.isFinite(roiVal);
      ctx.beginPath();
      if(hasExit){
        ctx.fillStyle = roiVal >= 0 ? (posColor || '#3ddc84') : (negColor || '#ff4d4d');
      } else if(ddVal !== null && Number.isFinite(ddVal)){
        ctx.fillStyle = mutedColor || '#9aa6b2';
      } else {
        ctx.fillStyle = accentColor || '#5ea0ff';
      }
      ctx.arc(end.x, end.y, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  }

  function createVirtualList(listEl, options){
    if(!listEl || !listEl.parentElement){
      return null;
    }
    const scroller = listEl.parentElement;
    const settings = options || {};
    const overscan = Number.isFinite(settings.overscan) ? settings.overscan : 6;
    let estimatedHeight = Number.isFinite(settings.estimatedHeight) ? settings.estimatedHeight : 52;
    listEl.innerHTML = '';
    const topSpacer = document.createElement('li');
    topSpacer.className = 'forward-history-virtual-spacer';
    topSpacer.setAttribute('aria-hidden', 'true');
    const bottomSpacer = document.createElement('li');
    bottomSpacer.className = 'forward-history-virtual-spacer';
    bottomSpacer.setAttribute('aria-hidden', 'true');
    listEl.appendChild(topSpacer);
    listEl.appendChild(bottomSpacer);
    const rendered = new Map();
    let items = [];

    function detach(index){
      const node = rendered.get(index);
      if(node && node.parentNode === listEl){
        listEl.removeChild(node);
      }
      rendered.delete(index);
    }

    function update(){
      if(!items.length){
        topSpacer.style.height = '0px';
        bottomSpacer.style.height = '0px';
        rendered.forEach((node) => {
          if(node.parentNode === listEl){
            listEl.removeChild(node);
          }
        });
        rendered.clear();
        return;
      }
      const viewport = scroller.clientHeight || estimatedHeight;
      const scrollTop = scroller.scrollTop || 0;
      const itemHeight = estimatedHeight || 1;
      const start = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan);
      const end = Math.min(items.length, Math.ceil((scrollTop + viewport) / itemHeight) + overscan);
      topSpacer.style.height = `${start * itemHeight}px`;
      bottomSpacer.style.height = `${Math.max(0, items.length - end) * itemHeight}px`;

      rendered.forEach((node, index) => {
        if(index < start || index >= end){
          detach(index);
        }
      });

      for(let idx = start; idx < end; idx += 1){
        let node = rendered.get(idx);
        if(!node){
          node = settings.renderItem(items[idx], idx);
          if(!(node instanceof HTMLElement)){
            continue;
          }
          rendered.set(idx, node);
          listEl.insertBefore(node, bottomSpacer);
          if(!estimatedHeight){
            const rect = node.getBoundingClientRect();
            if(rect && rect.height){
              estimatedHeight = rect.height;
            }
          }
        }
      }
    }

    function setItems(nextItems){
      items = Array.isArray(nextItems) ? nextItems.slice() : [];
      rendered.forEach((node) => {
        if(node.parentNode === listEl){
          listEl.removeChild(node);
        }
      });
      rendered.clear();
      update();
    }

    function appendItems(extra){
      if(Array.isArray(extra) && extra.length){
        items = items.concat(extra);
        update();
      }
    }

    scroller.addEventListener('scroll', () => update());
    window.addEventListener('resize', () => update());

    return {
      setItems,
      appendItems,
      reset(){
        setItems([]);
      },
      getVisibleCount(){
        return rendered.size;
      },
      forceUpdate: update,
    };
  }

  function buildHistoryModalRow(run){
    const item = document.createElement('li');
    item.className = 'forward-history-row';
    const text = document.createElement('div');
    text.className = 'forward-history-row-text';
    const entryLabel = formatHistoryTimestamp(run?.entry_ts) || '—';
    text.append(entryLabel);
    const tooltipParts = [entryLabel];

    const outcomeRaw = typeof run?.outcome === 'string' ? run.outcome.toLowerCase() : '';
    const hasExit = Boolean(run?.exit_ts) && Boolean(outcomeRaw);
    const extras = [];
    if(hasExit){
      text.append(' → ');
      const badgeType = ['hit', 'stop', 'timeout'].includes(outcomeRaw) ? outcomeRaw : 'neutral';
      const badge = document.createElement('span');
      badge.className = `timeline-badge timeline-badge-${badgeType} forward-history-row-outcome`;
      const outcomeLabel = (outcomeRaw || 'EXIT').toUpperCase();
      badge.textContent = outcomeLabel;
      text.appendChild(badge);
      tooltipParts.push(`→ ${outcomeLabel}`);

      const roiText = formatSignedPercent(run?.roi);
      if(roiText){
        extras.push(roiText);
      }
      const barText = formatBars(run?.tt_bars);
      if(barText){
        extras.push(barText);
      }
      const ddText = formatPercentRatio(run?.dd, 2);
      if(ddText !== '—'){
        extras.push(`DD ${ddText}`);
      }
      if(extras.length){
        const extraLabel = ` (${extras.join(' | ')})`;
        text.append(extraLabel);
        tooltipParts.push(extraLabel.trim());
      }
    } else {
      text.append(' → …');
      tooltipParts.push('→ …');
    }

    if(run?.rule_mismatch){
      const badge = document.createElement('span');
      badge.className = 'history-badge-old';
      badge.textContent = '(old rule)';
      text.append(' ');
      text.appendChild(badge);
      tooltipParts.push('(old rule)');
    }

    item.appendChild(text);
    const canvas = document.createElement('canvas');
    canvas.className = 'forward-history-spark';
    item.appendChild(canvas);
    drawHistorySparkline(canvas, run || {});
    item.title = tooltipParts.join(' ');
    return item;
  }

  if(historyModalList){
    historyVirtualizer = createVirtualList(historyModalList, {
      renderItem: buildHistoryModalRow,
      estimatedHeight: 60,
      overscan: 6,
    });
  }

  async function loadHistoryModalRuns(reset=false){
    if(!historyModal || !historyModalState.favorite){
      return;
    }
    if(historyModalState.loading){
      return;
    }
    if(!reset && historyModalState.done){
      return;
    }

    if(reset){
      historyModalState.offset = 0;
      historyModalState.done = false;
      historyModalState.runs = [];
      if(historyVirtualizer){
        historyVirtualizer.reset();
      } else if(historyModalList){
        historyModalList.innerHTML = '';
      }
      setHistoryModalStatus('Loading runs…');
    }

    if(historyModalLoadMore && !reset){
      historyModalLoadMore.disabled = true;
      historyModalLoadMore.textContent = 'Loading…';
    }

    historyModalState.loading = true;
    try {
      const fav = historyModalState.favorite;
      const params = new URLSearchParams({
        limit: String(HISTORY_MODAL_LIMIT),
        offset: String(historyModalState.offset),
      });
      const res = await fetch(`/api/forward/${fav.id}?${params.toString()}`);
      if(!res.ok){
        throw new Error(`HTTP ${res.status}`);
      }
      let runs = await res.json();
      if(!Array.isArray(runs)){
        runs = [];
      }

      const currentFav = historyModalState.favorite;
      if(!historyModalState.open || !currentFav || currentFav.id !== fav.id){
        return;
      }

      if(historyModalList && runs.length){
        setHistoryModalStatus('');
      } else if((historyModalState.runs.length === 0) && runs.length === 0){
        setHistoryModalStatus('No forward runs yet—your history appears as windows complete.');
      }

      if(runs.length){
        if(historyVirtualizer){
          historyModalState.runs = historyModalState.runs.concat(runs);
          historyVirtualizer.setItems(historyModalState.runs);
        } else if(historyModalList){
          runs.forEach((run) => {
            const node = buildHistoryModalRow(run);
            historyModalList.appendChild(node);
          });
          historyModalState.runs = historyModalState.runs.concat(runs);
        }
      }

      historyModalState.offset += runs.length;
      const hasMore = runs.length === HISTORY_MODAL_LIMIT;
      historyModalState.done = !hasMore;

      if(historyModalLoadMore){
        if(hasMore){
          historyModalLoadMore.hidden = false;
          historyModalLoadMore.disabled = false;
          historyModalLoadMore.textContent = 'Load more';
        } else {
          historyModalLoadMore.hidden = true;
          historyModalLoadMore.disabled = false;
          historyModalLoadMore.textContent = 'Load more';
        }
      }
    } catch (err){
      console.error('Failed to load forward runs', err);
      historyModalState.done = false;
      if(historyModalLoadMore){
        historyModalLoadMore.hidden = false;
        historyModalLoadMore.disabled = false;
        historyModalLoadMore.textContent = 'Retry';
      }
      if(historyModalState.runs.length === 0){
        setHistoryModalStatus('Failed to load forward runs.', true);
      } else {
        setHistoryModalStatus('Failed to load more runs.', true);
      }
    } finally {
      historyModalState.loading = false;
    }
  }

  function openHistoryModal(fav){
    if(!historyModal || !fav || fav.id === undefined || fav.id === null){
      return;
    }
    historyModalState.favorite = fav;
    historyModalState.offset = 0;
    historyModalState.done = false;
    historyModalState.loading = false;
    historyModalState.runs = [];
    if(historyVirtualizer){
      historyVirtualizer.reset();
    } else if(historyModalList){
      historyModalList.innerHTML = '';
    }
    setHistoryModalStatus('Loading runs…');
    if(historyModalTitle){
      const ticker = (fav.ticker || '').toUpperCase();
      const direction = (fav.direction || 'UP').toUpperCase();
      historyModalTitle.textContent = `${ticker} ${direction} — Forward Runs`;
    }
    if(historyModalExport){
      historyModalExport.href = `/forward/export.csv?favorite_id=${fav.id}&limit=1000`;
    }
    if(historyModalLoadMore){
      historyModalLoadMore.hidden = true;
      historyModalLoadMore.disabled = true;
      historyModalLoadMore.textContent = 'Loading…';
    }
    historyModal.hidden = false;
    historyModalState.open = true;
    document.body.classList.add('modal-open');
    loadHistoryModalRuns(true);
  }

  function buildHistoryCell(fav){
    const td = document.createElement('td');
    td.className = 'history-td';
    const history = Array.isArray(fav.forward_history) ? fav.forward_history : [];

    if(!history.length){
      const dash = document.createElement('span');
      dash.className = 'history-empty muted';
      dash.textContent = '—';
      td.appendChild(dash);
      return td;
    }

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'history-chip';
    button.textContent = `Last ${HISTORY_LIMIT}`;
    button.title = 'Show forward test history';
    button.setAttribute('aria-haspopup', 'true');
    button.setAttribute('aria-expanded', 'false');
    td.appendChild(button);

    const popover = document.createElement('div');
    popover.className = 'history-popover';
    popover.hidden = true;
    popover.setAttribute('role', 'dialog');
    popover.setAttribute('aria-label', 'Forward test history');

    const list = document.createElement('ul');
    list.className = 'history-list';

    history.forEach((run) => {
      const item = document.createElement('li');
      item.className = 'history-item';
      const line = document.createElement('div');
      line.className = 'history-line';

      const entryLabel = formatHistoryTimestamp(run.entry_ts);
      line.append(entryLabel || '—');

      const outcome = typeof run.outcome === 'string' ? run.outcome.toLowerCase() : '';
      const hasExit = Boolean(run.exit_ts) && Boolean(outcome);
      if(hasExit){
        line.append(' → ');
        const badgeType = outcome === 'hit' || outcome === 'stop' || outcome === 'timeout' ? outcome : 'neutral';
        const badge = document.createElement('span');
        badge.className = `timeline-badge timeline-badge-${badgeType} history-outcome`;
        badge.textContent = (outcome || 'EXIT').toUpperCase();
        line.appendChild(badge);
        const extras = [];
        const roiText = formatSignedPercent(run.roi);
        if(roiText){
          extras.push(roiText);
        }
        const barText = formatBars(run.tt_bars);
        if(barText){
          extras.push(barText);
        }
        if(extras.length){
          line.append(` (${extras.join(' | ')})`);
        }
      } else {
        line.append(' → …');
      }

      if(run.rule_mismatch){
        const badge = document.createElement('span');
        badge.className = 'history-badge-old';
        badge.textContent = '(old rule)';
        line.append(' ');
        line.appendChild(badge);
      }

      item.appendChild(line);
      list.appendChild(item);
    });

    popover.appendChild(list);
    const actions = document.createElement('div');
    actions.className = 'history-popover-actions';
    const viewAllBtn = document.createElement('button');
    viewAllBtn.type = 'button';
    viewAllBtn.className = 'history-view-all';
    viewAllBtn.textContent = 'View all';
    viewAllBtn.title = 'View full forward history';
    viewAllBtn.setAttribute('data-testid', 'history-open');
    viewAllBtn.addEventListener('click', (event) => {
      event.stopPropagation();
      closeActiveHistory();
      openHistoryModal(fav);
    });
    actions.appendChild(viewAllBtn);
    popover.appendChild(actions);
    td.appendChild(popover);

    button.addEventListener('click', (event) => {
      event.stopPropagation();
      const isActive = activeHistoryPopover && activeHistoryPopover.button === button;
      if(isActive){
        closeActiveHistory();
      } else {
        closeActiveHistory();
        popover.hidden = false;
        button.classList.add('history-chip-active');
        button.setAttribute('aria-expanded', 'true');
        activeHistoryPopover = { button, popover };
      }
    });

    popover.addEventListener('click', (event) => {
      event.stopPropagation();
    });

    return td;
  }

  function buildSummaryCell(fav){
    const cell = createClampCell('summary-td', 'Expand summary');
    const summary = (fav && typeof fav.summary === 'object' && fav.summary) ? fav.summary : {};
    const unweighted = (summary && typeof summary.unweighted === 'object' && summary.unweighted)
      ? summary.unweighted
      : summary;
    const summaryModeRaw = typeof summary.mode === 'string' ? summary.mode : '';
    const summaryMode = summaryModeRaw ? summaryModeRaw.toLowerCase() : 'off';
    const weighted = summaryMode === 'exp' && summary && typeof summary.weighted === 'object' && summary.weighted
      ? summary.weighted
      : null;
    const metrics = weighted || unweighted;

    const supportValue = coerceNumber(metrics?.n);
    const fallbackSupport = coerceNumber(unweighted?.n ?? summary?.n);
    const support = supportValue !== null ? supportValue : fallbackSupport;
    if(support === null || support <= 0){
      const dash = document.createElement('span');
      dash.className = 'summary-empty muted';
      dash.textContent = '—';
      cell.content.appendChild(dash);
      cell.toggle.hidden = true;
      return cell;
    }

    const hits = coerceNumber(metrics?.hits) ?? 0;
    const hitRateText = formatPercentRatio(metrics?.hit_rate, 2);
    const lbText = formatPercentRatio(metrics?.hit_lb95, 2);
    const chips = document.createElement('div');
    chips.className = 'summary-chips';

    let recencyBadge = null;
    const hasRecency = summaryMode === 'exp' && !!weighted;
    if(hasRecency){
      recencyBadge = document.createElement('span');
      recencyBadge.className = 'summary-recency-badge';
      const halfLifeValue = coerceNumber(summary.half_life_days);
      const halfLifeLabel = formatCount(summary.half_life_days);
      const halfLifeText = halfLifeLabel ? `${halfLifeLabel} day${halfLifeValue !== null && Math.abs(halfLifeValue - 1) < 1e-6 ? '' : 's'}` : '';
      const badgeSuffix = halfLifeValue !== null ? `${Math.round(halfLifeValue)}d` : '';
      recencyBadge.textContent = badgeSuffix ? `RECENCY ON (HL=${badgeSuffix})` : 'RECENCY ON';
      recencyBadge.title = halfLifeText ? `Exp weighting, half-life ${halfLifeText}` : 'Exp weighting active';
      recencyBadge.setAttribute('data-testid', 'recency-badge');
      chips.appendChild(recencyBadge);
    }

    const hitChip = document.createElement('span');
    hitChip.className = 'summary-chip summary-chip-success';
    const hitMain = document.createElement('span');
    hitMain.className = 'summary-chip-main';
    hitMain.textContent = hitRateText === '—' ? 'Hit —' : `Hit ${hitRateText}`;
    hitChip.appendChild(hitMain);
    if(lbText !== '—'){
      const hitSub = document.createElement('span');
      hitSub.className = 'summary-chip-sub';
      hitSub.textContent = `LB ${lbText}`;
      hitChip.appendChild(hitSub);
    }
    const supportLabel = formatCount(metrics?.n);
    const hitsLabel = formatCount(metrics?.hits);
    const unweightedSupportLabel = formatCount(unweighted?.n ?? summary?.n);
    const unweightedHitsLabel = formatCount(unweighted?.hits ?? summary?.hits);
    const hitDetails = [];
    if(hasRecency){
      if(supportLabel){ hitDetails.push(`Weighted runs ${supportLabel}`); }
      if(hitsLabel){ hitDetails.push(`Weighted hits ${hitsLabel}`); }
      if(unweightedSupportLabel){ hitDetails.push(`Unweighted runs ${unweightedSupportLabel}`); }
      if(unweightedHitsLabel){ hitDetails.push(`Unweighted hits ${unweightedHitsLabel}`); }
    } else {
      if(supportLabel){ hitDetails.push(`Runs ${supportLabel}`); }
      if(hitsLabel){ hitDetails.push(`Hits ${hitsLabel}`); }
    }
    hitChip.title = hitDetails.length ? hitDetails.join(' · ') : '';
    chips.appendChild(hitChip);

    const avgRoiText = formatPercentRatio(metrics?.avg_roi, 2);
    const avgRoi = coerceNumber(metrics?.avg_roi);
    let roiTone = 'neutral';
    if(avgRoi !== null){
      if(avgRoi > 0){
        roiTone = 'success';
      } else if(avgRoi < 0){
        roiTone = 'warn';
      }
    }
    const roiChip = document.createElement('span');
    roiChip.className = `summary-chip summary-chip-${roiTone}`;
    const roiMain = document.createElement('span');
    roiMain.className = 'summary-chip-main';
    roiMain.textContent = avgRoiText === '—' ? 'Avg ROI —' : `Avg ROI ${avgRoiText}`;
    roiChip.appendChild(roiMain);
    if(avgRoiText === '—'){
      roiChip.title = hasRecency ? 'Weighted average ROI unavailable' : 'Average ROI unavailable';
    } else {
      roiChip.title = hasRecency ? `Weighted average ROI ${avgRoiText}` : `Average ROI ${avgRoiText}`;
    }
    chips.appendChild(roiChip);

    const medianBarsText = formatMedianBars(metrics?.median_tt_bars);
    const ddText = formatPercentRatio(metrics?.avg_dd, 2);
    const speedChip = document.createElement('span');
    const hasDd = ddText !== '—';
    speedChip.className = `summary-chip summary-chip-${hasDd ? 'warn' : 'neutral'}`;
    const speedMain = document.createElement('span');
    speedMain.className = 'summary-chip-main';
    speedMain.textContent = medianBarsText ? `Median TT ${medianBarsText}` : 'Median TT —';
    speedChip.appendChild(speedMain);
    if(hasDd){
      const speedSub = document.createElement('span');
      speedSub.className = 'summary-chip-sub';
      speedSub.textContent = `Avg DD ${ddText}`;
      speedChip.appendChild(speedSub);
    }
    const speedTooltipParts = [];
    if(medianBarsText){
      speedTooltipParts.push(`Median time-to-target ${medianBarsText}`);
    }
    if(hasDd){
      speedTooltipParts.push(`Average drawdown ${ddText}`);
    }
    if(speedTooltipParts.length){
      speedChip.title = speedTooltipParts.join(' · ');
    }
    chips.appendChild(speedChip);

    cell.content.appendChild(chips);
    const summaryTooltip = [];
    if(recencyBadge){
      summaryTooltip.push(recencyBadge.title);
    }
    const hitTooltipParts = [hitMain.textContent || 'Hit —'];
    if(lbText !== '—'){
      hitTooltipParts.push(`LB ${lbText}`);
    }
    if(hitDetails.length){
      summaryTooltip.push(hitDetails.join(' · '));
    }
    if(hasRecency){
      if(unweightedSupportLabel){
        hitTooltipParts.push(`Unweighted runs ${unweightedSupportLabel}`);
      }
      if(unweightedHitsLabel){
        hitTooltipParts.push(`Unweighted hits ${unweightedHitsLabel}`);
      }
    } else {
      if(supportLabel){ hitTooltipParts.push(`Runs ${supportLabel}`); }
      if(hitsLabel){ hitTooltipParts.push(`Hits ${hitsLabel}`); }
    }
    summaryTooltip.push(hitTooltipParts.join(' · '));
    summaryTooltip.push(roiChip.title);
    if(speedTooltipParts.length){
      summaryTooltip.push(speedTooltipParts.join(' · '));
    }
    cell.content.title = summaryTooltip.filter(Boolean).join(' | ');
    const recentRaw = Array.isArray(summary?.recent) ? summary.recent : Array.isArray(summary?.recent3) ? summary.recent3 : [];
    const recentEntries = Array.isArray(recentRaw) ? recentRaw.filter(Boolean) : [];
    if(recentEntries.length){
      const recentWrap = document.createElement('div');
      recentWrap.className = 'summary-recent';
      recentWrap.setAttribute('data-testid', 'recent-preview');
      recentEntries.slice(0, 2).forEach((entry) => {
        const text = formatRecentEntry(entry);
        if(!text){
          return;
        }
        const chip = document.createElement('span');
        chip.className = 'summary-recent-item';
        chip.textContent = text;
        recentWrap.appendChild(chip);
      });
      if(recentEntries.length > 2){
        const more = document.createElement('span');
        more.className = 'summary-recent-more';
        more.textContent = `+${recentEntries.length - 2} more`;
        recentWrap.appendChild(more);
      }
      cell.content.appendChild(recentWrap);
    }

    return cell;
  }

  function buildTimelineDisplay(fav){
    const container = document.createElement('span');
    container.className = 'timeline-text timeline-chip';
    const forward = fav.forward;
    const events = Array.isArray(forward?.events) ? forward.events : [];
    if(events.length === 0){
      container.textContent = '—';
      return {node: container, label: '—'};
    }

    const entry = events[0] || {};
    const entryTime = formatEventTime(entry.ts);
    const entryType = (typeof entry.t === 'string' ? entry.t : 'detect').toLowerCase() || 'detect';
    const labelParts = [];
    if(entryTime){
      labelParts.push(`${entryTime} ${entryType}`);
    } else {
      labelParts.push('—');
    }

    if(events.length === 1){
      labelParts.push('→ …');
      const labelSingle = labelParts.join(' ').replace(/\s+/g, ' ').trim();
      container.textContent = labelSingle;
      container.title = labelSingle;
      return {node: container, label: labelSingle};
    }

    const exit = events[1] || {};
    const exitTime = formatEventTime(exit.ts);
    const exitType = (typeof exit.t === 'string' ? exit.t : 'exit').toLowerCase() || 'exit';
    const exitSegment = exitTime ? `${exitTime} ${exitType}` : exitType;
    labelParts.push('→', exitSegment);

    const extras = [];
    const roiText = formatSignedPercent(exit.roi);
    if(roiText){
      extras.push(roiText);
    }
    const barsValue = coerceNumber(exit.tt_bars);
    if(barsValue !== null){
      const rounded = Math.round(barsValue);
      const barsLabel = Number.isFinite(rounded) ? `${rounded} bar${rounded === 1 ? '' : 's'}` : '';
      if(barsLabel){
        extras.push(`in ${barsLabel}`);
      }
    }
    if(extras.length){
      labelParts.push(`(${extras.join(' ')})`);
    }

    const label = labelParts.join(' ').replace(/\s+/g, ' ').trim();
    container.textContent = label;
    container.title = label;
    return {node: container, label: label};
  }

  function createClampCell(extraClass='', toggleLabel='Expand cell'){
    const td = document.createElement('td');
    td.className = ['clamp-cell', extraClass].filter(Boolean).join(' ');
    const wrap = document.createElement('div');
    wrap.className = 'clamp-wrap';
    wrap.dataset.expanded = '0';
    const content = document.createElement('div');
    content.className = 'clamp-content';
    wrap.appendChild(content);
    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'clamp-toggle';
    toggle.textContent = '▸';
    toggle.title = toggleLabel;
    toggle.setAttribute('aria-label', toggleLabel);
    toggle.setAttribute('aria-expanded', 'false');
    toggle.addEventListener('click', withCooldown(() => {
      const expanded = wrap.dataset.expanded === '1';
      wrap.dataset.expanded = expanded ? '0' : '1';
      toggle.textContent = expanded ? '▸' : '▾';
      toggle.setAttribute('aria-expanded', expanded ? 'false' : 'true');
    }, 120));
    wrap.appendChild(toggle);
    td.appendChild(wrap);
    return {td, wrap, content, toggle};
  }

  function formatSupportLabel(fav){
    const display = typeof fav.support_display === 'string' ? fav.support_display.trim() : '';
    if(display && display !== '—'){
      return display;
    }
    if(fav.support_count === 0){
      return '0';
    }
    if(fav.support_count !== undefined && fav.support_count !== null){
      return String(fav.support_count);
    }
    return '';
  }

  function formatForwardHit(fav){
    const hitText = formatPercent(fav.forward?.hit_pct, 0);
    const support = formatSupportLabel(fav);
    if(hitText === '—' && !support){
      return '—';
    }
    if(support){
      if(hitText === '—'){
        return `— / ${support}`;
      }
      return `${hitText} / ${support}`;
    }
    return hitText;
  }

  function formatForwardRoi(fav){
    return formatPercent(fav.forward?.roi_pct, 1);
  }

  function formatForwardUpdated(forward){
    if(!forward){
      return '';
    }
    const raw = forward.updated_at || forward.last_run_at || forward.created_at;
    if(!raw){
      return '';
    }
    const date = new Date(raw);
    if(Number.isNaN(date.getTime())){
      return String(raw);
    }
    if(dateFormatter){
      return dateFormatter.format(date);
    }
    return date.toLocaleString();
  }

  function formatForwardStatus(fav){
    const forward = fav.forward;
    if(!forward){
      return '—';
    }
    const rawStatus = typeof forward.status === 'string' ? forward.status.trim() : '';
    const status = rawStatus ? rawStatus.charAt(0).toUpperCase() + rawStatus.slice(1) : '';
    const updated = formatForwardUpdated(forward);
    if(status && updated){
      return `${status} · ${updated}`;
    }
    if(updated){
      return updated;
    }
    return status || '—';
  }

  function renderFavorites(items){
    favorites = Array.isArray(items) ? items : [];
    loading = false;
    loadingEl.hidden = true;
    errorBox.hidden = true;
    closeActiveHistory();
    closeHistoryModal();

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
      tr.setAttribute('data-testid', 'forward-row');

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

      const timelineCell = createClampCell('timeline-td', 'Expand timeline');
      const timelineDisplay = buildTimelineDisplay(fav);
      timelineCell.content.appendChild(timelineDisplay.node);
      if(timelineDisplay.label){
        timelineCell.content.title = timelineDisplay.label;
      }
      if(!timelineDisplay.label || timelineDisplay.label === '—'){
        timelineCell.toggle.hidden = true;
      }
      timelineCell.content.setAttribute('data-testid', 'timeline-chip');

      const historyTd = buildHistoryCell(fav);
      const summaryCell = buildSummaryCell(fav);

      const hitsTd = document.createElement('td');
      hitsTd.textContent = formatForwardHit(fav);

      const roiTd = document.createElement('td');
      roiTd.textContent = formatForwardRoi(fav);

      const statusTd = document.createElement('td');
      statusTd.textContent = formatForwardStatus(fav);

      const ruleCell = createClampCell('rule-td', 'Expand rule');
      const ruleText = fav.rule || fav.rule_snapshot || '';
      const code = document.createElement('code');
      code.textContent = ruleText;
      ruleCell.content.appendChild(code);
      if(ruleText){
        ruleCell.content.title = ruleText;
      } else {
        ruleCell.toggle.hidden = true;
      }

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

      [
        tickerTd,
        directionTd,
        intervalTd,
        lookbackTd,
        targetTd,
        stopTd,
        windowTd,
        timelineCell.td,
        historyTd,
        summaryCell.td,
        hitsTd,
        roiTd,
        statusTd,
        ruleCell.td,
        actionsTd,
      ].forEach((td) => {
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
    table.hidden = false;
    emptyState.hidden = true;
    errorBox.hidden = true;
    loadingEl.hidden = false;
    renderSkeletonRows();

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
      favorites = [];
      tbody.innerHTML = '';
      table.hidden = true;
      errorBox.hidden = false;
      showToast('Failed to load forward favorites', false, {
        actionLabel: 'Retry',
        onAction: () => {
          if(!loading){
            loadFavorites();
          }
        },
      });
    }
  }

  document.addEventListener('click', () => {
    closeActiveHistory();
  });

  document.addEventListener('keydown', (event) => {
    if(event.key === 'Escape'){
      closeActiveHistory();
      closeHistoryModal();
    }
  });

  if(historyModal){
    historyModal.addEventListener('click', (event) => {
      if(event.target === historyModal){
        closeHistoryModal();
      }
    });
  }

  if(historyModalDialog){
    historyModalDialog.addEventListener('click', (event) => {
      event.stopPropagation();
    });
  }

  if(historyModalClose){
    historyModalClose.addEventListener('click', (event) => {
      event.preventDefault();
      closeHistoryModal();
    });
  }

  if(historyModalLoadMore){
    historyModalLoadMore.addEventListener('click', () => {
      if(historyModalState.loading){
        return;
      }
      if(historyModalState.runs.length === 0){
        setHistoryModalStatus('Loading runs…');
      } else {
        setHistoryModalStatus('');
      }
      loadHistoryModalRuns(false);
    });
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

  const forwardTestHooks = {
    loadFavorites,
    renderFavorites,
    getFavorites: () => favorites.slice(),
    getSkeletonCount: () => tbody ? tbody.querySelectorAll('.forward-skeleton-row').length : 0,
    openHistoryModal,
    closeHistoryModal,
    loadHistoryModalRuns,
    getHistoryCount: () => historyModalState.runs.length,
    historyState: historyModalState,
    virtualVisible: () => (historyVirtualizer ? historyVirtualizer.getVisibleCount() : 0),
  };

  if(typeof window !== 'undefined'){
    window.__forwardTestHooks = forwardTestHooks;
  }

  if(!isTestMode){
    loadFavorites();
  }
})();
