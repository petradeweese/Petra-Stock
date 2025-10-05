(function () {
  const statusSeed = safeParseJson('favorites-status-seed');
  const summarySeed = safeParseJson('favorites-summary-seed');
  const equitySeed = safeParseJson('favorites-equity-seed');
  const activitySeed = safeParseJson('favorites-activity-seed');

  const sectionEl = document.querySelector('section[data-has-universe]');
  const hasUniverse = (sectionEl?.dataset.hasUniverse || 'false') === 'true';
  const metricsEl = document.getElementById('favorites-metrics');
  const pillEl = document.getElementById('favorites-status-pill');
  const statusLineEl = document.getElementById('favorites-status-line');
  const rangeButtons = Array.from(document.querySelectorAll('[data-range]'));
  const actionButtons = Array.from(document.querySelectorAll('[data-favorites-action]'));
  actionButtons.forEach((btn) => {
    btn.dataset.disabledInitial = btn.disabled ? '1' : '0';
  });
  const chartCanvas = document.getElementById('favorites-equity-chart');
  const activityBody = document.getElementById('favorites-activity-body');
  const toast = document.getElementById('toast');

  const currencyFmt = new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD' });
  const percentFmt = new Intl.NumberFormat(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const shortPercentFmt = new Intl.NumberFormat(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  const dateTimeFmt = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' });

  let currentRange = '1m';
  let chartPoints = Array.isArray(equitySeed) ? equitySeed : [];

  function safeParseJson(id) {
    const el = document.getElementById(id);
    if (!el) return null;
    try {
      return JSON.parse(el.textContent || 'null');
    } catch (err) {
      return null;
    }
  }

  function showToast(message, ok = true) {
    if (!toast) return;
    toast.textContent = message;
    toast.style.borderColor = ok ? '#2e7d32' : '#8b0000';
    toast.style.background = ok ? '#0f3311' : '#2b0f0f';
    toast.hidden = false;
    setTimeout(() => {
      toast.hidden = true;
    }, 2400);
  }

  function formatCurrency(value) {
    return currencyFmt.format(Number(value) || 0);
  }

  function formatPercent(value) {
    return `${percentFmt.format(Number(value) || 0)}%`;
  }

  function formatShortPercent(value) {
    return `${shortPercentFmt.format(Number(value) || 0)}%`;
  }

  function updateMetrics(summary, universeCount) {
    if (!metricsEl) return;
    const data = summary || {};
    const map = {
      equity: formatCurrency(data.balance),
      pnl: formatCurrency(data.pnl),
      pnl_pct: formatPercent(data.pnl_pct),
      win_rate: formatShortPercent(data.win_rate),
      trades: Number(data.total_trades || 0).toString(),
      avg_win: formatCurrency(data.avg_win),
      avg_loss: formatCurrency(data.avg_loss),
      max_drawdown: formatPercent(data.max_drawdown),
      universe: Number(universeCount || data.universe || 0).toString(),
    };
    Object.entries(map).forEach(([key, value]) => {
      const el = metricsEl.querySelector(`[data-metric="${key}"]`);
      if (el) {
        el.textContent = value;
      }
    });
  }

  function updateStatus(status) {
    if (!statusLineEl) return;
    if (!hasUniverse) {
      statusLineEl.textContent = 'No favorites available. Add favorites from the scanner to begin simulating.';
      if (pillEl) {
        pillEl.classList.remove('active');
        pillEl.textContent = 'Inactive';
      }
      return;
    }
    const state = (status.status || '').toLowerCase();
    const startedRaw = status.started_at;
    let startedLabel = startedRaw;
    if (startedRaw) {
      const parsed = new Date(startedRaw);
      if (!Number.isNaN(parsed.valueOf())) {
        startedLabel = dateTimeFmt.format(parsed);
      }
    }
    if (state === 'active' && startedLabel) {
      statusLineEl.textContent = `Active since ${startedLabel}`;
    } else if (state === 'active') {
      statusLineEl.textContent = 'Active and monitoring favorites hits.';
    } else {
      statusLineEl.textContent = 'Simulator is currently inactive.';
    }
    if (pillEl) {
      pillEl.classList.toggle('active', state === 'active');
      pillEl.textContent = state === 'active' ? 'Active' : 'Inactive';
    }
  }

  function setActionsDisabled(disabled) {
    actionButtons.forEach((btn) => {
      if (disabled) {
        btn.disabled = true;
      } else {
        btn.disabled = btn.dataset.disabledInitial === '1';
      }
    });
  }

  function renderActivity(rows) {
    if (!activityBody) return;
    const items = Array.isArray(rows) ? rows : [];
    if (!items.length) {
      activityBody.innerHTML = '<tr><td colspan="8" class="muted">No simulated trades yet.</td></tr>';
      return;
    }
    const fragment = document.createDocumentFragment();
    items.forEach((row) => {
      const tr = document.createElement('tr');
      const ts = row.entry_time || row.ts;
      const exitTime = row.exit_time;
      const pnl = Number(row.pnl);
      const roi = Number(row.roi_pct);
      const qty = Number(row.quantity || row.qty || 0);
      const entryPrice = row.entry_price != null ? formatCurrency(row.entry_price) : '—';
      const exitPrice = row.exit_price != null ? formatCurrency(row.exit_price) : '—';
      const timeCell = document.createElement('td');
      if (ts) {
        const parsed = new Date(ts);
        timeCell.textContent = Number.isNaN(parsed.valueOf()) ? ts : dateTimeFmt.format(parsed);
        timeCell.title = ts;
      } else {
        timeCell.textContent = '—';
      }
      const exitCell = document.createElement('td');
      if (exitTime) {
        const parsedExit = new Date(exitTime);
        exitCell.textContent = Number.isNaN(parsedExit.valueOf()) ? exitTime : dateTimeFmt.format(parsedExit);
        exitCell.title = exitTime;
      } else {
        exitCell.textContent = '—';
      }
      const cells = [
        timeCell,
        row.symbol || '—',
        row.side ? String(row.side).toUpperCase() : '—',
        qty.toString(),
        entryPrice,
        exitPrice,
        formatCurrency(pnl || 0),
        Number.isFinite(roi) ? `${percentFmt.format(roi)}%` : '—',
      ];
      cells.forEach((cell) => {
        let td;
        if (cell instanceof HTMLElement) {
          td = cell;
        } else {
          td = document.createElement('td');
          td.textContent = cell;
        }
        tr.appendChild(td);
      });
      fragment.appendChild(tr);
    });
    activityBody.innerHTML = '';
    activityBody.appendChild(fragment);
  }

  function renderChart(points) {
    if (!chartCanvas) return;
    const ctx = chartCanvas.getContext('2d');
    if (!ctx) return;
    const dataset = Array.isArray(points) ? points : [];
    const dpr = window.devicePixelRatio || 1;
    const rect = chartCanvas.getBoundingClientRect();
    const width = rect.width || 640;
    const height = rect.height || 260;
    chartCanvas.width = width * dpr;
    chartCanvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    if (!dataset.length) {
      return;
    }
    const parsed = dataset
      .map((pt) => ({
        ts: pt.ts,
        date: new Date(pt.ts),
        balance: Number(pt.balance) || 0,
      }))
      .filter((pt) => !Number.isNaN(pt.date.valueOf()));
    if (!parsed.length) {
      return;
    }
    const minBal = Math.min(...parsed.map((pt) => pt.balance));
    const maxBal = Math.max(...parsed.map((pt) => pt.balance));
    const minTime = Math.min(...parsed.map((pt) => pt.date.valueOf()));
    const maxTime = Math.max(...parsed.map((pt) => pt.date.valueOf()));
    const padX = 24;
    const padY = 18;
    const chartW = width - padX * 2;
    const chartH = height - padY * 2;
    const spanBal = Math.max(1e-3, maxBal - minBal);
    const spanTime = Math.max(1e-3, maxTime - minTime);
    const path = parsed.map((pt) => {
      const x = padX + ((pt.date.valueOf() - minTime) / spanTime) * chartW;
      const y = padY + (1 - (pt.balance - minBal) / spanBal) * chartH;
      return { x, y, balance: pt.balance, ts: pt.ts };
    });
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(94,160,255,0.85)';
    ctx.beginPath();
    path.forEach((pt, idx) => {
      if (idx === 0) {
        ctx.moveTo(pt.x, pt.y);
      } else {
        ctx.lineTo(pt.x, pt.y);
      }
    });
    ctx.stroke();
    ctx.fillStyle = 'rgba(94,160,255,0.15)';
    ctx.lineTo(path[path.length - 1].x, height - padY);
    ctx.lineTo(path[0].x, height - padY);
    ctx.closePath();
    ctx.fill();
  }

  function fetchStatus() {
    return fetch('/api/paper/favorites/status', { credentials: 'same-origin' })
      .then((resp) => {
        if (!resp.ok) throw new Error('status_failed');
        return resp.json();
      })
      .then((payload) => {
        updateMetrics(payload, payload.favorites_universe_count);
        updateStatus(payload);
        return payload;
      })
      .catch((err) => {
        console.error('favorites_status_error', err);
        showToast('Unable to load simulator status.', false);
        return null;
      });
  }

  function fetchEquity(rangeKey) {
    const key = rangeKey || currentRange || '1m';
    return fetch(`/api/paper/favorites/equity.json?range=${encodeURIComponent(key)}`, {
      credentials: 'same-origin',
    })
      .then((resp) => {
        if (!resp.ok) throw new Error('equity_failed');
        return resp.json();
      })
      .then((payload) => {
        chartPoints = payload.points || [];
        renderChart(chartPoints);
        return payload;
      })
      .catch((err) => {
        console.error('favorites_equity_error', err);
        showToast('Unable to load equity curve.', false);
        return null;
      });
  }

  function fetchActivity() {
    return fetch('/api/paper/favorites/activity', { credentials: 'same-origin' })
      .then((resp) => {
        if (!resp.ok) throw new Error('activity_failed');
        return resp.json();
      })
      .then((payload) => {
        renderActivity(payload.rows || []);
        return payload;
      })
      .catch((err) => {
        console.error('favorites_activity_error', err);
        showToast('Unable to load simulator activity.', false);
        return null;
      });
  }

  function requestAction(action) {
    if (!action) return;
    setActionsDisabled(true);
    fetch(`/api/paper/favorites/${action}`, {
      method: 'POST',
      credentials: 'same-origin',
    })
      .then((resp) => {
        if (!resp.ok) throw new Error('action_failed');
        return resp.json();
      })
      .then((payload) => {
        updateMetrics(payload, payload.favorites_universe_count);
        updateStatus(payload);
        const verb = action === 'start' ? 'started' : action === 'stop' ? 'stopped' : 'restarted';
        showToast(`Favorites simulator ${verb}.`, true);
      })
      .catch((err) => {
        console.error('favorites_action_error', err);
        showToast('Unable to update simulator state.', false);
      })
      .finally(() => {
        setActionsDisabled(false);
        fetchEquity(currentRange);
        fetchActivity();
      });
  }

  rangeButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const range = btn.dataset.range;
      if (!range) return;
      rangeButtons.forEach((other) => other.classList.remove('is-active'));
      btn.classList.add('is-active');
      currentRange = range;
      fetchEquity(range);
    });
  });

  actionButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const action = btn.dataset.favoritesAction;
      requestAction(action);
    });
  });

  if (summarySeed) {
    updateMetrics(summarySeed, summarySeed.universe || 0);
  }
  if (statusSeed) {
    updateStatus(statusSeed);
  }
  renderChart(chartPoints);
  if (activitySeed && typeof activitySeed === 'object') {
    renderActivity(activitySeed.rows || []);
  }

  fetchStatus();
  fetchActivity();
})();
