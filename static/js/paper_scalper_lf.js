(function(){
  const modes = [
    {
      key: 'lf',
      statusSeedId: 'lf-status-seed',
      equitySeedId: 'lf-equity-seed',
      activitySeedId: 'lf-activity-seed',
      metricsEl: document.getElementById('lf-metrics'),
      secondaryEl: document.getElementById('lf-secondary-metrics'),
      statusLine: document.getElementById('lf-status-line'),
      rangeButtons: Array.from(document.querySelectorAll('.paper-range-btn[data-mode="lf"]')),
      tableBody: document.getElementById('lf-activity-body'),
      canvas: document.getElementById('lf-equity-chart'),
      metricsMode: 'lf',
      statusUrl: '/api/paper/scalper/lf/status',
      equityUrl: (range) => `/api/paper/scalper/lf/equity.json?range=${encodeURIComponent(range)}`,
      activityUrl: '/api/paper/scalper/lf/activity',
    },
    {
      key: 'hf',
      statusSeedId: 'hf-status-seed',
      equitySeedId: 'hf-equity-seed',
      activitySeedId: 'hf-activity-seed',
      metricsEl: document.getElementById('hf-metrics'),
      secondaryEl: document.getElementById('hf-secondary-metrics'),
      statusLine: document.getElementById('hf-status-line'),
      rangeButtons: Array.from(document.querySelectorAll('.paper-range-btn[data-mode="hf"]')),
      tableBody: document.getElementById('hf-activity-body'),
      canvas: document.getElementById('hf-equity-chart'),
      metricsMode: 'hf',
      statusUrl: '/api/paper/scalper/hf/status',
      equityUrl: (range) => `/api/paper/scalper/hf/equity.json?range=${encodeURIComponent(range)}`,
      activityUrl: '/api/paper/scalper/hf/activity',
    },
  ];

  const currencyFmt = new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD' });
  const percentFmt = new Intl.NumberFormat(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  const decimalFmt = new Intl.NumberFormat(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const oneDecimalFmt = new Intl.NumberFormat(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  const dateTimeFmt = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' });

  const chartState = new Map();

  function readSeed(id) {
    const el = document.getElementById(id);
    if (!el) return null;
    try {
      return JSON.parse(el.textContent || 'null');
    } catch (err) {
      return null;
    }
  }

  function formatStrikeExpiry(strike, expiry) {
    const strikeNum = Number(strike);
    const strikeText = Number.isFinite(strikeNum) ? `$${strikeNum.toFixed(2).replace(/\.00$/, '')}` : '—';
    if (!expiry) {
      return `${strikeText}-—`;
    }
    const maybeDate = new Date(expiry);
    if (!Number.isNaN(maybeDate.valueOf())) {
      const month = String(maybeDate.getMonth() + 1);
      const day = String(maybeDate.getDate());
      return `${strikeText}-${month}/${day}`;
    }
    return `${strikeText}-${expiry}`;
  }

  function formatTs(value) {
    if (!value) return { text: '—', title: '' };
    const dt = new Date(value);
    if (Number.isNaN(dt.valueOf())) {
      return { text: value, title: value };
    }
    return { text: dateTimeFmt.format(dt), title: dt.toISOString() };
  }

  function renderPrimaryMetrics(mode, data) {
    const metricsEl = mode.metricsEl;
    const statusLine = mode.statusLine;
    if (!metricsEl || !data) return;
    const equityEl = metricsEl.querySelector('[data-metric="equity"]');
    const openEl = metricsEl.querySelector('[data-metric="open_positions"]');
    const realizedEl = metricsEl.querySelector('[data-metric="realized"]');
    const unrealizedEl = metricsEl.querySelector('[data-metric="unrealized"]');
    const winEl = metricsEl.querySelector('[data-metric="win_rate"]');
    const started = data.started_at;
    if (equityEl) equityEl.textContent = currencyFmt.format(Number(data.account_equity || 0));
    if (openEl) openEl.textContent = String(data.open_positions || 0);
    if (realizedEl) realizedEl.textContent = currencyFmt.format(Number(data.realized_pl_day || 0));
    if (unrealizedEl) unrealizedEl.textContent = currencyFmt.format(Number(data.unrealized_pl || 0));
    if (winEl) winEl.textContent = `${percentFmt.format(Number(data.win_rate_pct || 0))}%`;
    metricsEl.dataset.status = data.status || '';
    metricsEl.dataset.started = started || '';
    if ('halted' in data) {
      metricsEl.dataset.halted = data.halted ? 'true' : 'false';
    }
    if (statusLine) {
      const statusKey = (data.status || '').toLowerCase();
      if (statusKey === 'active') {
        if (data.halted) {
          statusLine.textContent = 'Halted — awaiting manual restart';
        } else if (started) {
          const dt = new Date(started);
          const label = Number.isNaN(dt.valueOf()) ? started : dateTimeFmt.format(dt);
          statusLine.textContent = `Active since ${label}`;
        } else {
          statusLine.textContent = 'Active';
        }
      } else if (statusKey === 'halted') {
        statusLine.textContent = 'Halted — risk guard triggered';
      } else {
        statusLine.textContent = 'Inactive';
      }
    }
  }

  function renderSecondaryMetrics(container, metrics) {
    if (!container) return;
    const data = metrics || {};
    const avgWin = container.querySelector('[data-metric="avg_win"]');
    const avgLoss = container.querySelector('[data-metric="avg_loss"]');
    const profitFactor = container.querySelector('[data-metric="profit_factor"]');
    const sharpe = container.querySelector('[data-metric="sharpe"]');
    const maxDd = container.querySelector('[data-metric="max_drawdown"]');
    const tradesDay = container.querySelector('[data-metric="trades_per_day"]');
    if (avgWin) avgWin.textContent = currencyFmt.format(Number(data.avg_win || 0));
    if (avgLoss) avgLoss.textContent = currencyFmt.format(Number(data.avg_loss || 0));
    if (profitFactor) profitFactor.textContent = decimalFmt.format(Number(data.profit_factor || 0));
    if (sharpe) sharpe.textContent = decimalFmt.format(Number(data.sharpe || 0));
    if (maxDd) maxDd.textContent = `${percentFmt.format(Number(data.max_drawdown || 0))}%`;
    if (tradesDay) tradesDay.textContent = oneDecimalFmt.format(Number(data.trades_per_day || 0));
  }

  function renderTable(body, rows) {
    if (!body) return;
    const data = Array.isArray(rows) ? rows : [];
    if (!data.length) {
      body.innerHTML = '<tr><td colspan="11" class="muted">No scalper trades yet.</td></tr>';
      return;
    }
    const frag = document.createDocumentFragment();
    data.forEach((row) => {
      const tr = document.createElement('tr');
      const entryMeta = formatTs(row.entry_time);
      const exitMeta = formatTs(row.exit_time);
      const roiValue = Number(row.roi_pct);
      const roiCell = document.createElement('td');
      if (Number.isFinite(roiValue)) {
        roiCell.textContent = `${percentFmt.format(roiValue)}%`;
        roiCell.className = roiValue >= 0 ? 'paper-roi-pos' : 'paper-roi-neg';
      } else {
        roiCell.textContent = '—';
      }
      const cells = [
        row.date || '—',
        row.ticker || '—',
        row.call_put || '—',
        formatStrikeExpiry(row.strike, row.expiry),
        Number(row.qty) || 0,
        entryMeta,
        row.entry_price != null ? currencyFmt.format(Number(row.entry_price)) : '—',
        exitMeta,
        row.exit_price != null ? currencyFmt.format(Number(row.exit_price)) : '—',
        roiCell,
        currencyFmt.format(Number(row.fees || 0)),
      ];
      cells.forEach((cell) => {
        let td;
        if (cell instanceof HTMLElement) {
          td = cell;
        } else if (typeof cell === 'object' && cell && 'text' in cell) {
          td = document.createElement('td');
          td.textContent = cell.text;
          if (cell.title) td.title = cell.title;
        } else {
          td = document.createElement('td');
          td.textContent = String(cell);
        }
        tr.appendChild(td);
      });
      frag.appendChild(tr);
    });
    body.innerHTML = '';
    body.appendChild(frag);
  }

  function renderChart(canvas, stateKey, points) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dataset = Array.isArray(points) ? points : [];
    const rect = canvas.getBoundingClientRect();
    const width = rect.width || 600;
    const height = rect.height || 260;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    if (!dataset.length) {
      chartState.set(stateKey, { points: [], rendered: true, range: (chartState.get(stateKey) || {}).range || '1m' });
      return;
    }
    const times = dataset.map((pt) => new Date(pt.ts || pt.time || Date.now()));
    const balances = dataset.map((pt) => Number(pt.balance || 0));
    const minTime = Math.min(...times.map((d) => d.valueOf()));
    const maxTime = Math.max(...times.map((d) => d.valueOf()));
    const minBal = Math.min(...balances);
    const maxBal = Math.max(...balances);
    const padX = 32;
    const padY = 24;
    const plotW = width - padX * 2;
    const plotH = height - padY * 2;
    const spanT = Math.max(1, maxTime - minTime);
    const spanB = Math.max(1, maxBal - minBal);
    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(94,160,255,0.9)';
    dataset.forEach((pt, idx) => {
      const t = times[idx].valueOf();
      const b = balances[idx];
      const x = padX + ((t - minTime) / spanT) * plotW;
      const y = padY + (1 - (b - minBal) / spanB) * plotH;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    const current = chartState.get(stateKey) || {};
    chartState.set(stateKey, { points: dataset, rendered: true, range: current.range || '1m' });
  }

  function fetchStatus(mode) {
    return fetch(mode.statusUrl, { credentials: 'same-origin' })
      .then((resp) => (resp.ok ? resp.json() : Promise.reject()))
      .then((payload) => {
        renderPrimaryMetrics(mode, payload);
        return payload;
      })
      .catch(() => null);
  }

  function fetchEquity(mode, range) {
    const state = chartState.get(mode.key) || { range: '1m' };
    const key = range || state.range || '1m';
    return fetch(mode.equityUrl(key), { credentials: 'same-origin' })
      .then((resp) => (resp.ok ? resp.json() : Promise.reject()))
      .then((payload) => {
        chartState.set(mode.key, { range: payload.range, rendered: true, points: payload.points || [] });
        renderChart(mode.canvas, mode.key, payload.points || []);
        return payload;
      })
      .catch(() => null);
  }

  function fetchActivity(mode) {
    return fetch(mode.activityUrl, { credentials: 'same-origin' })
      .then((resp) => (resp.ok ? resp.json() : Promise.reject()))
      .then((payload) => {
        renderTable(mode.tableBody, payload.rows || []);
        return payload;
      })
      .catch(() => null);
  }

  function fetchMetrics(mode) {
    return fetch(`/api/paper/scalper/metrics.json?mode=${encodeURIComponent(mode.metricsMode)}`, { credentials: 'same-origin' })
      .then((resp) => (resp.ok ? resp.json() : Promise.reject()))
      .then((payload) => {
        renderSecondaryMetrics(mode.secondaryEl, payload.metrics || {});
        return payload;
      })
      .catch(() => null);
  }

  modes.forEach((mode) => {
    chartState.set(mode.key, { points: [], rendered: false, range: '1m' });
    const statusSeed = readSeed(mode.statusSeedId) || {};
    const equitySeedRaw = readSeed(mode.equitySeedId);
    const equitySeed = Array.isArray(equitySeedRaw) ? equitySeedRaw : [];
    const activitySeedRaw = readSeed(mode.activitySeedId);
    const activitySeed = Array.isArray(activitySeedRaw) ? activitySeedRaw : [];

    renderPrimaryMetrics(mode, statusSeed);
    renderChart(mode.canvas, mode.key, equitySeed);
    renderTable(mode.tableBody, activitySeed);
    renderSecondaryMetrics(mode.secondaryEl, null);

    mode.rangeButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        mode.rangeButtons.forEach((b) => b.classList.remove('is-active'));
        btn.classList.add('is-active');
        const key = btn.dataset.range || '1m';
        const state = chartState.get(mode.key) || {};
        chartState.set(mode.key, { ...state, range: key });
        fetchEquity(mode, key);
      });
    });

    fetchStatus(mode);
    fetchEquity(mode, chartState.get(mode.key)?.range || '1m');
    fetchActivity(mode);
    fetchMetrics(mode);
  });
})();
