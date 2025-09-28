(function(){
  const summarySeedEl = document.getElementById('paper-summary-seed');
  let summarySeed = {};
  if (summarySeedEl) {
    try {
      summarySeed = JSON.parse(summarySeedEl.textContent || '{}') || {};
    } catch (err) {
      summarySeed = {};
    }
  }

  const balanceEl = document.getElementById('paper-balance');
  const roiChip = document.getElementById('paper-roi-chip');
  const statusLine = document.getElementById('paper-status-line');
  const rangeButtons = Array.from(document.querySelectorAll('.paper-range-btn'));
  const tableBody = document.getElementById('paper-trades-body');
  const canvas = document.getElementById('paper-equity-chart');
  const tooltip = document.createElement('div');
  tooltip.id = 'paper-chart-tooltip';
  document.body.appendChild(tooltip);

  const currencyFmt = new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD' });
  const percentFmt = new Intl.NumberFormat(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const dateTimeFmt = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' });
  const dateShortFmt = new Intl.DateTimeFormat(undefined, { month: 'numeric', day: 'numeric' });

  let currentRange = '1m';
  let chartState = { points: [], baseBalance: 0, rendered: false };

  function formatCurrency(value) {
    const num = Number(value) || 0;
    return currencyFmt.format(num);
  }

  function formatPercent(value) {
    const num = Number(value) || 0;
    return `${percentFmt.format(num)}%`;
  }

  function applySummary(data) {
    if (!data) return;
    const balance = Number(data.balance) || 0;
    const roi = Number(data.roi_pct) || 0;
    if (balanceEl) {
      balanceEl.dataset.balance = String(balance);
      balanceEl.textContent = formatCurrency(balance);
    }
    if (roiChip) {
      roiChip.textContent = formatPercent(roi);
      roiChip.classList.remove('chip-neg', 'chip-pos');
      roiChip.classList.add(roi < 0 ? 'chip-neg' : 'chip-pos');
    }
    if (statusLine) {
      const status = (data.status || '').toLowerCase();
      const startedRaw = data.started_at;
      let startedLabel = startedRaw;
      if (startedRaw) {
        const parsed = new Date(startedRaw);
        if (!Number.isNaN(parsed.valueOf())) {
          startedLabel = dateTimeFmt.format(parsed);
        }
      }
      if (status === 'active' && startedLabel) {
        statusLine.textContent = `Active and running since ${startedLabel}`;
      } else if (status === 'active') {
        statusLine.textContent = 'Active and running';
      } else {
        statusLine.textContent = 'Inactive';
      }
    }
  }

  function formatStrikeExp(strike, expiry) {
    const strikeNum = Number(strike);
    const strikeText = Number.isFinite(strikeNum) ? `$${strikeNum.toFixed(2).replace(/\.00$/, '')}` : '—';
    if (!expiry) {
      return `${strikeText}–—`;
    }
    const asDate = new Date(expiry);
    if (!Number.isNaN(asDate.valueOf())) {
      return `${strikeText}–${dateShortFmt.format(asDate)}`;
    }
    return `${strikeText}–${expiry}`;
  }

  function formatDateCell(value) {
    if (!value) {
      return { text: '—', title: '' };
    }
    const dt = new Date(value);
    if (Number.isNaN(dt.valueOf())) {
      return { text: value, title: value };
    }
    return { text: dateTimeFmt.format(dt), title: dt.toISOString() };
  }

  function renderTrades(rows) {
    if (!tableBody) return;
    const items = Array.isArray(rows) ? rows : [];
    if (!items.length) {
      tableBody.innerHTML = '<tr><td colspan="10" class="muted">No paper trades yet.</td></tr>';
      return;
    }
    const frag = document.createDocumentFragment();
    items.forEach((row) => {
      const tr = document.createElement('tr');
      const entryMeta = formatDateCell(row.entry_time);
      const exitMeta = formatDateCell(row.exit_time);
      const roi = Number(row.roi_pct);
      const roiCell = document.createElement('td');
      if (Number.isFinite(roi)) {
        roiCell.textContent = `${percentFmt.format(roi)}%`;
        roiCell.className = roi >= 0 ? 'paper-roi-pos' : 'paper-roi-neg';
      } else {
        roiCell.textContent = '—';
      }
      const statusText = row.status ? String(row.status).replace(/\b\w/g, (c) => c.toUpperCase()) : '';

      const cells = [
        row.ticker || '—',
        row.call_put || '—',
        formatStrikeExp(row.strike, row.expiry),
        Number(row.qty) || 0,
        entryMeta,
        row.entry_price != null ? formatCurrency(row.entry_price) : '—',
        exitMeta,
        row.exit_price != null ? formatCurrency(row.exit_price) : '—',
        roiCell,
        statusText || '—',
      ];

      cells.forEach((cell, idx) => {
        let td;
        if (cell instanceof HTMLElement) {
          td = cell;
        } else if (typeof cell === 'object' && cell && 'text' in cell) {
          td = document.createElement('td');
          td.textContent = cell.text;
          if (cell.title) {
            td.title = cell.title;
          }
        } else {
          td = document.createElement('td');
          td.textContent = String(cell);
        }
        tr.appendChild(td);
      });

      frag.appendChild(tr);
    });
    tableBody.innerHTML = '';
    tableBody.appendChild(frag);
  }

  function renderChart(points) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const deviceRatio = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width || 600;
    const height = rect.height || 260;
    canvas.width = width * deviceRatio;
    canvas.height = height * deviceRatio;
    ctx.setTransform(deviceRatio, 0, 0, deviceRatio, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const dataset = Array.isArray(points) ? points : [];
    if (!dataset.length) {
      chartState = { points: [], baseBalance: 0, rendered: false };
      return;
    }
    const balances = dataset.map((d) => Number(d.balance) || 0);
    const times = dataset.map((d) => new Date(d.ts));
    const minBal = Math.min(...balances);
    const maxBal = Math.max(...balances);
    const minTime = Math.min(...times.map((d) => d.valueOf()));
    const maxTime = Math.max(...times.map((d) => d.valueOf()));
    const padX = 20;
    const padY = 20;
    const chartW = width - padX * 2;
    const chartH = height - padY * 2;
    const spanBal = Math.max(1e-6, maxBal - minBal);
    const spanTime = Math.max(1e-6, maxTime - minTime);

    const plotted = dataset.map((point, index) => {
      const t = times[index].valueOf();
      const value = balances[index];
      const x = padX + ((t - minTime) / spanTime) * chartW;
      const y = padY + (1 - (value - minBal) / spanBal) * chartH;
      return { x, y, rawX: t, balance: value, ts: point.ts };
    });

    ctx.lineWidth = 2;
    const gradient = ctx.createLinearGradient(0, padY, 0, height - padY);
    gradient.addColorStop(0, 'rgba(94,160,255,0.7)');
    gradient.addColorStop(1, 'rgba(94,160,255,0.1)');

    ctx.beginPath();
    plotted.forEach((pt, idx) => {
      if (idx === 0) {
        ctx.moveTo(pt.x, pt.y);
      } else {
        ctx.lineTo(pt.x, pt.y);
      }
    });
    ctx.strokeStyle = '#5ea0ff';
    ctx.stroke();

    ctx.lineTo(plotted[plotted.length - 1].x, height - padY);
    ctx.lineTo(plotted[0].x, height - padY);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    chartState = {
      points: plotted,
      baseBalance: balances[0],
      rendered: true,
    };
  }

  function nearestPoint(canvasX) {
    if (!chartState.rendered || !chartState.points.length) return null;
    let best = chartState.points[0];
    let bestDist = Math.abs(canvasX - best.x);
    for (let i = 1; i < chartState.points.length; i += 1) {
      const pt = chartState.points[i];
      const dist = Math.abs(canvasX - pt.x);
      if (dist < bestDist) {
        best = pt;
        bestDist = dist;
      }
    }
    return best;
  }

  function showTooltip(pt, evt) {
    if (!pt) return;
    const base = chartState.baseBalance || 0;
    const delta = pt.balance - base;
    const pct = base === 0 ? 0 : (delta / base) * 100;
    const dt = new Date(pt.ts);
    const positionX = evt.clientX + 12;
    const positionY = evt.clientY + 12;
    tooltip.innerHTML = `
      <strong>${Number.isNaN(dt.valueOf()) ? pt.ts : dateTimeFmt.format(dt)}</strong>
      <div>Balance: ${formatCurrency(pt.balance)}</div>
      <div>Range Δ: ${formatCurrency(delta)} (${percentFmt.format(pct)}%)</div>
    `;
    tooltip.style.left = `${positionX}px`;
    tooltip.style.top = `${positionY}px`;
    tooltip.style.display = 'block';
  }

  function hideTooltip() {
    tooltip.style.display = 'none';
  }

  if (canvas) {
    canvas.addEventListener('mousemove', (evt) => {
      if (!chartState.rendered) return;
      const rect = canvas.getBoundingClientRect();
      const canvasX = evt.clientX - rect.left;
      const pt = nearestPoint(canvasX);
      if (pt) {
        showTooltip(pt, evt);
      } else {
        hideTooltip();
      }
    });
    canvas.addEventListener('mouseleave', hideTooltip);
  }

  function fetchSummary() {
    return fetch('/paper/summary', { credentials: 'same-origin' })
      .then((resp) => (resp.ok ? resp.json() : null))
      .then((data) => {
        if (data) {
          applySummary(data);
        }
      })
      .catch(() => {});
  }

  function fetchEquity(rangeKey) {
    const key = rangeKey || currentRange;
    return fetch(`/paper/equity?range=${encodeURIComponent(key)}`, { credentials: 'same-origin' })
      .then((resp) => (resp.ok ? resp.json() : null))
      .then((payload) => {
        if (payload && Array.isArray(payload.points)) {
          renderChart(payload.points);
        }
      })
      .catch(() => {});
  }

  function fetchTrades() {
    return fetch('/paper/trades?status=all', { credentials: 'same-origin' })
      .then((resp) => (resp.ok ? resp.json() : null))
      .then((payload) => {
        if (payload && Array.isArray(payload.trades)) {
          renderTrades(payload.trades);
        }
      })
      .catch(() => {});
  }

  rangeButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const range = btn.dataset.range || '1m';
      currentRange = range;
      rangeButtons.forEach((el) => el.classList.toggle('is-active', el === btn));
      fetchEquity(range);
    });
  });

  if (summarySeed && Object.keys(summarySeed).length) {
    applySummary(summarySeed);
  }
  fetchSummary();
  fetchEquity(currentRange);
  fetchTrades();
})();
