(function () {
  const page = document.querySelector('.forecast-page');
  if (!page) {
    return;
  }

  const form = document.getElementById('forecast-form');
  const symbolInput = document.getElementById('forecast-symbol');
  const resultsBody = document.getElementById('forecast-results');
  const errorEl = document.getElementById('forecast-error');
  const table = document.getElementById('forecast-table');
  const summaryMap = {
    matches: document.querySelector('[data-summary="matches"]'),
    median_close_pct: document.querySelector('[data-summary="median_close_pct"]'),
    expected_move_iqr: document.querySelector('[data-summary="expected_move_iqr"]'),
    median_high_pct: document.querySelector('[data-summary="median_high_pct"]'),
    median_low_pct: document.querySelector('[data-summary="median_low_pct"]')
  };
  const gauge = {
    root: document.querySelector('[data-confidence-gauge]'),
    value: document.querySelector('[data-confidence-value]'),
    bias: document.querySelector('[data-bias-label]')
  };
  const rangeRoot = document.querySelector('[data-forecast-range]');
  const range = {
    root: rangeRoot,
    bar: rangeRoot ? rangeRoot.querySelector('.forecast-range-bar') : null,
    iqr: rangeRoot ? rangeRoot.querySelector('[data-range-iqr]') : null,
    median: rangeRoot ? rangeRoot.querySelector('[data-range-median]') : null,
    zero: rangeRoot ? rangeRoot.querySelector('[data-range-zero]') : null,
    lowLabel: rangeRoot ? rangeRoot.querySelector('[data-range-low]') : null,
    closeLabel: rangeRoot ? rangeRoot.querySelector('[data-range-close]') : null,
    highLabel: rangeRoot ? rangeRoot.querySelector('[data-range-high]') : null,
    pill: rangeRoot ? rangeRoot.querySelector('[data-iv-pill]') : null
  };
  const submitBtn = form.querySelector('button[type="submit"]');

  let currentTicker = '';
  let matches = [];
  let sortState = { key: 'similarity', direction: 'desc' };

  function formatPercent(value, digits = 2) {
    if (typeof value !== 'number' || Number.isNaN(value)) {
      return '—';
    }
    const fixed = value.toFixed(digits);
    const prefix = Number(fixed) > 0 ? '+' : '';
    return `${prefix}${fixed}%`;
  }

  function formatNumber(value, digits = 2) {
    if (typeof value !== 'number' || Number.isNaN(value)) {
      return '—';
    }
    return value.toFixed(digits);
  }

  function toNumber(value) {
    if (typeof value === 'number') {
      return Number.isFinite(value) ? value : null;
    }
    if (typeof value === 'string' && value.trim() !== '') {
      const parsed = Number.parseFloat(value);
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
    return null;
  }

  function clearSummary() {
    Object.values(summaryMap).forEach((el) => {
      if (el) {
        el.textContent = '—';
      }
    });
  }

  function clearVisuals() {
    updateGauge(null);
    updateRange(null);
  }

  function showError(message) {
    if (errorEl) {
      errorEl.textContent = message || 'No forecast available.';
      errorEl.hidden = false;
    }
    clearSummary();
    clearVisuals();
    resultsBody.innerHTML = '';
  }

  function hideError() {
    if (errorEl) {
      errorEl.hidden = true;
    }
  }

  function updateGauge(data) {
    if (!gauge.root || !gauge.value || !gauge.bias) {
      return;
    }
    let pct = null;
    if (data && typeof data.confidence_pct === 'number') {
      pct = data.confidence_pct;
    } else if (data && typeof data.confidence === 'number') {
      pct = data.confidence * 100;
    }
    if (pct === null || Number.isNaN(pct)) {
      gauge.root.classList.add('is-empty');
      gauge.root.classList.remove('is-strong', 'is-moderate', 'is-weak');
      gauge.root.style.removeProperty('--confidence');
      gauge.value.textContent = '—';
    } else {
      const clamped = Math.min(100, Math.max(0, pct));
      gauge.root.classList.remove('is-empty');
      gauge.root.style.setProperty('--confidence', clamped.toFixed(1));
      gauge.value.textContent = `${clamped.toFixed(1)}%`;
      gauge.root.classList.remove('is-strong', 'is-moderate', 'is-weak');
      if (clamped > 70) {
        gauge.root.classList.add('is-strong');
      } else if (clamped >= 40) {
        gauge.root.classList.add('is-moderate');
      } else {
        gauge.root.classList.add('is-weak');
      }
    }
    const bias = data && typeof data.bias === 'string' ? data.bias : null;
    gauge.root.classList.toggle('bias-up', bias === 'Up');
    gauge.root.classList.toggle('bias-down', bias === 'Down');
    gauge.bias.textContent = bias ? `Bias ${bias}` : 'Bias —';
  }

  function updateRange(data) {
    if (!range.root || !range.bar) {
      return;
    }
    const summary = (data && data.summary) || {};
    const iqrRaw = Array.isArray(summary.expected_move_iqr)
      ? summary.expected_move_iqr
      : summary.iqr_close_pct;
    const iqrLow = Array.isArray(iqrRaw) && iqrRaw.length === 2 ? toNumber(iqrRaw[0]) : null;
    const iqrHigh = Array.isArray(iqrRaw) && iqrRaw.length === 2 ? toNumber(iqrRaw[1]) : null;
    const medianClose = toNumber(summary.median_close_pct);
    const medianHigh = toNumber(summary.median_high_pct);
    const medianLow = toNumber(summary.median_low_pct);

    if (range.lowLabel) {
      range.lowLabel.textContent = `Median Low ${formatPercent(
        typeof medianLow === 'number' ? medianLow : Number.NaN,
        2
      )}`;
    }
    if (range.closeLabel) {
      range.closeLabel.textContent = `Median Close ${formatPercent(
        typeof medianClose === 'number' ? medianClose : Number.NaN,
        2
      )}`;
    }
    if (range.highLabel) {
      range.highLabel.textContent = `Median High ${formatPercent(
        typeof medianHigh === 'number' ? medianHigh : Number.NaN,
        2
      )}`;
    }

    const ivHint = (summary.iv_rank_hint || data?.iv_rank_hint || '').toString().toLowerCase();
    if (range.pill) {
      range.pill.classList.remove('iv-pill-cheap', 'iv-pill-neutral', 'iv-pill-rich');
      const normalized = ivHint === 'cheap' || ivHint === 'rich' ? ivHint : 'neutral';
      range.pill.classList.add(`iv-pill-${normalized}`);
      const label = normalized.charAt(0).toUpperCase() + normalized.slice(1);
      range.pill.textContent = `IV: ${label}`;
    }

    const bias = data && typeof data.bias === 'string' ? data.bias : null;
    range.root.classList.toggle('bias-up', bias === 'Up');
    range.root.classList.toggle('bias-down', bias === 'Down');

    const values = [0];
    if (typeof iqrLow === 'number') {
      values.push(iqrLow);
    }
    if (typeof iqrHigh === 'number') {
      values.push(iqrHigh);
    }
    if (typeof medianClose === 'number') {
      values.push(medianClose);
    }
    if (typeof medianHigh === 'number') {
      values.push(medianHigh);
    }
    if (typeof medianLow === 'number') {
      values.push(medianLow);
    }

    if (values.length <= 1) {
      range.root.classList.add('is-empty');
      if (range.iqr) {
        range.iqr.style.width = '0%';
      }
      if (range.median) {
        range.median.style.left = '50%';
      }
      if (range.zero) {
        range.zero.style.left = '50%';
      }
      return;
    }

    let minVal = Math.min(...values);
    let maxVal = Math.max(...values);
    if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) {
      minVal = -1;
      maxVal = 1;
    }
    if (Math.abs(maxVal - minVal) < 1e-6) {
      maxVal = minVal + 1;
      minVal = minVal - 1;
    }
    const span = maxVal - minVal || 1;
    const toPercent = (val) => ((val - minVal) / span) * 100;

    range.root.classList.remove('is-empty');

    if (range.zero) {
      range.zero.style.left = `${Math.min(100, Math.max(0, toPercent(0)))}%`;
    }

    if (range.median) {
      if (typeof medianClose === 'number') {
        range.median.style.left = `${Math.min(100, Math.max(0, toPercent(medianClose)))}%`;
        range.median.hidden = false;
      } else {
        range.median.hidden = true;
      }
    }

    if (range.iqr) {
      if (typeof iqrLow === 'number' && typeof iqrHigh === 'number') {
        const start = Math.min(iqrLow, iqrHigh);
        const end = Math.max(iqrLow, iqrHigh);
        const startPct = Math.min(100, Math.max(0, toPercent(start)));
        const endPct = Math.min(100, Math.max(0, toPercent(end)));
        const width = Math.max(1.5, endPct - startPct);
        range.iqr.style.left = `${startPct}%`;
        range.iqr.style.width = `${width}%`;
        range.iqr.hidden = false;
      } else {
        range.iqr.style.width = '0%';
        range.iqr.hidden = true;
      }
    }
  }

  function updateSummary(data) {
    if (!data) {
      clearSummary();
      clearVisuals();
      return;
    }
    const summary = data.summary || {};
    updateGauge(data);
    updateRange(data);
    if (summaryMap.matches) {
      summaryMap.matches.textContent = `${data.n ?? matches.length}`;
    }
    if (summaryMap.median_close_pct) {
      summaryMap.median_close_pct.textContent = formatPercent(summary.median_close_pct ?? NaN, 2);
    }
    if (summaryMap.expected_move_iqr) {
      const iqr = Array.isArray(summary.expected_move_iqr)
        ? summary.expected_move_iqr
        : summary.iqr_close_pct;
      if (Array.isArray(iqr) && iqr.length === 2) {
        summaryMap.expected_move_iqr.textContent = `${formatPercent(
          iqr[0] ?? NaN,
          2
        )} to ${formatPercent(iqr[1] ?? NaN, 2)}`;
      } else {
        summaryMap.expected_move_iqr.textContent = '—';
      }
    }
    if (summaryMap.median_high_pct) {
      summaryMap.median_high_pct.textContent = formatPercent(summary.median_high_pct ?? NaN, 2);
    }
    if (summaryMap.median_low_pct) {
      summaryMap.median_low_pct.textContent = formatPercent(summary.median_low_pct ?? NaN, 2);
    }
  }

  function buildMatchLink(match) {
    if (!match || !match.timestamp) {
      return '#';
    }
    const encoded = encodeURIComponent(match.timestamp);
    return `/api/forecast/${encodeURIComponent(currentTicker)}?asof=${encoded}`;
  }

  function renderMatches(list) {
    resultsBody.innerHTML = '';
    if (!list.length) {
      const emptyRow = document.createElement('tr');
      const cell = document.createElement('td');
      cell.colSpan = 6;
      cell.textContent = 'No matches available.';
      emptyRow.appendChild(cell);
      resultsBody.appendChild(emptyRow);
      return;
    }

    list.forEach((match) => {
      const row = document.createElement('tr');
      const dateCell = document.createElement('td');
      const link = document.createElement('a');
      link.href = buildMatchLink(match);
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = match.date || '—';
      dateCell.appendChild(link);
      row.appendChild(dateCell);

      const similarityCell = document.createElement('td');
      similarityCell.classList.add('num');
      similarityCell.textContent = formatNumber(match.similarity ?? NaN, 4);
      row.appendChild(similarityCell);

      const timeCell = document.createElement('td');
      timeCell.textContent = match.time_of_match || '—';
      row.appendChild(timeCell);

      const closeCell = document.createElement('td');
      closeCell.classList.add('num');
      closeCell.textContent = formatPercent(match.close_pct ?? NaN, 2);
      row.appendChild(closeCell);

      const highCell = document.createElement('td');
      highCell.classList.add('num');
      highCell.textContent = formatPercent(match.high_pct ?? NaN, 2);
      row.appendChild(highCell);

      const lowCell = document.createElement('td');
      lowCell.classList.add('num');
      lowCell.textContent = formatPercent(match.low_pct ?? NaN, 2);
      row.appendChild(lowCell);

      resultsBody.appendChild(row);
    });
  }

  function applySort(key, toggle = true) {
    if (!matches.length) {
      renderMatches([]);
      return;
    }
    if (sortState.key === key) {
      if (toggle) {
        sortState.direction = sortState.direction === 'asc' ? 'desc' : 'asc';
      }
    } else {
      sortState = { key, direction: key === 'date' ? 'desc' : 'desc' };
    }

    const sorted = [...matches];
    sorted.sort((a, b) => {
      const dir = sortState.direction === 'asc' ? 1 : -1;
      if (key === 'date') {
        return dir * ((a.timestamp || '').localeCompare(b.timestamp || ''));
      }
      if (key === 'time') {
        return dir * ((a.time_of_match || '').localeCompare(b.time_of_match || ''));
      }
      if (key === 'similarity') {
        return dir * ((a.similarity ?? 0) - (b.similarity ?? 0));
      }
      if (key === 'close') {
        return dir * ((a.close_pct ?? 0) - (b.close_pct ?? 0));
      }
      if (key === 'high') {
        return dir * ((a.high_pct ?? 0) - (b.high_pct ?? 0));
      }
      if (key === 'low') {
        return dir * ((a.low_pct ?? 0) - (b.low_pct ?? 0));
      }
      return 0;
    });
    renderMatches(sorted);
  }

  function setLoading(loading) {
    if (!submitBtn) {
      return;
    }
    submitBtn.disabled = loading;
    submitBtn.textContent = loading ? 'Loading…' : 'Run';
  }

  document.body.addEventListener('htmx:configRequest', (event) => {
    if (event.target !== form) {
      return;
    }
    const symbol = (symbolInput.value || '').trim().toUpperCase();
    symbolInput.value = symbol;
    if (!symbol) {
      event.preventDefault();
      showError('Enter a ticker to see a forecast.');
      return;
    }
    hideError();
    const params = new URLSearchParams();
    params.set('limit', '20');
    event.detail.path = `/api/forecast/${encodeURIComponent(symbol)}?${params.toString()}`;
    event.detail.headers['Accept'] = 'application/json';
  });

  document.body.addEventListener('htmx:beforeRequest', (event) => {
    if (event.target === form) {
      setLoading(true);
    }
  });

  document.body.addEventListener('htmx:afterRequest', (event) => {
    if (event.target !== form) {
      return;
    }
    setLoading(false);
    if (!event.detail.successful) {
      showError('No forecast available.');
      return;
    }
    try {
      const data = JSON.parse(event.detail.xhr.responseText || '{}');
      currentTicker = data.ticker || symbolInput.value;
      matches = Array.isArray(data.matches) ? data.matches : [];
      updateSummary(data);
      applySort(sortState.key, false);
      hideError();
    } catch (err) {
      console.error('Failed to parse forecast response', err); // eslint-disable-line no-console
      showError('No forecast available.');
    }
  });

  document.body.addEventListener('htmx:responseError', (event) => {
    if (event.target === form) {
      setLoading(false);
      showError('No forecast available.');
    }
  });

  if (table) {
    table.querySelectorAll('th').forEach((th) => {
      th.addEventListener('click', () => {
        const key = th.dataset.sort;
        if (key) {
          applySort(key);
        }
      });
    });
  }

  const initialSymbol = page.dataset.symbol || '';
  if (initialSymbol) {
    symbolInput.value = initialSymbol;
    htmx.trigger(form, 'submit');
  }
})();
