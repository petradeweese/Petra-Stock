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
    confidence: document.querySelector('[data-summary="confidence"]'),
    matches: document.querySelector('[data-summary="matches"]'),
    median_close_pct: document.querySelector('[data-summary="median_close_pct"]'),
    iqr_close_pct: document.querySelector('[data-summary="iqr_close_pct"]'),
    median_high_pct: document.querySelector('[data-summary="median_high_pct"]'),
    median_low_pct: document.querySelector('[data-summary="median_low_pct"]')
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

  function clearSummary() {
    Object.values(summaryMap).forEach((el) => {
      if (el) {
        el.textContent = '—';
      }
    });
  }

  function showError(message) {
    if (errorEl) {
      errorEl.textContent = message || 'No forecast available.';
      errorEl.hidden = false;
    }
    clearSummary();
    resultsBody.innerHTML = '';
  }

  function hideError() {
    if (errorEl) {
      errorEl.hidden = true;
    }
  }

  function updateSummary(data) {
    if (!data) {
      clearSummary();
      return;
    }
    const summary = data.summary || {};
    if (summaryMap.confidence) {
      summaryMap.confidence.textContent = formatPercent((data.confidence || 0) * 100, 1);
    }
    if (summaryMap.matches) {
      summaryMap.matches.textContent = `${data.n ?? matches.length}`;
    }
    if (summaryMap.median_close_pct) {
      summaryMap.median_close_pct.textContent = formatPercent(summary.median_close_pct ?? NaN, 2);
    }
    if (summaryMap.iqr_close_pct) {
      const iqr = summary.iqr_close_pct;
      if (Array.isArray(iqr) && iqr.length === 2) {
        summaryMap.iqr_close_pct.textContent = `${formatPercent(iqr[0] ?? NaN, 2)} to ${formatPercent(iqr[1] ?? NaN, 2)}`;
      } else {
        summaryMap.iqr_close_pct.textContent = '—';
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
