function appendLine(text, who) {
  const log = document.getElementById('chat-log');
  const row = document.createElement('div');
  row.innerHTML = `<strong>${who}:</strong> ${text}`;
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
}

async function refreshDashboard() {
  const res = await fetch('/api/dashboard');
  const data = await res.json();
  const map = {
    'runtime-summary':'runtime_summary',
    'ticker-cards':'ticker_cards',
    'radar-queue':'radar_queue',
    'market-intelligence':'market_intelligence',
    'premarket-gameplan':'premarket_gameplan',
    'trade-idea':'trade_idea',
    'target-move':'target_move_board',
    'trade-plan':'trade_plan',
    'trade-plan-tracking':'trade_plan_tracking',
    'trade-review':'trade_review',
    'visual':'visual_explainability',
    'strategy':'strategy_intelligence',
    'execution':'execution_monitor',
    'timeline':'timeline',
    'providers':'provider_diagnostics',
  };
  for (const [id, key] of Object.entries(map)) {
    const el = document.getElementById(id);
    if (el) el.textContent = JSON.stringify(data[key] ?? {}, null, 2);
  }
}

document.getElementById('chat-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const input = document.getElementById('chat-input');
  const message = input.value.trim();
  if (!message) return;
  appendLine(message, 'You');
  input.value = '';
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message })
  });
  const data = await res.json();
  appendLine(data.formatted_response, 'Kade');
  document.getElementById('latest-intent').textContent = JSON.stringify(data.interpreted_intent, null, 2);
  document.getElementById('latest-raw').textContent = JSON.stringify(data.structured_result, null, 2);
  await refreshDashboard();
});

document.getElementById('refresh').addEventListener('click', refreshDashboard);
setInterval(refreshDashboard, 20000);
