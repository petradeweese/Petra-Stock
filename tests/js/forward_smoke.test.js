const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { createTestEnv, makeFavorite } = require('./dom-shim');

function loadForwardScript(env) {
  const scriptPath = path.resolve(__dirname, '../../static/js/forward.js');
  const source = fs.readFileSync(scriptPath, 'utf8');
  const context = vm.createContext({ ...env });
  const script = new vm.Script(source, { filename: 'forward.js' });
  script.runInContext(context);
  return env.window.__forwardTestHooks;
}

async function testSkeletonToTable() {
  const env = createTestEnv();
  env.window.fetch = () =>
    new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          ok: true,
          json: async () => ({ favorites: [makeFavorite(1), makeFavorite(2)] }),
        });
      }, 0);
    });
  env.fetch = env.window.fetch;
  const hooks = loadForwardScript(env);
  const promise = hooks.loadFavorites();
  assert.ok(promise && typeof promise.then === 'function', 'loadFavorites returns a promise');
  assert.ok(hooks.getSkeletonCount() > 0, 'skeleton rows render while loading');
  await promise;
  assert.strictEqual(hooks.getSkeletonCount(), 0, 'skeleton rows removed after load');
  assert.strictEqual(env.document.getElementById('forward-table').hidden, false, 'table visible');
  assert.strictEqual(hooks.getFavorites().length, 2, 'favorites rendered');
}

async function flush() {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

async function testHistoryModalPagination() {
  const env = createTestEnv();
  const firstPage = Array.from({ length: 50 }).map((_, index) => ({
    entry_ts: `2025-01-${String(index + 1).padStart(2, '0')}T09:30:00Z`,
    exit_ts: `2025-01-${String(index + 1).padStart(2, '0')}T11:00:00Z`,
    outcome: index % 3 === 0 ? 'hit' : index % 3 === 1 ? 'stop' : 'timeout',
    roi: index % 2 === 0 ? 0.01 : -0.005,
    tt_bars: 4 + (index % 3),
    dd: 0.002,
  }));
  const secondPage = [
    {
      entry_ts: '2025-03-01T09:30:00Z',
      exit_ts: '2025-03-01T11:00:00Z',
      outcome: 'hit',
      roi: 0.02,
      tt_bars: 6,
      dd: 0.003,
    },
    {
      entry_ts: '2025-03-02T09:30:00Z',
      exit_ts: '2025-03-02T11:00:00Z',
      outcome: 'stop',
      roi: -0.01,
      tt_bars: 5,
      dd: 0.004,
    },
  ];
  let callIndex = 0;
  env.window.fetch = async () => {
    const page = callIndex === 0 ? firstPage : callIndex === 1 ? secondPage : [];
    callIndex += 1;
    return {
      ok: true,
      json: async () => page,
    };
  };
  env.fetch = env.window.fetch;
  const hooks = loadForwardScript(env);
  const favorite = makeFavorite(9);
  hooks.openHistoryModal(favorite);
  await flush();
  await flush();
  assert.strictEqual(hooks.getHistoryCount(), firstPage.length, 'first page loaded');
  assert.ok(hooks.virtualVisible() > 0, 'virtual list rendered items');
  await hooks.loadHistoryModalRuns(false);
  assert.strictEqual(
    hooks.getHistoryCount(),
    firstPage.length + secondPage.length,
    'second page appended'
  );
}

async function run() {
  await testSkeletonToTable();
  await testHistoryModalPagination();
}

run().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
