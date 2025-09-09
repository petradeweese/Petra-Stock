import routes
from concurrent.futures import Future

def test_perform_scan_progress_every(monkeypatch):
    tickers = [f"T{i}" for i in range(12)]

    # stub compute_scan_for_ticker
    monkeypatch.setattr(routes, "compute_scan_for_ticker", lambda t, p: {"ticker": t})
    # no-op preload
    monkeypatch.setattr(routes, "preload_prices", lambda t, i, l: None)

    class ImmediateExecutor:
        def submit(self, fn, *args, **kwargs):
            fut = Future()
            try:
                fut.set_result(fn(*args, **kwargs))
            except Exception as e:
                fut.set_exception(e)
            return fut
    monkeypatch.setattr(routes, "_get_scan_executor", lambda: ImmediateExecutor())

    updates = []

    def prog(d, total, msg):
        if d:
            updates.append(d)

    routes._perform_scan(
        tickers,
        {},
        "",
        progress_cb=prog,
        progress_every=5,
    )
    assert updates == [5, 10, 12]
