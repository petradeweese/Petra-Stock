import logging
import os
import pickle
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context

from scanner import compute_scan_for_ticker


def _max_workers(default_cap: int = 8) -> int:
    import os as _os

    cap = int(os.getenv("EXECUTOR_MAX_WORKERS", "0") or 0)
    return cap if cap > 0 else min(_os.cpu_count() or 2, default_cap)


def _picklable(fn) -> bool:
    try:
        pickle.dumps(fn)
        return True
    except Exception:
        return False


_MODE = os.getenv("SCAN_EXECUTOR_MODE", "process").lower()
_LOGGER = logging.getLogger(__name__)

if _MODE == "thread" or not _picklable(compute_scan_for_ticker):
    MODE = "thread"
    WORKERS = _max_workers(32)
    EXECUTOR: Executor = ThreadPoolExecutor(max_workers=WORKERS)
else:
    MODE = "process"
    _MP_CTX = get_context("spawn")
    WORKERS = _max_workers()
    EXECUTOR = ProcessPoolExecutor(max_workers=WORKERS, mp_context=_MP_CTX)

_LOGGER.info("executor mode=%s workers=%d", MODE, WORKERS)
