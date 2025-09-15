import os
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context

from scanner import compute_scan_for_ticker

_MODE = os.getenv("SCAN_EXECUTOR_MODE", "process").lower()


def _picklable(fn) -> bool:
    try:
        pickle.dumps(fn)
        return True
    except Exception:
        return False


if _MODE == "thread" or not _picklable(compute_scan_for_ticker):
    EXECUTOR = ThreadPoolExecutor(
        max_workers=min(32, 4 * (os.cpu_count() or 2))
    )
else:
    _MP_CTX = get_context("spawn")
    EXECUTOR = ProcessPoolExecutor(
        max_workers=min(os.cpu_count() or 2, 8),
        mp_context=_MP_CTX,
    )
