"""Keep the public owner's graceful checkpoint/stop handlers across Kit launch."""

import signal
from contextlib import contextmanager


@contextmanager
def preserve_worker_signals():
    original = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for s, handler in original.items():
            signal.signal(s, handler)
