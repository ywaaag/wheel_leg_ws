"""Opt-in wall timings and small in-memory counters; no default CUDA work.

Sections are inclusive. ``parents`` describes nesting; never sum parents and
children. Only outer sections synchronize CUDA unless explicitly requested.
No GPU event/kernel time is inferred from a wall clock.
"""

from contextlib import contextmanager
from time import perf_counter

import torch


class Measurement:
    def __init__(self, enabled=False, device="cpu"):
        self.enabled = bool(enabled)
        self.device = torch.device(device)
        self.seconds = {}
        self.counters = {}
        self.parents = {}
        self.clocks = {}
        self._stack = []

    def boundary(self, *, synchronize=True):
        if self.enabled and synchronize and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return perf_counter()

    @contextmanager
    def section(self, name, *, parent=None, synchronize=None):
        if not self.enabled:
            yield
            return
        # Hot-path children measure CPU dispatch wall time, not GPU execution.
        sync = not self._stack if synchronize is None else synchronize
        parent = parent if parent is not None else (self._stack[-1] if self._stack else None)
        clock = "synchronized_stage_wall" if sync and self.device.type == "cuda" else "cpu_wall"
        self.parents[name] = parent
        self.clocks[name] = clock
        started = self.boundary(synchronize=sync)
        self._stack.append(name)
        try:
            yield
        except BaseException as primary:
            try:
                self.seconds[name] = (
                    self.seconds.get(name, 0.0) + self.boundary(synchronize=sync) - started
                )
            except BaseException as timing:
                primary.add_note("Performance boundary failed: " + repr(timing))
            raise
        else:
            self.seconds[name] = (
                self.seconds.get(name, 0.0) + self.boundary(synchronize=sync) - started
            )
        finally:
            self._stack.pop()

    def count(self, name, amount=1, *, reason=None):
        """Accumulate host-known counts only; callers must not read back tensors."""
        if not self.enabled:
            return
        if not isinstance(amount, int | float) or isinstance(amount, bool):
            raise TypeError("Measurement counters require a host int/float")
        key = name if reason is None else f"{name}:{reason}"
        self.counters[key] = self.counters.get(key, 0) + amount

    def reset(self):
        """Start a reporting window without changing the shared object identity."""
        if self._stack:
            raise RuntimeError("Cannot reset an active measurement section")
        self.seconds.clear()
        self.counters.clear()
        self.parents.clear()
        self.clocks.clear()

    def summary(self):
        return dict(
            seconds=dict(self.seconds),
            counters=dict(self.counters),
            parents=dict(self.parents),
            clocks=dict(self.clocks),
            aggregation="inclusive; do not add child sections to their parents",
        )

    def batches(self, generator):
        iterator = iter(generator)
        while True:
            with self.section("minibatch_data"):
                batch = next(iterator, None)
            if batch is None:
                return
            yield batch
