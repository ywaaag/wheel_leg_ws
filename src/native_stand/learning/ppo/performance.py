"""Opt-in boundary wall timings; CUDA synchronization never occurs by default."""

from contextlib import contextmanager
from time import perf_counter

import torch


class Measurement:
    def __init__(self, enabled=False, device="cpu"):
        self.enabled = bool(enabled)
        self.device = torch.device(device)
        self.seconds = {}

    def boundary(self):
        if self.enabled and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return perf_counter()

    @contextmanager
    def section(self, name):
        if not self.enabled:
            yield
            return
        started = self.boundary()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + self.boundary() - started

    def batches(self, generator):
        iterator = iter(generator)
        while True:
            with self.section("minibatch_data"):
                batch = next(iterator, None)
            if batch is None:
                return
            yield batch
