"""Read-only versioned source loading, outside tensor computations."""

from __future__ import annotations

import importlib.util

from native_stand.artifacts.io import file_sha256


def load_response(identity):
    if file_sha256(identity["path"]) != identity["sha256"]:
        raise ValueError("Frozen V4 tensor source changed")
    spec = importlib.util.spec_from_file_location("_recovery_frozen_V4_response", identity["path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Response
