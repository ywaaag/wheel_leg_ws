"""Read-only versioned source loading, outside tensor computations."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from native_stand.artifacts.io import file_sha256


def load_response(identity):
    if file_sha256(identity["path"]) != identity["sha256"]:
        raise ValueError("Frozen V4 tensor source changed")
    spec = importlib.util.spec_from_file_location("_recovery_frozen_V4_response", identity["path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Response


def asset_module_root(template):
    """Resolve the declared package location without relaxing file identities."""
    path = Path(template["asset_module_path"]).resolve()
    if path.parent.name != "assets" or path.parent.parent.name != "agent_world":
        raise ValueError("Explicit agent_world/assets module package layout")
    if file_sha256(path) != template["asset_module_sha256"]:
        raise ValueError("Asset module identity differs at mapped location")
    for parent in (path.parent, path.parent.parent):
        if not (parent / "__init__.py").is_file():
            raise ValueError("Mapped asset package initializer missing: " + str(parent))
    return path.parents[2]
