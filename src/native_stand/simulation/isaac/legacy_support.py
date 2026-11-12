"""受限的冻结环境适配入口；不复制源码、不运行旧supervisor。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from native_stand.artifacts.io import file_sha256


def activate_view(view_root, bundle):
    import native_stand

    view = Path(view_root).resolve()
    package = view / "native_stand"
    if not (package / "students/fullchain_runtime_impl.py").is_file():
        raise ValueError("缺少已验证的完整V4冻结环境")
    for name in ("native_stand.students", "native_stand.recovery", "native_stand.command_v6"):
        existing = sys.modules.get(name)
        if existing is not None and not str(existing.__file__).startswith(str(package)):
            raise RuntimeError("冻结数值模块已从另一源码根导入: " + name)
    native_stand.__path__.insert(0, str(package))
    sys.path.insert(0, str(Path(bundle).resolve()))
    # Frozen reward/value consumers import this one stable numerical API.
    # The bridge preserves plane math and adds explicit per-instance surfaces.
    sys.path.insert(0, str(Path(__file__).with_name("numeric_bridge")))
    importlib.invalidate_caches()


def loaded_legacy_sources(view_root, bundle):
    roots = (str(Path(view_root).resolve()), str(Path(bundle).resolve()))
    result = {}
    for module in tuple(sys.modules.values()):
        value = getattr(module, "__file__", None)
        if value and str(value).endswith(".py") and str(value).startswith(roots):
            path = Path(value)
            if path.is_file():
                result[str(path)] = file_sha256(path)
    return result
