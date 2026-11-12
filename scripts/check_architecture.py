#!/usr/bin/env python
"""正式源码的轻量架构约束检查（纯标准库AST，不导入项目运行模块）。

规则（只覆盖能可靠静态识别的模式，不追求检测任意动态行为）：
  R1-entry   `if __name__ == "__main__"` 并行入口只允许：
             src/native_stand/{__main__.py, cli.py, experiments/worker.py}。
  R1-process 训练进程创建（按 import 绑定识别 subprocess.Popen / multiprocessing.Process /
             os.fork / os.spawn*，含 as 别名）只允许 experiments/owner.py 的准确包内路径。
             普通方法名（如 buffer.spawn()）不误判。
  R2-imports 模块依赖方向（绝对与相对导入同样解析）：
             tasks↛cli/experiments.owner/experiments.report；
             learning↛cli；simulation↛integrations；artifacts↛tasks；
             正式源码↛scripts。`experiments.config` 等数据类型放行。
  R3-rebind  不得通过导入别名改写项目模块的全局实现对象（如 workflow.DelayedPlant = ...）；
             别名经相对/绝对导入解析，并用项目源文件索引确认目标是模块。
             显式 factory 参数注入是唯一合法方式；普通对象属性赋值不受限。

退出码：0=通过；1=发现违规；2=参数或根目录错误。
基线：scripts/architecture_baseline.json，精确身份 = 相对路径+规则+限定符号+
规范化表达式（ast.dump，不含行号属性）+允许次数；行号仅定位。
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

PACKAGE_ROOT = "src/native_stand"
FORMAL_MODULES = ("tasks", "learning", "simulation", "artifacts", "experiments", "integrations")
ALLOWED_MAIN_FILES = {
    "src/native_stand/__main__.py",
    "src/native_stand/cli.py",
    "src/native_stand/experiments/worker.py",
}
ALLOWED_PROCESS_FILES = {"src/native_stand/experiments/owner.py"}


def _formal_files(root: Path):
    pkg = root / PACKAGE_ROOT
    if not pkg.is_dir():
        raise NotADirectoryError(f"找不到正式源码包: {root / PACKAGE_ROOT}")
    for module in FORMAL_MODULES:
        yield from sorted((pkg / module).rglob("*.py"))
    for name in ("cli.py", "__main__.py", "__init__.py"):
        candidate = pkg / name
        if candidate.is_file():
            yield candidate


def _rel(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _module_of(rel: str) -> str:
    """src/native_stand/tasks/motion/spec.py -> native_stand.tasks.motion.spec"""
    parts = rel.split("/")
    return ".".join(parts[1:]).removesuffix(".py").removesuffix(".__init__")


def _resolve_relative(rel: str, level: int, module: str) -> str:
    """把 from ..x import y 的相对导入解析为项目模块全名；解析不了返回空串。"""
    parts = _module_of(rel).split(".")
    # 包自身用全名，普通模块去掉文件名
    current = parts if Path(rel).name == "__init__.py" else parts[:-1]
    up = level - 1
    if up > len(current) - 1:
        return ""
    base = current[: len(current) - up] if up else current
    if module:
        base = base + module.split(".")
    return ".".join(base)


def _iter_imports(tree: ast.Module):
    """产出 (解析后的模块全名或'', 绑定名, node)。相对导入尽量解析。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, (alias.asname or alias.name.split(".")[0]), node
        elif isinstance(node, ast.ImportFrom):
            yield node.module or "", None, node


def check_file(path: Path, root: Path, project_modules: set[str]):
    violations: list[tuple[str, int, str, str, ast.AST]] = []
    rel = _rel(path, root)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    norm_rel = rel

    # 绑定表：本地名 -> 解析后的项目模块全名（只对可确认的项目内导入建立）
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                target = _resolve_relative(norm_rel, node.level, node.module or "")
            else:
                target = node.module or ""
            if not target or any(alias.name == "*" for alias in node.names):
                continue
            for alias in node.names:
                bindings[alias.asname or alias.name] = f"{target}.{alias.name}"

    def qualify(bound: str, attr: str) -> str:
        if bound in project_modules:
            return f"{bound}.{attr}"
        return ""  # 非项目目标不受R3约束

    for node in ast.walk(tree):
        # R2 依赖方向
        if isinstance(node, ast.Import | ast.ImportFrom):
            for mod, _bind, _import_node in _iter_imports(ast.Module(body=[node], type_ignores=[])):
                target = mod
                if isinstance(node, ast.ImportFrom) and node.level:
                    target = _resolve_relative(norm_rel, node.level, node.module or "")
                    if not target:
                        continue
                member_mod = target
                banned = {
                    "tasks": ("native_stand.cli", "experiments.owner", "experiments.report"),
                    "learning": ("native_stand.cli",),
                    "simulation": ("native_stand.integrations",),
                    "artifacts": ("native_stand.tasks",),
                }.get(norm_rel.split("/")[2] if norm_rel.count("/") >= 3 else "", ())
                if isinstance(node, ast.ImportFrom):
                    member_matched = False
                    for alias in node.names:
                        member = f"{target}.{alias.name}"
                        if member not in project_modules:
                            continue
                        # 每个成员单独判定并立即报告，不能被后续成员覆盖。
                        for banned_prefix in banned:
                            hit = (
                                member == banned_prefix
                                or member.startswith(banned_prefix + ".")
                                or member.endswith("." + banned_prefix)
                                or f".{banned_prefix}." in f".{member}."
                            )
                            if hit:
                                member_matched = True
                                violations.append(
                                    (norm_rel, node.lineno, "R2-imports", f"import:{member}", node)
                                )
                    if not member_matched:
                        # 成员不可解析（如目标模块不在fixture内）时，按包级目标判定。
                        for banned_prefix in banned:
                            hit = (
                                member_mod == banned_prefix
                                or member_mod.startswith(banned_prefix + ".")
                                or member_mod.endswith("." + banned_prefix)
                                or f".{banned_prefix}." in f".{member_mod}."
                            )
                            if hit:
                                violations.append(
                                    (
                                        norm_rel,
                                        node.lineno,
                                        "R2-imports",
                                        f"import:{member_mod}",
                                        node,
                                    )
                                )
                if (
                    member_mod.startswith("native_stand.scripts.")
                    or member_mod == "scripts"
                    or member_mod.startswith("scripts.")
                ):
                    violations.append(
                        (norm_rel, node.lineno, "R2-imports", f"import:{member_mod}", node)
                    )
        # R1 并行入口：真实 __main__ 判断
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
                and any(isinstance(op, ast.Eq) for op in test.ops)
                and any(
                    isinstance(c, ast.Constant) and c.value == "__main__" for c in test.comparators
                )
                and norm_rel not in ALLOWED_MAIN_FILES
            ):
                violations.append((norm_rel, node.lineno, "R1-entry", "entry:__main__", node))
        # R1 训练进程创建（按绑定识别）
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                bound = bindings.get(func.value.id)
                owner_module = func.value.id
                if bound:
                    owner_module = bound
                if (
                    func.attr in ("Popen", "Process", "fork", "spawn", "spawnv")
                    and owner_module
                    in (
                        "subprocess",
                        "multiprocessing",
                        "os",
                    )
                    and norm_rel not in ALLOWED_PROCESS_FILES
                ):
                    violations.append(
                        (
                            norm_rel,
                            node.lineno,
                            "R1-process",
                            f"process:{owner_module}.{func.attr}",
                            node,
                        )
                    )
            elif isinstance(func, ast.Name):
                bound_full = bindings.get(func.id, "")
                api = bound_full.rsplit(".", 1)[-1] if bound_full else ""
                api_module = bound_full.split(".")[0] if bound_full else ""
                if (
                    api in ("Popen", "Process")
                    and api_module in ("subprocess", "multiprocessing")
                    and norm_rel not in ALLOWED_PROCESS_FILES
                ):
                    violations.append(
                        (norm_rel, node.lineno, "R1-process", f"process:{bound_full}", node)
                    )
                elif (
                    func.id in ("fork", "spawn", "spawnv")
                    and "os" in bindings
                    and norm_rel not in ALLOWED_PROCESS_FILES
                ):
                    violations.append(
                        (norm_rel, node.lineno, "R1-process", f"process:os.{func.id}", node)
                    )
        # R3 别名覆写（绑定必须是已确认的项目模块）
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in bindings
                    and not target.attr.startswith("__")
                ):
                    continue
                qualified = qualify(bindings[target.value.id], target.attr)
                # 只约束已确认的项目模块目标
                if qualified:
                    violations.append(
                        (norm_rel, node.lineno, "R3-rebind", f"rebind:{qualified}", node)
                    )
    return violations, tree


def _qualified_symbol(tree: ast.Module, lineno: int) -> str:
    """所在限定符号：最近的顶层函数/类名（无则 '<module>'）。"""
    symbol = "<module>"
    for node in tree.body:
        if isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ) and node.lineno <= lineno <= getattr(node, "end_lineno", node.lineno):
            symbol = node.name
    return symbol


def load_baseline(root: Path) -> dict:
    baseline_path = root / "scripts" / "architecture_baseline.json"
    if not baseline_path.is_file():
        return {}
    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    return {
        (entry["path"], entry["rule"], entry["symbol"], entry["expression"]): entry
        for entry in data.get("exceptions", [])
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="待检查项目根目录（默认脚本上一级）")
    args = parser.parse_args(argv)
    root = Path(args.root or Path(__file__).resolve().parents[1]).resolve()
    if not root.is_dir():
        print(f"错误: 检查根目录不存在: {root}", file=sys.stderr)
        return 2
    try:
        files = list(_formal_files(root))
    except NotADirectoryError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    # 项目源文件索引：绝对导入的成员是否确为模块
    project_modules = set()
    for f in files:
        parts = _module_of(_rel(f, root)).removesuffix(".__init__").split(".")
        for depth in range(2, len(parts) + 1):
            project_modules.add(".".join(parts[:depth]))

    baseline = load_baseline(root)
    found: dict[tuple[str, str, str, str], dict] = {}
    for path in files:
        violations, tree = check_file(path, root, project_modules)
        for rel, lineno, rule, detail, _node in violations:
            symbol = _qualified_symbol(tree, lineno)
            # 基线身份：短描述 + 违规节点完整规范化AST（不含行号属性）
            identity = f"{detail}\n{ast.dump(_node, include_attributes=False)}"
            key = (rel, rule, symbol, identity)
            entry = found.setdefault(
                key,
                {"path": rel, "rule": rule, "symbol": symbol, "expression": identity, "lines": []},
            )
            entry["lines"].append(lineno)

    remaining = []
    for key, entry in found.items():
        allowed = baseline.get(key)
        if allowed is None:
            remaining.append((entry, "未登记"))
        elif len(entry["lines"]) > allowed["allowed"]:
            remaining.append((entry, f"超出基线允许 {allowed['allowed']} 次"))
    # 基线中已消失的问题：提示清理（不算失败，保持基线最小）
    stale = [key for key in baseline if key not in found]

    if remaining:
        print(f"发现 {len(remaining)} 处架构违规：")
        for entry, reason in sorted(remaining, key=lambda item: (item[0]["path"], item[0]["rule"])):
            lines = ",".join(str(line) for line in entry["lines"][:5])
            print(
                f"  {entry['path']}:{lines} [{entry['rule']}] {reason} "
                f"{entry['expression'].splitlines()[0][:120]}"
            )
            # 单行JSON编码，保证多行AST身份可被基线工具与测试精确取回
            print("    identity: " + json.dumps(entry["expression"]))
        return 1
    message = f"架构检查通过：{len(files)} 个正式源码文件，{len(baseline)} 条基线例外。"
    if stale:
        message += f"（{len(stale)} 条基线例外已失效，建议清理）"
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
