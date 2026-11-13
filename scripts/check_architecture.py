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
import os
import shlex
import subprocess
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
                        if member == "native_stand.scripts":
                            violations.append(
                                (norm_rel, node.lineno, "R2-imports", f"import:{member}", node)
                            )
                            continue
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
                    member_mod == "native_stand.scripts"
                    or member_mod.startswith("native_stand.scripts.")
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


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    if result.returncode:
        raise ValueError("Git基点/文件列表错误: " + result.stderr.decode(errors="replace").strip())
    return result.stdout


def changed_paths(root: Path, base: str, *, scripts_only=False) -> list[Path]:
    """REF→working tree, including earlier commits, renames and untracked additions.

    NUL-delimited Git output is never split on whitespace. Deleted files are
    ignored. No target program, external diff or textconv is executed.
    """
    top = Path(os.fsdecode(_git(root, "rev-parse", "--show-toplevel")).strip()).resolve()
    if top != root:
        raise ValueError("--base要求--root是Git仓库根目录")
    _git(root, "rev-parse", "--verify", "--end-of-options", base + "^{commit}")
    scope = ["scripts"] if scripts_only else []
    fields = _git(
        root,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--name-status",
        "-z",
        "--find-renames",
        base,
        "--",
        *scope,
    ).split(b"\0")
    paths = set()
    cursor = 0
    while cursor < len(fields) and fields[cursor]:
        status = fields[cursor].decode("ascii")
        cursor += 1
        name = fields[cursor]
        cursor += 1
        if status.startswith(("R", "C")):
            name = fields[cursor]  # destination, not deleted source
            cursor += 1
        if not status.startswith("D"):
            paths.add(os.fsdecode(name))
    paths.update(
        os.fsdecode(name)
        for name in _git(
            root, "ls-files", "--others", "--exclude-standard", "-z", "--", *scope
        ).split(b"\0")
        if name
    )
    return [root / name for name in sorted(paths) if (root / name).is_file()]


def _bound_name(node, bindings):
    if isinstance(node, ast.Name):
        return bindings.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return _bound_name(node.value, bindings) + "." + node.attr
    return ""


def _static_value(node, values):
    """Small literal/assignment evaluator; never eval/execute target code."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return values.get(node.id)
    if isinstance(node, ast.Attribute):
        return values.get(ast.unparse(node))
    if isinstance(node, ast.List | ast.Tuple):
        return [_static_value(value, values) for value in node.elts]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _static_value(node.left, values), _static_value(node.right, values)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        if isinstance(left, list) and isinstance(right, list):
            return left + right
    return None


def _training_command(value):
    """(violation, unresolved). Known compilers/tests/git/packaging remain legal."""
    if isinstance(value, str):
        try:
            lexer = shlex.shlex(value, posix=True, punctuation_chars=";&|")
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError:
            return False, True
    elif isinstance(value, list):
        tokens = value
    else:
        return False, True
    text = [token for token in tokens if isinstance(token, str)]
    # Shell sequences are examined separately, not hidden behind 'echo; python'.
    if any(token in (";", "&&", "||", "|") for token in text):
        split = next(i for i, token in enumerate(tokens) if token in (";", "&&", "||", "|"))
        left = _training_command(tokens[:split])
        right = _training_command(tokens[split + 1 :])
        return left[0] or right[0], left[1] or right[1]
    if not tokens or not isinstance(tokens[0], str):
        return False, True
    executable = Path(tokens[0]).name
    if executable in ("sh", "bash", "zsh") and "-c" in tokens:
        index = tokens.index("-c") + 1
        return _training_command(tokens[index] if index < len(tokens) else None)
    if executable in ("env", "uv", "conda"):
        return False, True  # Wrapper option grammar is deliberately not guessed.
    python = executable.startswith("python")
    cli = executable == "wheel-leg"
    arguments = tokens[1:]
    if python and "-m" in tokens:
        index = tokens.index("-m") + 1
        module = tokens[index] if index < len(tokens) else None
        if not isinstance(module, str):
            return False, True
        if module in ("native_stand.experiments.worker", "native_stand.command_v6.supervisor"):
            return True, False
        cli = module in ("native_stand", "native_stand.cli")
        arguments = tokens[index + 1 :]
        if module.rsplit(".", 1)[-1].startswith("train"):
            return True, False
        if not cli:
            return False, False  # pytest/build/analysis arguments are not launch targets.
    if cli:
        command = next(
            (
                token
                for token in arguments
                if not isinstance(token, str) or not token.startswith("-")
            ),
            None,
        )
        if command is None:
            return False, False
        if not isinstance(command, str):
            return False, True
        return command in ("train", "distill", "resume") and "--dry-run" not in arguments, False
    if python:
        if "-c" in text:
            return False, True
        target = next(
            (
                token
                for token in arguments
                if not isinstance(token, str) or not token.startswith("-")
            ),
            None,
        )
        if not isinstance(target, str):
            return False, True
        return Path(target).stem.startswith("train") or Path(target).name == "worker.py", False
    return False, False


def check_script(path: Path, root: Path):
    """Only definite training bypasses fail; dynamic process targets need review."""
    rel = _rel(path, root)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bindings = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom) and not node.level:
            for alias in node.names:
                bindings[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    values = {
        name + ".executable": "python" for name, module in bindings.items() if module == "sys"
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = _static_value(node.value, values)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = value
    findings, reviews = [], []
    process = {
        "subprocess.Popen",
        "subprocess.run",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "os.system",
        "os.popen",
    }
    training_modules = (
        "native_stand.experiments.worker",
        "native_stand.experiments.owner",
        "native_stand.command_v6.supervisor",
    )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _bound_name(node.func, bindings)
        if any(name == module or name.startswith(module + ".") for module in training_modules):
            findings.append((rel, node.lineno, "R4-script-training", name))
        elif name in process or name.startswith(("os.exec", "os.spawn")):
            command = (
                node.args[0]
                if node.args
                else next((item.value for item in node.keywords if item.arg == "args"), None)
            )
            if name.startswith(("os.exec", "os.spawn")):
                reviews.append((rel, node.lineno, "动态OS进程入口需审阅"))
                if any(_training_command(_static_value(arg, values))[0] for arg in node.args):
                    findings.append((rel, node.lineno, "R4-script-training", name))
                continue
            violation, dynamic = _training_command(_static_value(command, values))
            if violation:
                findings.append((rel, node.lineno, "R4-script-training", name))
            elif dynamic:
                reviews.append((rel, node.lineno, "动态进程命令需审阅；静态检查无法确认实际目标"))
        elif name == "multiprocessing.Process":
            target = next((item.value for item in node.keywords if item.arg == "target"), None)
            target_name = _bound_name(target, bindings)
            if any(target_name.startswith(module + ".") for module in training_modules):
                findings.append((rel, node.lineno, "R4-script-training", target_name))
            else:
                reviews.append((rel, node.lineno, "并行进程target需审阅"))
        elif name in ("native_stand.cli.main", "native_stand.__main__.main"):
            violation, dynamic = (
                _training_command(["wheel-leg", *(_static_value(node.args[0], values) or [])])
                if node.args and isinstance(_static_value(node.args[0], values), list)
                else (False, True)
            )
            if violation:
                findings.append((rel, node.lineno, "R4-script-training", name))
            elif dynamic:
                reviews.append((rel, node.lineno, "CLI动态参数需审阅"))
    return findings, reviews


def check_shell(path: Path, root: Path):
    findings, reviews = [], []
    for line, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip() or text.lstrip().startswith("#"):
            continue
        violation, dynamic = _training_command(text)
        if violation:
            findings.append((_rel(path, root), line, "R4-script-training", "shell训练旁路"))
        elif dynamic or any(token in text for token in ("$", "eval ", "exec ", "\\")):
            reviews.append((_rel(path, root), line, "动态shell命令需审阅；不执行目标脚本"))
    return findings, reviews


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="待检查项目根目录（默认脚本上一级）")
    parser.add_argument(
        "--base", help="Git基点：检查此提交之后新增/修改/重命名的scripts Python/shell工具"
    )
    args = parser.parse_args(argv)
    root = Path(args.root or Path(__file__).resolve().parents[1]).resolve()
    if not root.is_dir():
        print(f"错误: 检查根目录不存在: {root}", file=sys.stderr)
        return 2
    try:
        files = list(_formal_files(root))
        changed = changed_paths(root, args.base) if args.base is not None else []
        scripts = [
            path
            for path in changed
            if path.is_relative_to(root / "scripts") and path.suffix in (".py", ".sh", ".bash")
        ]
    except (NotADirectoryError, ValueError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    # 项目源文件索引：绝对导入的成员是否确为模块
    project_modules = set()
    for f in files:
        parts = _module_of(_rel(f, root)).removesuffix(".__init__").split(".")
        for depth in range(2, len(parts) + 1):
            project_modules.add(".".join(parts[:depth]))

    script_findings, reviews = [], []
    # Other executable/script languages are reported rather than silently
    # certified by a Python AST checker. Data files are not treated as scripts.
    try:
        for path in changed:
            if not path.is_relative_to(root / "scripts") or path in scripts:
                continue
            with path.open("rb") as handle:
                shebang = handle.read(2) == b"#!"
            if path.suffix in (".js", ".mjs", ".pl", ".rb", ".ps1") or shebang:
                reviews.append((_rel(path, root), 1, "未支持的脚本语言需人工审阅"))
        for path in scripts:
            findings, notes = (check_script if path.suffix == ".py" else check_shell)(path, root)
            script_findings.extend(findings)
            reviews.extend(notes)
    except (SyntaxError, OSError, ValueError) as exc:
        print(f"错误: 脚本静态解析失败: {exc}", file=sys.stderr)
        return 2
    for rel, line, note in reviews:
        print(f"  REVIEW {rel}:{line} {note}")
    public_paths = {
        "src/native_stand/cli.py",
        "src/native_stand/tasks/registry.py",
        "src/native_stand/learning/policies/factory.py",
        "src/native_stand/simulation/dispatch.py",
    }
    for path in changed:
        if path.relative_to(root).as_posix() in public_paths:
            print(
                f"  PUBLIC-REVIEW {path.relative_to(root)}: 核对公开组合/帮助/配置示例/当前文档；文档改动不是内容正确的证明"
            )
    for rel, line, rule, detail in script_findings:
        print(f"  {rel}:{line} [{rule}] {detail}")
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

    if remaining or script_findings:
        print(f"发现 {len(remaining) + len(script_findings)} 处架构违规：")
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
    if args.base is not None:
        message += f" 增量脚本 {len(scripts)} 个（基点 {args.base}），{len(reviews)} 处需人工审阅。"
    if stale:
        message += f"（{len(stale)} 条基线例外已失效，建议清理）"
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
