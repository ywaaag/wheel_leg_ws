"""架构约束检查器的行为验证；只使用标准库fixture，不导入训练或仿真依赖。

FROZEN_TEST_SCOPE = "PORTABLE"
"""

import json as _json
import subprocess
import sys
from pathlib import Path

CHECKER = Path(__file__).resolve().parents[1] / "scripts" / "check_architecture.py"

FROZEN_TEST_SCOPE = "PORTABLE"


def _fixture(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    return tmp_path


def run_checker(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CHECKER), "--root", str(root)],
        capture_output=True,
        text=True,
        timeout=60,
    )


WORKER_STUB = ("src/native_stand/experiments/worker.py", "x = 1\n")
WORKFLOW_MODULE = ("src/native_stand/simulation/workflow.py", "class DelayedPlant:\n    pass\n")


def test_legal_factory_and_owner_process_pass(tmp_path):
    root = _fixture(
        tmp_path,
        {
            "src/native_stand/experiments/owner.py": "import subprocess\nsubprocess.Popen(['ls'])\n",
            "src/native_stand/learning/loop.py": "def build(factory):\n    return factory()\n",
            "src/native_stand/tasks/motion.py": (
                "from ..experiments.config import JobConfig\n"
                "def use(job: JobConfig):\n    return job\n"
            ),
        },
    )
    assert run_checker(root).returncode == 0


def test_new_training_process_creation_fails(tmp_path):
    root = _fixture(
        tmp_path,
        {
            WORKER_STUB[0]: WORKER_STUB[1],
            "src/native_stand/learning/trainer.py": (
                "import subprocess\n"
                "def spawn():\n    return subprocess.Popen(['python', 'train.py'])\n"
            ),
        },
    )
    done = run_checker(root)
    assert done.returncode == 1 and "R1-process" in done.stdout


def test_process_api_aliases_are_recognized(tmp_path):
    root = _fixture(
        tmp_path,
        {
            WORKER_STUB[0]: WORKER_STUB[1],
            "src/native_stand/learning/aliased.py": (
                "from subprocess import Popen as launch\n"
                "def go():\n    return launch(['python', 'train.py'])\n"
            ),
            "src/native_stand/simulation/spawner.py": (
                "from multiprocessing import Process as WorkerProcess\n"
                "def go():\n    return WorkerProcess(target=print)\n"
            ),
        },
    )
    done = run_checker(root)
    assert done.returncode == 1
    assert done.stdout.count("R1-process") == 2


def test_object_method_spawn_is_not_flagged(tmp_path):
    root = _fixture(
        tmp_path,
        {
            WORKER_STUB[0]: WORKER_STUB[1],
            "src/native_stand/learning/buffer.py": (
                "class Buffer:\n"
                "    def spawn(self):\n        return self\n"
                "def use(buffer):\n    return buffer.spawn()\n"
            ),
        },
    )
    assert run_checker(root).returncode == 0


def test_new_experiment_entry_point_fails_but_analysis_main_passes(tmp_path):
    root = _fixture(
        tmp_path,
        {
            WORKER_STUB[0]: WORKER_STUB[1],
            "src/native_stand/learning/experiment_entry.py": (
                'if __name__ == "__main__":\n    train()\n'
            ),
        },
    )
    assert run_checker(root).returncode == 1

    # 同名不同路径不能继承worker入口权限
    root_same_name = _fixture(
        tmp_path / "same_name",
        {
            WORKER_STUB[0]: WORKER_STUB[1],
            "src/native_stand/learning/worker.py": 'if __name__ == "__main__":\n    train()\n',
        },
    )
    done = run_checker(root_same_name)
    assert done.returncode == 1 and "R1-entry" in done.stdout

    root2 = _fixture(
        tmp_path / "second",
        {
            WORKER_STUB[0]: WORKER_STUB[1],
            "scripts/analyze.py": 'if __name__ == "__main__":\n    print("analysis")\n',
        },
    )
    assert run_checker(root2).returncode == 0


def test_relative_imports_resolved_by_same_rules(tmp_path):
    # tasks内 from ..cli import main：解析为native_stand.cli → 违规
    root = _fixture(
        tmp_path,
        {
            WORKER_STUB[0]: WORKER_STUB[1],
            "src/native_stand/tasks/bad.py": "from ..cli import main\n",
        },
    )
    done = run_checker(root)
    assert done.returncode == 1 and "import:native_stand.cli" in done.stdout

    # tasks内 from ..experiments import owner as boss → 违规
    root2 = _fixture(
        tmp_path / "second",
        {
            WORKER_STUB[0]: WORKER_STUB[1],
            "src/native_stand/experiments/owner.py": "def spawn():\n    pass\n",
            "src/native_stand/tasks/boss.py": "from ..experiments import owner as boss\n",
        },
    )
    done2 = run_checker(root2)
    assert done2.returncode == 1 and "import:native_stand.experiments.owner" in done2.stdout

    # 合法：层内引用与配置数据类型
    root3 = _fixture(
        tmp_path / "third",
        {
            WORKER_STUB[0]: WORKER_STUB[1],
            "src/native_stand/tasks/motion.py": (
                "from ..experiments.config import JobConfig\n"
                "def use(job: JobConfig):\n    return job\n"
            ),
            "src/native_stand/tasks/__init__.py": "",
        },
    )
    assert run_checker(root3).returncode == 0


def test_cross_module_rebind_fails_but_object_attribute_passes(tmp_path):
    # 相对导入拿到的目标模块也受R3约束（fixture提供真实目标模块）
    root = _fixture(
        tmp_path,
        {
            WORKER_STUB[0]: WORKER_STUB[1],
            WORKFLOW_MODULE[0]: WORKFLOW_MODULE[1],
            "src/native_stand/learning/hack.py": (
                "from ..simulation import workflow\nworkflow.DelayedPlant = Actor20Plant\n"
            ),
        },
    )
    done = run_checker(root)
    assert (
        done.returncode == 1
        and "rebind:native_stand.simulation.workflow.DelayedPlant" in done.stdout
    )

    # 绝对导入成员为真实模块时同样识别
    root2 = _fixture(
        tmp_path / "second",
        {
            WORKER_STUB[0]: WORKER_STUB[1],
            WORKFLOW_MODULE[0]: WORKFLOW_MODULE[1],
            "src/native_stand/learning/hack2.py": (
                "from native_stand.simulation import workflow\n"
                "workflow.DelayedPlant = Actor20Plant\n"
            ),
        },
    )
    assert run_checker(root2).returncode == 1

    root3 = _fixture(
        tmp_path / "third",
        {
            WORKER_STUB[0]: WORKER_STUB[1],
            "src/native_stand/learning/fine.py": (
                "class Model:\n    def reset(self):\n        self.state = {}\n"
            ),
        },
    )
    assert run_checker(root3).returncode == 0


def test_bad_root_exit_code_2(tmp_path):
    done = subprocess.run(
        [sys.executable, str(CHECKER), "--root", str(tmp_path / "missing")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 2


def test_baseline_binds_symbol_and_expression(tmp_path):
    files = {
        WORKER_STUB[0]: WORKER_STUB[1],
        "src/native_stand/learning/legacy.py": (
            "import subprocess\ndef spawn():\n    return subprocess.Popen(['ls'])\n"
        ),
    }
    root = _fixture(tmp_path, files)
    # 先拿到检查器给出的精确身份，再登记基线（登记必须基于真实阅读）
    done = run_checker(root)
    assert done.returncode == 1
    out_lines = done.stdout.splitlines()
    line = next(ln for ln in out_lines if "R1-process" in ln)
    path_part, rule = line.split()[0], "R1-process"
    expression = _json.loads(out_lines[out_lines.index(line) + 1].split("identity: ", 1)[1])
    baseline = {
        "exceptions": [
            {
                "path": path_part.split(":")[0],
                "rule": rule,
                "symbol": "spawn",
                "expression": expression,
                "allowed": 1,
                "line": 3,
                "reason": "暂缓：历史分析工具，迁移单独安排",
            }
        ]
    }
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "architecture_baseline.json").write_text(_json.dumps(baseline))
    assert run_checker(root).returncode == 0

    # 同一问题出现次数增加 → 失败
    (root / "src/native_stand/learning/legacy.py").write_text(
        "import subprocess\n"
        "def spawn():\n    return subprocess.Popen(['ls'])\n"
        "def spawn_more():\n    return subprocess.Popen(['ls'])\n"
    )
    assert run_checker(root).returncode == 1

    # 移除旧问题、新增不同表达式（另一符号）→ 旧豁免不继承
    (root / "src/native_stand/learning/legacy.py").write_text(
        "import subprocess\ndef other_symbol():\n    return subprocess.Popen(['ls'])\n"
    )
    done = run_checker(root)
    assert done.returncode == 1


def test_checker_does_not_import_training_dependencies(tmp_path):
    import ast as _ast

    source = CHECKER.read_text()
    for node in _ast.walk(_ast.parse(source)):
        if isinstance(node, _ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in sys.stdlib_module_names, alias.name
        elif isinstance(node, _ast.ImportFrom) and node.level == 0 and node.module:
            assert node.module.split(".")[0] in sys.stdlib_module_names, node.module


def test_review_required_patterns(tmp_path):
    """审阅补充反例：from .. import cli；多成员；scripts.xxx；os.fork。"""
    root = _fixture(
        tmp_path,
        {
            WORKER_STUB[0]: WORKER_STUB[1],
            "src/native_stand/cli.py": "def main():\n    pass\n",
            "src/native_stand/tasks/a.py": "from .. import cli\n",
            "src/native_stand/experiments/owner.py": "def spawn():\n    pass\n",
            "src/native_stand/tasks/b.py": "from native_stand.experiments import owner, config\n",
            "src/native_stand/learning/c.py": "import scripts.analyze\n",
            "src/native_stand/learning/d.py": "import os\nos.fork()\n",
        },
    )
    done = run_checker(root)
    assert done.returncode == 1
    assert "import:native_stand.cli" in done.stdout
    assert "import:native_stand.experiments.owner" in done.stdout
    assert "import:scripts.analyze" in done.stdout
    assert "process:os.fork" in done.stdout


def test_baseline_identity_includes_full_ast_and_symbol(tmp_path):
    """同函数换命令仍失败；嵌套符号可区分。"""
    files = {
        WORKER_STUB[0]: WORKER_STUB[1],
        "src/native_stand/learning/legacy.py": (
            "import subprocess\ndef spawn():\n    return subprocess.Popen(['ls'])\n"
        ),
    }
    root = _fixture(tmp_path, files)
    done = run_checker(root)
    assert done.returncode == 1
    line = next(ln for ln in done.stdout.splitlines() if "R1-process" in ln)
    import json as _json

    path_part = line.split(":")[0].strip()
    # 从完整输出的身份行提取ast.dump（第二行是dump内容）
    out_lines = done.stdout.splitlines()
    idx = out_lines.index(line)
    # 检查器在违规行后输出 "identity: <json>" 行，即登记用的精确身份
    expression = _json.loads(out_lines[idx + 1].split("identity: ", 1)[1])
    baseline = {
        "exceptions": [
            {
                "path": path_part,
                "rule": "R1-process",
                "symbol": "spawn",
                "expression": expression,
                "allowed": 1,
                "line": 3,
                "reason": "暂缓",
            }
        ]
    }
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "architecture_baseline.json").write_text(_json.dumps(baseline))
    assert run_checker(root).returncode == 0

    # 同函数内换成不同命令：数量不变但AST不同 → 必须失败
    (root / "src/native_stand/learning/legacy.py").write_text(
        "import subprocess\ndef spawn():\n    return subprocess.Popen(['rm', '-rf'])\n"
    )
    assert run_checker(root).returncode == 1


# Incremental script review uses real, self-contained Git repositories. It never
# imports inspected code, creates engines, or executes training commands.
def git_fixture(tmp_path, scripts=None):
    root = _fixture(tmp_path, {WORKER_STUB[0]: WORKER_STUB[1], **(scripts or {})})
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", "--all"], cwd=root, check=True)
    commit_fixture(root)
    subprocess.run(["git", "tag", "refactor-migration-v1"], cwd=root, check=True)
    return root


def commit_fixture(root):
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=CPU Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )


def incremental(root, base="refactor-migration-v1"):
    return subprocess.run(
        [sys.executable, "-B", str(CHECKER), "--root", str(root), "--base", base],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_incremental_factory_analysis_compiler_and_test_tools_pass_without_execution(tmp_path):
    root = git_fixture(tmp_path)
    _fixture(
        root,
        {
            "src/native_stand/simulation/factory.py": "def make(factory, config):\n return factory(config)\n",
            "scripts/analyze.py": (
                "import subprocess\n"
                "subprocess.run(['cc', '-shared', 'control.c'])\n"
                "subprocess.run(['python', '-m', 'pytest', 'tests'])\n"
                "subprocess.run(['python', '-m', 'build'])\n"
                "raise RuntimeError('must not execute target script')\n"
            ),
        },
    )
    done = incremental(root)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "增量脚本 1 个" in done.stdout
    assert "Traceback" not in done.stderr


def test_incremental_worker_subprocess_alias_and_assigned_command_fail(tmp_path):
    root = git_fixture(tmp_path)
    _fixture(
        root,
        {
            "scripts/bypass.py": (
                "import sys as runtime\nfrom subprocess import Popen as launch\n"
                "command = [runtime.executable, '-m', 'native_stand.experiments.worker']\n"
                "launch(command)\n"
            )
        },
    )
    done = incremental(root)
    assert done.returncode == 1 and "R4-script-training" in done.stdout


def test_incremental_script_invokes_owner_worker_or_legacy_supervisor_fail(tmp_path):
    root = git_fixture(tmp_path)
    cases = {
        "owner": "from native_stand.experiments import owner as o\no.submit(job)\n",
        "worker": "from native_stand.experiments.worker import main as run\nrun()\n",
        "legacy": "from native_stand.command_v6 import supervisor as s\ns.main()\n",
        "mp": "from multiprocessing import Process as P\nfrom native_stand.experiments.worker import main\nP(target=main)\n",
        "train": "import subprocess\nsubprocess.run(['python', '-m', 'native_stand', 'train', '--config', 'x'])\n",
        "oldtrain": "import subprocess\nsubprocess.run(['python', 'train_new.py'])\n",
        "cli": "from native_stand.cli import main\nmain(['distill', '--config', 'x'])\n",
        "shell": "#!/bin/sh\npython -m native_stand.experiments.worker --run-dir x\n",
    }
    for name, code in cases.items():
        folder = root / "scripts"
        folder.mkdir(exist_ok=True)
        path = folder / (name + (".sh" if name == "shell" else ".py"))
        path.write_text(code)
        done = incremental(root)
        assert done.returncode == 1 and "R4-script-training" in done.stdout, (
            name,
            done.stdout,
            done.stderr,
        )
        path.unlink()


def test_incremental_cli_dry_run_and_import_without_launch_are_legal(tmp_path):
    root = git_fixture(tmp_path)
    _fixture(
        root,
        {
            "scripts/inspect_plan.py": (
                "from native_stand.cli import main\n"
                "from native_stand.experiments import owner\n"
                "main(['train', '--dry-run', '--config', 'x'])\n"
            )
        },
    )
    assert incremental(root).returncode == 0


def test_incremental_formal_scripts_import_still_fails(tmp_path):
    root = git_fixture(tmp_path)
    _fixture(root, {"src/native_stand/learning/bad.py": "import scripts.analyze\n"})
    done = incremental(root)
    assert done.returncode == 1 and "R2-imports" in done.stdout
    # Relative package import cannot hide behind the exact package name.
    (root / "src/native_stand/learning/bad.py").write_text("from ..scripts import analyze\n")
    assert incremental(root).returncode == 1


def test_incremental_base_missing_invalid_empty_and_non_git_fail_closed(tmp_path):
    root = git_fixture(tmp_path)
    for base in ("missing-tag", "HEAD:no-such-file", ""):
        done = incremental(root, base)
        assert done.returncode == 2 and "错误" in done.stderr
    no_git = _fixture(tmp_path / "plain", {WORKER_STUB[0]: WORKER_STUB[1]})
    assert incremental(no_git).returncode == 2


def test_incremental_history_rename_spaces_deletion_and_persisted_new_script(tmp_path):
    root = git_fixture(
        tmp_path,
        {
            "scripts/historical.py": "import subprocess\nsubprocess.run(['python', 'train.py'])\n",
            "scripts/old name.py": "import subprocess\nsubprocess.run(['python', 'train.py'])\n",
            "scripts/deleted.py": "this was never valid Python !\n",
        },
    )
    assert incremental(root).returncode == 0  # untouched history is not retroactively governed
    subprocess.run(
        ["git", "mv", "scripts/old name.py", "scripts/new name.py"], cwd=root, check=True
    )
    subprocess.run(["git", "rm", "-q", "scripts/deleted.py"], cwd=root, check=True)
    commit_fixture(root)
    done = incremental(root)
    assert done.returncode == 1 and "scripts/new name.py" in done.stdout
    assert "scripts/historical.py" not in done.stdout and "deleted.py" not in done.stdout
    # This script is already committed, not changed in the latest working tree.
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=root) == b""
    assert incremental(root).returncode == 1


def test_incremental_modified_and_untracked_scripts_both_checked(tmp_path):
    root = git_fixture(tmp_path, {"scripts/tool.py": "print('old')\n"})
    _fixture(
        root,
        {
            "scripts/tool.py": "import subprocess\nsubprocess.run(['wheel-leg', 'resume', '--run', 'x'])\n",
            "scripts/untracked name.py": "import subprocess\nsubprocess.run(['python', '-m', 'native_stand.experiments.worker'])\n",
        },
    )
    done = incremental(root)
    assert done.returncode == 1 and done.stdout.count("R4-script-training") == 2


def test_incremental_dynamic_process_reports_review_without_fake_certification(tmp_path):
    root = git_fixture(tmp_path)
    _fixture(root, {"scripts/dynamic.py": "import subprocess\nsubprocess.run(command_from_user)\n"})
    done = incremental(root)
    assert done.returncode == 0 and "REVIEW scripts/dynamic.py:2" in done.stdout
    assert "无法确认" in done.stdout


def test_incremental_public_surface_requires_content_review_even_with_doc_touch(tmp_path):
    root = git_fixture(tmp_path)
    _fixture(
        root,
        {
            "src/native_stand/cli.py": "def register(parser):\n parser.add_parser('new')\n",
            "README.md": "one arbitrary changed line\n",
        },
    )
    done = incremental(root)
    assert done.returncode == 0 and "PUBLIC-REVIEW" in done.stdout
    assert "文档改动不是内容正确的证明" in done.stdout


def test_incremental_target_syntax_error_is_not_silently_ignored(tmp_path):
    root = git_fixture(tmp_path)
    _fixture(root, {"scripts/bad.py": "def invalid(\n"})
    done = incremental(root)
    assert done.returncode == 2 and "静态解析失败" in done.stderr


def test_incremental_analysis_printing_training_names_is_not_a_launcher(tmp_path):
    root = git_fixture(tmp_path)
    _fixture(
        root,
        {
            "scripts/analyze_names.py": (
                "import subprocess\n"
                "subprocess.run(['echo', 'native_stand.experiments.worker'])\n"
                "subprocess.run(['python', 'analyze.py', '--source', 'train.py'])\n"
                "subprocess.run(['python', '-m', 'pytest', 'train.py'])\n"
                "subprocess.run(['wheel-leg', 'inspect', '--run', 'train'])\n"
            ),
            "scripts/analyze.sh": "#!/bin/sh\necho 'native_stand.experiments.worker'\ncc -shared core.c\n",
        },
    )
    assert incremental(root).returncode == 0


def test_incremental_shell_sequence_cannot_hide_explicit_worker_launch(tmp_path):
    root = git_fixture(tmp_path)
    _fixture(
        root,
        {"scripts/launch.sh": "#!/bin/sh\necho ready; python -m native_stand.experiments.worker\n"},
    )
    done = incremental(root)
    assert done.returncode == 1 and "scripts/launch.sh:2" in done.stdout


def test_closeout_ci_fetches_baseline_history_and_uses_same_static_command():
    root = CHECKER.parents[1]
    workflow = (root / ".github/workflows/cpu.yml").read_text()
    assert "fetch-depth: 0" in workflow and "fetch-tags: true" in workflow
    assert "python scripts/check_architecture.py --base refactor-migration-v1" in workflow
    template = (root / ".github/PULL_REQUEST_TEMPLATE.md").read_text()
    for topic in ("消费者", "复用", "替代", "文档", "验证", "尚未验证"):
        assert topic in template


def test_current_navigation_documents_exist_in_the_committed_source_boundary():
    import re

    root = CHECKER.parents[1]
    for name in ("AGENTS.md", "README.md", "ARCHITECTURE.md"):
        text = (root / name).read_text()
        # Only explicit document navigation, not runtime manifests or history inventory.
        references = re.findall(r"`([^`\n]+\.md)`", text)
        references += re.findall(r"\]\(([^)]+\.md)(?:#[^)]*)?\)", text)
        for reference in references:
            assert (root / reference).is_file(), (name, reference)
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", reference],
                cwd=root,
                capture_output=True,
            )
            assert tracked.returncode == 0, (name, reference, "document not staged/tracked")


def test_incremental_other_script_languages_are_reviewed_not_silently_certified(tmp_path):
    root = git_fixture(tmp_path)
    _fixture(root, {"scripts/new.js": "throw new Error('not executed');\n"})
    done = incremental(root)
    assert done.returncode == 0 and "REVIEW scripts/new.js:1" in done.stdout
    assert "未支持的脚本语言" in done.stdout
