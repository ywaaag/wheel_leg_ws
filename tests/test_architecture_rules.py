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
