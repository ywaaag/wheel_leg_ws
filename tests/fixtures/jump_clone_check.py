"""Clean-source + RuntimePaths acceptance, with old checkout reads forbidden."""

import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import mujoco
import torch
from fixtures.jump_cpu_io import mapped_job

from native_stand.artifacts.checkpoint import load_actor_source
from native_stand.artifacts.io import read_json, write_json
from native_stand.cli import main as cli
from native_stand.learning.policies.factory import build_policy
from native_stand.simulation.jump_backend import make_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deny", type=Path, action="append", default=[])
    args = parser.parse_args()
    denied = [str(p.resolve()) for p in args.deny]

    def audit(event, values):
        if event == "open" and isinstance(values[0], str | bytes):
            path = str(Path(values[0]).resolve())
            if any(path == root or path.startswith(root + "/") for root in denied):
                raise AssertionError("Forbidden historical/worktree read: " + path)

    def forbidden(*a, **k):
        raise AssertionError("Clean clone CPU check forbids integration/CUDA")

    sys.addaudithook(audit)
    for name in ("mj_step", "mj_step1", "mj_step2"):
        setattr(mujoco, name, forbidden)
    torch.cuda.init = torch.cuda._lazy_init = forbidden
    job = mapped_job(args.checkout, args.data_root)
    with tempfile.TemporaryDirectory(prefix="jump-clone-check-") as temporary:
        folder = Path(temporary)
        config = read_json(args.checkout / "configs/jobs/jump_unified_raw37_wheel48.json")
        record = job.as_run_record()
        for key in ("assets", "initialization", "source_root"):
            config[key] = record[key]
        write_json(folder / "job.json", config)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = cli(["train", "--config", str(folder / "job.json"), "--dry-run", "--json"])
        assert code == 0
        plan = json.loads(stdout.getvalue())
        batch = make_env(job, output=folder / "execution")
        observations = batch.observations()
        source = load_actor_source(job.initialization.checkpoint)
        assert source.sha256 == job.initialization.sha256
        handle = build_policy(job, observations, source=source)
        assert not handle["ppo"].optimizer.actor.state
        assert not handle["ppo"].optimizer.critic.state
        for key, value in source.payload["actor"].items():
            assert torch.equal(value, handle["ppo"].actor.state_dict()[key])
        saved = batch.runtime_state()
        batch.load_runtime_state(saved)
        for key in list(observations.keys()):
            assert torch.equal(observations[key], batch.observations()[key])
        receipt = dict(
            status="PASS",
            checkout=str(args.checkout),
            data_root=str(args.data_root),
            forbidden_read_roots=denied,
            actual_physics_steps=0,
            GPU_calls=0,
            actual_training_updates=0,
            actual_Actor_sha256=source.sha256,
            Actor_shape=list(observations["student_actor"].shape),
            Critic_shape=list(observations["student_critic"].shape),
            Critic_fields=len(batch.envs[0].schema.fields),
            dual_Adam_empty=True,
            static_restore_equal=True,
            CLI_plan=plan,
        )
        batch.close()
    write_json(args.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
