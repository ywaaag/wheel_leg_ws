"""Pinned real u3000 CPU load/capture; blocks engine/GPU construction."""

# ruff: noqa: E402 -- install the no-engine guard before project imports

import importlib.abc
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS


class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"isaaclab", "isaacsim", "omni", "mujoco"}:
            raise AssertionError("CPU fixture attempted an engine import: " + fullname)


sys.meta_path.insert(0, NoEngine())

import torch
from tensordict import TensorDict

from native_stand.artifacts.checkpoint import load_complete_state
from native_stand.experiments.config import load_job_config
from native_stand.learning.ppo.source_anchor import KEY
from native_stand.learning.ppo.supported import assemble_from_full
from native_stand.simulation.isaac.motion_backend import MotionProductionAdapter
from native_stand.students.resume import restore_rng, rng_state
from native_stand.tasks.motion.maneuver_statistics import RESPONSE_VERSION, episode_class

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_motion_spin_branches import equal_tree
from test_phase3_fixes import _production_context

torch.set_num_threads(1)
torch.cuda.init = lambda: (_ for _ in ()).throw(AssertionError("GPU forbidden"))
root = Path(__file__).resolve().parents[2]
receipts = []
for branch in ("motion", "spin"):
    job = load_job_config(root / f"configs/jobs/{branch}48_u3000_prepared.json")
    job = replace(
        job,
        runtime=replace(job.runtime, device="cpu"),
        storage=replace(job.storage, tensorboard=False),
    )
    original = load_complete_state(job.initialization.checkpoint)
    assert original.sha256 == "abe10db68b0ad62a3e0ceac5285ebb0041ff084134bc313edb235ae46d427d7e"
    with tempfile.TemporaryDirectory(prefix=f"motion_spin_{branch}_") as temporary:
        context = _production_context(Path(temporary), job, smoke=1)

        class Adapter(MotionProductionAdapter):
            def _open_environment(self):
                self.env = NS(
                    num_envs=self.job.runtime.envs,
                    device="cpu",
                    stage_spec=self.spec["curriculum"]["stages"][0],
                    transient_events=NS(summary=lambda: {}),
                    identified_dr=NS(
                        episodes=torch.tensor(
                            self.source["motion_course_state"]["DR_episode_counters"]
                        )
                    ),
                )
                self.env.onpolicy_episodes = episode_class(object, version=RESPONSE_VERSION)(
                    self.env
                )
                self.obs = TensorDict(
                    dict(
                        student_actor=torch.zeros(2, 31),
                        student_critic=torch.zeros(2, 877),
                        recovery_actor_valid=torch.ones(2, 1, dtype=torch.bool),
                        motion_diagnostic_cohort=torch.zeros(2, 1),
                        **{KEY: torch.full((2, 1), -1, dtype=torch.long)},
                    ),
                    [2],
                )
                self.modules = {"rng_state": rng_state}

            def _assemble_training(self):
                self.algorithm = assemble_from_full(self.source, self.obs, self.job)
                restore_rng({**self.source["training_rng"], "torch_cuda": None})
                self._verify_restore()
                self.fixed_std = self.algorithm.actor.log_std.detach().clone()

        adapter = Adapter()
        try:
            adapter.build(context, "smoke", None)
            data = adapter.capture(context, {})
            for key in ("actor", "critic", "optimizer", "std", "training_rng"):
                equal_tree(original.payload[key], data[key])
            assert adapter.algorithm.actor_update_enabled
            assert adapter.state["stage"] == 0
            assert not any(adapter.state["streaks"].values())
            assert (
                data["source_anchor_runtime_state"]
                == original.payload["source_anchor_runtime_state"]
            )
            assert data["counters"]["online_updates"] == 5111
            assert data["counters"]["new_run_updates"] == 0
            assert data["model_online_updates"] == 5097
            assert data["warmup_updates_done"] == 50
            reference = context.save(data, update=0, stage="smoke")
            adapter.after_save(context, reference)
            saved = load_complete_state(reference["checkpoint"])
            again = Adapter()
            again.build(context, "smoke", saved)
            for key, state in (
                ("actor", again.algorithm.actor.state_dict()),
                ("critic", again.algorithm.critic.state_dict()),
                ("optimizer", again.algorithm.optimizer.state_dict()),
            ):
                equal_tree(data[key], state)
            assert again.state == adapter.state
            receipts.append(
                dict(
                    branch=branch,
                    source_sha256=original.sha256,
                    actor_equal=True,
                    critic_equal=True,
                    two_Adam_equal=True,
                    training_RNG_preserved=True,
                    CUDA_RNG_activated=False,
                    warmup_done=data["warmup_updates_done"],
                    model_progress=5097,
                    consumed_authorization=5111,
                    branch_updates=0,
                    shared_balance_unallocated=4889,
                    cohorts=len(adapter.state["assessment_keys"]),
                    checkpoint_saved_and_restored=True,
                    physical_steps=0,
                    PPO_updates=0,
                    source_bank_rotation=data["source_anchor_runtime_state"],
                )
            )
        finally:
            context.metrics.close()
print(json.dumps(receipts))
