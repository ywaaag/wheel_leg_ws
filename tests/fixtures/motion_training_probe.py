"""Host-only source3302 → real adapter → save → publication, without engine I/O."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import torch
from tensordict import TensorDict

from native_stand.artifacts.io import file_sha256
from native_stand.experiments.config import load_job_config
from native_stand.simulation.isaac.motion_backend import MotionProductionAdapter
from native_stand.students.resume import rng_state
from native_stand.tasks.motion.progressive_episodes import episode_class

sys.path.insert(0, str(Path(__file__).parents[1]))
from test_phase3_fixes import _production_context

root = Path("/home/kellen/wheel_leg_native_train")
job = load_job_config(root / "configs/jobs/motion_C0_yaw1to4_full_KL10000.json")
work = Path(sys.argv[1])
work.mkdir(parents=True, exist_ok=True)
context = _production_context(work, job, smoke=100)
from native_stand.artifacts.model_refs import initialize_fixed

initialize_fixed(context.refs, Path(__file__).parents[2] / "configs/model_refs.json")


class CpuAdapter(MotionProductionAdapter):
    def _open_environment(self):
        stage = self.spec["curriculum"]["stages"][0]
        spec = importlib.util.spec_from_file_location(
            "frozen_episode_statistics", Path(job.assets["frozen_bundle"]) / "onpolicy.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from native_stand.tasks.motion.movement_evidence import episode_class as movement

        cls = movement(
            episode_class(module.Episodes),
            progression_gate=self.course_strategy.implementation.PROGRESSION,
        )
        self.env = NS(
            num_envs=job.runtime.envs,
            device="cpu",
            stage_spec=stage,
            identified_dr=NS(episodes=torch.zeros(job.runtime.envs, dtype=torch.long)),
            transient_events=NS(summary=lambda: {}),
        )
        self.env.onpolicy_episodes = cls(self.env)
        self.obs = TensorDict(
            {
                "student_actor": torch.zeros(2, 31),
                "student_critic": torch.zeros(2, 863),
                "recovery_actor_valid": torch.ones(2, 1, dtype=torch.bool),
                "motion_diagnostic_cohort": torch.zeros(2, 1),
                "supported_source_retention_cohort": torch.full((2, 1), -1, dtype=torch.long),
            },
            batch_size=[2],
        )
        self.modules = {"rng_state": rng_state}

    def _assemble_training(self):
        # CPU inspection cannot restore a GPU generator mapping. Keep source state intact.
        from native_stand.learning.ppo.supported import assemble_from_full
        from native_stand.students.resume import restore_rng

        self.algorithm = assemble_from_full(self.source, self.obs, self.job, fresh=False)
        restore_rng(dict(self.source["training_rng"], torch_cuda=None))
        self._verify_restore()
        self.fixed_std = self.algorithm.actor.log_std.detach().clone()

    def _write_restoration(self):
        pass

    def physical_readback(self):
        return {"physics_steps": 0}


adapter = CpuAdapter()
try:
    adapter.build(context, "smoke", None)
    assert adapter.source_sha == "739e38287a8a12508422291de0cc82c863ce3c4d3047f8ab2e5cc9867f947107"
    assert adapter.source.get("training_candidates")
    assert adapter.candidates["best_candidate"] is None
    # Normal rollout has produced no ended episodes in this fixture: UNKNOWN is valid.
    context.updates_used = 100
    adapter._update_scope = {
        "stats": NS(export=lambda: {}),
        "diagnostics": {},
        "detail": {},
        "channel_clips": torch.zeros(6),
        "valid_samples": torch.tensor(1.0),
    }
    adapter.after_update(context, 100, {})
    assert adapter._candidate_pending["eligible"] is False
    captured = adapter.capture(context, {"updates": 100})
    reference = context.save(captured, update=100, stage="smoke")
    assert reference and Path(reference["checkpoint"]).is_file()
    adapter.after_save(context, reference)
    authority = context.refs.candidate_index(context.run_dir.name)
    assert authority["last_review"]["checkpoint"] == reference
    assert authority["best_candidate"] is None
    assert (
        authority["comparison_sha256"] != adapter.source["training_candidates"]["comparison_sha256"]
    )
    assert (
        str(root / "artifacts/2026_11_09__motion_yaw1to4_u1600_preserved/model_update_1600.pt")
        in context.refs.consumer_paths()
    )
    from native_stand.artifacts.checkpoint import LoadedSource
    from native_stand.artifacts.io import read_json

    generation = read_json(context.refs.path)["generation"]
    payload = dict(captured, update=100, stage="smoke", run_identity=str(context.run_dir.resolve()))
    resumed = CpuAdapter()
    resumed.build(
        context, "smoke", LoadedSource(payload, Path(reference["checkpoint"]), reference["sha256"])
    )
    assert resumed.state["consumed_window"] == adapter.state["consumed_window"]
    assert resumed._candidate_pending is None
    assert read_json(context.refs.path)["generation"] == generation
    assert resumed.candidates == authority
    receipt = {
        "resume_idempotent": True,
        "u1600_protected": True,
        "source_sha256": adapter.source_sha,
        "first_publication": True,
        "extra_simulation_steps": 0,
        "PPO_updates": 0,
        "source_cohorts": 95,
        "target_cohorts": len(adapter.state["assessment_keys"]),
        "checkpoint_sha256": file_sha256(reference["checkpoint"]),
    }
    print(json.dumps(receipt))
finally:
    context.metrics.close()
