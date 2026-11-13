"""Recovery collection on a SHA-bound original-XML MuJoCo reference model.

One owner/worker uses the common collector and BC loop. Integration is 0.5ms;
execution/PD/policy remain 1/5/10ms. No rescue, projection, or teacher cache.
"""

import importlib.util
import sys
import types

import numpy as np
import torch

from native_stand.artifacts.io import file_sha256, read_json
from native_stand.tasks.recovery.cubic_randomization import draw

LEGS = [0, 1, 3, 4]
WHEELS = [2, 5]


def verified(ref):
    if file_sha256(ref["path"]) != ref["sha256"]:
        raise ValueError("Recovery MuJoCo source changed: " + ref["path"])
    return ref["path"]


def validate_job(job):
    c = job.assets["recovery_mujoco"]
    if job.task != "recovery" or job.algorithm != "distillation" or job.backend != "mujoco":
        raise ValueError("Recovery MuJoCo factory supports explicit distillation only")
    if c["version"] != "recovery_mujoco_collection_v1":
        raise ValueError("Unknown Recovery MuJoCo collection contract")
    if job.control.physics_dt_s != 0.0005 or job.control.policy_dt_s != 0.01:
        raise ValueError("MuJoCo integration/policy must be .5/10ms")
    if list(job.control.effort_limits_nm) != [40, 40, 4.8, 40, 40, 4.8]:
        raise ValueError("Frozen Recovery motor limits required")
    for key in ("model", "model_readback", "catalog", "V4_contract", "V4_tensor"):
        verified(c[key])
    for ref in c["execution_sources"].values():
        verified(ref)
    for name in ("train", "development", "holdout"):
        if len(c["cases"][name]) != job.runtime.envs:
            raise ValueError("Every declared partition must match actual environment count")
    return c


def available(job):
    try:
        validate_job(job)
        import importlib.util

        if importlib.util.find_spec("mujoco") is None:
            return False, "MuJoCo is not installed"
        return True, ""
    except (ValueError, KeyError, OSError) as exc:
        return False, str(exc)


def execution_factories(sources):
    """Load verified engine-independent modules into a private relative package."""
    import hashlib

    for ref in sources.values():
        verified(ref)
    package = (
        "_recovery_mujoco_"
        + hashlib.sha256("".join(ref["sha256"] for ref in sources.values()).encode()).hexdigest()[
            :16
        ]
    )
    if package not in sys.modules:
        namespace = types.ModuleType(package)
        namespace.__path__ = []
        sys.modules[package] = namespace
        for name in (
            "execution",
            "lifecycle",
            "legacy_lifecycle",
            "preentry_lifecycle",
            "settle_lifecycle",
        ):
            path = verified(sources[name])
            spec = importlib.util.spec_from_file_location(package + "." + name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
    return (
        sys.modules[package + ".execution"].Execution,
        sys.modules[package + ".settle_lifecycle"].Lifecycle,
    )


def catalog_reset(mj, model, refs, pose, yaw):
    data = mj.MjData(model)
    data.qpos[:3] = pose["root_pose_xyzw"][:3]
    quaternion = np.asarray(pose["root_pose_xyzw"][3:7])[[3, 0, 1, 2]]
    mj.mju_mulQuat(
        data.qpos[3:7], np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]), quaternion
    )
    rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )
    data.qpos[:3] = rotation @ data.qpos[:3]
    linear = rotation @ np.asarray(pose["root_velocity_linear_angular"][:3])
    angular = rotation @ np.asarray(pose["root_velocity_linear_angular"][3:])
    mj.mj_forward(model, data)
    data.qvel[:3] = linear - np.cross(angular, data.xipos[refs["root"]] - data.xpos[refs["root"]])
    data.qvel[3:6] = data.xmat[refs["root"]].reshape(3, 3).T @ angular
    for name, position in pose["joint_position"].items():
        j = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if j < 0 or model.jnt_type[j] != mj.mjtJoint.mjJNT_HINGE:
            raise ValueError("Catalog named hinge missing: " + name)
        data.qpos[model.jnt_qposadr[j]] = position
        data.qvel[model.jnt_dofadr[j]] = pose["joint_velocity"][name]
    mj.mj_forward(model, data)
    if max(np.linalg.norm(data.site_xpos[a] - data.site_xpos[b]) for a, b in refs["loops"]) > 0.005:
        raise ValueError("Catalog reset violates closure gate")
    return data


def measured(mj, model, data, refs):
    """Same CAD frame and root-reference velocity as the frozen teacher probe."""
    rot = data.xmat[refs["root"]].reshape(3, 3)
    velocity = np.zeros(6)
    mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, refs["root"], velocity, 0)
    gyro = rot.T @ velocity[:3]
    gravity = rot.T @ np.array([0.0, 0.0, -1.0])
    rootvel = velocity[3:] + np.cross(
        velocity[:3], data.xpos[refs["root"]] - data.xipos[refs["root"]]
    )
    forward = -rot[:, 2].copy()
    forward[2] = 0.0
    forward /= np.linalg.norm(forward)
    left = np.cross([0.0, 0.0, 1.0], forward)
    q, dq = data.qpos[refs["qpos"]].copy(), data.qvel[refs["dofs"]].copy()
    frame = np.r_[gyro * 0.25, gravity, q[LEGS] / 1.5, dq * 0.05]
    centers = data.xpos[refs["wheels"]]
    hips = data.xanchor[np.array(refs["joints"])[[0, 3]]]
    vectors = (centers - hips) @ rot
    lengths = np.linalg.norm(centers - hips, axis=-1)
    swing = np.arctan2(-vectors[:, 2], -vectors[:, 1])
    diff = abs(np.arctan2(np.sin(swing[0] - swing[1]), np.cos(swing[0] - swing[1])))
    closure = max(np.linalg.norm(data.site_xpos[a] - data.site_xpos[b]) for a, b in refs["loops"])
    support = np.zeros(2)
    for index in range(data.ncon):
        contact = data.contact[index]
        if refs["ground"] not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == refs["ground"] else contact.geom1
        body = int(model.geom_bodyid[other])
        if body in refs["wheels"]:
            force = np.zeros(6)
            mj.mj_contactForce(model, data, index, force)
            support[refs["wheels"].index(body)] += max(0.0, force[0] * abs(contact.frame[2]))
    state = np.r_[
        rootvel @ forward,
        rootvel @ left,
        gyro[1],
        data.xpos[refs["root"], 2],
        np.arctan2(gravity[2], -gravity[1]),
        np.arctan2(gravity[0], -gravity[1]),
        lengths[0] - lengths[1],
        (centers[0] - centers[1]) @ forward,
        diff,
        support,
        closure,
        -gravity[1],
    ]
    return frame, q, dq, state


class RecoveryMuJoCoCollection:
    def __init__(self, job, *, execution_factory=None, lifecycle_factory=None):
        import mujoco

        from .recovery_isaac.bindings import load_response

        self.mj, self.job = mujoco, job
        self.config = validate_job(job)
        meta = read_json(self.config["model_readback"]["path"])
        if meta["mujoco_version"] != mujoco.__version__:
            raise ValueError("Frozen binary model MuJoCo version differs")
        self.refs = meta["refs"]
        for key in ("qpos", "dofs", "gears"):
            self.refs[key] = np.array(self.refs[key])
        self.catalog = read_json(self.config["catalog"]["path"])
        self.v4 = read_json(self.config["V4_contract"]["path"])
        self.response = load_response(self.config["V4_tensor"])
        factories = execution_factories(self.config["execution_sources"])
        self.execution_factory = execution_factory or factories[0]
        self.lifecycle_factory = lifecycle_factory or factories[1]
        self.num_envs = job.runtime.envs
        self.segment = 0
        self.reset_collection()

    @staticmethod
    def tensor(x):
        return torch.as_tensor(np.asarray(x).copy(), dtype=torch.float32)

    def reset_collection(self, fresh=False, seed_offset=0):
        self.partition = "holdout" if seed_offset == 100000 else "development" if fresh else "train"
        self.cases = self.config["cases"][self.partition]
        self.execution = self.execution_factory(
            self.v4,
            self.response,
            self.num_envs,
            dtype=torch.float32,
            effort_caps=self.job.control.effort_limits_nm,
        )
        self.life = self.lifecycle_factory(self.num_envs, dtype=torch.float32)
        self.life.height_conditioned = True
        self.life.command_height = torch.full((self.num_envs,), 0.34)
        self.life.minimum_success_time = torch.zeros(self.num_envs)
        self.life.effort_caps = self.tensor(self.job.control.effort_limits_nm)
        self.models = [None] * self.num_envs
        self.data = [None] * self.num_envs
        self.episode_index = np.zeros(self.num_envs, dtype=int)
        self.elapsed = np.zeros(self.num_envs)
        self.pending = np.zeros(self.num_envs, dtype=bool)
        self.raw_hold = np.zeros(self.num_envs)
        self.raw_longest = np.zeros(self.num_envs)
        self.hf_power = np.zeros((self.num_envs, 4))
        self.hf_count = np.zeros(self.num_envs)
        self.phase = torch.zeros(self.num_envs, dtype=torch.long)
        self._reset_rows(range(self.num_envs))
        self.segment += 1
        self._observation = None

    def _reset_rows(self, ids):
        qrows = []
        vrows = []
        heights = []
        for i in ids:
            case = self.cases[i]
            m = self.mj.MjModel.from_binary_path(self.config["model"]["path"])
            if abs(m.opt.timestep - 0.0005) > 1e-12:
                raise ValueError("Frozen integration step differs")
            # Distinct reset-only DR draws, not fake nominal seed diversity.
            seed = (
                case["seed"]
                + self.episode_index[i] * 1009
                + (self.segment * 100003 if self.partition == "train" else 0)
            )
            dr = draw(case["DR_strength"], np.random.default_rng(seed))
            root = self.refs["root"]
            m.body_mass[root] *= dr["root_mass_scale"]
            m.body_inertia[root] *= dr["root_inertia_scale"]
            m.body_ipos[root] += dr["root_COM_m"]
            m.geom_friction[:] = [dr["friction"], 0.0, 0.0]
            if (
                case.get("contact_version", "original_XML_contact_reference")
                != "original_XML_contact_reference"
            ):
                raise ValueError(
                    "This collection contract only supports frozen original XML contact"
                )
            self.mj.mj_setConst(m, self.mj.MjData(m))
            d = catalog_reset(
                self.mj, m, self.refs, self.catalog["poses"][case["pose"]], case["yaw_rad"]
            )
            self.models[i] = m
            self.data[i] = d
            frame, q, v, s = measured(self.mj, m, d, self.refs)
            qrows.append(q)
            vrows.append(v)
            heights.append(s[3])
            self.execution.bias[i] = self.tensor(dr["encoder_bias_rad"])
            self.execution.packet_age[i] = dr["Actor_age_ms"]
            self.execution.target_delay[i] = dr["target_delay_ms"]
            self.execution.wheel_delay[i] = torch.as_tensor(dr["wheel_delay_ms"])
            self.execution.lower[i, LEGS] = self.tensor(
                m.jnt_range[np.array(self.refs["joints"])[LEGS], 0]
            )
            self.execution.upper[i, LEGS] = self.tensor(
                m.jnt_range[np.array(self.refs["joints"])[LEGS], 1]
            )
            self.elapsed[i] = 0.0
            self.pending[i] = False
            self.raw_hold[i] = 0.0
            self.raw_longest[i] = 0.0
            self.hf_power[i] = 0.0
            self.hf_count[i] = 0
        indices = torch.tensor(list(ids), dtype=torch.long)
        self.execution.reset(indices, self.tensor(qrows), self.tensor(vrows))
        self.life.reset(indices, self.tensor(heights), torch.zeros(len(indices)))

    def observations(self):
        if self._observation is None:
            if self.pending.any():
                ids = np.flatnonzero(self.pending)
                self.episode_index[ids] += 1
                self._reset_rows(ids)
            rows = [
                measured(self.mj, m, d, self.refs)
                for m, d in zip(self.models, self.data, strict=True)
            ]
            f, q, v, s = (self.tensor([r[k] for r in rows]) for k in range(4))
            self._observation = self.execution.report(f, q, v).clone()
            self.pre_state = s
            entered = self.life.entry >= 0
            self.phase = torch.where(
                entered, torch.where(self.life.hold < 0.5, 3, 4), torch.where(s[:, 3] < 0.27, 1, 2)
            ).long()
            self.phase[self.tensor(self.elapsed) < 0.1] = 0
        return {"student_actor": self._observation}

    def step(self, actions):
        self.observations()
        self.execution.submit(actions)
        fine = []
        fineq = []
        finedq = []
        finemotor = []
        finenet = []
        fineadopt = []
        finehp = []
        executed = np.zeros(self.num_envs, dtype=int)
        try:
            for _ in range(10):
                q = self.tensor([d.qpos[self.refs["qpos"]] for d in self.data])
                v = self.tensor([d.qvel[self.refs["dofs"]] for d in self.data])
                motor, net = self.execution.step(q, v)
                for i, (m, d) in enumerate(zip(self.models, self.data, strict=True)):
                    names = [
                        self.mj.mj_id2name(m, self.mj.mjtObj.mjOBJ_JOINT, j)
                        for j in self.refs["joints"]
                    ]
                    damping = np.array([self.catalog["mechanics"]["damping"][n] for n in names])
                    friction = np.array(
                        [self.catalog["mechanics"]["friction_loss_Nm"][n] for n in names]
                    )
                    physical_velocity = d.qvel[self.refs["dofs"]].copy()
                    passive = -damping * physical_velocity - friction * np.sign(physical_velocity)
                    d.ctrl[self.refs["actuators"]] = motor[i].numpy() / self.refs["gears"]
                    d.qfrc_applied[:] = 0.0
                    d.qfrc_applied[self.refs["dofs"]] = net[i].numpy() + passive
                    for _ in range(2):
                        self.mj.mj_step(m, d)
                        executed[i] += 1
                    if np.abs(d.qfrc_actuator[self.refs["dofs"]] - motor[i].numpy()).max() > 1e-4:
                        raise RuntimeError("Requested motor vs actual actuator mismatch")
                rows = [
                    measured(self.mj, m, d, self.refs)
                    for m, d in zip(self.models, self.data, strict=True)
                ]
                frame, qafter, vafter, s = (self.tensor([r[k] for r in rows]) for k in range(4))
                self.elapsed += 0.001
                hp = (self.execution.last_dq[:, LEGS] - self.execution.hp_low).clone()
                state = s.numpy()
                dual = (state[:, 9:11] > 2).all(1)
                quality = (
                    (state[:, 3] >= 0.30)
                    & (state[:, 3] <= 0.40)
                    & (abs(state[:, 3] - 0.34) <= 0.02)
                    & (abs(state[:, 4:6]) <= 0.18).all(1)
                    & (state[:, 11] <= 0.005)
                    & dual
                    & (self.life.entry.numpy() >= 0)
                )
                self.raw_hold = np.where(quality, self.raw_hold + 0.001, 0.0)
                self.raw_longest = np.maximum(self.raw_longest, self.raw_hold)
                eligible = (
                    (self.life.entry.numpy() >= 0)
                    & (self.elapsed >= self.life.entry.numpy() + 5)
                    & (self.elapsed < self.life.entry.numpy() + 10)
                )
                self.hf_power += hp.numpy() ** 2 * eligible[:, None]
                self.hf_count += eligible
                fine.append(s)
                fineq.append(q.clone())
                finedq.append(v.clone())
                finemotor.append(motor.clone())
                finenet.append(net.clone())
                fineadopt.append(self.execution.adopted.clone())
                finehp.append(hp)
            fault = (
                (~torch.isfinite(torch.cat((s, qafter, vafter), -1)).all(-1))
                | (s[:, 11] > 0.005)
                | (s[:, 12] < 0.5)
                | (s[:, 3] < 0.10)
            )
            result = self.life.observe(
                time_s=self.tensor(self.elapsed),
                state=s,
                target=self.execution.adopted,
                motor=motor,
                student=torch.ones(self.num_envs, dtype=torch.bool),
                terminal=fault,
                truncated=torch.zeros(self.num_envs, dtype=torch.bool),
                hp10=self.execution.physical_HP10(),
                angular_velocity=frame[:, :3] * 4,
                physical_q=qafter,
                clearance=torch.zeros(self.num_envs),
            )
            # Success is recorded but own controller continues to the declared full horizon.
            timeout = self.tensor(self.elapsed) >= self.config["episode_seconds"] - 1e-5
            self.pending = (fault | timeout).numpy()
            entered = self.life.entry >= 0
            hf = np.sqrt(self.hf_power / np.maximum(1, self.hf_count[:, None]))
            conditions = dict(
                recovery_phase=self.phase,
                initial_Swing=torch.tensor([c["pose"] == "Swing" for c in self.cases]),
                validation_cohort=torch.tensor([c.get("cohort", 0) for c in self.cases]),
                entered=entered,
                dual_contact=(s[:, 9:11] > 2).all(-1),
                continuous_hold_s=self.life.hold.clone(),
                HF_peak=self.tensor(hf.max(1)),
                raw_longest_hold_s=self.tensor(self.raw_longest),
                actor_age_ms=self.execution.effective_packet_age.clone(),
                target_delay_ms=self.execution.target_delay.clone(),
                wheel_left_ms=self.execution.wheel_delay[:, 0].clone(),
                wheel_right_ms=self.execution.wheel_delay[:, 1].clone(),
                root_height_m=s[:, 3].clone(),
            )
            traces = {
                k: torch.stack(values, dim=1)
                for k, values in dict(
                    state_1kHz=fine,
                    q_1kHz=fineq,
                    dq_1kHz=finedq,
                    motor_1kHz=finemotor,
                    NET_1kHz=finenet,
                    adopted_1kHz=fineadopt,
                    dq_HP10_1kHz=finehp,
                ).items()
            }
            self._observation = None
            return dict(
                reward=result["reward"].clone(),
                terminated=fault,
                truncated=timeout,
                extras=dict(
                    actual_physics_steps=torch.tensor(executed),
                    control_owner=torch.full((self.num_envs,), 2),
                    collection_conditions=conditions,
                    collection_trace=traces,
                ),
            )
        except BaseException as exc:
            exc.actual_physics_steps = executed
            raise

    def close(self):
        self.models = []
        self.data = []


def prepare_resources(job):
    """Reuse the Recovery owner/UUID/resource policy without launching Isaac."""
    if str(job.runtime.device).startswith("cuda"):
        from .recovery_isaac.backend import require_idle_gpu

        return require_idle_gpu(
            job.assets["physical_gpu_uuid"], job.assets["gpu_resource_policy"], distillation=True
        )
    return {"device": "cpu"}


def make_collection_env(job, resources=None):
    prepare_resources(job)
    return RecoveryMuJoCoCollection(job)
