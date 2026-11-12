"""Explicit scene input and world-Z ground query; no simulator is started here."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SceneConfig:
    version: str = "flat_scene_v1"
    ground_z_m: float = 0.0
    friction: float = 0.5
    initial_state: dict | None = None

    def __post_init__(self):
        import math

        if (
            self.version != "flat_scene_v1"
            or not all(math.isfinite(x) for x in (self.ground_z_m, self.friction))
            or self.friction < 0
        ):
            raise ValueError("Unsupported scene or material")


class FlatHeight:
    def __init__(self, config):
        self.config = config

    def __call__(self, xy):
        return xy[..., 0] * 0 + self.config.ground_z_m


def height_above_ground(root_position, provider):
    """Vertical distance from ground at root XY to root-link origin, not COM."""
    ground = provider(root_position[..., :2])
    if ground.shape != root_position.shape[:-1]:
        raise ValueError("Height provider returned wrong rows")
    return root_position[..., 2] - ground


def scene_for(job, *, height_provider=None):
    config = SceneConfig(**job.assets.get("scene", {}))
    return config, height_provider or FlatHeight(config)


def initial_catalog(config, default, *, paths=None):
    """An alternative catalog needs explicit content identity and matching mechanics."""
    if config.initial_state is None:
        return default
    from native_stand.artifacts.io import file_sha256, json_digest, read_json

    ref = config.initial_state
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("Scene initial_state requires path and SHA")
    path = paths.resolve(ref["path"]) if paths else ref["path"]
    if file_sha256(path) != ref["sha256"]:
        raise ValueError("Initial catalog SHA differs")
    selected = read_json(path)
    if json_digest(selected["mechanics"]) != json_digest(default["mechanics"]):
        raise ValueError("Scene cannot silently change the robot mechanics")
    if set(default["poses"]) - selected["poses"].keys():
        raise ValueError("Missing task-required legal initial poses")
    return selected


def validate_reset_pose(root, q, dq, ground):
    import torch

    if not bool(
        torch.isfinite(root).all() and torch.isfinite(q).all() and torch.isfinite(dq).all()
    ):
        raise ValueError("Nonfinite scene reset")
    if not torch.allclose(
        torch.linalg.vector_norm(root[:, 3:7], dim=-1),
        torch.ones(len(root), device=root.device),
        atol=1e-4,
        rtol=0,
    ):
        raise ValueError("Invalid reset quaternion")
    if bool((height_above_ground(root[:, :3], ground) <= 0).any()):
        raise ValueError("Reset root is below scene ground")
