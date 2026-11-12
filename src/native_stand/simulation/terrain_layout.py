"""Finite authored road layout and horizon checks, without an engine."""

import math


def validate(config):
    for key in ("repetitions", "lateral_repetitions"):
        v = config.get(key, 1)
        if type(v) is not int or v <= 0:
            raise ValueError("Terrain tile counts must be positive integers")
    if (
        config.get("lateral_repetitions", 1) > 1
        and config.get("seam_contract") != "periodic_C0_edges_v1"
    ):
        raise ValueError("Lateral tiling requires explicitly seam-matched collision assets")
    for key in ("length_m", "width_m", "cell_m", "spawn_x_m", "boundary_margin_m"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError("Terrain dimensions must be positive finite values")
    guard = config.get("horizon_guard")
    if guard:
        # Conservative bound includes entry/ramp as full speed; do not weaken
        # the real30s complete-episode criterion to fit a short road.
        travel = guard["forward_speed_mps"] * guard["episode_s"] * guard["distance_factor"]
        available = (
            config["length_m"] * config["repetitions"]
            - config["spawn_x_m"]
            - config["boundary_margin_m"]
        )
        if not math.isfinite(travel) or travel <= 0 or available < travel:
            raise ValueError(
                "Authored H1 collision extent cannot cover the declared episode horizon"
            )
    return config


def tile_offsets(config):
    validate(config)
    ny = config.get("lateral_repetitions", 1)
    return [
        (ix * config["length_m"], (iy - (ny - 1) / 2) * config["width_m"], 0.0)
        for ix in range(config["repetitions"])
        for iy in range(ny)
    ]


def periodic_edges(heights, *, seam_cells):
    """Blend a narrow side strip to identical endpoint heights; interior unchanged."""
    import numpy as np

    h = np.array(heights, dtype=np.float32, copy=True)
    if not 1 <= seam_cells < (len(h) - 1) // 2:
        raise ValueError("Invalid terrain seam strip")
    common = (h[0] + h[-1]) * 0.5
    for j in range(seam_cells + 1):
        # Smoothstep gives zero derivative for the blend weight at both ends.
        t = j / seam_cells
        w = 1 - 3 * t * t + 2 * t * t * t
        if j:
            h[j] += (common - heights[0]) * w
            h[-1 - j] += (common - heights[-1]) * w
    h[0] = h[-1] = common
    if not np.array_equal(h[:, 0], h[:, -1]):
        raise ValueError("Longitudinal source edges are not periodic")
    return h
