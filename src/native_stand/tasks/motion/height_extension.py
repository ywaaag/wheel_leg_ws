"""Explicit new-contract branch; old-domain physical function must be identical."""

import copy

import torch

from native_stand.artifacts.io import file_sha256, json_digest, read_json

from .action_contract import ActionDecoder

VERSION = "motion_height020_contract_extension_v1"
ACTION = "motion_multiheight020_support_center_015_80_v2"


def extend_source(source, job):
    declaration = job.assets.get("output_contract_extension")
    if declaration is None:
        return source
    old = source["output_contract"]
    new = read_json(job.assets["output_contract"])
    evidence = declaration["geometry_evidence"]
    binding_ref = job.assets["support_center_policy_binding"]
    binding = read_json(binding_ref["path"])
    if (
        declaration["version"] != VERSION
        or file_sha256(job.assets["output_contract"]) != job.assets["output_contract_sha256"]
        or file_sha256(evidence["path"]) != evidence["sha256"]
        or file_sha256(binding_ref["path"]) != binding_ref["sha256"]
        or binding["policy_sha256"] != job.initialization.sha256
        or binding["large_contract_sha256"] != job.assets["output_contract_sha256"]
        or binding["geometry_evidence_sha256"] != evidence["sha256"]
        or binding["center_definition_canonical_json_sha256"]
        != json_digest(
            new["center_formula"] | {"support_offset_table": new["support_offset_table"]}
        )
        or json_digest(old) != declaration["source_contract_canonical_sha256"]
        or new["action_version"] != ACTION
        or job.action.version != ACTION
        or new["height_command_range_m"] != [0.20, 0.40]
    ):
        raise ValueError("高度扩展缺少明确来源和新合同身份")
    before, after = ActionDecoder.from_contract(old), ActionDecoder.from_contract(new)
    heights = torch.linspace(0.30, 0.40, 101, dtype=torch.float64)
    raw = torch.linspace(-1.2, 1.2, 606, dtype=torch.float64).reshape(101, 6)
    a, b = before.decode(raw, heights), after.decode(raw, heights)
    error = (a["requested_target"] - b["requested_target"]).abs().max().item()
    if error > 1e-6:
        raise ValueError("新中心改变旧域物理函数")
    result = dict(source)
    result.update(
        action_version=ACTION,
        output_contract=copy.deepcopy(new),
        output_contract_extension_receipt=dict(
            declaration,
            center_binding_sha256=binding_ref["sha256"],
            legacy_max_difference_rad=error,
            preserves_actor_critic_two_Adam_std_rng=True,
            low_height_support="engineering_candidate_not_identified",
        ),
    )
    return result
