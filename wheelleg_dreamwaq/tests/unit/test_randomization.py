from __future__ import annotations

import hashlib

import pytest
import torch

from wheelleg_dreamwaq.kinematics.virtual_leg import (
    load_usd_anchor_audit,
    offset_geometry_from_audit,
    offset_leg_fk,
    wrap_to_pi,
)
from wheelleg_dreamwaq.schemas.randomization import (
    CLOSED_CHAIN_NOMINAL_BRANCH_SIGNATURE,
    CLOSED_CHAIN_PHYSICAL_PASSIVE_BRANCH_JOINTS,
    CLOSED_CHAIN_RELAXATION_VERSION,
    CLOSED_CHAIN_RESET_CACHE_SCHEMA_VERSION,
    CLOSED_CHAIN_ROOT_HEIGHT_ALIGNMENT_VERSION,
    CPU_STREAMS,
    FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1,
    NOMINAL_TRAINING_PROFILE_V1,
    PROCESS_START_RANDOMIZATION_STREAMS,
    RANDOMIZATION_STREAMS,
    RUNTIME_RANDOMIZATION_STREAMS,
    RandomizationProfileV1,
    canonical_tensor_sha256,
    closed_chain_reset_contract_payload,
    compute_closed_chain_root_height_offset,
    derive_stream_seed,
    generator_device_for_stream,
    profile_contract_hash,
    profile_contract_payload,
    seed_manifest,
    validate_closed_chain_reset_cache_artifact,
    validate_randomization_rng_state_payload,
)
from wheelleg_dreamwaq.schemas.action import CANONICAL_JOINT_ORDER
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.randomization import (
    WheelLegRandomizationRuntime,
    sample_leg_references,
)


def test_fudan_profile_matches_frozen_v1_ranges() -> None:
    profile = FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1
    assert profile.enabled is True
    assert profile.friction_range == (0.6, 1.4)
    assert profile.friction_bucket_count == 64
    assert profile.leg_reference_offset_range_rad == (-0.03, 0.03)
    assert profile.kp_scale_range == (0.95, 1.05)
    assert profile.kd_scale_range == (0.95, 1.05)
    assert profile.effort_limit_scale_range == (0.95, 1.05)


def test_nominal_profile_is_explicitly_disabled() -> None:
    assert NOMINAL_TRAINING_PROFILE_V1.enabled is False
    assert NOMINAL_TRAINING_PROFILE_V1.name == "NominalTrainingProfileV1"


def test_seed_derivation_matches_frozen_sha256_rule() -> None:
    payload = b"RandomizationSeedV1|42|command_rng"
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)
    assert derive_stream_seed(42, "command_rng") == (expected or 1)
    assert derive_stream_seed(42, "command_rng") != derive_stream_seed(42, "reset_velocity_rng")


def test_stream_manifest_uses_cpu_except_for_observation_noise() -> None:
    manifest = seed_manifest(123, "cuda:0")
    assert set(manifest) == set(RANDOMIZATION_STREAMS)
    for stream in CPU_STREAMS:
        assert manifest[stream]["device"] == "cpu"
    assert manifest["observation_noise_rng"]["device"] == "cuda:0"
    assert generator_device_for_stream("command_rng", "cuda:0") == "cpu"


def test_stream_lifecycle_partition_is_frozen_in_the_profile_contract() -> None:
    assert RANDOMIZATION_STREAMS == (
        PROCESS_START_RANDOMIZATION_STREAMS + RUNTIME_RANDOMIZATION_STREAMS
    )
    assert set(PROCESS_START_RANDOMIZATION_STREAMS).isdisjoint(RUNTIME_RANDOMIZATION_STREAMS)
    payload = profile_contract_payload(FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1)
    assert payload["stream_lifecycle"] == {
        "process_start": list(PROCESS_START_RANDOMIZATION_STREAMS),
        "runtime": list(RUNTIME_RANDOMIZATION_STREAMS),
    }


def _rng_state_payload(seed: int) -> dict:
    streams = {}
    for index, stream in enumerate(RANDOMIZATION_STREAMS):
        generator = torch.Generator(device="cpu").manual_seed(seed + index)
        streams[stream] = {"device": "cpu", "state": generator.get_state().clone()}
    return {
        "seed_derivation_version": "RandomizationSeedDerivationV1",
        "streams": streams,
    }


def _runtime_with_cpu_generators(seed: int) -> WheelLegRandomizationRuntime:
    runtime = WheelLegRandomizationRuntime.__new__(WheelLegRandomizationRuntime)
    runtime._generators = {
        stream: torch.Generator(device="cpu").manual_seed(seed + index)
        for index, stream in enumerate(RANDOMIZATION_STREAMS)
    }
    return runtime


def test_selective_rng_restore_preserves_target_process_start_streams() -> None:
    source = _rng_state_payload(100)
    runtime = _runtime_with_cpu_generators(1000)
    target_before = {
        stream: generator.get_state().clone()
        for stream, generator in runtime._generators.items()
    }

    runtime.set_rng_state(source, stream_names=RUNTIME_RANDOMIZATION_STREAMS)

    for stream in PROCESS_START_RANDOMIZATION_STREAMS:
        assert torch.equal(runtime._generators[stream].get_state(), target_before[stream])
    for stream in RUNTIME_RANDOMIZATION_STREAMS:
        assert torch.equal(runtime._generators[stream].get_state(), source["streams"][stream]["state"])


def test_full_rng_restore_replaces_all_streams() -> None:
    source = _rng_state_payload(200)
    runtime = _runtime_with_cpu_generators(2000)

    runtime.set_rng_state(source)

    for stream in RANDOMIZATION_STREAMS:
        assert torch.equal(runtime._generators[stream].get_state(), source["streams"][stream]["state"])


def test_selective_rng_restore_still_validates_unselected_streams() -> None:
    source = _rng_state_payload(300)
    source["streams"]["material_rng"]["state"] = torch.tensor([1], dtype=torch.uint8)
    runtime = _runtime_with_cpu_generators(3000)

    with pytest.raises(ValueError, match="material_rng"):
        runtime.set_rng_state(source, stream_names=RUNTIME_RANDOMIZATION_STREAMS)


def test_rng_state_payload_rejects_stream_device_rule_mismatch() -> None:
    source = _rng_state_payload(400)
    source["streams"]["material_rng"]["device"] = "cuda:0"

    with pytest.raises(ValueError, match="device mismatch"):
        validate_randomization_rng_state_payload(source)


def test_profile_contract_is_seed_independent_and_stable() -> None:
    payload = profile_contract_payload(FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1)
    encoded = repr(payload)
    assert "master_seed" not in encoded
    assert "31001" not in encoded
    assert profile_contract_hash(FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1) == profile_contract_hash(
        FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"friction_range": (1.0, 0.5)},
        {"friction_range": (-0.1, 1.0)},
        {"friction_bucket_count": 0},
        {"joint_reference_resample_limit": 0},
        {"joint_velocity_noise_rad_s": -1.0},
    ],
)
def test_profile_rejects_invalid_ranges(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        RandomizationProfileV1(name="invalid", enabled=True, **kwargs)


def test_unknown_stream_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        derive_stream_seed(1, "unknown")


def _audited_leg_geometries():
    audit = load_usd_anchor_audit()
    return (
        offset_geometry_from_audit(audit["legs"]["left"]),
        offset_geometry_from_audit(audit["legs"]["right"]),
    )


def test_leg_reference_sampling_is_deterministic_and_closed_chain_valid() -> None:
    q_nominal = torch.tensor((-0.33367134, 0.33367134, -0.33367134, 0.33367134))
    hard_limits = torch.tensor(((-1.0, 1.0),) * 4)

    def sample(seed: int):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return sample_leg_references(
            count=256,
            q_nominal=q_nominal,
            hard_limits=hard_limits,
            soft_limit_abs=1.0,
            geometries=_audited_leg_geometries(),
            profile=FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1,
            generator=generator,
            device="cpu",
        )

    first, first_audit = sample(17)
    repeated, repeated_audit = sample(17)
    different, _ = sample(18)
    assert torch.equal(first, repeated)
    assert first_audit == repeated_audit
    assert not torch.equal(first, different)
    assert torch.all(first >= hard_limits[:, 0])
    assert torch.all(first <= hard_limits[:, 1])
    assert torch.all(torch.abs(first) <= 1.0)

    left_geometry, right_geometry = _audited_leg_geometries()
    nominal_left = offset_leg_fk(q_nominal[:2].unsqueeze(0), left_geometry)
    nominal_right = offset_leg_fk(q_nominal[2:].unsqueeze(0), right_geometry)
    left = offset_leg_fk(first[:, :2], left_geometry)
    right = offset_leg_fk(first[:, 2:], right_geometry)
    profile = FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1
    assert torch.all(left.valid & right.valid)
    assert torch.all(torch.abs(left.length - nominal_left.length) <= profile.max_leg_length_delta_m)
    assert torch.all(torch.abs(right.length - nominal_right.length) <= profile.max_leg_length_delta_m)
    assert torch.all(torch.abs(wrap_to_pi(left.phi0 - nominal_left.phi0)) <= profile.max_leg_phi0_delta_rad)
    assert torch.all(torch.abs(wrap_to_pi(right.phi0 - nominal_right.phi0)) <= profile.max_leg_phi0_delta_rad)
    assert torch.all(torch.abs(wrap_to_pi(left.phi0 - right.phi0)) <= profile.max_lr_phi0_delta_rad)


def test_nominal_leg_reference_sampling_does_not_consume_randomness() -> None:
    q_nominal = torch.tensor((-0.3, 0.3, -0.3, 0.3))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(9)
    before = generator.get_state().clone()
    references, audit = sample_leg_references(
        count=4,
        q_nominal=q_nominal,
        hard_limits=torch.tensor(((-1.0, 1.0),) * 4),
        soft_limit_abs=1.0,
        geometries=_audited_leg_geometries(),
        profile=NOMINAL_TRAINING_PROFILE_V1,
        generator=generator,
        device="cpu",
    )
    assert torch.equal(references, q_nominal.repeat(4, 1))
    assert torch.equal(generator.get_state(), before)
    assert audit["acceptance_rate"] == 1.0


def _cache_artifact() -> dict:
    num_envs = 2
    num_joints = 26
    q_reset = torch.arange(num_envs * num_joints, dtype=torch.float32).reshape(num_envs, num_joints)
    root_height_offset = torch.tensor((0.001, -0.002), dtype=torch.float32)
    actuator_plan = {
        "active_joint_realized_stiffness": torch.ones((num_envs, 6), dtype=torch.float32),
        "active_joint_realized_damping": torch.full((num_envs, 6), 2.0, dtype=torch.float32),
        "active_joint_realized_effort_limit": torch.full((num_envs, 6), 3.0, dtype=torch.float32),
    }
    return {
        "schema_version": CLOSED_CHAIN_RESET_CACHE_SCHEMA_VERSION,
        "algorithm_version": CLOSED_CHAIN_RELAXATION_VERSION,
        "root_height_algorithm_version": CLOSED_CHAIN_ROOT_HEIGHT_ALIGNMENT_VERSION,
        "contract": closed_chain_reset_contract_payload(sim_dt=0.005),
        "identity": {
            "asset_bundle_version": "AssetBundleV2",
            "asset_bundle_hash": "A" * 64,
            "physics_schema_version": "PhysicsV4",
            "randomization_profile_hash": "B" * 64,
            "realized_plan_hash": "C" * 64,
            "actuator_plan_hash": canonical_tensor_sha256(actuator_plan),
            "master_seed": 42,
            "num_envs": num_envs,
            "joint_names": [f"joint_{index}" for index in range(num_joints)],
            "passive_joint_names": [f"passive_{index}" for index in range(20)],
            "physical_passive_branch_joint_names": list(CLOSED_CHAIN_PHYSICAL_PASSIVE_BRANCH_JOINTS),
        },
        "q_reset_projected_env": q_reset.contiguous(),
        "root_height_offset_env": root_height_offset.contiguous(),
        "tensor_sha256": canonical_tensor_sha256(
            {
                "q_reset_projected_env": q_reset,
                "root_height_offset_env": root_height_offset,
            }
        ),
        "actuator_plan": actuator_plan,
        "metrics": {
            "physical_passive_branch_signature": [list(CLOSED_CHAIN_NOMINAL_BRANCH_SIGNATURE)] * num_envs,
        },
        "trace": [],
    }


def test_canonical_tensor_hash_is_stable_and_content_sensitive() -> None:
    first = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    repeated = first.clone()
    changed = first.clone()
    changed[0, 0] = 99.0

    assert canonical_tensor_sha256({"field": first}) == canonical_tensor_sha256({"field": repeated})
    assert canonical_tensor_sha256({"field": first}) != canonical_tensor_sha256({"field": changed})
    assert canonical_tensor_sha256({"field": first}) != canonical_tensor_sha256({"renamed": first})


def test_root_height_alignment_is_canonical_cpu_float32() -> None:
    q_nominal = torch.tensor((-0.33367134, 0.33367134, -0.33367134, 0.33367134))
    references = torch.tensor(
        (
            (-0.3620099127292633, 0.3551750183105469, -0.33141160011291504, 0.356408953666687),
            (-0.32874682545661926, 0.31220877170562744, -0.32875359058380127, 0.3121575713157654),
        ),
        dtype=torch.float32,
    )
    geometries = _audited_leg_geometries()

    nominal = compute_closed_chain_root_height_offset(
        q_nominal.repeat(2, 1),
        q_nominal=q_nominal,
        geometries=geometries,
    )
    first = compute_closed_chain_root_height_offset(
        references,
        q_nominal=q_nominal,
        geometries=geometries,
    )
    repeated = compute_closed_chain_root_height_offset(
        references.clone(),
        q_nominal=q_nominal.clone(),
        geometries=geometries,
    )

    assert nominal.device.type == "cpu"
    assert nominal.dtype == torch.float32
    assert nominal.is_contiguous()
    assert torch.equal(nominal, torch.zeros(2, dtype=torch.float32))
    assert torch.equal(first, repeated)
    assert first.tolist() == pytest.approx([0.0071051097474992275, -0.0037394764367491007], abs=1.0e-9)

    if torch.cuda.is_available():
        from_cuda = compute_closed_chain_root_height_offset(
            references.cuda(),
            q_nominal=q_nominal.cuda(),
            geometries=geometries,
        )
        assert torch.equal(first, from_cuda)


def test_closed_chain_reset_cache_validation_rejects_tensor_or_actuator_drift() -> None:
    artifact = _cache_artifact()
    assert validate_closed_chain_reset_cache_artifact(artifact) is artifact

    changed_tensor = _cache_artifact()
    changed_tensor["q_reset_projected_env"][0, 0] += 1.0
    with pytest.raises(ValueError, match="tensor hash"):
        validate_closed_chain_reset_cache_artifact(changed_tensor)

    changed_root_offset = _cache_artifact()
    changed_root_offset["root_height_offset_env"][0] += 1.0
    with pytest.raises(ValueError, match="tensor hash"):
        validate_closed_chain_reset_cache_artifact(changed_root_offset)

    changed_actuator = _cache_artifact()
    changed_actuator["actuator_plan"]["active_joint_realized_damping"][0, 0] += 1.0
    with pytest.raises(ValueError, match="actuator plan hash"):
        validate_closed_chain_reset_cache_artifact(changed_actuator)
