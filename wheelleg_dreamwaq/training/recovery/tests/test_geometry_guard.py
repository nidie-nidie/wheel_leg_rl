from pathlib import Path
from copy import deepcopy
import sys
from time import time_ns

import mujoco
import numpy as np
import pytest

from recovery.contact_model import GeometryAudit, build_contact_model
from recovery.geometry_guard import GeometryGuardClient, GeometryInspector


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def setup():
    directory = ROOT / "recovery/artifacts/geometry_guard_tests" / str(time_ns())
    directory.mkdir(parents=True)
    model_path = build_contact_model(directory / "model")
    config = {"python_executable": sys.executable, "contact_manifest": str(model_path.with_name("contact_manifest.json")),
              "allowed_body_pairs": [], "maximum_ground_penetration_m": .002,
              "minimum_clearance_m": 0.0, "distance_limit_m": .05, "internal_clearance_mode": "strict",
              "diagnostic_envs_per_substep": 32}
    return directory, config


def body_state(model, data):
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) for index in range(1, model.nbody)]
    return names, data.xpos[None, 1:].copy(), data.xquat[None, 1:].copy()


def test_world_body_geometry_matches_native_queries_during_real_physics(setup):
    _, config = setup
    inspector = GeometryInspector(config)
    model, data = inspector.model, mujoco.MjData(inspector.model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    audit = GeometryAudit(model, [])
    for _ in range(5):
        for _ in range(10):
            mujoco.mj_step(model, data)
        names, positions, quaternions = body_state(model, data)
        expected = audit.inspect(data)
        report = inspector.inspect(names, positions, quaternions)[0]
        assert report["clearance"]["minimum_nonallowed_clearance_m"] == pytest.approx(
            expected["minimum_nonallowed_clearance_m"], abs=1e-10)
        native_floor = min(float(mujoco.mj_geomDistance(model, data, inspector.floor, geom, .05, None))
                           for geom in inspector.geoms)
        assert report["maximum_ground_penetration_m"] == pytest.approx(max(0.0, -native_floor), abs=1e-10)


def test_worker_uses_same_queries_and_rejects_missing_bodies(setup):
    directory, config = setup
    inspector = GeometryInspector(config)
    mujoco.mj_resetDataKeyframe(inspector.model, inspector.data, 0)
    mujoco.mj_forward(inspector.model, inspector.data)
    names, positions, quaternions = body_state(inspector.model, inspector.data)
    expected = inspector.inspect(names, positions, quaternions)
    client = GeometryGuardClient(config, directory)
    try:
        assert client.inspect(names, positions.tolist(), quaternions.tolist()) == expected
        assert client.sequence == 1
    finally:
        client.close()
    assert client.process.poll() == 0
    removed = names.index(inspector.body_names[0])
    kept = [index for index in range(len(names)) if index != removed]
    with pytest.raises(ValueError, match="刚体缺少"):
        inspector.inspect([names[index] for index in kept], positions[:, kept], quaternions[:, kept])


def test_world_body_queries_detect_ground_intersection(setup):
    _, config = setup
    inspector = GeometryInspector(config)
    mujoco.mj_resetDataKeyframe(inspector.model, inspector.data, 0)
    mujoco.mj_forward(inspector.model, inspector.data)
    names, positions, quaternions = body_state(inspector.model, inspector.data)
    positions[:, :, 2] -= .5
    report = inspector.inspect(names, positions, quaternions)[0]
    assert report["maximum_ground_penetration_m"] > .1
    assert report["failure_reason"] is not None


def test_diagnostic_sampling_preserves_unknowns_and_native_root_alias(setup):
    _, original = setup
    config = deepcopy(original)
    config.update(internal_clearance_mode="diagnostic", diagnostic_envs_per_substep=1)
    inspector = GeometryInspector(config)
    mujoco.mj_resetDataKeyframe(inspector.model, inspector.data, 0)
    mujoco.mj_forward(inspector.model, inspector.data)
    names, positions, quaternions = body_state(inspector.model, inspector.data)
    names[names.index("base")] = "base_link"
    positions = np.repeat(positions, 3, axis=0)
    quaternions = np.repeat(quaternions, 3, axis=0)
    for expected_index in range(3):
        reports = inspector.inspect(names, positions, quaternions)
        observed = [index for index, report in enumerate(reports)
                    if report["clearance"]["minimum_nonallowed_clearance_m"] is not None]
        assert observed == [expected_index]
        for index, report in enumerate(reports):
            if index != expected_index:
                assert report["internal_clearance_violation"] is None
                assert report["clearance"]["checked_pair_count"] == 0
            assert report["failure_reason"] is None
