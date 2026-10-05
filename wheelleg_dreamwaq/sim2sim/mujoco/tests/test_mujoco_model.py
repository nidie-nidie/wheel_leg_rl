from __future__ import annotations

import json
import struct
from pathlib import Path

import mujoco
import numpy as np

from wheelleg_mujoco.contract import sha256_file
from wheelleg_mujoco.versions import ACTION_ADAPTER_VERSION, OBSERVATION_ADAPTER_VERSION
from wheelleg_mujoco.model_semantics import compiled_model_semantics, stable_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "wheel_leg_urdf4_v1.xml"
MANIFEST_PATH = PROJECT_ROOT / "model_manifest.json"


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, index: int) -> str:
    value = mujoco.mj_id2name(model, object_type, index)
    assert value is not None
    return value


def test_generated_model_has_frozen_identity() -> None:
    assert MODEL_PATH.is_file()
    assert MANIFEST_PATH.is_file()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert (model.nbody, model.njnt, model.nq, model.nv, model.nu, model.neq, model.npair) == (
        28,
        27,
        33,
        32,
        6,
        8,
        2,
    )
    assert manifest["model_version"] == "MujocoModelV1"
    assert manifest["compiled_dimensions"] == {
        "nbody": 28,
        "njnt": 27,
        "nq": 33,
        "nv": 32,
        "nu": 6,
        "neq": 8,
        "npair": 2,
    }


def test_model_contains_exact_dummy_and_actuator_sets() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    body_names = {_name(model, mujoco.mjtObj.mjOBJ_BODY, index) for index in range(model.nbody)}
    joint_names = {_name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)}
    actuator_names = {_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in range(model.nu)}
    expected_dummy = {
        "jIO_dummy_child_link1",
        "jIO_dummy_child_link2",
        "jAG_dummy_child_link1",
        "jAG_dummy_child_link2",
        "jKN_dummy_child_link1",
        "jKN_dummy_child_link2",
        "jMK_dummy_child1",
        "jMK_dummy_child2",
        "jCF_dummy_child_link1",
        "jCF_dummy_child_link2",
        "jEC_dummy_child_link1",
        "jEC_dummy_child_link2",
    }
    assert expected_dummy <= body_names
    assert expected_dummy <= joint_names
    assert actuator_names == {
        "Left_front_joint_act",
        "Left_rear_joint_act",
        "Right_front_joint_act",
        "Right_rear_joint_act",
        "Left_Wheel_act",
        "Right_Wheel_act",
    }


def test_only_two_explicit_wheel_floor_pairs_are_enabled() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    pair_names = {_name(model, mujoco.mjtObj.mjOBJ_PAIR, index) for index in range(model.npair)}
    assert pair_names == {"floor_left_wheel", "floor_right_wheel"}
    assert np.all(model.geom_contype == 0)
    assert np.all(model.geom_conaffinity == 0)
    assert np.allclose(model.pair_friction[:, :2], 1.0)
    assert np.allclose(model.pair_friction[:, 2:], 0.0)


def test_model_mass_and_reset_keyframe_match_usd_contract() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    np.testing.assert_allclose(model.body_mass.sum(), 4.396253988146782, rtol=0.0, atol=1.0e-6)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "contract_v4_reset")
    assert key_id >= 0
    assert np.isfinite(model.key_qpos[key_id]).all()


def test_base_mesh_is_losslessly_partitioned_for_mujoco() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    partition = manifest["base_mesh_partition"]
    assert partition["source_face_count"] == 454_256
    assert sum(partition["part_face_counts"]) == partition["source_face_count"]
    assert max(partition["part_face_counts"]) <= 200_000
    assert len(partition["parts"]) == len(partition["part_face_counts"])
    for filename, expected_faces in zip(partition["parts"], partition["part_face_counts"], strict=True):
        path = MODEL_PATH.parent / "meshes" / filename
        with path.open("rb") as stream:
            stream.seek(80)
            actual_faces = struct.unpack("<I", stream.read(4))[0]
        assert actual_faces == expected_faces
        assert path.stat().st_size == 84 + 50 * expected_faces


def test_only_rounding_scale_inertia_corrections_are_applied() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    adjustments = manifest["inertia_adjustments"]
    assert {adjustment["body_name"] for adjustment in adjustments} == {"jMK", "jEC"}
    for adjustment in adjustments:
        assert adjustment["relative_violation"] < 1.0e-6
        assert abs(adjustment["absolute_correction_kg_m2"]) < 1.0e-10
        corrected = sorted(adjustment["mujoco_diagonal_inertia_kg_m2"])
        assert corrected[0] + corrected[1] >= corrected[2]


def test_manifest_freezes_ground_solver_and_adapter_semantics() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["ground"] == {
        "geom_name": "floor",
        "owner": "MujocoModelV1",
        "top_z_m": 0.0,
        "type": "plane",
    }
    assert manifest["com_kinematics"] == {
        "position_api": "data.xipos[base]",
        "velocity_api": "mj_jacBodyCom(base) @ data.qvel",
    }
    assert len(manifest["equality_constraints"]) == 8
    assert {item["name"] for item in manifest["equality_constraints"]} == {
        "connect_cf_gh_1",
        "connect_cf_gh_2",
        "connect_kn_op_1",
        "connect_kn_op_2",
        "connect_ec_ag_1",
        "connect_ec_ag_2",
        "connect_mk_io_1",
        "connect_mk_io_2",
    }
    adapters = manifest["adapters"]
    assert adapters["observation"]["version"] == OBSERVATION_ADAPTER_VERSION
    assert adapters["action"]["version"] == ACTION_ADAPTER_VERSION
    assert adapters["observation"]["implementation_sha256"] == sha256_file(
        PROJECT_ROOT / "wheelleg_mujoco" / "observation.py"
    )
    assert adapters["action"]["implementation_sha256"] == sha256_file(
        PROJECT_ROOT / "wheelleg_mujoco" / "control.py"
    )


def test_lossless_partition_preserves_base_local_aabb() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    partition = manifest["base_mesh_partition"]
    assert partition["source_local_aabb"] == partition["parts_combined_local_aabb"]
    assert partition["max_abs_local_aabb_error_m"] == 0.0


def test_usd_and_mujoco_visual_aabb_match_within_one_millimeter() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    audit = manifest["visual_aabb_audit"]
    assert audit["passed"]
    assert audit["tolerance_m"] == 1.0e-3
    assert max(audit["center_abs_error_m"]) <= audit["tolerance_m"]
    assert max(audit["size_abs_error_m"]) <= audit["tolerance_m"]


def test_compiled_dynamics_semantics_match_frozen_hash() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    actual = compiled_model_semantics(model)
    assert actual == manifest["dynamics_semantics"]
    assert stable_hash(actual) == manifest["dynamics_semantics_hash"]
