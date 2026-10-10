from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = PROJECT_ROOT / "debug" / "sim2sim" / "models"


def _joint_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> tuple[str, ...]:
    return tuple(mujoco.mj_id2name(model, object_type, index) for index in range(count))


def _assert_physical_parameters_match(formal: mujoco.MjModel, debug: mujoco.MjModel) -> None:
    assert _names(formal, mujoco.mjtObj.mjOBJ_BODY, formal.nbody) == _names(
        debug, mujoco.mjtObj.mjOBJ_BODY, debug.nbody
    )
    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
        np.testing.assert_allclose(getattr(debug, field), getattr(formal, field), rtol=0.0, atol=0.0)

    assert _names(formal, mujoco.mjtObj.mjOBJ_GEOM, formal.ngeom) == _names(
        debug, mujoco.mjtObj.mjOBJ_GEOM, debug.ngeom
    )
    for field in (
        "geom_type",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_friction",
        "geom_contype",
        "geom_conaffinity",
    ):
        np.testing.assert_allclose(getattr(debug, field), getattr(formal, field), rtol=0.0, atol=0.0)

    formal_joint_names = set(_names(formal, mujoco.mjtObj.mjOBJ_JOINT, formal.njnt)) - {"base_free"}
    debug_joint_names = set(_names(debug, mujoco.mjtObj.mjOBJ_JOINT, debug.njnt)) - {"base_free"}
    assert debug_joint_names == formal_joint_names
    for name in sorted(formal_joint_names):
        formal_id = _joint_id(formal, name)
        debug_id = _joint_id(debug, name)
        assert debug.jnt_type[debug_id] == formal.jnt_type[formal_id]
        for field in ("jnt_pos", "jnt_axis", "jnt_range", "jnt_limited"):
            np.testing.assert_allclose(
                getattr(debug, field)[debug_id],
                getattr(formal, field)[formal_id],
                rtol=0.0,
                atol=0.0,
            )
        formal_dof = int(formal.jnt_dofadr[formal_id])
        debug_dof = int(debug.jnt_dofadr[debug_id])
        for field in ("dof_damping", "dof_frictionloss", "dof_armature"):
            np.testing.assert_allclose(
                getattr(debug, field)[debug_dof],
                getattr(formal, field)[formal_dof],
                rtol=0.0,
                atol=0.0,
            )

    formal_actuator_names = _names(formal, mujoco.mjtObj.mjOBJ_ACTUATOR, formal.nu)
    assert _names(debug, mujoco.mjtObj.mjOBJ_ACTUATOR, debug.nu) == formal_actuator_names
    for name in formal_actuator_names:
        formal_id = mujoco.mj_name2id(formal, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        debug_id = mujoco.mj_name2id(debug, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        formal_joint = mujoco.mj_id2name(
            formal, mujoco.mjtObj.mjOBJ_JOINT, int(formal.actuator_trnid[formal_id, 0])
        )
        debug_joint = mujoco.mj_id2name(
            debug, mujoco.mjtObj.mjOBJ_JOINT, int(debug.actuator_trnid[debug_id, 0])
        )
        assert debug_joint == formal_joint
        for field in (
            "actuator_gear",
            "actuator_ctrlrange",
            "actuator_forcerange",
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_dynprm",
        ):
            np.testing.assert_allclose(
                getattr(debug, field)[debug_id],
                getattr(formal, field)[formal_id],
                rtol=0.0,
                atol=0.0,
            )

    assert debug.neq == formal.neq
    for field in ("eq_type", "eq_obj1id", "eq_obj2id", "eq_data", "eq_solref", "eq_solimp"):
        np.testing.assert_allclose(getattr(debug, field), getattr(formal, field), rtol=0.0, atol=0.0)


def test_fixed_base_model_has_no_free_joint_contact_or_gravity() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_ROOT / "wheel_leg_urdf4_fixed_base_debug.xml"))

    assert _joint_id(model, "base_free") == -1
    np.testing.assert_array_equal(model.opt.gravity, 0.0)
    assert model.npair == 0
    assert model.nu == 6
    assert model.nkey == 1


def test_suspended_model_keeps_free_joint_but_removes_contact_and_gravity() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_ROOT / "wheel_leg_urdf4_suspended_debug.xml"))
    free_joint = _joint_id(model, "base_free")

    assert free_joint >= 0
    assert model.jnt_type[free_joint] == mujoco.mjtJoint.mjJNT_FREE
    np.testing.assert_array_equal(model.opt.gravity, 0.0)
    assert model.npair == 0
    assert model.nu == 6


def test_formal_model_remains_grounded_and_unmodified() -> None:
    model = mujoco.MjModel.from_xml_path(
        str(PROJECT_ROOT / "sim2sim" / "mujoco" / "models" / "wheel_leg_urdf4_v1.xml")
    )

    assert _joint_id(model, "base_free") >= 0
    np.testing.assert_allclose(model.opt.gravity, (0.0, 0.0, -9.81), rtol=0.0, atol=0.0)
    assert model.npair == 2


def test_debug_models_preserve_formal_physical_parameters() -> None:
    formal = mujoco.MjModel.from_xml_path(
        str(PROJECT_ROOT / "sim2sim" / "mujoco" / "models" / "wheel_leg_urdf4_v1.xml")
    )
    fixed = mujoco.MjModel.from_xml_path(str(MODEL_ROOT / "wheel_leg_urdf4_fixed_base_debug.xml"))
    suspended = mujoco.MjModel.from_xml_path(
        str(MODEL_ROOT / "wheel_leg_urdf4_suspended_debug.xml")
    )

    _assert_physical_parameters_match(formal, fixed)
    _assert_physical_parameters_match(formal, suspended)
