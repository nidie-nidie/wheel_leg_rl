"""Generate a full-length known-parameter A1 PACE trajectory."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch


DT_S = 0.002
SAMPLE_COUNT = 10000
ALLOWED_DELAYS = frozenset((0, 1, 5, 9, 10))
PACE_JOINT_ORDER = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    value = json.loads(
        path.read_text(encoding="ascii"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )
    _require(isinstance(value, dict), "parameter file must contain one object")

    def require_finite(item: Any) -> None:
        if isinstance(item, float):
            _require(math.isfinite(item), "parameter JSON must be finite")
        elif isinstance(item, dict):
            for child in item.values():
                require_finite(child)
        elif isinstance(item, list):
            for child in item:
                require_finite(child)

    require_finite(value)
    return value


def load_parameter_vector(path: Path | str, delay_steps: int) -> torch.Tensor:
    """Load one strict physical vector and replace only its delay endpoint."""

    _require(delay_steps in ALLOWED_DELAYS, "delay is not in the allowed set")
    path = Path(path)
    payload = _strict_json(path)
    _require(
        set(payload)
        == {"schema_version", "joint_order", "physical_parameter_vector"},
        "parameter file keys differ from the frozen contract",
    )
    _require(
        payload["schema_version"] == "a1_pace_synthetic_parameters/v1",
        "parameter file schema version mismatch",
    )
    _require(
        tuple(payload["joint_order"]) == PACE_JOINT_ORDER,
        "parameter file joint order mismatch",
    )
    vector = torch.tensor(payload["physical_parameter_vector"], dtype=torch.float32)
    _require(vector.shape == (49,), "physical parameter vector must have shape (49,)")
    _require(bool(torch.isfinite(vector).all()), "physical parameter vector is non-finite")
    vector[48] = float(delay_steps)
    from pace_sim2real.tasks.manager_based.pace.a1_mean import parse_a1_mean

    parsed = parse_a1_mean(vector)
    _require(parsed.delay_steps == delay_steps, "delay truncation changed the endpoint")
    return vector.contiguous()


def build_excitation_contract(*, seed: int, amplitude_scale: float) -> dict[str, Any]:
    """Realize the same linear-chirp fields consumed by the A1_Base collector."""

    _require(isinstance(seed, int) and seed >= 0, "seed must be a nonnegative integer")
    _require(
        amplitude_scale in (0.5, 1.0),
        "synthetic amplitude_scale must be exactly 0.5 or 1.0",
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    phase_offsets = torch.rand(12, generator=generator, dtype=torch.float64) * (
        2.0 * math.pi
    )
    direction = torch.where(
        torch.rand(12, generator=generator) >= 0.5,
        torch.ones(12),
        -torch.ones(12),
    ).to(torch.float64)
    return {
        "schema_version": "a1_pace_excitation/v1",
        "formula": "center + direction * amplitude * sin(2*pi*(f0*t + 0.5*((f1-f0)/duration)*t^2) + phase)",
        "dt_s": DT_S,
        "duration_s": SAMPLE_COUNT * DT_S,
        "f0_hz": 0.1,
        "f1_hz": 10.0,
        "center_rad": [0.0] * 4 + [0.8] * 4 + [-1.5] * 4,
        "nominal_amplitude_rad": [0.12] * 4 + [0.18] * 4 + [0.20] * 4,
        "amplitude_scale": amplitude_scale,
        "realized_amplitude_rad": [
            amplitude_scale * value
            for value in ([0.12] * 4 + [0.18] * 4 + [0.20] * 4)
        ],
        "direction": [float(value) for value in direction],
        "phase_rad": [float(value) for value in phase_offsets],
        "joint_order": list(PACE_JOINT_ORDER),
    }


def build_absolute_targets(
    *, seed: int, num_samples: int, amplitude_scale: float = 1.0
) -> torch.Tensor:
    """Build the A1_Base-compatible linear chirp in PACE joint order."""

    _require(num_samples == SAMPLE_COUNT, "synthetic data requires exactly 10000 targets")
    contract = build_excitation_contract(seed=seed, amplitude_scale=amplitude_scale)
    centers = torch.tensor(contract["center_rad"], dtype=torch.float64)
    amplitudes = torch.tensor(contract["realized_amplitude_rad"], dtype=torch.float64)
    direction = torch.tensor(contract["direction"], dtype=torch.float64)
    phase_offsets = torch.tensor(contract["phase_rad"], dtype=torch.float64)
    f0 = float(contract["f0_hz"])
    f1 = float(contract["f1_hz"])
    time = torch.arange(num_samples, dtype=torch.float64) * DT_S
    duration = num_samples * DT_S
    beta = (f1 - f0) / duration
    phase = 2.0 * math.pi * (
        time * f0 + 0.5 * time.square() * beta
    )
    targets = centers[None, :] + amplitudes[None, :] * direction[None, :] * torch.sin(
        phase[:, None] + phase_offsets[None, :]
    )
    return targets.to(dtype=torch.float32).contiguous()


def _joint_ids(joint_names: Sequence[str], device: torch.device) -> torch.Tensor:
    _require(len(set(joint_names)) == len(joint_names), "articulation joint names repeat")
    try:
        indices = [joint_names.index(name) for name in PACE_JOINT_ORDER]
    except ValueError as exc:
        raise ValueError("articulation does not contain every A1 PACE joint") from exc
    _require(len(set(indices)) == 12, "A1 PACE joint mapping is not one-to-one")
    return torch.tensor(indices, device=device, dtype=torch.long)


def collect_synthetic_trajectory(
    env: Any,
    targets: torch.Tensor,
    mean: Any,
    *,
    reset_seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Collect q[k] before sending the same row's absolute target[k]."""

    _require(isinstance(targets, torch.Tensor), "targets must be a tensor")
    _require(targets.ndim == 2 and targets.shape[1] == 12, "targets need shape (T, 12)")
    _require(torch.is_floating_point(targets), "targets must use a floating dtype")
    _require(bool(torch.isfinite(targets).all()), "targets must be finite")
    unwrapped = env.unwrapped
    _require(unwrapped.num_envs == 1, "synthetic generation requires one environment")
    _require(
        abs(float(unwrapped.sim.cfg.dt) - DT_S) <= 1.0e-12,
        "synthetic environment dt is not 0.002 s",
    )
    articulation = unwrapped.scene["robot"]
    device = torch.device(unwrapped.device)
    dtype = articulation.data.joint_pos.dtype
    joint_ids = _joint_ids(articulation.joint_names, device)
    targets_device = targets.detach().to(device=device, dtype=dtype).contiguous()
    bias = torch.tensor(
        [mean.encoder_bias[name] for name in PACE_JOINT_ORDER],
        device=device,
        dtype=dtype,
    )

    if reset_seed is None:
        env.reset()
    else:
        env.reset(seed=reset_seed)
    initial_physical = (targets_device[0] + bias).unsqueeze(0)
    articulation.write_joint_position_to_sim(initial_physical, joint_ids=joint_ids)
    articulation.write_joint_velocity_to_sim(
        torch.zeros_like(initial_physical), joint_ids=joint_ids
    )

    measured = torch.empty_like(targets_device)
    desired = torch.empty_like(targets_device)
    with torch.inference_mode():
        for index in range(targets_device.shape[0]):
            physical = articulation.data.joint_pos[0, joint_ids]
            measured[index] = physical - bias
            action = torch.zeros(env.action_space.shape, device=device, dtype=dtype)
            action[:, joint_ids] = targets_device[index].unsqueeze(0)
            desired[index] = action[0, joint_ids]
            env.step(action)

    return {
        "time": (
            torch.arange(targets.shape[0], dtype=torch.float64) * DT_S
        ).contiguous(),
        "dof_pos": measured.detach().cpu().to(torch.float32).contiguous(),
        "des_dof_pos": desired.detach().cpu().to(torch.float32).contiguous(),
    }


def validate_synthetic_data(data: dict[str, torch.Tensor]) -> None:
    from pace_sim2real.tasks.manager_based.pace.a1_replay import (
        validate_a1_pace_data,
    )

    validate_a1_pace_data(data)
    time = data["time"]
    measured = data["dof_pos"]
    desired = data["des_dof_pos"]
    _require(time.dtype == torch.float64, "synthetic time must use float64")
    _require(
        measured.dtype == desired.dtype == torch.float32,
        "synthetic positions must use float32",
    )
    _require(time.shape == (SAMPLE_COUNT,), "synthetic time must contain 10000 samples")
    _require(
        measured.shape == desired.shape == (SAMPLE_COUNT, 12),
        "synthetic positions must have shape (10000, 12)",
    )
    for name, value in data.items():
        _require(value.device.type == "cpu", f"synthetic {name} must be on CPU")
        _require(value.is_contiguous(), f"synthetic {name} must be contiguous")
        _require(bool(torch.isfinite(value).all()), f"synthetic {name} is non-finite")
    expected_time = torch.arange(SAMPLE_COUNT, dtype=torch.float64) * DT_S
    _require(torch.equal(time, expected_time), "synthetic time grid is not exact")


def tensor_payload_sha256(data: dict[str, torch.Tensor]) -> str:
    validate_synthetic_data(data)
    digest = hashlib.sha256()
    for key in ("time", "dof_pos", "des_dof_pos"):
        value = data[key].detach().cpu().contiguous()
        digest.update(key.encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(
            json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
            + b"\0"
        )
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _atomic_torch_save(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".pt", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(value, temporary, _use_new_zipfile_serialization=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(value: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".json", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
            newline="\n",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def _exclusive_destination_locks(
    destinations: Sequence[Path],
) -> Iterator[None]:
    locks: list[Path] = []
    try:
        for destination in sorted(
            {path.resolve() for path in destinations}, key=str
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            lock_path = destination.parent / f".{destination.name}.a1-pace.lock"
            try:
                descriptor = os.open(
                    lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
                )
            except FileExistsError as exc:
                raise ValueError(
                    f"synthetic destination is locked: {destination}"
                ) from exc
            locks.append(lock_path)
            try:
                os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
            finally:
                os.close(descriptor)
        yield
    finally:
        for lock_path in reversed(locks):
            lock_path.unlink(missing_ok=True)


def _git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _numeric_values(value: Any) -> list[float]:
    values = value.values() if isinstance(value, dict) else (value,)
    return sorted({float(item) for item in values})


def _actuator_control_contract(env_cfg: Any) -> dict[str, Any]:
    actuator = env_cfg.scene.robot.actuators["base_legs"]
    class_type = actuator.class_type
    return {
        "class_type": f"{class_type.__module__}.{class_type.__qualname__}",
        "saturation_effort_nm": float(actuator.saturation_effort),
        "effort_limit_nm": float(actuator.effort_limit),
        "velocity_limit_rad_s": float(actuator.velocity_limit),
        "stiffness_values": _numeric_values(actuator.stiffness),
        "damping_values": _numeric_values(actuator.damping),
        "configured_delay_steps": int(actuator.max_delay),
    }


def _a1_extension_hashes(repository_root: Path) -> dict[str, str]:
    relative_paths = (
        "source/pace_sim2real/pace_sim2real/tasks/manager_based/__init__.py",
        "source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/a1_mean.py",
        "source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/a1_pace_env_cfg.py",
        "source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/a1_replay.py",
    )
    return {
        relative: sha256_path(repository_root / relative)
        for relative in relative_paths
    }


def _runtime_contract(env_cfg: Any) -> dict[str, Any]:
    return {
        "dt_s": float(env_cfg.sim.dt),
        "decimation": int(env_cfg.decimation),
        "data_dir": str(env_cfg.sim2real.data_dir),
        "joint_order": list(env_cfg.sim2real.joint_order),
        "fix_root_link": bool(
            env_cfg.scene.robot.spawn.articulation_props.fix_root_link
        ),
        "root_initial_position": list(env_cfg.scene.robot.init_state.pos),
        "action_scale": float(env_cfg.actions.joint_pos.scale),
        "use_default_offset": bool(env_cfg.actions.joint_pos.use_default_offset),
        "actuator_control": _actuator_control_contract(env_cfg),
    }


def generate(args: argparse.Namespace) -> dict[str, Any]:
    """Run one registered A1 task and atomically publish all artifacts."""

    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    import isaaclab_tasks  # noqa: F401
    import pace_sim2real.tasks  # noqa: F401
    from pace_sim2real.tasks.manager_based.pace.a1_mean import parse_a1_mean
    from pace_sim2real.tasks.manager_based.pace.a1_replay import (
        inject_a1_actuator_before_make,
        make_a1_pace_actuator_cfg,
    )

    output = Path(args.output).resolve()
    manifest_output = Path(args.manifest_output).resolve()
    mean_output = Path(args.mean_output).resolve()
    params_path = Path(args.params).resolve()
    destinations = {output, manifest_output, mean_output}
    _require(len(destinations) == 3, "synthetic output paths must differ")
    _require(params_path not in destinations, "an output path aliases the parameter file")
    generator_path = Path(__file__).resolve()
    generator_hash = sha256_path(generator_path)
    repository_root = generator_path.parents[2]
    extension_hashes = _a1_extension_hashes(repository_root)
    params_hash = sha256_path(params_path)
    vector = load_parameter_vector(params_path, args.delay_steps)
    _require(sha256_path(params_path) == params_hash, "parameter file changed while loading")
    mean = parse_a1_mean(vector)
    excitation = build_excitation_contract(
        seed=args.seed, amplitude_scale=args.amplitude_scale
    )
    targets = build_absolute_targets(
        seed=args.seed,
        num_samples=args.num_samples,
        amplitude_scale=args.amplitude_scale,
    )

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    _require(tuple(env_cfg.sim2real.joint_order) == PACE_JOINT_ORDER, "task joint order mismatch")
    _require(abs(float(env_cfg.sim.dt) - DT_S) <= 1.0e-12, "task dt mismatch")
    inject_a1_actuator_before_make(env_cfg, make_a1_pace_actuator_cfg(mean))
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    try:
        data = collect_synthetic_trajectory(
            env, targets, mean, reset_seed=args.seed
        )
    finally:
        env.close()
    validate_synthetic_data(data)
    _require(
        torch.equal(data["des_dof_pos"], targets),
        "environment changed an absolute target before collection",
    )
    _require(sha256_path(params_path) == params_hash, "parameter file changed during simulation")
    _require(sha256_path(generator_path) == generator_hash, "generator changed during simulation")
    _require(
        _a1_extension_hashes(repository_root) == extension_hashes,
        "A1 extension source changed during simulation",
    )

    _atomic_torch_save(data, output)
    _atomic_torch_save(vector.detach().cpu().contiguous(), mean_output)
    reloaded = torch.load(output, map_location="cpu", weights_only=True)
    validate_synthetic_data(reloaded)
    for key in data:
        _require(torch.equal(data[key], reloaded[key]), f"saved {key} changed")
    reloaded_mean = torch.load(mean_output, map_location="cpu", weights_only=True)
    _require(torch.equal(vector.cpu(), reloaded_mean), "saved known mean changed")

    manifest = {
        "schema_version": "a1_pace_synthetic/v1",
        "scope": "synthetic_software_gate",
        "physical_motion_authorized": False,
        "task": args.task,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "dt_s": DT_S,
        "delay_steps": args.delay_steps,
        "pace_joint_order": list(PACE_JOINT_ORDER),
        "physical_parameter_vector": [float(value) for value in vector],
        "parameter_file_sha256": params_hash,
        "target_tensor_sha256": hashlib.sha256(
            targets.numpy().tobytes(order="C")
        ).hexdigest(),
        "excitation": excitation,
        "tensor_payload_sha256": tensor_payload_sha256(data),
        "output_sha256": sha256_path(output),
        "known_mean_sha256": sha256_path(mean_output),
        "generator_sha256": generator_hash,
        "a1_extension_source_sha256": extension_hashes,
        "exact_command": shlex.join(sys.argv),
        "environment": _runtime_contract(env_cfg),
        "revisions": {
            "pace_sim2real": _git_revision(repository_root),
            "isaaclab": _git_revision(Path(args.isaaclab_root).resolve()),
        },
        "acceptance": {"accepted": True, "status": "PASS"},
    }
    _require(sha256_path(params_path) == params_hash, "parameter file changed before manifest publish")
    _require(sha256_path(generator_path) == generator_hash, "generator changed before manifest publish")
    _require(
        _a1_extension_hashes(repository_root) == extension_hashes,
        "A1 extension source changed before manifest publish",
    )
    _atomic_json(manifest, manifest_output)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a known-parameter full-length A1 PACE dataset"
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--delay-steps", type=int, choices=sorted(ALLOWED_DELAYS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--amplitude-scale", type=float, choices=(0.5, 1.0), default=1.0
    )
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--mean-output", type=Path, required=True)
    parser.add_argument(
        "--isaaclab-root", type=Path, default=Path("/home/changba01/IsaacLab")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from isaaclab.app import AppLauncher

    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        with _exclusive_destination_locks(
            (Path(args.output), Path(args.manifest_output), Path(args.mean_output))
        ):
            manifest = generate(args)
        print(json.dumps(manifest, sort_keys=True))
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
