from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


EXPECTED_ISAAC_LAB_COMMIT = "37ddf626871758333d6ed89cf64ad702aef127d0"
EXPECTED_VERSIONS = {
    "isaacsim": "5.1.0.0",
    "rsl-rl-lib": "3.1.2",
    "tensordict": "0.14.2",
}
EXPECTED_TORCH = "2.7.0+cu128"


def _distribution_version(name: str) -> str:
    return importlib.metadata.version(name)


def validate_runtime(project_root: Path, *, device: str = "cuda:0") -> dict[str, Any]:
    import isaaclab
    import isaaclab_rl
    import rsl_rl
    import tensordict

    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Python must be 3.11, got {sys.version.split()[0]}")
    if torch.__version__ != EXPECTED_TORCH:
        raise RuntimeError(f"Unexpected torch build: {torch.__version__}")
    for distribution, expected in EXPECTED_VERSIONS.items():
        actual = _distribution_version(distribution)
        if actual != expected:
            raise RuntimeError(f"{distribution} must be exactly {expected}, got {actual}")

    isaac_lab_root = project_root / "dependencies" / "IsaacLab-v2.3.2"
    commit = subprocess.check_output(
        ["git", "-C", str(isaac_lab_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_ISAAC_LAB_COMMIT:
        raise RuntimeError(f"Isaac Lab commit mismatch: {commit}")

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise RuntimeError(f"Phase 1 requires a CUDA device, got {device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    device_index = torch_device.index if torch_device.index is not None else torch.cuda.current_device()
    probe = torch.arange(1024, dtype=torch.float32, device=torch_device)
    probe_result = torch.sum(probe * probe)
    torch.cuda.synchronize(torch_device)
    if not torch.isfinite(probe_result) or probe_result.item() <= 0.0:
        raise RuntimeError("CUDA execution probe returned an invalid result")

    capability = torch.cuda.get_device_capability(device_index)
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(device_index),
        "cuda_capability": [capability[0], capability[1]],
        "torch_arch_list": torch.cuda.get_arch_list(),
        "isaac_sim": _distribution_version("isaacsim"),
        "isaaclab_path": str(Path(isaaclab.__file__).resolve()),
        "isaaclab_rl_path": str(Path(isaaclab_rl.__file__).resolve()),
        "rsl_rl_path": str(Path(rsl_rl.__file__).resolve()),
        "rsl_rl": _distribution_version("rsl-rl-lib"),
        "tensordict_path": str(Path(tensordict.__file__).resolve()),
        "tensordict": _distribution_version("tensordict"),
        "isaac_lab_commit": commit,
        "cuda_probe": float(probe_result.item()),
    }
