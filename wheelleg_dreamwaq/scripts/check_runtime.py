from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from wheelleg_dreamwaq.training.runtime import validate_runtime


def main() -> None:
    info = validate_runtime(PROJECT_ROOT)
    capability = info["cuda_capability"]
    print(f"python={info['python']}")
    print(f"torch={info['torch']} cuda_runtime={info['torch_cuda_runtime']}")
    print(f"gpu={info['cuda_device_name']} capability={capability[0]}.{capability[1]}")
    print(f"cuda_probe={info['cuda_probe']}")
    print(f"isaacsim={info['isaac_sim']}")
    print(f"isaaclab={info['isaaclab_path']}")
    print(f"isaaclab_rl={info['isaaclab_rl_path']}")
    print(f"rsl_rl={info['rsl_rl_path']} version={info['rsl_rl']}")
    print(f"tensordict={info['tensordict_path']} version={info['tensordict']}")
    print(f"torch_arch_list={info['torch_arch_list']}")
    print(f"isaaclab_commit={info['isaac_lab_commit']}")


if __name__ == "__main__":
    main()
