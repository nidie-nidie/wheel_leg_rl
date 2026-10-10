from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .trace_schema import (
    build_file_hashes,
    load_npz,
    save_npz,
    sha256_array,
    sha256_file,
    write_json,
)


def evaluate_actor_cpu(
    *,
    actor_path: str | Path,
    input_path: str | Path,
    output_directory: str | Path,
    input_key: str = "actor_obs_policy",
) -> dict[str, Any]:
    actor = Path(actor_path).resolve()
    actor_input_path = Path(input_path).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = load_npz(actor_input_path)
    if input_key not in inputs:
        raise KeyError(f"Actor input key {input_key!r} is missing")
    actor_input = np.asarray(inputs[input_key])
    if actor_input.shape != (25,) or actor_input.dtype != np.float32 or not np.isfinite(actor_input).all():
        raise ValueError(f"Actor input must be finite float32[25], got {actor_input.dtype} {actor_input.shape}")
    module = torch.jit.load(str(actor), map_location="cpu").eval()
    with torch.inference_mode():
        result = module(torch.from_numpy(actor_input.copy()).unsqueeze(0))
    actor_output = result.detach().cpu().numpy().reshape(-1)
    if actor_output.shape != (6,) or actor_output.dtype != np.float32 or not np.isfinite(actor_output).all():
        raise ValueError(f"Actor output must be finite float32[6], got {actor_output.dtype} {actor_output.shape}")
    output_path = save_npz(output / "actor_output.npz", {"actor_output": actor_output})
    metadata = {
        "schema_version": "ActorCpuEvaluationV1",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "device": "cpu",
        "dtype": "float32",
        "actor_sha256": sha256_file(actor),
        "input_file_sha256": sha256_file(actor_input_path),
        "input_array_sha256": sha256_array(actor_input),
        "output_array_sha256": sha256_array(actor_output),
    }
    write_json(output / "metadata.json", metadata)
    write_json(
        output / "file_hashes.json",
        build_file_hashes(
            {
                "actor": actor,
                "input": actor_input_path,
                "actor_output": output_path,
                "metadata": output / "metadata.json",
                "evaluator": Path(__file__),
            }
        ),
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen TorchScript Actor on CPU.")
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-key", default="actor_obs_policy")
    args = parser.parse_args()
    metadata = evaluate_actor_cpu(
        actor_path=args.actor,
        input_path=args.input,
        output_directory=args.output,
        input_key=args.input_key,
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
