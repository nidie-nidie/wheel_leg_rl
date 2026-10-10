from __future__ import annotations

import json

import pytest
import torch
from tensordict import TensorDict

from wheelleg_dreamwaq.algorithms.dreamwaq.actor_critic import DreamWaQActorCritic
from wheelleg_dreamwaq.deployment.dreamwaq_inference import DreamWaQInferenceActorV1
from wheelleg_dreamwaq.deployment.manifest import (
    EXPORT_ARTIFACT_NAMES,
    stable_manifest_hash,
    export_dreamwaq_inference_package,
)
from wheelleg_dreamwaq.deployment.observation_adapter import FrameMajorHistoryV1
from wheelleg_dreamwaq.schemas.dreamwaq_manifest import DREAMWAQ_EXPORT_CONTRACT_VERSION


def _policy() -> DreamWaQActorCritic:
    obs = TensorDict(
        {
            "policy": torch.zeros(2, 25),
            "policy_history": torch.zeros(2, 125),
            "critic": torch.zeros(2, 41),
        },
        batch_size=[2],
    )
    return DreamWaQActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, 6).eval()


def test_frame_major_history_reset_append_done_and_flatten() -> None:
    first = torch.arange(50, dtype=torch.float32).reshape(2, 25)
    runtime = FrameMajorHistoryV1(first)
    assert torch.equal(runtime.frames(), first[:, None, :].expand(-1, 5, -1))

    next_obs = first + 100.0
    runtime.append(next_obs, torch.tensor([False, True]))
    frames = runtime.frames()
    assert torch.equal(frames[0, -1], next_obs[0])
    assert torch.equal(frames[0, -2], first[0])
    assert torch.equal(frames[1], next_obs[1].expand(5, -1))
    assert torch.equal(runtime.flat(), frames.reshape(2, 125))


def test_export_package_is_exact_scripted_dynamic_batch_and_golden_replay(tmp_path) -> None:
    actor = DreamWaQInferenceActorV1.from_policy(_policy())
    output = tmp_path / "export"
    summary = export_dreamwaq_inference_package(
        actor,
        output,
        {
            "dreamwaq_export_contract_version": DREAMWAQ_EXPORT_CONTRACT_VERSION,
            "dreamwaq_export_contract_hash": "A" * 64,
            "base_task_contract_hash": "B" * 64,
            "dreamwaq_algorithm_contract_hash": "C" * 64,
        },
    )

    assert {path.name for path in output.iterdir()} == EXPORT_ARTIFACT_NAMES
    assert summary["verification_max_abs_error"] <= 1.0e-7
    manifest = json.loads((output / "policy_manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_hash"] == stable_manifest_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    golden = torch.load(output / "golden_vectors.pt", map_location="cpu", weights_only=False)
    scripted = torch.jit.load(str(output / "actor.ts"), map_location="cpu")
    with torch.inference_mode():
        actual = scripted(golden["history"])
        one = scripted(golden["history"][:1])
    assert torch.max(torch.abs(actual - golden["expected_action_mean"])).item() <= 1.0e-7
    assert one.shape == (1, 6)


def test_export_refuses_nonempty_directory(tmp_path) -> None:
    output = tmp_path / "export"
    output.mkdir()
    (output / "stale.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError):
        export_dreamwaq_inference_package(
            DreamWaQInferenceActorV1.from_policy(_policy()),
            output,
            {"dreamwaq_export_contract_version": DREAMWAQ_EXPORT_CONTRACT_VERSION},
        )
