import json
from pathlib import Path

import pytest
import torch

from recovery.bounded_actor_critic import BoundedActorCritic
from recovery.compact_metrics import CompactStandingMetrics
from recovery.contracts import file_sha256, stable_hash
from recovery.independent_checkpoint import IDENTITY_HASH_KEYS, load_independent_checkpoint, save_independent_checkpoint
from test_bounded_actor_critic import REFERENCE, actual_observations


def test_sync_log_barrier_resume_tail_validation_and_finalization(tmp_path):
    metrics = CompactStandingMetrics(tmp_path)
    metrics.append_episodes([{"reward": 20., "length": 150, "source": "standing", "success": True, "reason": "success"}], 1)
    metrics.append_update({"update": 1, "mean_step_reward": 0.1})
    state = metrics.snapshot()
    assert state["submitted"] == state["consumed"] == 2 and state["pending"] == 0
    metrics.close("interface_segment")
    resumed = CompactStandingMetrics(tmp_path, state)
    resumed.append_update({"update": 2, "mean_step_reward": 0.2})
    final = resumed.close()
    assert final["finalized"] and final["reason"] == "training_run_end"
    assert final["worker_state"] == "synchronous" and final["submitted"] == final["consumed"] == 3
    assert json.loads((tmp_path / "metrics.json").read_text())["logging_barrier"] == final
    with pytest.raises(ValueError, match="长度"):
        CompactStandingMetrics(tmp_path, state)


def test_real_model_checkpoint_identity_hash_and_rng_restore(tmp_path):
    actor, critic, median = actual_observations()
    policy = BoundedActorCritic(median)
    evidence_hash = file_sha256(REFERENCE / "manifest.json")
    identity = {"schema_version": "RecoveryIndependentStandV1", **{key: evidence_hash for key in IDENTITY_HASH_KEYS}}
    identity["source_sha256"] = file_sha256(Path(__file__).parents[1] / "bounded_actor_critic.py")
    identity["reference_sha256"] = file_sha256(REFERENCE / "recovery_contact_paired_reference.npz")
    metrics = CompactStandingMetrics(tmp_path / "logs")
    payload = {"completed_updates": 0, "policy": policy.state_dict(), "logging_barrier": metrics.snapshot()}
    path = tmp_path / "model_0.pt"
    index = save_independent_checkpoint(path, payload, identity)
    assert index["checkpoint_sha256"] == file_sha256(path)
    expected = torch.rand(7)
    restored = load_independent_checkpoint(path, identity)
    assert torch.equal(torch.rand(7), expected)
    policy.load_state_dict(restored["policy"])
    assert torch.isfinite(policy.sample(actor, critic).action).all()
    changed = dict(identity, reference_sha256=stable_hash({"different_reference": True}))
    with pytest.raises(ValueError, match="身份"):
        load_independent_checkpoint(path, changed)
    with pytest.raises(ValueError, match="独立站立"):
        load_independent_checkpoint(path, {**identity, "schema_version": "RecoveryResolvedConfigV3"})
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="哈希"):
        load_independent_checkpoint(path, identity)
    metrics.close("component_verification")
