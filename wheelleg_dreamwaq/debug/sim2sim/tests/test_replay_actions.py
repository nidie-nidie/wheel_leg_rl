from __future__ import annotations

import numpy as np

from debug.sim2sim.extract_replay_actions import extract_replay_actions
from debug.sim2sim.trace_schema import load_npz, save_npz


def test_extract_replay_actions_preserves_clipped_sequence(tmp_path) -> None:
    source = tmp_path / "control_trace.npz"
    output = tmp_path / "replay.npz"
    actions = np.arange(18, dtype=np.float32).reshape(3, 6)
    save_npz(source, {"action_clipped": actions})
    metadata = extract_replay_actions(source, output)
    np.testing.assert_array_equal(load_npz(output)["isaac_policy_replay"], actions)
    assert metadata["control_ticks"] == 3
    assert output.with_suffix(".json").is_file()
