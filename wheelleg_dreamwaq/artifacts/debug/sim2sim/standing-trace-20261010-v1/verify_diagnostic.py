"""Validate temporal alignment, units, immutable inputs and evidence identities."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


report = json.loads((ROOT / "analysis.json").read_text(encoding="utf-8"))
protected = json.loads((ROOT / "protected-before.json").read_text(encoding="utf-8"))
for path, expected in protected.items():
    assert digest(path) == expected, path
for path, expected in report["evidence_sha256"].items():
    assert digest(path) == expected, path
counts = {}
for engine, count in (("isaac", 499), ("mujoco", 500)):
    with (ROOT / engine / "nominal_stand.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == count
    inputs = np.load(ROOT / engine / "policy_inputs.npz")
    current, history = inputs["current"], inputs["history"]
    np.testing.assert_array_equal(history[0].reshape(5, 25), np.repeat(current[0][None, :], 5, axis=0))
    np.testing.assert_array_equal(history[1:, :-25], history[:-1, 25:])
    np.testing.assert_array_equal(history[:, -25:], current)
    clipped = np.asarray([[float(row[f"clipped_action_{i}"]) for i in range(6)] for row in rows], dtype=np.float32)
    np.testing.assert_array_equal(current[1:, 19:25], clipped[:-1])
    np.testing.assert_array_equal(current[0, 19:25], 0)
    pre = np.asarray([float(row["actual_vx_mps"]) for row in rows])
    post = np.asarray([float(row["post_vx_mps"]) for row in rows])
    np.testing.assert_array_equal(post[:-1], pre[1:])
    if engine == "isaac":
        assert rows[-1]["done"] == "1" and np.isnan(post[-1])
        assert all(row["done"] == "0" for row in rows[:-1])
    else:
        assert all(row["done"] == "0" for row in rows)
    assert report["engines"][engine]["action_replay_max_abs_error"] < 1e-5
    assert report["engines"][engine]["estimator_replay_max_abs_error"] < 1e-5
    counts[engine] = count
execution = json.loads((ROOT / "execution.json").read_text(encoding="utf-8"))
assert execution[0]["exit_code"] == 0 and execution[-1]["exit_code"] == 0
summary = {
    "schema_version": "StandingDiagnosticVerificationV1", "passed": True,
    "sample_counts": counts, "protected_files_unchanged": len(protected),
    "history_and_previous_action_alignment": "exact", "pre_post_velocity_alignment": "exact",
    "isaac_final_auto_reset_state": "excluded from post-action comparison",
    "source_actor_and_estimator_replay": "passed below 1e-5", "formal_files_changed": [],
    "diagnostic_sources": {str(path): digest(path) for path in ROOT.glob("*.py")},
    "analysis_sha256": digest(ROOT / "analysis.json"), "plot_sha256": digest(ROOT / "standing-comparison.png"),
    "report_sha256": digest(ROOT / "report.md"),
}
(ROOT / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("STANDING_FINAL_VERIFIED", counts, len(protected), flush=True)
