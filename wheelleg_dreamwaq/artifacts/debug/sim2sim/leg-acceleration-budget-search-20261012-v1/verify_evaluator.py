from pathlib import Path
import json
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from evaluate_leg_acceleration_mujoco import _step_or_failure, _window_stats, _write_csv

class NonfiniteRuntime:
    def step(self, action, command):
        raise FloatingPointError("synthetic nonfinite metrics")

result, failure = _step_or_failure(NonfiniteRuntime(), None, None)
assert result is None and failure == "non_finite_state"
short = _window_stats(np.zeros((49, 4)), 50)
assert short == {"window_count": 0, "rms_mean": [None]*4, "rms_p90": [None]*4}
json.dumps(short, allow_nan=False)
full = _window_stats(np.ones((100, 4))*3, 50)
assert full == {"window_count": 2, "rms_mean": [3.]*4, "rms_p90": [3.]*4}
out=Path(__file__).with_name("failure-record-smoke.csv")
_write_csv(out, [{"tick":1,"vx_mps":0.,"survived":1,"failure_reason":""},
                 {"tick":2,"survived":0,"failure_reason":failure}])
assert failure in out.read_text()
print("evaluator nonfinite, incomplete window, RMS, mixed-row CSV checks passed")
