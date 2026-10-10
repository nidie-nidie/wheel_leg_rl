from __future__ import annotations

import json
import hashlib

import torch

import pytest

from conftest import run_project_script


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["formal", "randomized", "debug"])
def test_reset_observation_coherence(tmp_path, kind):
    generated = tmp_path / kind
    for output, extra in ((generated, []), (tmp_path / f"{kind}-loaded", [
        "--reset-cache", str(generated / "reset-cache.pt"),
    ])):
        run_project_script([
            "tests/integration/probes/reset_observation_coherence.py", "--kind", kind,
            "--output", str(output), "--headless", *extra,
        ], timeout=900)
        report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert report["passed"]
        assert report["checks"] and all(check["passed"] for check in report["checks"])
        assert report["kind"] == kind
        evidence_path = output / "evidence.pt"
        assert hashlib.sha256(evidence_path.read_bytes()).hexdigest().upper() == report["evidence_sha256"]
        evidence = torch.load(evidence_path, map_location="cpu", weights_only=False)
        by_name = {item["name"]: item for item in evidence}
        assert set(by_name) == {"constructor", "repeat0", "repeat1", "repeat2", "repeat3", "partial", "timeout", "failure"}
        if kind == "debug":
            for name in ("constructor", "repeat0", "repeat1", "repeat2", "repeat3"):
                phases = by_name[name]["debug_reset_phases"]
                assert {"reset_written_pre_forward", "reset_forwarded_post_forward",
                        "reset_returned_actor_obs_policy"} <= set(phases)
