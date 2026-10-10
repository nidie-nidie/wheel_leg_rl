"""Reproducible SHA256 and source diff audit against the pre-edit snapshots."""
from pathlib import Path
import hashlib
import json
import difflib


ROOT = Path(__file__).resolve().parents[5]
ARTIFACT = Path(__file__).resolve().parent
NEW_SOURCE_FILES = (
    "wheelleg_dreamwaq/scripts/check_unrestricted_dynamics.py",
    "wheelleg_dreamwaq/sim2sim/mujoco/wheelleg_mujoco/physics.py",
    "wheelleg_dreamwaq/sim2sim/mujoco/tests/test_physics_contract.py",
    "wheelleg_dreamwaq/tests/integration/test_unrestricted_dynamics.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    before = json.loads((ARTIFACT / "before.json").read_text(encoding="utf-8"))
    protected = {rel: {"before": old, "after": sha(ROOT / rel)} for rel, old in before["protected"].items()}
    assert all(record["before"] == record["after"] for record in protected.values())
    changes, diffs = [], []
    for rel in (*before["sources"], *NEW_SOURCE_FILES):
        path = ROOT / rel
        old_sha = before["sources"].get(rel)
        new_sha = sha(path) if path.exists() else None
        if old_sha == new_sha:
            continue
        old = (ARTIFACT / "source-before" / rel).read_text(encoding="utf-8").splitlines(keepends=True) if old_sha else []
        new = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.exists() else []
        diff = list(difflib.unified_diff(old, new, fromfile="before/" + rel, tofile="after/" + rel))
        changes.append({
            "file": rel, "status": "deleted" if not path.exists() else "modified" if old_sha else "added",
            "added_lines": sum(line.startswith("+") and not line.startswith("+++") for line in diff),
            "removed_lines": sum(line.startswith("-") and not line.startswith("---") for line in diff),
            "before_sha256": old_sha, "after_sha256": new_sha,
        })
        diffs.extend(diff)
    rel = "wheelleg_dreamwaq/sim2sim/mujoco/model_manifest.json"
    old = json.loads((ARTIFACT / "source-before" / rel).read_text(encoding="utf-8"))
    new = json.loads((ROOT / rel).read_text(encoding="utf-8"))
    assert {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)} == {"adapters"}
    assert old["adapters"]["observation"] == new["adapters"]["observation"]
    assert {key for key in old["adapters"]["action"].keys() | new["adapters"]["action"].keys()
            if old["adapters"]["action"].get(key) != new["adapters"]["action"].get(key)} == {"version", "implementation_sha256"}
    report = {
        "baseline_source_count": len(before["sources"]), "protected_count": len(protected),
        "protected_unchanged": True, "protected": protected, "changes": changes,
        "model_manifest_only_action_adapter_version_and_hash_changed": True,
        "architecture_sha256": sha(ROOT / "docs/2026-10-03-wheelleg-dreamwaq-architecture.md"),
    }
    (ARTIFACT / "change-audit.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (ARTIFACT / "changes.diff").write_text("".join(diffs), encoding="utf-8")
    print("AUDIT_VERIFIED", len(protected), "protected files unchanged;", len(changes), "source changes")
    print("architecture_sha256", report["architecture_sha256"])


if __name__ == "__main__":
    main()
