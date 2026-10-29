from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from recovery.checkpoint import checkpoint_index_path, load_checkpoint, prepare_compatible_import, save_checkpoint, validate_geometry_barrier, _validate_performance_report
from recovery.contracts import ARTIFACT_ROOT, file_sha256, source_sha256, stable_hash
from recovery.geometry_archive import prefix_sha256


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        root = ARTIFACT_ROOT / "test_work"
        root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "model_1.pt"
        self.identity = {"component": "checkpoint_optimizer_roundtrip", "device": "cpu", "num_envs": 1}
        self.model = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.inputs = torch.tensor([[0.3, -0.8], [-0.4, 0.2]], dtype=torch.float32)
        self.targets = torch.tensor([[0.7], [-0.2]], dtype=torch.float32)
        self.update(self.model, self.optimizer)

    def update(self, model, optimizer):
        optimizer.zero_grad()
        loss = (model(self.inputs) - self.targets).square().mean()
        loss.backward()
        optimizer.step()

    def payload(self):
        return {
            "model_state_dict": deepcopy(self.model.state_dict()),
            "optimizer_state_dict": deepcopy(self.optimizer.state_dict()),
            "completed_updates": 1,
            "learning_rate": 0.001,
            "environment_state": {"component_inputs": self.inputs, "component_targets": self.targets},
        }

    def test_roundtrip_reproduces_next_real_optimizer_update(self):
        index = save_checkpoint(self.path, self.payload(), self.identity)
        self.assertEqual(index["checkpoint_sha256"], file_sha256(self.path))
        self.assertEqual(index["completed_updates"], 1)
        self.assertNotIn("checkpoint_sha256", torch.load(self.path, weights_only=False))
        self.update(self.model, self.optimizer)
        expected = deepcopy(self.model.state_dict())
        recovered_model = torch.nn.Linear(2, 1)
        recovered_optimizer = torch.optim.Adam(recovered_model.parameters(), lr=0.9)
        recovered = load_checkpoint(self.path, self.identity)
        recovered_model.load_state_dict(recovered["model_state_dict"])
        recovered_optimizer.load_state_dict(recovered["optimizer_state_dict"])
        self.update(recovered_model, recovered_optimizer)
        self.assertTrue(all(torch.equal(expected[key], value) for key, value in recovered_model.state_dict().items()))
        self.assertEqual(recovered_optimizer.param_groups[0]["lr"], 0.001)

    def test_python_numpy_torch_and_cuda_rng_restore_exactly(self):
        random.seed(177)
        np.random.seed(178)
        torch.manual_seed(179)
        save_checkpoint(self.path, self.payload(), self.identity)
        expected = (random.random(), np.random.random(4), torch.rand(4))
        expected_cuda = torch.rand(4, device="cuda") if torch.cuda.is_available() else None
        random.seed(280)
        np.random.seed(281)
        torch.manual_seed(282)
        load_checkpoint(self.path, self.identity)
        self.assertEqual(random.random(), expected[0])
        np.testing.assert_array_equal(np.random.random(4), expected[1])
        self.assertTrue(torch.equal(torch.rand(4), expected[2]))
        if expected_cuda is not None:
            self.assertTrue(torch.equal(torch.rand(4, device="cuda"), expected_cuda))

    def test_read_without_rng_restore_preserves_current_sequence(self):
        save_checkpoint(self.path, self.payload(), self.identity)
        torch.manual_seed(991)
        before = torch.get_rng_state().clone()
        load_checkpoint(self.path, self.identity, restore_rng=False)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_complete_identity_mismatch_and_unknown_index_fields_are_rejected(self):
        save_checkpoint(self.path, self.payload(), self.identity)
        with self.assertRaises(ValueError):
            load_checkpoint(self.path, {**self.identity, "device": "cuda:0"})
        with self.assertRaises(ValueError):
            load_checkpoint(self.path, {"component": self.identity["component"]})
        path = checkpoint_index_path(self.path)
        index = json.loads(path.read_text(encoding="utf-8"))
        index["extra"] = True
        path.write_text(json.dumps(index), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_checkpoint(self.path, self.identity)

    def test_existing_checkpoint_or_index_cannot_be_overwritten(self):
        save_checkpoint(self.path, self.payload(), self.identity)
        with self.assertRaises(FileExistsError):
            save_checkpoint(self.path, self.payload(), self.identity)
        orphan = Path(self.directory.name) / "orphan.pt"
        checkpoint_index_path(orphan).write_text("{}", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            save_checkpoint(orphan, self.payload(), self.identity)
        self.assertFalse(orphan.exists())

    def test_changed_checkpoint_bytes_rejected_before_deserialization(self):
        save_checkpoint(self.path, self.payload(), self.identity)
        with self.path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaises(ValueError):
            load_checkpoint(self.path, self.identity)

    def test_nonfinite_parameters_or_incomplete_progress_cannot_be_saved(self):
        payload = self.payload()
        payload["model_state_dict"]["weight"][0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            save_checkpoint(self.path, payload, self.identity)
        payload = self.payload()
        payload["completed_updates"] = 0
        with self.assertRaises(ValueError):
            save_checkpoint(self.path, payload, self.identity)
        self.assertFalse(self.path.exists())

    def test_geometry_barrier_accepts_later_tail_and_rejects_changed_saved_prefix_or_pending(self):
        source = Path(self.directory.name) / "geometry.bin"
        source.write_bytes(b"actual-geometry-record\n")
        cutoff = source.stat().st_size
        barrier = {"schema_version": "RecoveryGeometryBarrierV1", "all_requests_acknowledged": True,
            "pending_requests": 0, "submitted_sequence": 8, "archived_diagnostic_sequence": 7,
            "archive_path": str(source), "byte_cutoff": cutoff, "prefix_sha256": prefix_sha256(source, cutoff)}
        with source.open("ab") as stream:
            stream.write(b"later-live-record\n")
        validate_geometry_barrier({"geometry_barrier": barrier})
        for key, value in (("pending_requests", 1), ("all_requests_acknowledged", False), ("archived_diagnostic_sequence", 9)):
            with self.assertRaises(ValueError):
                validate_geometry_barrier({"geometry_barrier": {**barrier, key: value}})
        with source.open("r+b") as stream:
            stream.write(b"!")
        with self.assertRaises(ValueError):
            validate_geometry_barrier({"geometry_barrier": barrier})


class CompatibleImportRejectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = ARTIFACT_ROOT / "bootstrap-6144-20261022/run/model_50.pt"
        if not cls.parent.exists():
            raise unittest.SkipTest("本地真实6144环境checkpoint50尚未存在")
        cls.parent_identity = json.loads((cls.parent.parent / "run_manifest.json").read_text(encoding="utf-8"))["identity"]
        cls.parent_config = json.loads((cls.parent.parent / "resolved_config.json").read_text(encoding="utf-8"))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ARTIFACT_ROOT / "test_work")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config = deepcopy(self.parent_config)
        self.config["environment"]["run_dir"] = str(self.directory / "new_run")
        self.identity = {**self.parent_identity, "training_config_sha256": stable_hash(self.config), "source_sha256": source_sha256()}
        self.evidence = {"schema_version": "RecoveryPerformanceEquivalenceEvidenceV1",
            "source_sha256": self.identity["source_sha256"], "parent_identity_sha256": stable_hash(self.parent_identity),
            "new_identity_sha256": stable_hash(self.identity), "checks": {}}

    def reject(self, expression):
        path = self.directory / "deliberately_incomplete_evidence.json"
        path.write_text(json.dumps(self.evidence), encoding="utf-8")
        destination = self.directory / "import_manifest.json"
        with self.assertRaisesRegex(ValueError, expression):
            prepare_compatible_import(self.parent, self.parent_identity, self.identity, path, destination, new_config=self.config)
        self.assertFalse(destination.exists())

    def test_real_checkpoint50_rejects_evidence_for_other_source_parent_or_new_identity(self):
        for key in ("source_sha256", "parent_identity_sha256", "new_identity_sha256"):
            with self.subTest(key=key):
                saved = self.evidence[key]
                self.evidence[key] = "0" * 64
                self.reject("未绑定父身份")
                self.evidence[key] = saved

    def test_real_checkpoint50_rejects_incomplete_actual_interface_evidence(self):
        self.reject("缺少完整几何")

    def test_actual_early_interface_report_rejects_changed_source_and_wrong_scale(self):
        path = ARTIFACT_ROOT / "gpu-optimization-20261023/interface4-debug3/run/training_verification.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "未绑定当前受验源码"):
            _validate_performance_report("interface4", path, self.identity, self.config, set())
        with self.assertRaisesRegex(ValueError, "规模或恢复更新验收不完整"):
            _validate_performance_report("interface256", path, report["identity"], self.config, set())

    def test_real_checkpoint50_rejects_physical_config_change_and_missing_identity(self):
        self.config["environment"]["num_envs"] -= 1
        self.identity["training_config_sha256"] = stable_hash(self.config)
        self.reject("改变了训练物理")
        self.config = deepcopy(self.parent_config)
        self.config["environment"]["run_dir"] = str(self.directory / "new_run")
        self.identity["training_config_sha256"] = stable_hash(self.config)
        del self.identity["isaac_contact_model_sha256"]
        self.reject("缺少完整固定物理身份")


if __name__ == "__main__":
    unittest.main()
