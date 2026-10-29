from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

from recovery.checkpoint import load_checkpoint, save_checkpoint, validate_guidance_checkpoint
from recovery.contracts import (
    ARTIFACT_ROOT, GuidanceConfigV1, GuidanceTrainingContractV1, RecoveryContractV2,
    default_guidance_resolved_config, default_resolved_config, file_sha256,
    source_sha256, stable_hash, validate_resolved_config,
)
from recovery.export import export_actor, golden_inputs_from_pose_bank, load_export
from recovery.guidance_networks import GuidanceCostCritic
from recovery.guidance_ppo import GuidancePPO
from recovery.pose_bank import PoseBank, source_pose_records


class GuidanceContractTests(unittest.TestCase):
    def test_v3_validation_preserves_original_v2_dictionary(self):
        original = default_resolved_config()
        original_hash = stable_hash(original)
        config = default_guidance_resolved_config()
        before = deepcopy(config)
        validate_resolved_config(config)
        self.assertEqual(config, before)
        self.assertEqual(stable_hash(default_resolved_config()), original_hash)
        self.assertNotIn("guidance", original)
        self.assertEqual(config["contract"], original["contract"])
        self.assertEqual(config["training_contract"]["physical_contract_sha256"], RecoveryContractV2().sha256)

    def test_guidance_config_has_complete_strict_fields(self):
        self.assertEqual(GuidanceConfigV1().end_control_steps, 72000)
        self.assertEqual(GuidanceConfigV1.from_dict(asdict(GuidanceConfigV1())), GuidanceConfigV1())
        for key, value in (("mu", float("inf")), ("max_force_body_weights", -1.0),
                           ("beta_force", True), ("end_updates", True), ("force_enabled", 1),
                           ("torque_enabled", True), ("max_torque_nm", 1.0),
                           ("rollout_steps", 25), ("height_domain", "raw")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                GuidanceConfigV1.from_dict({**asdict(GuidanceConfigV1()), key: value})
        fields = asdict(GuidanceConfigV1())
        fields.pop("mu")
        with self.assertRaises(ValueError):
            GuidanceConfigV1.from_dict(fields)
        self.assertEqual(GuidanceConfigV1(beta_force=0).beta_force, 0)

    def test_training_contract_rejects_network_or_formula_changes(self):
        for key, value in (("actor_input_dim", 110), ("critic_input_dim", 34),
                           ("deployment_alpha", 0.5), ("formula_version", "unknown"),
                           ("actor_input_dim", 111.0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                GuidanceTrainingContractV1.from_dict({**asdict(GuidanceTrainingContractV1()), key: value})

    def test_v3_rejects_missing_fields_or_mismatched_ppo(self):
        for key in ("guidance", "training_contract"):
            config = default_guidance_resolved_config()
            del config[key]
            with self.assertRaises(ValueError):
                validate_resolved_config(config)
        config = default_guidance_resolved_config()
        config["guidance"]["gae_lambda"] = 0.90
        with self.assertRaises(ValueError):
            validate_resolved_config(config)

    def test_old_v2_rejects_guidance_fields(self):
        config = default_resolved_config()
        config["guidance"] = asdict(GuidanceConfigV1())
        with self.assertRaises(ValueError):
            validate_resolved_config(config)


class GuidanceCheckpointExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank_path = ARTIFACT_ROOT / "bootstrap_pose_bank_v1" / "pose_bank.json"
        cls.bank = PoseBank.load(cls.bank_path)
        cls.inputs, cls.provenance = golden_inputs_from_pose_bank(cls.bank_path)
        cls.inputs = cls.inputs[:32].contiguous()
        cls.provenance["record_ids"] = cls.provenance["record_ids"][:32]
        cls.config = default_guidance_resolved_config()
        cls.config["environment"]["device"] = "cpu"
        cls.config["environment"]["pose_bank_path"] = str(cls.bank_path)
        cls.identity = {"source_sha256": source_sha256(), "training_config_sha256": stable_hash(cls.config),
                        "pose_bank_sha256": cls.bank.sha256,
                        "source_pose_records_sha256": stable_hash(source_pose_records(cls.bank)),
                        "training_contract_sha256": GuidanceTrainingContractV1().sha256,
                        "guidance_config_sha256": GuidanceConfigV1().sha256}
        alpha = torch.ones(32, 1)
        policy_inputs = torch.cat((cls.inputs, alpha), dim=1)
        # 此处只验证网络、优化器和文件组件，35维输入由真实历史采样构成，不作为环境critic证据。
        critic_inputs = torch.cat((cls.inputs[:, :34], alpha), dim=1)
        observations = TensorDict({"policy": policy_inputs, "critic": critic_inputs}, batch_size=[32])
        policy = ActorCritic(observations, {"policy": ["policy"], "critic": ["critic"]}, 6,
                             actor_hidden_dims=[256, 128, 64], critic_hidden_dims=[256, 128, 64],
                             activation="elu", init_noise_std=0.5, noise_std_type="log")
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
        targets = torch.tensor([record.prior_actions[-1] for record in cls.bank.records[:32]], dtype=torch.float32)
        optimizer.zero_grad()
        (policy.actor(policy_inputs) - targets).square().mean().backward()
        optimizer.step()
        cls.cost = GuidanceCostCritic()
        cls.cost_optimizer = torch.optim.Adam(cls.cost.parameters(), lr=0.001)
        cls.critic_inputs = critic_inputs
        height = torch.tensor([record.metadata["com_height_m"] for record in cls.bank.records[:32]])
        cls.cost_targets = (-torch.expm1(-20.0 * (0.20 - height).clamp_min(0)) * 0.5).unsqueeze(-1)
        cls.update_cost(cls.cost, cls.cost_optimizer)
        cls.base_payload = {
            "model_state_dict": deepcopy(policy.state_dict()), "optimizer_state_dict": optimizer.state_dict(),
            "completed_updates": 1, "learning_rate": 0.001,
            "environment_state": {"validation_kind": "recorded_history_file_component"},
            "guidance": {"schema_version": "RecoveryGuidanceLearnerStateV1",
                         "cost_critic_state_dict": deepcopy(cls.cost.state_dict()),
                         "cost_optimizer_state_dict": deepcopy(cls.cost_optimizer.state_dict()),
                         "global_control_steps": 24, "alpha": 1.0 - 24 / 72000,
                         "guidance_config": asdict(GuidanceConfigV1()),
                         "training_contract": asdict(GuidanceTrainingContractV1())},
        }
        cls.policy = policy

    @classmethod
    def update_cost(cls, cost, optimizer):
        optimizer.zero_grad()
        (cost(cls.critic_inputs) - cls.cost_targets).square().mean().backward()
        optimizer.step()

    def setUp(self):
        root = ARTIFACT_ROOT / "test_work"
        root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "model_1.pt"

    def test_roundtrip_restores_cost_optimizer_and_reproduces_next_gradient_update(self):
        payload = deepcopy(self.base_payload)
        index = save_checkpoint(self.path, payload, self.identity)
        self.assertEqual(index["schema_version"], "RecoveryCheckpointIndexV2")
        self.assertEqual(index["checkpoint_sha256"], file_sha256(self.path))
        loaded = load_checkpoint(self.path, self.identity, restore_rng=False)
        first = GuidanceCostCritic()
        second = GuidanceCostCritic()
        first.load_state_dict(payload["guidance"]["cost_critic_state_dict"])
        second.load_state_dict(loaded["guidance"]["cost_critic_state_dict"])
        optimizers = [torch.optim.Adam(model.parameters(), lr=0.1) for model in (first, second)]
        optimizers[0].load_state_dict(payload["guidance"]["cost_optimizer_state_dict"])
        optimizers[1].load_state_dict(loaded["guidance"]["cost_optimizer_state_dict"])
        for model, optimizer in zip((first, second), optimizers):
            self.update_cost(model, optimizer)
        self.assertTrue(all(torch.equal(first.state_dict()[key], value) for key, value in second.state_dict().items()))
        self.assertEqual(loaded["guidance"]["global_control_steps"], 24)

    def test_missing_guidance_state_is_rejected_before_writing(self):
        payload = deepcopy(self.base_payload)
        del payload["guidance"]
        with self.assertRaises(ValueError):
            save_checkpoint(self.path, payload, self.identity)
        self.assertFalse(self.path.exists())

    def test_checkpoint_accepts_actual_float32_observation_alpha_at_update_one(self):
        payload = deepcopy(self.base_payload)
        payload["guidance"]["alpha"] = float(torch.tensor(1.0 - 24 / 72000, dtype=torch.float32))
        save_checkpoint(self.path, payload, self.identity)
        loaded = load_checkpoint(self.path, self.identity, restore_rng=False)
        self.assertEqual(loaded["guidance"]["global_control_steps"], 24)
        self.assertEqual(loaded["guidance"]["alpha"], payload["guidance"]["alpha"])

    def test_missing_cost_network_optimizer_or_progress_is_rejected(self):
        for name in self.base_payload["guidance"]:
            payload = deepcopy(self.base_payload)
            del payload["guidance"][name]
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_guidance_checkpoint(payload, self.identity)

    def test_mismatched_progress_alpha_identity_and_shape_are_rejected(self):
        for name, value in (("global_control_steps", 23), ("global_control_steps", True),
                            ("alpha", 1.0), ("alpha", float("nan"))):
            payload = deepcopy(self.base_payload)
            payload["guidance"][name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                validate_guidance_checkpoint(payload, self.identity)
        payload = deepcopy(self.base_payload)
        payload["guidance"]["cost_critic_state_dict"]["network.0.weight"] = torch.zeros(256, 34)
        with self.assertRaises(ValueError):
            validate_guidance_checkpoint(payload, self.identity)
        for key in ("source_sha256", "training_contract_sha256", "guidance_config_sha256"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_guidance_checkpoint(deepcopy(self.base_payload), {**self.identity, key: "invalid"})

    def test_nonfinite_cost_network_and_optimizer_are_rejected(self):
        payload = deepcopy(self.base_payload)
        payload["guidance"]["cost_critic_state_dict"]["network.0.weight"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            validate_guidance_checkpoint(payload, self.identity)
        payload = deepcopy(self.base_payload)
        first = next(iter(payload["guidance"]["cost_optimizer_state_dict"]["state"].values()))
        first["exp_avg"].flatten()[0] = float("inf")
        with self.assertRaises(ValueError):
            validate_guidance_checkpoint(payload, self.identity)

    def test_empty_cost_optimizer_state_is_rejected(self):
        payload = deepcopy(self.base_payload)
        payload["guidance"]["cost_optimizer_state_dict"]["state"] = {}
        with self.assertRaises(ValueError):
            validate_guidance_checkpoint(payload, self.identity)

    def test_disabled_guidance_rejects_partial_optimizer_state_and_invalid_parameter_groups(self):
        payload = deepcopy(self.base_payload)
        disabled = asdict(GuidanceConfigV1(force_enabled=False))
        payload["guidance"]["guidance_config"] = disabled
        payload["guidance"]["alpha"] = 0.0
        identity = {**self.identity, "guidance_config_sha256": stable_hash(disabled)}
        optimizer = payload["guidance"]["cost_optimizer_state_dict"]
        optimizer["state"].pop(next(iter(optimizer["state"])))
        with self.assertRaises(ValueError):
            validate_guidance_checkpoint(payload, identity)
        optimizer["state"] = {}
        optimizer["param_groups"][0]["params"].pop()
        with self.assertRaises(ValueError):
            validate_guidance_checkpoint(payload, identity)
        payload = deepcopy(self.base_payload)
        payload["guidance"].update({"guidance_config": disabled, "alpha": 0.0})
        payload["guidance"]["cost_optimizer_state_dict"]["state"] = {}
        payload["guidance"]["cost_optimizer_state_dict"]["param_groups"][0]["lr"] = 0.9
        with self.assertRaises(ValueError):
            validate_guidance_checkpoint(payload, identity)

    def test_real_disabled_learner_update_checkpoint_restore_and_export(self):
        config = deepcopy(self.config)
        config["guidance"]["force_enabled"] = False
        policy_inputs = torch.cat((self.inputs, torch.zeros(32, 1)), dim=1)
        critic_inputs = torch.cat((self.inputs[:, :34], torch.zeros(32, 1)), dim=1)
        observations = TensorDict({"policy": policy_inputs, "critic": critic_inputs}, batch_size=[32])
        policy = ActorCritic(observations, {"policy": ["policy"], "critic": ["critic"]}, 6,
                             **config["ppo"]["policy"])
        algorithm = GuidancePPO(policy, config["guidance"], device="cpu", **config["ppo"]["algorithm"])
        algorithm.init_storage("rl", 32, 24, observations, [6])
        targets = torch.tensor([record.prior_actions[-1] for record in self.bank.records[:32]], dtype=torch.float32)
        before = deepcopy(policy.actor.state_dict())
        # 使用记录历史和动作构造学习器组件数据；本检查不声明物理回合或起身能力已经通过。
        with torch.inference_mode():
            for k in range(1, 25):
                actions = algorithm.act(observations)
                rewards = -(actions - targets).square().mean(dim=1)
                extras = {"guidance": {"schema_version": "RecoveryGuidanceTransitionV1",
                                       "cost_force_n": torch.zeros(32), "max_force_n": torch.full((32,), 21.563625812),
                                       "alpha_t": torch.zeros(32), "alpha_next": torch.zeros(32),
                                       "global_control_steps": k}}
                algorithm.process_env_step(observations, rewards, torch.zeros(32, dtype=torch.bool), extras)
            algorithm.compute_returns(observations)
        statistics = algorithm.update()
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in statistics.values()))
        self.assertTrue(any(not torch.equal(before[key], value) for key, value in policy.actor.state_dict().items()))
        self.assertEqual(algorithm.cost_optimizer.state_dict()["state"], {})
        payload = {"model_state_dict": policy.state_dict(), "optimizer_state_dict": algorithm.optimizer.state_dict(),
                   "completed_updates": 1, "learning_rate": algorithm.learning_rate,
                   "environment_state": {"validation_kind": "disabled_guidance_learner_component"},
                   "guidance": {"schema_version": "RecoveryGuidanceLearnerStateV1",
                                **algorithm.guidance_state_dict(), "global_control_steps": 24, "alpha": 0.0,
                                "guidance_config": config["guidance"], "training_contract": config["training_contract"]}}
        identity = {**self.identity, "training_config_sha256": stable_hash(config),
                    "guidance_config_sha256": stable_hash(config["guidance"])}
        save_checkpoint(self.path, payload, identity)
        loaded = load_checkpoint(self.path, identity, restore_rng=False)
        restored_policy = ActorCritic(observations, {"policy": ["policy"], "critic": ["critic"]}, 6,
                                      **config["ppo"]["policy"])
        restored = GuidancePPO(restored_policy, config["guidance"], device="cpu", **config["ppo"]["algorithm"])
        restored_policy.load_state_dict(loaded["model_state_dict"])
        restored.optimizer.load_state_dict(loaded["optimizer_state_dict"])
        restored.load_guidance_state_dict(loaded["guidance"])
        self.assertEqual(restored.cost_optimizer.state_dict()["state"], {})
        output = self.root / "disabled_export"
        export_actor(self.path, output, resolved_config=config, expected_identity=identity,
                     golden_inputs=self.inputs, golden_provenance=self.provenance)
        actor, manifest = load_export(output, expected_identity=identity)
        with torch.inference_mode():
            torch.testing.assert_close(actor(self.inputs), restored_policy.actor(policy_inputs), atol=1e-7, rtol=0)
        self.assertFalse(manifest["guidance"]["force_enabled"])

    def test_zero_alpha_only_matches_withdrawn_or_disabled_guidance(self):
        payload = deepcopy(self.base_payload)
        payload["completed_updates"] = 3000
        payload["guidance"]["global_control_steps"] = 72000
        payload["guidance"]["alpha"] = 0.0
        self.assertTrue(validate_guidance_checkpoint(payload, self.identity))

    def export(self):
        save_checkpoint(self.path, deepcopy(self.base_payload), self.identity)
        output = self.root / "export"
        manifest = export_actor(self.path, output, resolved_config=self.config, expected_identity=self.identity,
                                golden_inputs=self.inputs, golden_provenance=self.provenance)
        return output, manifest

    def test_guidance_export_matches_training_actor_at_zero_assistance_and_is_portable(self):
        output, manifest = self.export()
        self.assertEqual(manifest["schema_version"], "RecoveryExportV3")
        self.assertEqual(manifest["contract"]["actor_input_dim"], 110)
        self.assertEqual(manifest["training_actor_input_dim"], 111)
        self.assertEqual(manifest["external_input_dim"], 110)
        self.path.rename(self.path.with_suffix(".saved"))
        actor, loaded = load_export(output, expected_identity=self.identity)
        with torch.inference_mode():
            for count in (1, 3, 32):
                states = self.inputs[:count]
                expected = self.policy.actor(torch.cat((states, torch.zeros(count, 1)), dim=1))
                torch.testing.assert_close(actor(states), expected, atol=1e-7, rtol=0)
        self.assertEqual(loaded["deployment_alpha"], 0.0)
        self.assertNotIn("cost_critic_state_dict", loaded)
        with self.assertRaises(torch.jit.Error):
            actor(torch.zeros(1, 111))

    def test_changed_training_actor_bytes_are_rejected(self):
        output, _ = self.export()
        with (output / "actor_111.ts").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaises(ValueError):
            load_export(output)

    def test_changed_external_dimension_is_rejected(self):
        output, _ = self.export()
        path = output / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["external_input_dim"] = 111
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_export(output)


if __name__ == "__main__":
    unittest.main()
