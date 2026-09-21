import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
OBSERVATION_DIM = 63
NUM_ACTIONS = 7
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_piper_rsl as trainer


class RslArgumentTests(unittest.TestCase):
    def test_defaults_are_gpu_and_batched(self):
        args = trainer.parse_args([])
        self.assertEqual(args.num_envs, 128)
        self.assertEqual(args.steps_per_env, 64)
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.tensorboard_port, 6006)
        self.assertFalse(args.no_tensorboard)
        self.assertEqual(args.start_mode, "curriculum")
        self.assertFalse(args.no_domain_randomization)
        self.assertFalse(args.no_sensor_noise)
        self.assertFalse(args.no_curriculum)
        self.assertFalse(args.rgb)
        self.assertEqual(args.rgb_camera_fps, 30.0)

    def test_aliases_and_bounded_smoke_flags(self):
        args = trainer.parse_args(
            [
                "--n-envs",
                "64",
                "--iterations",
                "5",
                "--steps-per-env",
                "64",
                "--headless",
                "--no-tensorboard",
                "--tensorboard-port",
                "16006",
                "--start-mode",
                "home",
                "--no-domain-randomization",
                "--no-sensor-noise",
                "--no-curriculum",
                "--rgb",
                "--rgb-camera-fps",
                "30",
                "--rgb-encoder-checkpoint",
                "encoder.pt",
            ]
        )
        self.assertEqual((args.num_envs, args.iterations, args.steps_per_env), (64, 5, 64))
        self.assertTrue(args.headless)
        self.assertTrue(args.no_tensorboard)
        self.assertEqual(args.tensorboard_port, 16006)
        self.assertEqual(args.start_mode, "home")
        self.assertTrue(args.no_domain_randomization)
        self.assertTrue(args.no_sensor_noise)
        self.assertTrue(args.no_curriculum)
        self.assertTrue(args.rgb)
        self.assertEqual(args.rgb_camera_fps, 30.0)
        self.assertEqual(args.rgb_encoder_checkpoint, Path("encoder.pt"))

    def test_cpu_is_rejected_by_parser(self):
        with self.assertRaises(SystemExit) as result:
            trainer.parse_args(["--device", "cpu"])
        self.assertEqual(result.exception.code, 2)

    def test_help_does_not_import_cuda_stack(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/train_piper_rsl.py"), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--num-envs", result.stdout)
        self.assertIn("--tensorboard-port", result.stdout)

    def test_mlp_actor_and_critic_use_tanh(self):
        args = trainer.parse_args(["--headless", "--no-tensorboard"])
        cfg = trainer._make_train_cfg(args, num_obs=OBSERVATION_DIM, num_actions=NUM_ACTIONS)
        self.assertEqual(cfg["actor"]["activation"], "tanh")
        self.assertEqual(cfg["critic"]["activation"], "tanh")
        self.assertEqual(cfg["obs_groups"], {"actor": ["policy"], "critic": ["critic"]})
        self.assertAlmostEqual(cfg["algorithm"]["gamma"], 0.99**0.5)
        self.assertAlmostEqual(cfg["algorithm"]["lam"], 0.95**0.5)

    def test_rgb_mlp_uses_policy_and_vision_actor_groups(self):
        args = trainer.parse_args(["--headless", "--no-tensorboard"])
        cfg = trainer._make_train_cfg(
            args,
            num_obs=36,
            num_actions=NUM_ACTIONS,
            obs_groups={"actor": ["policy", "vision"], "critic": ["critic"]},
        )
        self.assertEqual(cfg["obs_groups"], {"actor": ["policy", "vision"], "critic": ["critic"]})


class RslCheckpointTests(unittest.TestCase):
    class _FakeTorch:
        def __init__(self):
            self.saved = None

        def save(self, payload, path):
            self.saved = (payload, Path(path))
            Path(path).write_text(json.dumps({"saved": True}))

        def load(self, path, **_kwargs):
            return self.payload

    @staticmethod
    def _env():
        args = trainer.parse_args(["--headless", "--no-tensorboard"])
        return SimpleNamespace(
            num_actions=NUM_ACTIONS,
            num_observations=OBSERVATION_DIM,
            reward_version=2,
            recontact_penalty=0.2,
            frame_skip=10,
            max_episode_length=600,
            model=SimpleNamespace(opt=SimpleNamespace(timestep=0.002)),
            training_steps=0,
            training_config=trainer._make_training_config(args),
            cfg={
                "observation_version": "policy_critic_sensor_history_v1",
                "action_semantics": "absolute_joint_targets_v1",
            },
        )

    @staticmethod
    def _obs():
        return {
            "policy": SimpleNamespace(shape=(8, OBSERVATION_DIM)),
            "critic": SimpleNamespace(shape=(8, OBSERVATION_DIM)),
        }

    @staticmethod
    def _states():
        # RSL's MLPModel persists EmpiricalNormalization buffers alongside
        # its weights.  Keep fake checkpoints shaped like that contract.
        return {
            "actor_state_dict": {"obs_normalizer._mean": 0.0},
            "critic_state_dict": {"obs_normalizer._mean": 0.0},
        }

    def test_checkpoint_contains_schema_and_transition_count(self):
        args = trainer.parse_args(["--headless", "--no-tensorboard"])
        env = self._env()
        obs = self._obs()
        cfg = trainer._make_train_cfg(args, num_obs=OBSERVATION_DIM, num_actions=NUM_ACTIONS)
        fake_torch = self._FakeTorch()
        runner = SimpleNamespace(alg=SimpleNamespace(save=lambda: self._states()))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pt"
            trainer._save_checkpoint(
                runner,
                path,
                args,
                env,
                obs,
                cfg,
                torch=fake_torch,
                iteration=3,
                total_steps=2048,
            )
            payload, saved_path = fake_torch.saved
            self.assertNotEqual(saved_path, path)
            self.assertTrue(path.is_file())
            self.assertEqual(payload["iter"], 3)
            self.assertEqual(payload["infos"]["total_steps"], 2048)
            self.assertEqual(payload["piper_schema"]["num_obs"], OBSERVATION_DIM)
            self.assertEqual(payload["piper_schema"]["reward_version"], 2)
            self.assertEqual(payload["piper_schema"]["recontact_penalty"], 0.2)
            self.assertEqual(payload["piper_schema"]["total_steps"], 2048)
            self.assertEqual(payload["piper_schema"]["schema_version"], 2)
            self.assertEqual(payload["piper_schema"]["training_steps"], 0)
            self.assertEqual(payload["infos"]["training_steps"], 0)
            self.assertEqual(payload["piper_schema"]["obs_groups"], {"actor": ["policy"], "critic": ["critic"]})
            self.assertEqual(
                [stage["step"] for stage in payload["piper_schema"]["training_config"]["stages"]],
                [0, 32_000, 64_000, 96_000],
            )
            self.assertEqual(payload["piper_schema"]["timing"]["control_hz"], 50.0)

    def test_resume_rejects_old_or_missing_activation_schema(self):
        args = trainer.parse_args(["--headless", "--no-tensorboard"])
        env = self._env()
        obs = self._obs()
        cfg = trainer._make_train_cfg(args, num_obs=OBSERVATION_DIM, num_actions=NUM_ACTIONS)
        expected = trainer._checkpoint_schema(args, env, obs, iteration=0, train_cfg=cfg)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "old.pt"
            path.write_bytes(b"placeholder")
            for change, expected_text in (
                ({"actor_activation": "elu", "critic_activation": "elu"}, "actor_activation"),
                ({"actor_activation": None}, "actor_activation"),
            ):
                with self.subTest(change=change):
                    fake_torch = self._FakeTorch()
                    fake_torch.payload = {"piper_schema": {**expected, **change}, **self._states()}
                    with self.assertRaisesRegex(ValueError, expected_text):
                        trainer._load_resume_schema(
                            path, args, env, obs, cfg, torch=fake_torch
                        )

    def test_resume_rejects_old_observation_and_reward_contract(self):
        args = trainer.parse_args(["--headless", "--no-tensorboard"])
        env = self._env()
        obs = self._obs()
        cfg = trainer._make_train_cfg(args, num_obs=OBSERVATION_DIM, num_actions=NUM_ACTIONS)
        expected = trainer._checkpoint_schema(args, env, obs, iteration=0, train_cfg=cfg)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "incompatible.pt"
            path.write_bytes(b"placeholder")
            for change, expected_text in (
                ({"num_obs": 58}, "num_obs"),
                ({"reward_version": 1}, "reward_version"),
                ({"recontact_penalty": 0.0}, "recontact_penalty"),
            ):
                with self.subTest(change=change):
                    fake_torch = self._FakeTorch()
                    fake_torch.payload = {"piper_schema": {**expected, **change}, **self._states()}
                    with self.assertRaisesRegex(ValueError, expected_text):
                        trainer._load_resume_schema(
                            path, args, env, obs, cfg, torch=fake_torch
                        )

    def test_resume_accepts_current_observation_and_reward_contract(self):
        args = trainer.parse_args(["--headless", "--no-tensorboard"])
        env = self._env()
        obs = self._obs()
        cfg = trainer._make_train_cfg(args, num_obs=OBSERVATION_DIM, num_actions=NUM_ACTIONS)
        payload = {
            "piper_schema": trainer._checkpoint_schema(args, env, obs, iteration=5, train_cfg=cfg),
            **self._states(),
        }
        fake_torch = self._FakeTorch()
        fake_torch.payload = payload
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "compatible.pt"
            path.write_bytes(b"placeholder")
            self.assertIs(
                trainer._load_resume_schema(path, args, env, obs, cfg, torch=fake_torch),
                payload,
            )

    def test_resume_rejects_timing_observation_group_or_config_changes(self):
        args = trainer.parse_args(["--headless", "--no-tensorboard"])
        env = self._env()
        obs = self._obs()
        cfg = trainer._make_train_cfg(args, num_obs=OBSERVATION_DIM, num_actions=NUM_ACTIONS)
        expected = trainer._checkpoint_schema(args, env, obs, iteration=0, train_cfg=cfg)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "incompatible.pt"
            path.write_bytes(b"placeholder")
            for change, expected_text in (
                ({"timing": {**expected["timing"], "control_substeps": 20}}, "timing"),
                ({"obs_groups": {"actor": ["policy"], "critic": ["policy"]}}, "obs_groups"),
                ({"normalization": {"actor": False, "critic": True}}, "normalization"),
                ({"training_config": {**expected["training_config"], "sensor_noise": False}}, "training_config"),
            ):
                with self.subTest(change=change):
                    fake_torch = self._FakeTorch()
                    fake_torch.payload = {"piper_schema": {**expected, **change}, **self._states()}
                    with self.assertRaisesRegex(ValueError, expected_text):
                        trainer._load_resume_schema(path, args, env, obs, cfg, torch=fake_torch)

    def test_training_steps_are_saved_and_restored_explicitly(self):
        args = trainer.parse_args(["--headless", "--no-tensorboard"])
        env = self._env()
        env.training_steps = 64_000
        obs = self._obs()
        cfg = trainer._make_train_cfg(args, num_obs=OBSERVATION_DIM, num_actions=NUM_ACTIONS)
        fake_torch = self._FakeTorch()
        runner = SimpleNamespace(alg=SimpleNamespace(save=lambda: self._states()))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "progress.pt"
            trainer._save_checkpoint(
                runner, path, args, env, obs, cfg, torch=fake_torch,
                iteration=4, total_steps=13, lifetime_total_steps=77,
            )
            payload, _ = fake_torch.saved
            self.assertEqual(payload["training_steps"], 64_000)
            self.assertEqual(payload["infos"]["training_steps"], 64_000)
            self.assertEqual(payload["piper_schema"]["training_steps"], 64_000)

        # Progress is intentionally a resumable value, not a compatibility
        # field: the newly constructed env starts at zero before restore.
        fake_torch.payload = payload
        with tempfile.TemporaryDirectory() as temp_dir:
            resume_path = Path(temp_dir) / "progress.pt"
            resume_path.write_bytes(b"placeholder")
            self.assertIs(
                trainer._load_resume_schema(resume_path, args, env, obs, cfg, torch=fake_torch),
                payload,
            )

        restored = SimpleNamespace(training_steps=0)
        restored.set_training_steps = lambda count: setattr(restored, "training_steps", int(count))
        trainer._set_env_training_steps(restored, payload["piper_schema"]["training_steps"])
        self.assertEqual(restored.training_steps, 64_000)

    def test_chained_resume_preserves_lifetime_transition_count(self):
        args = trainer.parse_args(["--headless", "--no-tensorboard"])
        env = self._env()
        obs = self._obs()
        cfg = trainer._make_train_cfg(args, num_obs=OBSERVATION_DIM, num_actions=NUM_ACTIONS)
        fake_torch = self._FakeTorch()
        runner = SimpleNamespace(alg=SimpleNamespace(save=lambda: self._states()))
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.pt"
            second = Path(temp_dir) / "second.pt"
            trainer._save_checkpoint(
                runner, first, args, env, obs, cfg, torch=fake_torch,
                iteration=4, total_steps=20480, lifetime_total_steps=20480,
            )
            trainer._save_checkpoint(
                runner, second, args, env, obs, cfg, torch=fake_torch,
                iteration=5, total_steps=2048, lifetime_total_steps=22528,
            )
            payload, _ = fake_torch.saved
            self.assertEqual(payload["infos"]["total_steps"], 2048)
            self.assertEqual(payload["infos"]["lifetime_total_steps"], 22528)
            self.assertEqual(payload["piper_schema"]["lifetime_total_steps"], 22528)

    def test_terminate_only_owned_child(self):
        process = subprocess.Popen(["sleep", "30"], start_new_session=True)
        try:
            trainer._terminate_process_group(process, timeout=1)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


class RslNormalizationIntegrationTests(unittest.TestCase):
    def test_mlp_normalization_stats_round_trip_and_inference(self):
        try:
            import torch
            from tensordict import TensorDict
            from rsl_rl.models import MLPModel
        except ImportError as exc:
            self.skipTest(f"RSL-RL CPU stack is unavailable: {exc}")

        torch.manual_seed(7)
        obs = TensorDict(
            {
                "policy": torch.randn(32, OBSERVATION_DIM),
                "critic": torch.randn(32, OBSERVATION_DIM),
            },
            batch_size=[32],
        )
        groups = {"actor": ["policy"], "critic": ["critic"]}
        model = MLPModel(
            obs,
            groups,
            "actor",
            NUM_ACTIONS,
            hidden_dims=[16, 8],
            activation="tanh",
            obs_normalization=True,
        )
        model.update_normalization(obs)
        sample = TensorDict(
            {"policy": torch.randn(4, OBSERVATION_DIM)},
            batch_size=[4],
        )
        model.eval()
        expected = model(sample).detach()
        state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        self.assertTrue(any("normalizer" in key for key in state))
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "mlp.pt"
            torch.save(state, checkpoint)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)

        restored = MLPModel(
            obs,
            groups,
            "actor",
            NUM_ACTIONS,
            hidden_dims=[16, 8],
            activation="tanh",
            obs_normalization=True,
        )
        restored.load_state_dict(state)
        restored.eval()
        self.assertTrue(torch.equal(expected, restored(sample)))
        exported = restored.as_jit()
        self.assertTrue(torch.allclose(expected, exported(sample["policy"]), atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
