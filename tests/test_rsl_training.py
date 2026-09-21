import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
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
            ]
        )
        self.assertEqual((args.num_envs, args.iterations, args.steps_per_env), (64, 5, 64))
        self.assertTrue(args.headless)
        self.assertTrue(args.no_tensorboard)
        self.assertEqual(args.tensorboard_port, 16006)

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
        cfg = trainer._make_train_cfg(args, num_obs=58, num_actions=4)
        self.assertEqual(cfg["actor"]["activation"], "tanh")
        self.assertEqual(cfg["critic"]["activation"], "tanh")


class RslCheckpointTests(unittest.TestCase):
    class _FakeTorch:
        def __init__(self):
            self.saved = None

        def save(self, payload, path):
            self.saved = (payload, Path(path))
            Path(path).write_text(json.dumps({"saved": True}))

        def load(self, path, **_kwargs):
            return self.payload

    def test_checkpoint_contains_schema_and_transition_count(self):
        args = trainer.parse_args(["--headless", "--no-tensorboard"])
        env = SimpleNamespace(num_actions=4)
        obs = {"policy": SimpleNamespace(shape=(8, 58))}
        cfg = trainer._make_train_cfg(args, num_obs=58, num_actions=4)
        fake_torch = self._FakeTorch()
        runner = SimpleNamespace(alg=SimpleNamespace(save=lambda: {"actor_state_dict": {}, "critic_state_dict": {}}))
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
            self.assertEqual(payload["piper_schema"]["num_obs"], 58)
            self.assertEqual(payload["piper_schema"]["total_steps"], 2048)

    def test_resume_rejects_old_or_missing_activation_schema(self):
        args = trainer.parse_args(["--headless", "--no-tensorboard"])
        env = SimpleNamespace(num_actions=4)
        obs = {"policy": SimpleNamespace(shape=(8, 58))}
        cfg = trainer._make_train_cfg(args, num_obs=58, num_actions=4)
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
                    fake_torch.payload = {
                        "piper_schema": {**expected, **change},
                        "actor_state_dict": {},
                        "critic_state_dict": {},
                    }
                    with self.assertRaisesRegex(ValueError, expected_text):
                        trainer._load_resume_schema(
                            path, args, env, obs, cfg, torch=fake_torch
                        )

    def test_chained_resume_preserves_lifetime_transition_count(self):
        args = trainer.parse_args(["--headless", "--no-tensorboard"])
        env = SimpleNamespace(num_actions=4)
        obs = {"policy": SimpleNamespace(shape=(8, 58))}
        cfg = trainer._make_train_cfg(args, num_obs=58, num_actions=4)
        fake_torch = self._FakeTorch()
        runner = SimpleNamespace(alg=SimpleNamespace(save=lambda: {"actor_state_dict": {}, "critic_state_dict": {}}))
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


if __name__ == "__main__":
    unittest.main()
