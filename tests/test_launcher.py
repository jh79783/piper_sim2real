import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LauncherShellTests(unittest.TestCase):
    def _run_with_fake_podman(self, *args, fake_ss=False):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            capture = temp / "podman-args"
            fake = temp / "podman"
            fake.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$@\" > \"$PIPER_TEST_CAPTURE\"\n"
            )
            fake.chmod(0o755)
            if fake_ss:
                ss = temp / "ss"
                ss.write_text(
                    "#!/usr/bin/env bash\n"
                    "echo 'LISTEN 0 4096 127.0.0.1:16007 0.0.0.0:*'\n"
                )
                ss.chmod(0o755)
            env = dict(os.environ)
            env["PATH"] = f"{temp}:{env.get('PATH', '')}"
            env["PIPER_TEST_CAPTURE"] = str(capture)
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/run_piper_pick_place.sh"), *args],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            return result, capture.read_text().splitlines() if capture.exists() else []

    def test_wrapper_is_syntax_valid(self):
        result = subprocess.run(
            ["bash", "-n", str(ROOT / "scripts/run_piper_pick_place.sh"), str(ROOT / "scripts/run_tensorboard.sh")],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_headless_no_tensorboard_does_not_publish_a_port(self):
        result, command = self._run_with_fake_podman("--headless", "--no-tensorboard", "--iterations", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("-p", command)
        self.assertIn("-e", command)
        self.assertIn("WARP_CACHE_PATH=/workspace/.cache/warp", command)
        self.assertIn("localhost/piper-rsl:gpu", command)
        self.assertIn("scripts.train_piper_rsl", "\n".join(command))

    def test_custom_tensorboard_port_is_loopback_published(self):
        result, command = self._run_with_fake_podman(
            "--headless", "--tensorboard-port", "16006", "--iterations", "1"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-p", command)
        self.assertIn("127.0.0.1:16006:16006", command)

    def test_invalid_port_fails_before_podman(self):
        result, command = self._run_with_fake_podman("--headless", "--tensorboard-port", "80")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(command)
        self.assertIn("1024", result.stderr)

    def test_occupied_port_fails_without_touching_existing_process(self):
        result, command = self._run_with_fake_podman(
            "--headless", "--tensorboard-port", "16007", "--iterations", "1", fake_ss=True
        )
        self.assertEqual(result.returncode, 2)
        self.assertFalse(command)
        self.assertIn("already in use", result.stderr)
        self.assertIn("No existing process was stopped", result.stderr)


if __name__ == "__main__":
    unittest.main()
