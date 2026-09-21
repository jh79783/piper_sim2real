import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LauncherShellTests(unittest.TestCase):
    def _run_with_fake_podman(self, *args, fake_ss=False, renderer=None, extra_env=None):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            capture = temp / "podman-args"
            fake = temp / "podman"
            fake.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$@\" > \"$PIPER_TEST_CAPTURE\"\n"
            )
            fake.chmod(0o755)
            ss = temp / "ss"
            ss.write_text(
                "#!/usr/bin/env bash\n"
                + ("echo 'LISTEN 0 4096 127.0.0.1:16007 0.0.0.0:*'\n" if fake_ss else ":\n")
            )
            ss.chmod(0o755)
            env = dict(os.environ)
            env["PATH"] = f"{temp}:{env.get('PATH', '')}"
            env["PIPER_TEST_CAPTURE"] = str(capture)
            if renderer is not None:
                env["PIPER_RENDERER"] = renderer
            if extra_env:
                env.update(extra_env)
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
        self.assertNotIn("DISPLAY=:0", command)
        self.assertNotIn("GALLIUM_DRIVER=d3d12", command)
        self.assertNotIn("LIBGL_ALWAYS_SOFTWARE=1", command)
        self.assertNotIn("/usr/lib/wsl", command)
        self.assertIn("-e", command)
        self.assertIn("WARP_CACHE_PATH=/workspace/.cache/warp", command)
        self.assertIn("localhost/piper-rsl:gpu", command)
        self.assertIn("scripts.train_piper_rsl", "\n".join(command))

    def test_help_does_not_require_graphics(self):
        result, command = self._run_with_fake_podman("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("DISPLAY=:0", command)
        self.assertNotIn("GALLIUM_DRIVER=d3d12", command)
        self.assertNotIn("/usr/lib/wsl", command)

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

    def test_invalid_renderer_fails_before_podman(self):
        result, command = self._run_with_fake_podman(
            "--headless", "--no-tensorboard", renderer="vulkan"
        )
        self.assertEqual(result.returncode, 2)
        self.assertFalse(command)
        self.assertIn("PIPER_RENDERER", result.stderr)

    @unittest.skipUnless(Path("/tmp/.X11-unix/X0").is_socket(), "WSLg X11 socket unavailable")
    def test_gpu_renderer_sets_wslg_flags_and_no_software_fallback(self):
        with tempfile.TemporaryDirectory() as gpu_dir:
            wsl_root = Path(gpu_dir) / "wsl"
            (wsl_root / "lib").mkdir(parents=True)
            (wsl_root / "lib/libd3d12.so").touch()
            (wsl_root / "lib/libdxcore.so").touch()
            dxg = Path(gpu_dir) / "dxg"
            dxg.touch()
            result, command = self._run_with_fake_podman(
                "--no-tensorboard",
                "--iterations",
                "1",
                renderer="gpu",
                extra_env={
                    "PIPER_WSL_ROOT": str(wsl_root),
                    "PIPER_DXG_DEVICE": str(dxg),
                },
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("GALLIUM_DRIVER=d3d12", command)
        self.assertIn("MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA", command)
        self.assertIn("LD_LIBRARY_PATH=/usr/lib/wsl/lib", command)
        self.assertIn(f"{wsl_root}:/usr/lib/wsl:ro", command)
        self.assertNotIn("LIBGL_ALWAYS_SOFTWARE=1", command)

    @unittest.skipUnless(Path("/tmp/.X11-unix/X0").is_socket(), "WSLg X11 socket unavailable")
    def test_software_renderer_is_explicit_and_loopback_safe(self):
        result, command = self._run_with_fake_podman(
            "--no-tensorboard", "--iterations", "1", renderer="software"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LIBGL_ALWAYS_SOFTWARE=1", command)
        self.assertNotIn("GALLIUM_DRIVER=d3d12", command)

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
