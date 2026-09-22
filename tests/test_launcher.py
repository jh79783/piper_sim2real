import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LauncherShellTests(unittest.TestCase):
    def _run_with_fake_docker(
        self, *args, script="run_piper_pick_place.sh", fake_ss=False, renderer=None, extra_env=None
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            capture = temp / "docker-args"
            fake = temp / "docker"
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
            for key in ("PIPER_RSL_IMAGE", "PIPER_RL_IMAGE", "PIPER_RENDERER", "PIPER_WSL_ROOT", "PIPER_DXG_DEVICE"):
                env.pop(key, None)
            env["PATH"] = f"{temp}:{env.get('PATH', '')}"
            env["PIPER_TEST_CAPTURE"] = str(capture)
            if renderer is not None:
                env["PIPER_RENDERER"] = renderer
            if extra_env:
                env.update(extra_env)
            result = subprocess.run(
                ["bash", str(ROOT / "scripts" / script), *args],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            return result, capture.read_text().splitlines() if capture.exists() else []

    def test_wrapper_is_syntax_valid(self):
        for script in ("run_piper_pick_place.sh", "run_reacher_parallel.sh", "run_tensorboard.sh"):
            with self.subTest(script=script):
                result = subprocess.run(
                    ["bash", "-n", str(ROOT / "scripts" / script)],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_launchers_preserve_host_file_ownership(self):
        cases = (
            ("run_piper_pick_place.sh", ["--headless", "--no-tensorboard"]),
            ("run_reacher_parallel.sh", ["--headless", "--device", "cpu"]),
            ("run_tensorboard.sh", ["16006"]),
        )
        for script, args in cases:
            with self.subTest(script=script):
                result, command = self._run_with_fake_docker(*args, script=script)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(command[0], "run")
                self.assertIn("--rm", command)
                self.assertIn("--init", command)
                self.assertEqual(command[command.index("--user") + 1], f"{os.getuid()}:{os.getgid()}")
                self.assertIn("HOME=/tmp", command)
                self.assertFalse(any(arg.startswith("--userns") for arg in command))
                self.assertNotIn("--security-opt=label=disable", command)
                self.assertNotIn("nvidia.com/gpu=all", command)

    def test_headless_no_tensorboard_does_not_publish_a_port(self):
        result, command = self._run_with_fake_docker("--headless", "--no-tensorboard", "--iterations", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("-p", command)
        self.assertNotIn("DISPLAY=:0", command)
        self.assertNotIn("GALLIUM_DRIVER=d3d12", command)
        self.assertNotIn("LIBGL_ALWAYS_SOFTWARE=1", command)
        self.assertNotIn("/usr/lib/wsl", command)
        self.assertIn("-e", command)
        self.assertIn("WARP_CACHE_PATH=/workspace/.cache/warp", command)
        self.assertIn("XDG_CACHE_HOME=/workspace/.cache", command)
        self.assertEqual(command[command.index("--gpus") + 1], "all")
        self.assertNotIn("--device", command)
        self.assertIn("piper-rsl:gpu", command)
        self.assertIn("scripts.train_piper_rsl", "\n".join(command))

    def test_headless_rgb_leaves_vision_image_on_egl_backend(self):
        result, command = self._run_with_fake_docker("--rgb", "--headless", "--no-tensorboard")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("MUJOCO_GL=glfw", command)
        self.assertNotIn("LIBGL_ALWAYS_SOFTWARE=1", command)

    def test_native_software_gui_uses_display_and_scoped_xauthority(self):
        display = 900 + (os.getpid() % 50)
        socket_path = Path(f"/tmp/.X11-unix/X{display}")
        x11_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            x11_socket.bind(str(socket_path))
            x11_socket.listen(1)
            with tempfile.TemporaryDirectory() as temp_dir:
                xauthority = Path(temp_dir) / ".Xauthority"
                subprocess.run(
                    [
                        "xauth",
                        "-f",
                        str(xauthority),
                        "add",
                        f":{display}",
                        "MIT-MAGIC-COOKIE-1",
                        "00112233445566778899aabbccddeeff",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                result, command = self._run_with_fake_docker(
                    "--rgb",
                    "--no-tensorboard",
                    renderer="software",
                    extra_env={
                        "DISPLAY": f":{display}",
                        "XAUTHORITY": str(xauthority),
                    },
                )
        finally:
            x11_socket.close()
            socket_path.unlink(missing_ok=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"DISPLAY=:{display}", command)
        self.assertIn("MUJOCO_GL=egl", command)
        self.assertIn("PIPER_PREVIEW_SOFTWARE=1", command)
        self.assertIn("XAUTHORITY=/tmp/piper.xauthority", command)
        auth_mount = next(value for value in command if value.endswith(":/tmp/piper.xauthority:ro"))
        self.assertRegex(auth_mount, r"^/tmp/piper-xauth\.[^:]+:/tmp/piper\.xauthority:ro$")
        self.assertNotIn(f"{xauthority}:/tmp/piper.xauthority:ro", command)
        self.assertIn("/tmp/.X11-unix:/tmp/.X11-unix:ro", command)
        self.assertNotIn("LIBGL_ALWAYS_SOFTWARE=1", command)
        self.assertNotIn("GALLIUM_DRIVER=d3d12", command)
        self.assertNotIn("--device", command)

    def test_localhost_display_is_normalized_to_mounted_unix_socket(self):
        display = 900 + (os.getpid() % 50)
        socket_path = Path(f"/tmp/.X11-unix/X{display}")
        x11_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            x11_socket.bind(str(socket_path))
            x11_socket.listen(1)
            with tempfile.TemporaryDirectory() as temp_dir:
                xauthority = Path(temp_dir) / ".Xauthority"
                subprocess.run(
                    [
                        "xauth", "-f", str(xauthority), "add", f":{display}",
                        "MIT-MAGIC-COOKIE-1", "00112233445566778899aabbccddeeff",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                result, command = self._run_with_fake_docker(
                    "--no-tensorboard",
                    renderer="software",
                    extra_env={
                        "DISPLAY": f"localhost:{display}.0",
                        "XAUTHORITY": str(xauthority),
                    },
                )
        finally:
            x11_socket.close()
            socket_path.unlink(missing_ok=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"DISPLAY=:{display}.0", command)

    def test_native_gpu_gui_keeps_glfw_acceleration_without_wslg_flags(self):
        display = 900 + (os.getpid() % 50)
        socket_path = Path(f"/tmp/.X11-unix/X{display}")
        x11_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            x11_socket.bind(str(socket_path))
            x11_socket.listen(1)
            with tempfile.TemporaryDirectory() as temp_dir:
                xauthority = Path(temp_dir) / ".Xauthority"
                subprocess.run(
                    [
                        "xauth",
                        "-f",
                        str(xauthority),
                        "add",
                        f":{display}",
                        "MIT-MAGIC-COOKIE-1",
                        "00112233445566778899aabbccddeeff",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                result, command = self._run_with_fake_docker(
                    "--no-tensorboard",
                    renderer="gpu",
                    extra_env={
                        "DISPLAY": f":{display}",
                        "XAUTHORITY": str(xauthority),
                    },
                )
        finally:
            x11_socket.close()
            socket_path.unlink(missing_ok=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"DISPLAY=:{display}", command)
        self.assertIn("MUJOCO_GL=glfw", command)
        self.assertNotIn("LIBGL_ALWAYS_SOFTWARE=1", command)
        self.assertNotIn("GALLIUM_DRIVER=d3d12", command)
        self.assertNotIn("--device", command)

    def test_gui_fails_before_docker_when_xauthority_is_missing(self):
        display = 900 + (os.getpid() % 50)
        socket_path = Path(f"/tmp/.X11-unix/X{display}")
        x11_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            x11_socket.bind(str(socket_path))
            x11_socket.listen(1)
            result, command = self._run_with_fake_docker(
                "--no-tensorboard",
                renderer="software",
                extra_env={
                    "DISPLAY": f":{display}",
                    "XAUTHORITY": str(socket_path) + ".missing",
                },
            )
        finally:
            x11_socket.close()
            socket_path.unlink(missing_ok=True)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(command)
        self.assertIn("Xauthority", result.stderr)

    def test_help_does_not_require_graphics(self):
        result, command = self._run_with_fake_docker("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("DISPLAY=:0", command)
        self.assertNotIn("GALLIUM_DRIVER=d3d12", command)
        self.assertNotIn("/usr/lib/wsl", command)
        self.assertNotIn("--gpus", command)
        self.assertNotIn("--device", command)
        self.assertNotIn("-p", command)

    def test_custom_tensorboard_port_is_loopback_published(self):
        result, command = self._run_with_fake_docker(
            "--headless", "--tensorboard-port", "16006", "--iterations", "1"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-p", command)
        self.assertIn("127.0.0.1:16006:16006", command)

    def test_invalid_port_fails_before_docker(self):
        result, command = self._run_with_fake_docker("--headless", "--tensorboard-port", "80")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(command)
        self.assertIn("1024", result.stderr)

    def test_invalid_renderer_fails_before_docker(self):
        result, command = self._run_with_fake_docker(
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
            result, command = self._run_with_fake_docker(
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
        self.assertEqual(command[command.index("--device") + 1], f"{dxg}:/dev/dxg")
        self.assertNotIn("LIBGL_ALWAYS_SOFTWARE=1", command)

    @unittest.skipUnless(Path("/tmp/.X11-unix/X0").is_socket(), "WSLg X11 socket unavailable")
    def test_software_renderer_is_explicit_and_loopback_safe(self):
        result, command = self._run_with_fake_docker(
            "--no-tensorboard", "--iterations", "1", renderer="software"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LIBGL_ALWAYS_SOFTWARE=1", command)
        self.assertNotIn("GALLIUM_DRIVER=d3d12", command)
        self.assertNotIn("--device", command)

    def test_rgb_selects_vision_image_and_graphics_drivers(self):
        result, command = self._run_with_fake_docker("--rgb", "--headless", "--no-tensorboard")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("piper-rsl:vision", command)
        self.assertIn("NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display", command)

    def test_image_override_and_training_arguments_are_preserved(self):
        args = ["--rgb", "--headless", "--no-tensorboard", "--resume", "runs/test run/model.pt"]
        result, command = self._run_with_fake_docker(*args, extra_env={"PIPER_RSL_IMAGE": "custom/piper:dev"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("custom/piper:dev", command)
        self.assertEqual(command[-len(args):], args)

    def test_reacher_requests_gpu_only_for_cuda(self):
        for args, needs_gpu in ((["--headless"], True), (["--headless", "--device=cpu"], False), (["--help"], False)):
            with self.subTest(args=args):
                result, command = self._run_with_fake_docker(*args, script="run_reacher_parallel.sh")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("piper-rl:gpu", command)
                self.assertIn("scripts.train_reacher_parallel", command)
                self.assertEqual("--gpus" in command, needs_gpu)
                if needs_gpu:
                    self.assertEqual(command[command.index("--gpus") + 1], "all")
                self.assertNotIn("DISPLAY=:0", command)

    def test_reacher_image_override(self):
        result, command = self._run_with_fake_docker(
            "--headless", script="run_reacher_parallel.sh", extra_env={"PIPER_RL_IMAGE": "custom/reacher:dev"}
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("custom/reacher:dev", command)

    def test_standalone_tensorboard_is_loopback_only_without_gpu(self):
        for args in (["16006"], ["--port", "16006"], ["--port=16006"]):
            with self.subTest(args=args):
                result, command = self._run_with_fake_docker(*args, script="run_tensorboard.sh")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("piper-rsl:gpu", command)
                self.assertIn("127.0.0.1:16006:6006", command)
                self.assertIn(f"{ROOT}/runs:/logs:ro", command)
                self.assertNotIn("--gpus", command)

    def test_standalone_tensorboard_image_override_precedence(self):
        cases = (
            ({"PIPER_RL_IMAGE": "custom/reacher:dev"}, "custom/reacher:dev"),
            ({"PIPER_RL_IMAGE": "custom/reacher:dev", "PIPER_RSL_IMAGE": "custom/piper:dev"}, "custom/piper:dev"),
        )
        for env, expected in cases:
            with self.subTest(env=env):
                result, command = self._run_with_fake_docker("16006", script="run_tensorboard.sh", extra_env=env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(expected, command)

    def test_standalone_tensorboard_rejects_invalid_or_occupied_ports(self):
        for port, fake_ss in (("80", False), ("16007", True)):
            with self.subTest(port=port):
                result, command = self._run_with_fake_docker(port, script="run_tensorboard.sh", fake_ss=fake_ss)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(command)

    def test_occupied_port_fails_without_touching_existing_process(self):
        result, command = self._run_with_fake_docker(
            "--headless", "--tensorboard-port", "16007", "--iterations", "1", fake_ss=True
        )
        self.assertEqual(result.returncode, 2)
        self.assertFalse(command)
        self.assertIn("already in use", result.stderr)
        self.assertIn("No existing process was stopped", result.stderr)


if __name__ == "__main__":
    unittest.main()
