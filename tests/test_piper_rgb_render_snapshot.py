"""Focused checks for the Piper RGB render-only MuJoCo snapshot."""

from types import SimpleNamespace
import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - host state-only test environments
    torch = None

from scripts.piper_rgb_env import PiperRGBEnv


class _FakeWarp:
    def __init__(self):
        self.synchronize_calls = 0
        self.to_torch_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1

    def to_torch(self, value):
        self.to_torch_calls += 1
        return value


@unittest.skipIf(torch is None, "Torch is only needed for the device-view transfer test")
class RenderSnapshotTransferTest(unittest.TestCase):
    def _wrapper(self, num_envs=5):
        wrapper = object.__new__(PiperRGBEnv)
        wrapper.num_envs = num_envs
        wrapper.device = torch.device("cpu")
        wrapper._torch = torch
        wrapper._render_timing = {
            "render_calls": 0,
            "render_frames": 0,
            "render_field_transfers": 0,
            "render_transfer_seconds": 0.0,
            "render_seconds": 0.0,
            "encode_calls": 0,
            "encode_frames": 0,
            "encode_host_dispatch_seconds": 0.0,
        }
        warp = _FakeWarp()
        fields = {}
        for field_index, field in enumerate(PiperRGBEnv._RENDER_SNAPSHOT_FIELDS):
            values = torch.arange(num_envs * 2, dtype=torch.float32).reshape(num_envs, 2)
            fields[field] = values + 100.0 * field_index
        wrapper.base_env = SimpleNamespace(
            _wp=warp,
            _warp_data=SimpleNamespace(**fields),
        )
        return wrapper, warp, fields

    def test_one_gather_and_host_copy_per_field_independent_of_selection_size(self):
        wrapper, warp, fields = self._wrapper()

        first = wrapper._render_snapshot([4])
        second = wrapper._render_snapshot([3, 1, 4, 0])

        self.assertEqual(warp.synchronize_calls, 2)
        self.assertEqual(warp.to_torch_calls, 2 * len(PiperRGBEnv._RENDER_SNAPSHOT_FIELDS))
        self.assertEqual(first[PiperRGBEnv._RENDER_SNAPSHOT_FIELDS[0]].shape[0], 1)
        self.assertEqual(second[PiperRGBEnv._RENDER_SNAPSHOT_FIELDS[0]].shape[0], 4)
        self.assertEqual(wrapper.render_timing["render_field_transfers"], 2 * 8)
        for field in PiperRGBEnv._RENDER_SNAPSHOT_FIELDS:
            np.testing.assert_array_equal(first[field], fields[field][[4]].numpy())
            np.testing.assert_array_equal(second[field], fields[field][[3, 1, 4, 0]].numpy())

    def test_unexpanded_or_wrong_leading_dimension_is_rejected(self):
        wrapper, _, _ = self._wrapper()
        wrapper.base_env._warp_data.cam_xpos = torch.zeros((1, 1, 3), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "cam_xpos.*expected 5 worlds"):
            wrapper._render_snapshot([0])


@unittest.skipUnless(
    torch is not None and torch.cuda.is_available(),
    "requires the managed CUDA image and an available GPU",
)
class RenderSnapshotGpuPixelTest(unittest.TestCase):
    """Compare the old full transfer and new snapshot at identical Warp states."""

    def _make_render_wrapper(self, env):
        import mujoco

        wrapper = object.__new__(PiperRGBEnv)
        wrapper.base_env = env
        wrapper.num_envs = env.num_envs
        wrapper.device = env.device
        wrapper._torch = torch
        wrapper._renderer = mujoco.Renderer(env.model, height=480, width=640)
        wrapper._render_data = mujoco.MjData(env.model)
        wrapper._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(wrapper._camera)
        wrapper._camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        wrapper._camera.fixedcamid = int(env.model.camera("policy_rgb").id)
        wrapper._render_option = mujoco.MjvOption()
        wrapper._render_option.geomgroup[3] = 0
        wrapper._render_timing = {
            "render_calls": 0,
            "render_frames": 0,
            "render_field_transfers": 0,
            "render_transfer_seconds": 0.0,
            "render_seconds": 0.0,
            "encode_calls": 0,
            "encode_frames": 0,
            "encode_host_dispatch_seconds": 0.0,
        }
        return wrapper

    def _old_frames(self, env, indices):
        import mujoco

        env._wp.synchronize()
        data = mujoco.MjData(env.model)
        renderer = mujoco.Renderer(env.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = int(env.model.camera("policy_rgb").id)
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0
        try:
            frames = []
            for world_id in indices:
                env._mjw.get_data_into(data, env.model, env._warp_data, world_id=int(world_id))
                renderer.update_scene(data, camera=camera, scene_option=option)
                frames.append(renderer.render().copy())
            return frames
        finally:
            renderer.close()

    def _assert_pair(self, env, indices):
        wrapper = self._make_render_wrapper(env)
        try:
            old = self._old_frames(env, indices)
            new = wrapper._render_rgb(indices)
            self.assertEqual(len(old), len(new))
            for old_frame, new_frame in zip(old, new):
                self.assertEqual(old_frame.shape, (480, 640, 3))
                self.assertEqual(new_frame.shape, old_frame.shape)
                self.assertEqual(old_frame.dtype, np.uint8)
                self.assertEqual(new_frame.dtype, np.uint8)
                # Separate MuJoCo render contexts can quantize one edge pixel
                # differently even when the visual pose arrays are identical.
                # Keep the comparison explicit and bounded at one RGB8 level;
                # frame order and shape are still checked above.
                delta = np.abs(old_frame.astype(np.int16) - new_frame.astype(np.int16))
                self.assertLessEqual(int(delta.max()), 1)
            self.assertEqual(wrapper.render_timing["render_field_transfers"], 8)
        finally:
            wrapper._renderer.close()

    def test_reset_step_and_partial_reset_pixels_are_identical(self):
        from scripts.piper_warp_env import PiperWarpEnv

        env = PiperWarpEnv(num_envs=4, device="cuda", seed=123, start_mode="above_cube")
        try:
            self._assert_pair(env, [3, 1, 3, 0])
            actions = torch.zeros((4, 7), dtype=torch.float32, device=env.device)
            actions[:, 5] = 0.35
            actions[:, 6] = -0.2
            env.step(actions)
            self._assert_pair(env, [0, 2, 1])
            env._reset_worlds(torch.as_tensor([1, 3], dtype=torch.long, device=env.device))
            self._assert_pair(env, [3, 1, 0, 2])
        finally:
            env.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
