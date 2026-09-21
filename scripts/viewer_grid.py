"""Small GLFW/MuJoCo RGB grid viewer used by the Warp training launcher.

The viewer intentionally knows nothing about a simulator or a policy.  The
training process supplies already-rendered RGB frames, normally copied from
four selected GPU environments.  MuJoCo is used only for the pixel blit and
text overlays; it does not step any physics here.
"""

from __future__ import annotations

import time


def make_grid(frames):
    """Return a row-major 2x2 RGB grid, leaving unused cells black.

    Imports are local so this helper remains usable in a host-side test suite
    that does not install NumPy.  The training image supplies NumPy at runtime.
    """

    import numpy as np

    frames = list(frames)
    if not 1 <= len(frames) <= 4:
        raise ValueError("expected 1 to 4 RGB frames")
    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[2] != 3 or first.dtype != np.uint8:
        raise ValueError("expected uint8 RGB frames")
    height, width, _ = first.shape
    grid = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        frame = np.asarray(frame)
        if frame.shape != first.shape or frame.dtype != np.uint8:
            raise ValueError("all RGB frames must have the same shape and dtype")
        row, col = divmod(index, 2)
        grid[row * height : (row + 1) * height, col * width : (col + 1) * width] = frame
    return grid


def _as_rgb_frames(frames) -> list:
    """Convert a tensor/array/list returned by a Warp renderer to RGB arrays."""

    # Avoid importing torch/numpy until the GUI path is actually requested.
    if hasattr(frames, "detach"):
        frames = frames.detach()
        if hasattr(frames, "to"):
            frames = frames.to("cpu")
        if hasattr(frames, "numpy"):
            frames = frames.numpy()
    if hasattr(frames, "numpy"):
        frames = frames.numpy()
    if getattr(frames, "ndim", None) == 3:
        frames = [frames]
    return list(frames)


class GridWindow:
    """Draw four RGB snapshots in one GLFW window."""

    def __init__(self, title: str = "Piper Pick & Place PPO", width: int = 960, height: int = 720):
        import glfw
        import mujoco

        self.glfw = glfw
        self.mujoco = mujoco
        self.title = title
        self.window = None
        self.context = None
        self.frames_drawn = 0
        self.stop_requested = False
        self._closed = False
        self._window_closed = False
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed: check DISPLAY and the WSLg X11 mount")
        try:
            glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
            self.window = glfw.create_window(width, height, f"{title} | 2x2 | Esc: save and exit", None, None)
            if not self.window:
                raise RuntimeError("Could not open the WSLg preview window")
            glfw.make_context_current(self.window)
            glfw.swap_interval(0)
            # This empty model provides a valid MuJoCo OpenGL drawing context;
            # no simulation state is created or stepped by the viewer.
            self.display_model = mujoco.MjModel.from_xml_string("<mujoco/>")
            self.context = mujoco.MjrContext(self.display_model, mujoco.mjtFontScale.mjFONTSCALE_100)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)
        except BaseException:
            self.close()
            raise

    def poll(self) -> bool:
        if self._closed or self._window_closed or not self.window:
            return False
        self.glfw.poll_events()
        self.stop_requested = bool(
            self.glfw.window_should_close(self.window)
            or self.glfw.get_key(self.window, self.glfw.KEY_ESCAPE) == self.glfw.PRESS
        )
        return not self.stop_requested

    def draw(self, frames, steps: int, device: str, total_envs: int | None = None):
        if self._closed or self._window_closed or not self.window:
            return False
        frames = _as_rgb_frames(frames)
        grid = make_grid(frames)
        glfw, mj = self.glfw, self.mujoco
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0:  # Minimized window.
            return True
        # Accommodate resizing and WSLg/Windows HiDPI framebuffer sizes.
        import numpy as np

        rows = np.arange(height) * grid.shape[0] // height
        cols = np.arange(width) * grid.shape[1] // width
        # OpenGL's origin is at the bottom; MuJoCo's pixel API expects that
        # orientation even though RGB screenshots use a top-left origin.
        pixels = np.ascontiguousarray(grid[rows[:, None], cols[None, :]][::-1])
        mj.mjr_drawPixels(pixels.ravel(), None, mj.MjrRect(0, 0, width, height), self.context)
        for index in range(4):
            row, col = divmod(index, 2)
            left, right = col * width // 2, (col + 1) * width // 2
            bottom, top = (1 - row) * height // 2, (2 - row) * height // 2
            rect = mj.MjrRect(left, bottom, right - left, top - bottom)
            label = f"Env {index + 1}" if index < len(frames) else "Unused"
            mj.mjr_overlay(
                mj.mjtFontScale.mjFONTSCALE_100,
                mj.mjtGridPos.mjGRID_TOPLEFT,
                rect,
                label,
                "",
                self.context,
            )
        shown = len(frames)
        env_label = f"showing {shown}/{total_envs}" if total_envs is not None else f"{shown} envs"
        glfw.set_window_title(
            self.window,
            f"{self.title} | {env_label} | {device} | {steps:,} total steps | Esc: save and exit",
        )
        glfw.swap_buffers(self.window)
        self.frames_drawn += 1
        return True

    def close_window(self):
        """Destroy this window/context but leave GLFW initialized.

        This is used when preview rendering fails while the Warp environment
        is still running. The environment's MuJoCo RGB renderer is closed
        first by the trainer; only then does :meth:`close` terminate GLFW.
        """

        if self._window_closed:
            return
        self._window_closed = True
        if self.window:
            self.glfw.make_context_current(self.window)
        if self.context:
            self.context.free()
            self.context = None
        if self.window:
            self.glfw.destroy_window(self.window)
            self.window = None

    def close(self):
        if self._closed:
            return
        self._closed = True
        # During a partially constructed window, glfw may still be importable
        # while one of context/window is absent.
        try:
            self.close_window()
        finally:
            try:
                self.glfw.terminate()
            except Exception:
                pass


class PreviewController:
    """Rate-limit rendering and keep GUI event polling responsive."""

    def __init__(self, window: GridWindow, fps: float, total_envs: int | None = None):
        if fps <= 0:
            raise ValueError("fps must be greater than zero")
        self.window = window
        self.total_envs = total_envs
        self.disabled = False
        self.interval = 1.0 / fps
        self.last_draw = float("-inf")
        self.last_grid = None

    def poll(self) -> bool:
        if self.disabled:
            return True
        return self.window.poll()

    def maybe_draw(self, frames, steps: int, device: str) -> bool:
        if self.disabled:
            return True
        now = time.monotonic()
        if now - self.last_draw < self.interval:
            return True
        if not self.window.draw(frames, steps, device, total_envs=self.total_envs):
            return False
        self.last_draw = now
        self.last_grid = make_grid(_as_rgb_frames(frames))
        return True

    def close(self):
        self.window.close()

    def disable(self):
        """Turn off preview while retaining ownership for orderly final close."""

        if not self.disabled:
            self.window.close_window()
        self.disabled = True
