"""Lifecycle checks for the isolated RGB preview child."""

from __future__ import annotations

import queue
import unittest

import numpy as np

from scripts.viewer_grid import PreviewController


class _FakeProcess:
    def __init__(self):
        self.alive = True
        self.join_calls = 0
        self.terminate_calls = 0

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        del timeout
        self.join_calls += 1

    def terminate(self):
        self.terminate_calls += 1
        self.alive = False


def _controller():
    result = object.__new__(PreviewController)
    result.window = None
    result.total_envs = 1
    result.disabled = False
    result.interval = 0.0
    result.last_draw = float("-inf")
    result.last_grid = None
    result.frames_drawn = 0
    result._isolated_process = _FakeProcess()
    result._frame_queue = queue.Queue(maxsize=1)
    result._event_queue = queue.Queue(maxsize=4)
    result._child_closed = False
    result._child_ready = True
    result._child_error = None
    return result


class ViewerLifecycleTests(unittest.TestCase):
    def test_full_queue_drops_stale_frame_without_stopping_training(self):
        controller = _controller()
        controller._frame_queue.put_nowait(("stale",))
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
        self.assertTrue(controller.maybe_draw(frames, 1, "cuda"))
        self.assertTrue(controller.poll())
        self.assertEqual(controller._frame_queue.qsize(), 1)

    def test_child_error_is_reported_and_closed_event_stops_normally(self):
        controller = _controller()
        controller._event_queue.put_nowait(("error", "GLFW initialization failed"))
        with self.assertRaisesRegex(RuntimeError, "GLFW initialization failed"):
            controller.poll()

        controller = _controller()
        controller._event_queue.put_nowait(("closed", None))
        self.assertFalse(controller.poll())

    def test_draw_acknowledgement_and_close_are_idempotent(self):
        controller = _controller()
        controller._event_queue.put_nowait(("drawn", None))
        self.assertTrue(controller.poll())
        self.assertEqual(controller.frames_drawn, 1)
        controller._frame_queue.put_nowait(("pending",))
        controller.close()
        controller.close()
        self.assertEqual(controller._isolated_process, None)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
