import unittest

try:
    import stable_baselines3  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name == "stable_baselines3":
        raise unittest.SkipTest("legacy Reacher tests require the SB3 image") from exc
    raise

import numpy as np

from scripts.train_reacher_parallel import make_grid, parse_args


class GridTests(unittest.TestCase):
    def test_row_major_layout_and_orientation(self):
        frames = [np.full((2, 3, 3), index, dtype=np.uint8) for index in range(1, 5)]
        frames[0][0, 0] = [255, 0, 0]
        grid = make_grid(frames)
        self.assertEqual(grid.shape, (4, 6, 3))
        for index, frame in enumerate(frames):
            row, col = divmod(index, 2)
            np.testing.assert_array_equal(grid[row * 2:(row + 1) * 2, col * 3:(col + 1) * 3], frame)

    def test_unused_cells_are_black(self):
        grid = make_grid([np.ones((2, 3, 3), dtype=np.uint8)])
        self.assertFalse(grid[2:].any())
        self.assertFalse(grid[:2, 3:].any())

    def test_invalid_frames(self):
        for frames in ([], [None], [np.zeros((2, 3))], [np.zeros((2, 3, 3))]):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                make_grid(frames)
        with self.assertRaises(ValueError):
            make_grid([np.zeros((2, 3, 3), dtype=np.uint8)] * 5)

    def test_defaults(self):
        args = parse_args([])
        self.assertEqual(args.n_envs, 4)
        self.assertEqual(args.device, "cuda")
        self.assertFalse(args.headless)

    def test_headless_allows_more_environments(self):
        self.assertEqual(parse_args(["--headless", "--n-envs", "8"]).n_envs, 8)


if __name__ == "__main__":
    unittest.main()
