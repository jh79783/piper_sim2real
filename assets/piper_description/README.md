# piper_description

AgileX Piper arm + parallel gripper description, vendored from
[agilexrobotics/piper_ros](https://github.com/agilexrobotics/piper_ros)
(`src/piper_description`, commit `ac41fcbcdda598f01b51cf6175ed9a24d0dacadc`, MIT, see `LICENSE`).

Changes from upstream:

- `urdf/piper_description.urdf`: `package://piper_description/meshes/` mesh paths
  rewritten to `../meshes/` so the Isaac Sim URDF importer resolves them without ROS.
- The massless, geometry-free `dummy_link` root and its fixed joint are removed, so
  `base_link` is the articulation root (the importer otherwise warns about its empty visuals).
- Only the STL meshes referenced by that URDF are copied.
