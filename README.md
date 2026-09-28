# piper_sim2real

AgileX Piper sim2real on Isaac Sim 5.1 / Isaac Lab 2.3.2.

## Docker

The dev container is long-lived: start it once, then `exec` into it. Code is
bind-mounted from the host; Kit shader/kernel caches, downloaded assets and the
home directory live in named volumes, so `down`/`up` keeps them.

```bash
bash scripts/docker.sh build                           # once, or after Dockerfile/requirements change
bash scripts/docker.sh up                              # start the container
bash scripts/docker.sh shell                           # interactive shell
bash scripts/docker.sh exec python scripts/check_env.py  # headless GPU smoke test
bash scripts/docker.sh run python scripts/check_env.py   # same, in a throwaway container
bash scripts/docker.sh down                            # stop (volumes kept)
```

Inside the container `python`, `pip` and `tensorboard` point at Isaac Sim's
bundled interpreter, and the process runs as the host UID so files written to
the project stay host-owned. Add Python dependencies to
`docker/requirements.txt` and rebuild rather than `pip install`-ing into the
running container.

GUI: the container gets the host `DISPLAY`, X11 socket and Xauthority cookie,
so Isaac Lab scripts launched without `--headless` open a window on the host
display. Training should stay headless.

## Piper asset

`assets/piper_description` holds the official AgileX URDF and meshes
(see its README for the upstream commit and local changes).
`piper_sim2real.assets.PIPER_CFG` defines the articulation; the URDF is converted to USD
on first spawn and cached under `.cache/assets/piper`.

```bash
bash scripts/docker.sh exec python scripts/spawn_piper.py                   # GUI, joint sweep
bash scripts/docker.sh exec python scripts/spawn_piper.py --headless \
    --steps 300 --snapshot runs/spawn_piper.png                             # headless screenshot
```

## Layout

Generated from the Isaac Lab external-project template (`isaaclab.sh --new`).

```
source/piper_sim2real/          # Isaac Lab extension, pip install -e'd in the image
  piper_sim2real/assets/        # robot configs (PIPER_CFG)
  piper_sim2real/tasks/manager_based/<task>/   # env cfg + agents/, gym.register in __init__.py
scripts/rsl_rl/                 # template train.py / play.py
scripts/list_envs.py            # registered Piper-* tasks
scripts/spawn_piper.py          # spawn + joint sweep
assets/piper_description/       # AgileX URDF + meshes
docker/                         # Dockerfile, compose
```

Rebuild the image after changing `source/piper_sim2real/setup.py`; Python
sources are imported live from the bind mount.
