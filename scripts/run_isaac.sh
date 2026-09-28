#!/usr/bin/env bash
set -euo pipefail

# Isaac Sim launcher: every command runs inside the piper-isaac image as the
# host UID, with the project bind-mounted at /workspace/piper_sim2real.
#
#   bash scripts/run_isaac.sh convert [--force]      # Menagerie MJCF -> USD
#   bash scripts/run_isaac.sh train [args...]        # headless RSL-RL PPO
#   bash scripts/run_isaac.sh play  [args...]        # roll out a checkpoint
#   bash scripts/run_isaac.sh test  [pytest args]    # Isaac-side tests
#   bash scripts/run_isaac.sh tensorboard [--port N] # view runs/isaac
#   bash scripts/run_isaac.sh python <file> [args]   # any script in the image

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image="${PIPER_ISAAC_IMAGE:-piper-isaac:5.1}"
container_project=/workspace/piper_sim2real
isaac_python=(/workspace/isaaclab/isaaclab.sh -p)

usage() {
    sed -n '4,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

command="${1:-}"
if [[ -z "$command" || "$command" == "-h" || "$command" == "--help" ]]; then
    usage
    exit 0
fi
shift

docker_args=()
tensorboard_port=6006
case "$command" in
    convert) run=("${isaac_python[@]}" scripts/isaac/convert_piper_asset.py "$@") ;;
    train) run=("${isaac_python[@]}" scripts/isaac/train.py "$@") ;;
    play) run=("${isaac_python[@]}" scripts/isaac/play.py "$@") ;;
    test) run=("${isaac_python[@]}" -m pytest -q tests/isaac "$@") ;;
    python)
        if [[ $# -lt 1 ]]; then
            echo "python requires a script path" >&2
            exit 2
        fi
        run=("${isaac_python[@]}" "$@")
        ;;
    tensorboard)
        if [[ "${1:-}" == "--port" ]]; then
            tensorboard_port="${2:?--port requires a value}"
            shift 2
        fi
        docker_args+=(-p "127.0.0.1:${tensorboard_port}:${tensorboard_port}")
        run=("${isaac_python[@]}" -m tensorboard.main --logdir runs/isaac
             --host 0.0.0.0 --port "$tensorboard_port" "$@")
        ;;
    *)
        echo "unknown command: $command" >&2
        usage >&2
        exit 2
        ;;
esac

if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "Docker image '$image' not found. Build it first:" >&2
    echo "  docker build -t $image -f Dockerfile.isaac ." >&2
    exit 1
fi

# Persistent caches (shader/kernel/texture) keep the second launch fast.
cache_dir="$project_dir/.cache/isaac"
mkdir -p "$cache_dir"/{home,kit-cache,kit-logs} "$project_dir/runs/isaac"

tty_args=()
if [[ -t 0 && -t 1 ]]; then
    tty_args=(-it)
fi

exec docker run --rm "${tty_args[@]}" \
    --gpus all \
    --ipc host \
    --user "$(id -u):$(id -g)" \
    -e HOME="$container_project/.cache/isaac/home" \
    -e PYTHONPATH="$container_project" \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -v "$project_dir:$container_project" \
    -v "$cache_dir/kit-cache:/isaac-sim/kit/cache" \
    -v "$cache_dir/kit-logs:/isaac-sim/kit/logs" \
    -w "$container_project" \
    "${docker_args[@]}" \
    "$image" \
    "${run[@]}"
