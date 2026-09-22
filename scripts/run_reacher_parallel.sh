#!/usr/bin/env bash
set -euo pipefail

piper_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
piper_image="${PIPER_RL_IMAGE:-piper-rl:gpu}"
piper_headless=false
piper_device=cuda
for ((piper_arg=1; piper_arg<=$#; piper_arg++)); do
    piper_value="${!piper_arg}"
    case "$piper_value" in
        --headless) piper_headless=true ;;
        --device=*) piper_device="${piper_value#--device=}" ;;
        --device)
            piper_next=$((piper_arg + 1))
            if ((piper_next <= $#)); then piper_device="${!piper_next}"; fi
            ;;
        -h|--help)
            # Help needs neither GPU access nor an X11 socket.
            piper_headless=true
            piper_device=cpu
            ;;
    esac
done

mkdir -p "$piper_project_dir/.cache"
piper_options=(--rm --init --user "$(id -u):$(id -g)"
    -e HOME=/tmp -e XDG_CACHE_HOME=/workspace/.cache
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1
    -v "$piper_project_dir:/workspace" -w /workspace)
if [[ "$piper_device" == cuda ]]; then
    piper_options+=(--gpus all)
fi
if [[ "$piper_headless" == false ]]; then
    if [[ ! -S /tmp/.X11-unix/X0 ]]; then
        echo "WSLg X11 socket /tmp/.X11-unix/X0 is missing. Reopen WSL or use --headless." >&2
        exit 1
    fi
    piper_options+=(-e DISPLAY=:0 -e MUJOCO_GL=glfw -e LIBGL_ALWAYS_SOFTWARE=1
        -v /tmp/.X11-unix:/tmp/.X11-unix:ro)
fi
if [[ -t 0 && -t 1 ]]; then piper_options+=(-it); fi

exec docker run "${piper_options[@]}" "$piper_image" \
    python -m scripts.train_reacher_parallel "$@"
