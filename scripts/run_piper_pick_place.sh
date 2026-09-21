#!/usr/bin/env bash
set -euo pipefail

# Default Piper command: one managed GPU container owns Warp/PPO, its child
# TensorBoard, and (when enabled) the four-frame GLFW preview.
piper_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
piper_image="${PIPER_RSL_IMAGE:-localhost/piper-rsl:gpu}"
piper_headless=false
piper_tensorboard=true
piper_tensorboard_port=6006
piper_help=false

piper_port_is_listening() {
    local piper_port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -H -ltn 2>/dev/null | awk -v port="$piper_port" '$4 ~ (":" port "$") { found = 1 } END { exit !found }'
        return $?
    fi
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$piper_port" -sTCP:LISTEN >/dev/null 2>&1
        return $?
    fi
    # This fallback only probes; it never sends data or modifies the service.
    if command -v timeout >/dev/null 2>&1; then
        timeout 1 bash -c "</dev/tcp/127.0.0.1/$piper_port" >/dev/null 2>&1
        return $?
    fi
    return 1
}

piper_arg=1
while ((piper_arg <= $#)); do
    piper_value="${!piper_arg}"
    case "$piper_value" in
        --headless) piper_headless=true ;;
        --no-tensorboard) piper_tensorboard=false ;;
        --tensorboard-port=*) piper_tensorboard_port="${piper_value#--tensorboard-port=}" ;;
        --tensorboard-port)
            piper_next=$((piper_arg + 1))
            if ((piper_next > $#)); then
                echo "--tensorboard-port requires a value" >&2
                exit 2
            fi
            piper_tensorboard_port="${!piper_next}"
            piper_arg=$piper_next
            ;;
        -h|--help)
            # Help is a parser-only operation and needs neither GPU access,
            # WSLg nor a host port reservation.
            piper_help=true
            piper_headless=true
            piper_tensorboard=false
            ;;
    esac
    ((piper_arg += 1))
done

if [[ ! "$piper_tensorboard_port" =~ ^[0-9]+$ ]] ||
   ((10#$piper_tensorboard_port < 1024 || 10#$piper_tensorboard_port > 65535)); then
    echo "TensorBoard port must be an integer from 1024 through 65535 (got '$piper_tensorboard_port')." >&2
    exit 2
fi

if [[ "$piper_tensorboard" == true ]] && piper_port_is_listening "$piper_tensorboard_port"; then
    echo "TensorBoard host port $piper_tensorboard_port is already in use; " \
         "choose --tensorboard-port <free-port> or use --no-tensorboard. " \
         "No existing process was stopped." >&2
    exit 2
fi

mkdir -p "$piper_project_dir/.cache/warp"
piper_options=(
    --rm --init --userns=keep-id --security-opt=label=disable
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1
    -e WARP_CACHE_PATH=/workspace/.cache/warp
    -v "$piper_project_dir:/workspace" -w /workspace
)
if [[ "$piper_help" == false ]]; then
    piper_options+=(--device nvidia.com/gpu=all)
fi
if [[ "$piper_tensorboard" == true ]]; then
    # The host side is deliberately loopback-only. TensorBoard listens on the
    # container interface because Podman forwards to the container namespace.
    piper_options+=(-p "127.0.0.1:${piper_tensorboard_port}:${piper_tensorboard_port}")
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

exec podman run "${piper_options[@]}" "$piper_image" \
    python -m scripts.train_piper_rsl "$@"
