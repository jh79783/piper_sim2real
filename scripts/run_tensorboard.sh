#!/usr/bin/env bash
set -euo pipefail

piper_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
piper_image="${PIPER_RSL_IMAGE:-${PIPER_RL_IMAGE:-localhost/piper-rsl:gpu}}"
piper_port="${1:-6006}"
if [[ "${1:-}" == "--port" ]]; then piper_port="${2:-}"; fi
if [[ "${1:-}" == --port=* ]]; then piper_port="${1#--port=}"; fi
if [[ ! "$piper_port" =~ ^[0-9]+$ ]] || ((10#$piper_port < 1024 || 10#$piper_port > 65535)); then
    echo "Usage: bash scripts/run_tensorboard.sh [port|--port port: 1024-65535]" >&2
    exit 2
fi

piper_port_is_listening() {
    local piper_check_port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -H -ltn 2>/dev/null | awk -v port="$piper_check_port" '$4 ~ (":" port "$") { found = 1 } END { exit !found }'
        return $?
    fi
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$piper_check_port" -sTCP:LISTEN >/dev/null 2>&1
        return $?
    fi
    if command -v timeout >/dev/null 2>&1; then
        timeout 1 bash -c "</dev/tcp/127.0.0.1/$piper_check_port" >/dev/null 2>&1
        return $?
    fi
    return 1
}

if piper_port_is_listening "$piper_port"; then
    echo "TensorBoard host port $piper_port is already in use; choose another port. " \
         "No existing process was stopped." >&2
    exit 2
fi
mkdir -p "$piper_project_dir/runs"
echo "TensorBoard: http://localhost:$piper_port (Ctrl+C to stop)"
exec podman run --rm --init --userns=keep-id --security-opt=label=disable \
    -p "127.0.0.1:$piper_port:6006" \
    -v "$piper_project_dir/runs:/logs:ro" \
    "$piper_image" \
    python -m tensorboard.main --logdir /logs --host 0.0.0.0 --port 6006 --reload_interval 5
