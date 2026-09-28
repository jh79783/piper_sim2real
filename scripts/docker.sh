#!/usr/bin/env bash
set -euo pipefail

# Isaac Sim dev container helper.
#
#   bash scripts/docker.sh build            # build the image (host UID/GID baked in)
#   bash scripts/docker.sh up               # start the long-lived container
#   bash scripts/docker.sh shell            # bash inside it (starts it if needed)
#   bash scripts/docker.sh exec <cmd...>    # run a command in the running container
#   bash scripts/docker.sh run <cmd...>     # one-off throwaway container (--rm)
#   bash scripts/docker.sh down             # stop and remove the container (volumes kept)
#   bash scripts/docker.sh status           # container state
#   bash scripts/docker.sh purge            # remove the container AND cache volumes

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
compose=(docker compose -f "$project_dir/docker/compose.yaml")
service=isaac

export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
xauth="${XAUTHORITY:-$HOME/.Xauthority}"
if [[ -f "$xauth" ]]; then
    export XAUTH_FILE="$xauth"
else
    export XAUTH_FILE=/dev/null
fi

usage() {
    sed -n '4,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

tty_args=()
if [[ ! -t 0 || ! -t 1 ]]; then
    tty_args=(-T)
fi

ensure_up() {
    if [[ -z "$("${compose[@]}" ps -q --status running "$service" 2>/dev/null)" ]]; then
        "${compose[@]}" up -d "$service"
    fi
}

command="${1:-}"
[[ $# -gt 0 ]] && shift
case "$command" in
    build) exec "${compose[@]}" build "$@" "$service" ;;
    up) exec "${compose[@]}" up -d "$@" "$service" ;;
    shell)
        ensure_up
        exec "${compose[@]}" exec "${tty_args[@]}" "$service" bash "$@"
        ;;
    exec)
        [[ $# -ge 1 ]] || { echo "exec requires a command" >&2; exit 2; }
        ensure_up
        exec "${compose[@]}" exec "${tty_args[@]}" "$service" "$@"
        ;;
    run)
        [[ $# -ge 1 ]] || { echo "run requires a command" >&2; exit 2; }
        exec "${compose[@]}" run --rm "${tty_args[@]}" "$service" "$@"
        ;;
    down) exec "${compose[@]}" down "$@" ;;
    status) exec "${compose[@]}" ps -a ;;
    purge)
        read -r -p "Remove the container and all cache volumes? [y/N] " answer
        [[ "$answer" == [yY] ]] || exit 1
        exec "${compose[@]}" down -v
        ;;
    ""|-h|--help) usage ;;
    *)
        echo "unknown command: $command" >&2
        usage >&2
        exit 2
        ;;
esac
