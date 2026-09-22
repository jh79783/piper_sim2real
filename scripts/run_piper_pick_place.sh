#!/usr/bin/env bash
set -euo pipefail

# Default Piper command: one managed GPU container owns Warp/PPO, its child
# TensorBoard, and (when enabled) the four-frame GLFW preview.
piper_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
piper_image="${PIPER_RSL_IMAGE:-}"
piper_rgb=false
piper_headless=false
piper_tensorboard=true
piper_tensorboard_port=6006
piper_renderer="${PIPER_RENDERER:-gpu}"
piper_wsl_root="${PIPER_WSL_ROOT:-/usr/lib/wsl}"
piper_dxg_device="${PIPER_DXG_DEVICE:-/dev/dxg}"
piper_display="${DISPLAY:-}"
piper_display_for_container=""
piper_xauthority="${XAUTHORITY:-${HOME:-}/.Xauthority}"
piper_xauthority_temp=""
piper_wslg_gui=false
piper_help=false

piper_cleanup_xauthority() {
    if [[ -n "$piper_xauthority_temp" ]]; then
        rm -f -- "$piper_xauthority_temp"
        piper_xauthority_temp=""
    fi
}

trap piper_cleanup_xauthority EXIT

case "$piper_renderer" in
    gpu|software) ;;
    *)
        echo "PIPER_RENDERER must be 'gpu' (default) or 'software' (got '$piper_renderer')." >&2
        exit 2
        ;;
esac

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
        --rgb) piper_rgb=true ;;
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

if [[ -z "$piper_image" ]]; then
    if [[ "$piper_rgb" == true ]]; then
        piper_image="piper-rsl:vision"
    else
        piper_image="piper-rsl:gpu"
    fi
fi

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
    --rm --init --user "$(id -u):$(id -g)"
    -e HOME=/tmp -e XDG_CACHE_HOME=/workspace/.cache
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1
    -e WARP_CACHE_PATH=/workspace/.cache/warp
    -v "$piper_project_dir:/workspace" -w /workspace
)
if [[ "$piper_help" == false ]]; then
    piper_options+=(--gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display)
fi
if [[ "$piper_tensorboard" == true ]]; then
    # The host side is deliberately loopback-only. TensorBoard listens on the
    # container interface because Docker forwards to the container namespace.
    piper_options+=(-p "127.0.0.1:${piper_tensorboard_port}:${piper_tensorboard_port}")
fi
if [[ "$piper_headless" == false ]]; then
    if [[ -z "$piper_display" ]]; then
        echo "DISPLAY is unset; set DISPLAY to a local X11 display or use --headless." >&2
        exit 1
    fi
    # Resolve the local X11 socket from DISPLAY instead of assuming WSLg's
    # :0.  Native Ubuntu/Xrdp sessions commonly use :10 or another display.
    case "$piper_display" in
        :*|unix:*|localhost:*)
            piper_display_suffix="${piper_display##*:}"
            piper_display_number="$piper_display_suffix"
            piper_display_number="${piper_display_number%%.*}"
            if [[ ! "$piper_display_number" =~ ^[0-9]+$ ]]; then
                echo "Unsupported local DISPLAY '$piper_display'; expected :N or localhost:N." >&2
                exit 1
            fi
            piper_x11_socket="/tmp/.X11-unix/X${piper_display_number}"
            if [[ ! -S "$piper_x11_socket" ]]; then
                echo "X11 socket $piper_x11_socket for DISPLAY=$piper_display is missing; " \
                    "use --headless or start the display server." >&2
                exit 1
            fi
            piper_display_screen=""
            if [[ "$piper_display_suffix" == *.* ]]; then
                piper_display_screen_number="${piper_display_suffix#*.}"
                if [[ ! "$piper_display_screen_number" =~ ^[0-9]+$ ]]; then
                    echo "Unsupported local DISPLAY '$piper_display'; screen suffix must be numeric." >&2
                    exit 1
                fi
                piper_display_screen=".${piper_display_screen_number}"
            fi
            # Normalize :N, unix/:N, and localhost:N to the mounted Unix
            # socket. Passing localhost:N through would make Xlib use TCP
            # inside the container, where the host's localhost is absent.
            piper_display_for_container=":${piper_display_number}${piper_display_screen}"
            ;;
        *)
            echo "Unsupported DISPLAY '$piper_display'; remote TCP X11 is not enabled by this launcher." >&2
            exit 1
            ;;
    esac
    # WSLg's X server can be configured to authorize local socket peers
    # without a cookie. Preserve that existing path; native X11 still
    # requires a scoped cookie and fails clearly when one is unavailable.
    if [[ "$piper_display_number" == "0" &&
          -r "$piper_wsl_root/lib/libd3d12.so" && -r "$piper_wsl_root/lib/libdxcore.so" ]]; then
        piper_wslg_gui=true
    fi
    if [[ -n "$piper_xauthority" && -f "$piper_xauthority" && -r "$piper_xauthority" ]]; then
        if ! command -v xauth >/dev/null 2>&1; then
            echo "xauth is required to create a scoped X11 authority file for DISPLAY=$piper_display; " \
                "install xauth or use --headless." >&2
            exit 1
        fi
        # FamilyLocal records encode the host name. Docker gives the training
        # container a different hostname, so export only the selected
        # display's cookie as FamilyWild instead of mounting the host authority
        # database unchanged or weakening the X server with xhost.
        piper_xauth_listing="$(xauth -f "$piper_xauthority" nlist "$piper_display_for_container" 2>/dev/null || true)"
        if [[ -n "$piper_xauth_listing" ]]; then
            piper_xauthority_temp="$(mktemp "${TMPDIR:-/tmp}/piper-xauth.XXXXXX")"
            if ! printf '%s\n' "$piper_xauth_listing" | \
                sed -E 's/^[[:xdigit:]]{4} /ffff /' | \
                xauth -f "$piper_xauthority_temp" nmerge -; then
                echo "Could not extract Xauthority for DISPLAY=$piper_display; use --headless." >&2
                exit 1
            fi
            piper_xauthority="$piper_xauthority_temp"
        elif [[ "$piper_wslg_gui" == false ]]; then
            echo "No Xauthority cookie was found for DISPLAY=$piper_display; use --headless." >&2
            exit 1
        else
            piper_xauthority=""
        fi
    elif [[ "$piper_wslg_gui" == false ]]; then
        echo "Xauthority file '$piper_xauthority' is missing or unreadable for DISPLAY=$piper_display; " \
            "set XAUTHORITY or use --headless. No xhost change is performed." >&2
        exit 1
    else
        piper_xauthority=""
    fi

    # MuJoCo selects one OpenGL backend per Python process.  RGB policy frames
    # stay on EGL even with a GUI; the preview process initializes its own
    # GLFW backend and receives already-rendered pixels through a one-frame
    # drop-old queue. State-only GUI runs retain the original single GLFW path.
    piper_options+=(
        -e "DISPLAY=$piper_display_for_container"
        -v /tmp/.X11-unix:/tmp/.X11-unix:ro
    )
    if [[ -n "$piper_xauthority" ]]; then
        piper_options+=(-e XAUTHORITY=/tmp/piper.xauthority -v "$piper_xauthority:/tmp/piper.xauthority:ro")
    fi
    if [[ "$piper_rgb" == true ]]; then
        piper_options+=(-e MUJOCO_GL=egl)
        if [[ "$piper_renderer" == software ]]; then
            piper_options+=(-e PIPER_PREVIEW_SOFTWARE=1)
        fi
    else
        piper_options+=(-e MUJOCO_GL=glfw)
    fi
    if [[ "$piper_renderer" == gpu && -e "$piper_dxg_device" &&
          -r "$piper_wsl_root/lib/libd3d12.so" && -r "$piper_wsl_root/lib/libdxcore.so" ]]; then
        # WSLg's GPU path keeps the existing D3D12/GLFW integration for the
        # isolated preview. RGB policy EGL must not inherit these GLX/Mesa
        # selectors, because its context is created in the parent process.
        piper_options+=(
            --device "$piper_dxg_device:/dev/dxg"
            -v "$piper_wsl_root:/usr/lib/wsl:ro"
        )
        if [[ "$piper_rgb" == true ]]; then
            piper_options+=(-e PIPER_PREVIEW_WSLG=1)
        else
            piper_options+=(
                -e GALLIUM_DRIVER=d3d12
                -e MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
                -e LD_LIBRARY_PATH=/usr/lib/wsl/lib
            )
        fi
    elif [[ "$piper_renderer" == software ]]; then
        # State-only software GUI has one GLFW backend. RGB software applies
        # only to the isolated preview; the policy remains on GPU EGL.
        if [[ "$piper_rgb" == false ]]; then
            piper_options+=(-e LIBGL_ALWAYS_SOFTWARE=1)
        fi
    fi
fi
if [[ -t 0 && -t 1 ]]; then piper_options+=(-it); fi

docker run "${piper_options[@]}" "$piper_image" \
    python -m scripts.train_piper_rsl "$@"
piper_exit=$?
exit "$piper_exit"
