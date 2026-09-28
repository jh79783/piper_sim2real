# Isaac Sim 5.1 + Isaac Lab 2.3.2 development image for Piper sim2real.
#
# The NGC Isaac Lab image already ships Isaac Sim 5.1, Isaac Lab 2.3.2,
# torch 2.7.0+cu128 (RTX 50 capable) and rsl-rl-lib.  This layer only adds a
# non-root user matching the host UID/GID, so files written to the bind-mounted
# project stay owned by the host user, and python/pip wrappers that point at
# Isaac Sim's bundled interpreter.
FROM nvcr.io/nvidia/isaac-lab:2.3.2

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    PYTHONUNBUFFERED=1 \
    ISAACSIM_PATH=/isaac-sim \
    ISAACLAB_PATH=/workspace/isaaclab \
    PROJECT_DIR=/workspace/piper_sim2real \
    PYTHONPATH=/workspace/piper_sim2real

RUN apt-get update \
    && apt-get install -y --no-install-recommends sudo git less vim \
    && rm -rf /var/lib/apt/lists/*

# Reuse whatever account already owns the host UID/GID (Ubuntu 24.04 images
# ship "ubuntu" as 1000:1000) by renaming it, otherwise create a fresh one.
RUN set -eux; \
    if getent group "${USER_GID}" >/dev/null; then \
        groupmod -n "${USERNAME}" "$(getent group "${USER_GID}" | cut -d: -f1)"; \
    else \
        groupadd -g "${USER_GID}" "${USERNAME}"; \
    fi; \
    if getent passwd "${USER_UID}" >/dev/null; then \
        usermod -l "${USERNAME}" -g "${USER_GID}" -d "/home/${USERNAME}" -m -s /bin/bash \
            "$(getent passwd "${USER_UID}" | cut -d: -f1)"; \
    else \
        useradd -m -u "${USER_UID}" -g "${USER_GID}" -s /bin/bash "${USERNAME}"; \
    fi; \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}"; \
    chmod 0440 "/etc/sudoers.d/${USERNAME}"

# The NGC image ships /isaac-sim as 0750 (uid 1234) and Kit writes its caches
# and logs inside the install tree.  Open the top directory and hand the
# writable Kit paths to the dev user; these paths are also volume mount points,
# and Docker seeds a fresh named volume with this ownership.
RUN set -eux; \
    chmod a+rx /isaac-sim; \
    mkdir -p /isaac-sim/kit/cache /isaac-sim/kit/logs /isaac-sim/kit/data; \
    chown -R "${USER_UID}:${USER_GID}" /isaac-sim/kit/cache /isaac-sim/kit/logs /isaac-sim/kit/data; \
    chmod a+rwX /workspace/isaaclab; \
    mkdir -p "/home/${USERNAME}/.cache" "/home/${USERNAME}/.nv" \
             "/home/${USERNAME}/.nvidia-omniverse" "/home/${USERNAME}/.local/share"; \
    chown -R "${USER_UID}:${USER_GID}" "/home/${USERNAME}"

# There is no system python in the image; route python/pip to Isaac Sim's
# interpreter so non-interactive `docker compose exec isaac python ...` works.
RUN set -eux; \
    for name in python python3; do \
        printf '#!/bin/sh\nexec /isaac-sim/python.sh "$@"\n' > "/usr/local/bin/${name}"; \
    done; \
    for name in pip pip3; do \
        printf '#!/bin/sh\nexec /isaac-sim/python.sh -m pip "$@"\n' > "/usr/local/bin/${name}"; \
    done; \
    printf '#!/bin/sh\nexec /isaac-sim/python.sh -m tensorboard.main "$@"\n' > /usr/local/bin/tensorboard; \
    ln -s /workspace/isaaclab/isaaclab.sh /usr/local/bin/isaaclab; \
    chmod a+rx /usr/local/bin/python /usr/local/bin/python3 /usr/local/bin/pip \
               /usr/local/bin/pip3 /usr/local/bin/tensorboard

# Project dependencies are baked into the image so every container starts
# from the same environment.  Packages installed ad hoc inside a running
# container land in the home volume (pip --user) and should be copied here.
COPY docker/requirements.txt /tmp/requirements.txt
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

USER ${USERNAME}
WORKDIR /workspace/piper_sim2real
ENTRYPOINT []
CMD ["sleep", "infinity"]
