FROM docker.io/library/ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3 python3-venv \
    libgl1 libegl1 libglx-mesa0 libgl1-mesa-dri \
    libglfw3 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN python -m pip install --no-cache-dir \
    mujoco==3.3.7 \
    numpy scipy gymnasium imageio

RUN python -m pip install --no-cache-dir \
    torch==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

RUN python -m pip install --no-cache-dir \
    stable-baselines3==2.9.0 \
    tensorboard tqdm rich

WORKDIR /workspace
CMD ["bash"]
