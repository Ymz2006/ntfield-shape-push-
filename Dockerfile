FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

# RTX 3090 is Ampere (sm_86)
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV CUDA_HOME=/usr/local/cuda
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    x11-apps \
    ffmpeg \
    v4l-utils \
    libusb-1.0-0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    && rm -rf /var/lib/apt/lists/*

# numpy stays on 1.x: the base image's torch 2.2.2 is built against it, and importing
# torch under numpy 2 fails outright.  opencv-contrib-python >= 4.12 hard-requires numpy
# >= 2, so it is pinned to the last 1.x-compatible release.
RUN pip install --no-cache-dir \
    "numpy<2" \
    libigl \
    pytorch_kinematics \
    configargparse \
    ezdxf \
    shapely \
    descartes \
    plotly \
    wandb \
    ur_rtde \
    "opencv-contrib-python==4.11.0.86" \
    pyrealsense2 \
    viser \
    pymunk

WORKDIR /workspace
