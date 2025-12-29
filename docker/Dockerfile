FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"
ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender-dev \
    ffmpeg \
    libgl1-mesa-glx \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    alsa-utils \
    libasound2 \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libfreetype6 \
    libpng16-16 \
    libjpeg-dev \
    libbz2-dev \
    liblzma-dev \
    libopenmpi-dev \
    libgomp1 \
    libx11-6 \
    libzstd-dev \
    zlib1g-dev \
    ca-certificates \
    curl \
    gnupg \
    build-essential \
    libssl-dev \
    libffi-dev \
    ninja-build \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/local/bin/pip /usr/bin/pip

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install --ignore-installed blinker streamlit

RUN pip uninstall -y torch torchvision numpy || true

RUN pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124

RUN pip install numpy==1.26.4 scipy==1.13.1
RUN pip install fvcore iopath

RUN pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git

RUN pip install --no-build-isolation fast_jl || echo "Warning: fast_jl installation failed, will use fallback naive projection"

WORKDIR /workspace/src

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command to run the UI
# CMD ["streamlit", "run", "ui.py", "--server.address", "0.0.0.0"]
