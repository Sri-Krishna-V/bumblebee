# bumblebee GPU image for bare-metal/SSH GPU boxes.
#
#   docker build -t bumblebee .
#   docker run --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface \
#     bumblebee --source s3://bucket/in --target s3://bucket/out
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, reproducible installs and Python 3.12 management.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN uv venv --python 3.12 /opt/venv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# Install the GPU stack + cloud storage backends from the checked-in uv lockfile.
RUN uv sync --frozen --no-dev --extra gpu --extra trt --extra azure --extra s3 --extra gcs

ENV HF_XET_HIGH_PERFORMANCE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    VLLM_MEDIA_LOADING_THREAD_COUNT=8 \
    BUMBLEBEE_TRT_LAYOUT_CACHE=/root/.cache/vllm/trt_layout

CMD ["bumblebee", "--help"]
