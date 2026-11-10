# Image nộp bài ZAC2026 LifeMate. Dựng bằng vm_build.sh (cần weights đã gộp trong saved_models/).
# CHƯA kiểm chứng trên RTX 5090: tag CUDA và wheel torch lấy theo Docker Guide / môi trường Vast, phải thử thật.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y python3 python3-pip libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*
ARG TORCH_SPEC="torch==2.9.1"
ARG TORCH_INDEX="https://download.pytorch.org/whl/cu128"
RUN pip3 install ${TORCH_SPEC} --index-url ${TORCH_INDEX}
COPY requirements.txt /code/requirements.txt
RUN pip3 install -r /code/requirements.txt
COPY . /code/
WORKDIR /code
