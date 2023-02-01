FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# To fix GPG key error when running apt-get update
RUN apt-get update && apt-get install -y gnupg2
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# (Optional)
RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN apt-get update \
    && apt-get install -y gcc ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install openmim \
    && mim install "mmengine>=0.3.1" \
    && mim install "mmcv>=2.0.0rc1,<2.1.0" \
    && mim install "mmdet>=3.0.0rc5,<3.1.0"

# Install MMYOLO
COPY . /app
RUN cd /app && \
    pip install --no-cache-dir -r requirements.txt && \
    mkdir -p /img-man && \
    mv ymir/img-man/*.yaml /img-man && \
    mkdir /weights && \
    mv ymir/weights/*.pth /weights && \
    mkdir -p /root/.cache/torch/hub/checkpoints && \
    mv ymir/weights/imagenet/*.pth /root/.cache/torch/hub/checkpoints

ENV PYTHONPATH=.
WORKDIR /app

RUN echo "python3 ymir/start.py" > /usr/bin/start.sh
CMD bash /usr/bin/start.sh
