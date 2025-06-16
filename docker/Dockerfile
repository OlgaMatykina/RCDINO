FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

RUN DEBIAN_FRONTEND=noninteractive TZ=Europe/UTC apt-get update 
RUN DEBIAN_FRONTEND=noninteractive TZ=Europe/UTC apt-get install gcc g++ make nvidia-cuda-toolkit -y

RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    sudo

ARG USER=docker_rctrans
ARG UID=1001
ARG GID=1001
ARG PW=user
RUN id

RUN sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends python3.8-dev git wget unzip bzip2 sudo build-essential ca-certificates && \
    sudo apt-get install ffmpeg libsm6 libxext6  -y && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*

RUN sudo useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | sudo chpasswd && sudo adduser ${USER} sudo
RUN id
USER ${USER}
WORKDIR /home/${USER}
USER ${UID}:${UID}

RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN echo $CUDA_HOME
ENV CUDA_HOME=/usr/local/cuda-11.1

RUN pip install flash-attn==0.2.2 --no-build-isolation && \
    pip install mmdet==2.28.2 && \
    pip install mmsegmentation==0.30.0

RUN git clone https://github.com/OlgaMatykina/RCDINO

RUN cd ./RCDINO/mmdetection3d && \
    pip install -v -e . && \
    cd ..

RUN pip install ipython && \
    pip install fvcore && \
    pip install spconv-cu111==2.1.21 && \
    pip install yapf==0.40.0 && \
    pip install setuptools==59.5.0 && \
    pip install ccimport==0.3.7 && \
    pip install pccm==0.3.4 && \
    pip install timm && \
    pip3 install wandb
    pip install loguru

RUN pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html && \
    pip install mmengine