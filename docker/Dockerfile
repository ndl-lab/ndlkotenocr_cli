FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV PROJECT_DIR=/root/kotenocr_cli
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="7.5+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV DEBIAN_FRONTEND=noninteractive

RUN set -x \
    && apt update \
    && apt upgrade -y

RUN set -x \
    && apt update \
    && apt -y install locales \
    && locale-gen ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL=ja_JP.UTF-8
RUN localedef -f UTF-8 -i ja_JP ja_JP.utf8

RUN set -x && apt -y install libgl1-mesa-dev libglib2.0-0 zip git python3-pip wget

COPY . ${PROJECT_DIR}


RUN set -x && pip3 install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 torchtext==0.16.1 --index-url https://download.pytorch.org/whl/cu121
RUN set -x \
    && pip3 install -r ${PROJECT_DIR}/requirements.txt


RUN set -x && wget -nc https://lab.ndl.go.jp/dataset/ndlkotensekiocr/trocr/model-ver2.zip -P ${PROJECT_DIR}/src/text_kotenseki_recognition/
RUN set -x && wget -nc https://lab.ndl.go.jp/dataset/ndlkotensekiocr/layoutmodel/ndl_kotenseki_layout_ver3.pth -P ${PROJECT_DIR}/src/ndl_kotenseki_layout/models/
RUN set -x && cd ${PROJECT_DIR}/src/text_kotenseki_recognition/ && unzip -o model-ver2.zip
RUN set -x && pip3 install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

WORKDIR ${PROJECT_DIR}
