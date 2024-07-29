FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

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

RUN set -x && apt -y install libgl1-mesa-dev libglib2.0-0 zip git
RUN set -x \
    && apt -y install python3.8 python3.8-dev \
    && ln -s /usr/bin/python3.8 /usr/bin/python \
    && apt -y install wget python3-distutils && wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && python -m pip install --upgrade pip==24.0

COPY . ${PROJECT_DIR}


RUN set -x \
    && pip install -r ${PROJECT_DIR}/requirements.txt
RUN set -x && pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

RUN set -x && pip uninstall -y mmdet
RUN set -x \
    && cd ${PROJECT_DIR}/src/ndl_kotenseki_layout \
    && git clone https://github.com/open-mmlab/mmdetection.git -b v3.0.0 \
    && cd ${PROJECT_DIR}/src/ndl_kotenseki_layout/mmdetection \
    && sed -i -e 's/GPU_MEM_LIMIT = 1024\*\*3/GPU_MEM_LIMIT = 1024\*\*3\/\/4/' mmdet/models/roi_heads/mask_heads/fcn_mask_head.py \
    && python setup.py bdist_wheel \
    && pip install dist/*.whl

RUN set -x && wget -nc https://lab.ndl.go.jp/dataset/ndlkotensekiocr/trocr/model-ver2.zip -P ${PROJECT_DIR}/src/text_kotenseki_recognition/
RUN set -x && wget -nc https://lab.ndl.go.jp/dataset/ndlkotensekiocr/layoutmodel/ndl_kotenseki_layout_ver3.pth -P ${PROJECT_DIR}/src/ndl_kotenseki_layout/models/
RUN set -x && cd ${PROJECT_DIR}/src/text_kotenseki_recognition/ && unzip -o model-ver2.zip
RUN set -x && pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

WORKDIR ${PROJECT_DIR}
